import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 关键：导入并应用patch，使模型使用自定义的MoE实现
from slime.backends.fsdp_utils.models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe


def apply_fsdp2(model, mesh=None, cpu_offload=False):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
           or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
        "reshard_after_forward": True,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model


def main():
    # 获取分布式信息
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}/{world_size}] Starting on GPU {local_rank}")

    model_path = "/sgl-workspace/Qwen3-30B-A3B"
    # model_path = "/mnt/o1_alicloud/personal/zzl/hf_checkpoint/Qwen3-4B"

    if rank == 0:
        print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "<|im_start|>system\nYou are a helpful assistant. Please put the answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nStefan goes to a restaurant to eat dinner with his family. They order an appetizer that costs $10 and 4 entrees that are $20 each. If they tip 20% of the total for the waiter, what is the total amount of money that they spend at the restaurant?<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor(tokens, dtype=torch.long).cuda()

    # 关键步骤：在加载模型前应用patch，替换transformers的MoE实现为自定义实现
    enable_self_moe = os.environ.get("ENABLE_SELF_MOE", "false").lower() == "true"
    if rank == 0:
        if enable_self_moe:
            print("Applying custom MoE patch...")
        else:
            print("Using default transformers MoE implementation (ENABLE_SELF_MOE=false)")

    if enable_self_moe:
        apply_true_on_policy_patch_for_qwen3_moe()
        if rank == 0:
            print("✓ Custom MoE patch applied - will use fused_experts_impl with custom forward/backward")
    else:
        if rank == 0:
            print("✓ Using standard MoE implementation")

    # Use bfloat16 as it is common for newer models
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,  # 添加这行，避免在 CPU 上加载 float32
    )

    # 先不移动到 CUDA，让 FSDP 处理
    model.train()

    if rank == 0:
        print("Applying FSDP...")
    model = apply_fsdp2(model, cpu_offload=False)

    if rank == 0:
        print("Running forward pass...")

    # Measure forward pass time
    torch.cuda.synchronize()
    forward_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids=tokens.unsqueeze(0))
        logits = outputs.logits.squeeze(0)
        logits = logits.to(torch.float32)

        # Calculate log probabilities
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        # Logits shape: [batch_size, seq_len, vocab_size]
        # We want the logits for the last token in the sequence to predict the next token
        next_token_logits = logits[-1, :]

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        # Get top 1
        top1_logprob, top1_token_id = torch.max(log_probs, dim=-1)
        top1_token = tokenizer.decode(top1_token_id)

    torch.cuda.synchronize()
    forward_end = time.perf_counter()
    forward_time = forward_end - forward_start

    if rank == 0:
        print("-" * 20)
        print(f"Next Token Prediction:")
        print(f"Token ID: {top1_token_id.item()}")
        print(f"Token: '{top1_token}'")
        print(f"Logprob: {top1_logprob.item()}")
        print(f"Forward Time: {forward_time:.6f} seconds ({forward_time*1000:.2f} ms)")
        print("-" * 20)

    # Test backward pass to verify gradients computation
    if rank == 0:
        print("\nTesting backward pass...")
    model.zero_grad()

    # Forward with gradients enabled
    outputs = model(input_ids=tokens.unsqueeze(0))
    loss = outputs.logits.sum()

    if rank == 0:
        print("-" * 20)
        print(f"Backward Pass Debug Info:")
        print(f"Loss value: {loss.item()}")
        print(f"Loss dtype: {loss.dtype}")
        print(f"Loss shape: {loss.shape}")
        print("-" * 20)
        print("Running backward...")

    # Measure backward pass time
    torch.cuda.synchronize()
    backward_start = time.perf_counter()

    loss.backward()

    torch.cuda.synchronize()
    backward_end = time.perf_counter()
    backward_time = backward_end - backward_start

    if rank == 0:
        print("-" * 20)
        print("✓ Backward pass completed - custom backward functions were called")

        # Print gradient statistics
        total_grad_norm = 0.0
        num_params_with_grad = 0
        max_grad = float('-inf')
        min_grad = float('inf')

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                num_params_with_grad += 1
                max_grad = max(max_grad, param.grad.max().item())
                min_grad = min(min_grad, param.grad.min().item())

        total_grad_norm = total_grad_norm ** 0.5

        print(f"Gradient Statistics:")
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        print(f"  Params with gradients: {num_params_with_grad}")
        print(f"  Max gradient value: {max_grad:.6f}")
        print(f"  Min gradient value: {min_grad:.6f}")
        print(f"Backward Time: {backward_time:.6f} seconds ({backward_time*1000:.2f} ms)")
        print("-" * 20)

    # 清理
    torch.distributed.barrier()
    if rank == 0:
        print("All ranks completed successfully!")


if __name__ == "__main__":
    # 设置环境变量（这些会被 torchrun 自动设置，但保留作为默认值）
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 不要手动设置这些，让 torchrun 自动设置
    # os.environ["WORLD_SIZE"] = "2"
    # os.environ["RANK"] = "0"
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "29500"

    # 获取 LOCAL_RANK（由 torchrun 自动设置）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 初始化进程组
    torch.distributed.init_process_group(backend="nccl")

    # deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    enable_batch_invariant_mode(
        # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
        # and disabling it will make it aligned
        enable_bmm=False,
    )

    try:
        main()
    finally:
        # 清理进程组
        torch.distributed.destroy_process_group()
