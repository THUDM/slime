"""Tiny Qwen3 MoE, real Megatron training and P2P SGLang updates.

Requires RDMA-capable GPUs and Mooncake in the test image. Generates a random
checkpoint/tokenizer locally, so the test needs no model or dataset download.
The rollout hook compares serving logprobs to the matching HF checkpoint after
successive optimizer steps and verifies that expert weights actually changed.
Uses four GPUs normally, or six with --pd (two each for train/prefill/decode).
--offload-rollout also exercises KV deregistration and registration on resume.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

NUM_GPUS = 6
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

    model_path = directory / "model"
    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(model_path)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({str(i): i + 4 for i in range(252)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>"
    ).save_pretrained(model_path)
    with (directory / "prompts.jsonl").open("w") as stream:
        for i in range(12):
            stream.write(json.dumps({"prompt": f"{i} {i + 1} {i + 2} {i + 3}"}) + "\n")
    return model_path


async def reward(args, sample, **kwargs):
    return float(sum(sample.tokens[-4:]) % 7)


def check_rollout(rollout_id, args, samples, extra_metrics, rollout_time):
    import torch
    from transformers import AutoModelForCausalLM

    # save() runs before each post-training update, so this is independent of
    # the P2P sender and receiver and covers native inference TP/EP slicing.
    directory = Path(args.hf_checkpoint).parent
    previous = directory / f"hf-{rollout_id - 1}"
    checkpoint = previous if previous.exists() else Path(args.hf_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    assert samples and all(sample.response_length > 0 for sample in samples)
    max_error = 0.0
    with torch.no_grad():
        for sample in samples:
            assert sample.weight_versions and set(sample.weight_versions) == {str(rollout_id + 1)}
            tokens = torch.tensor([sample.tokens])
            logits = model(tokens).logits[0, :-1].float()
            expected = logits.log_softmax(-1).gather(1, tokens[0, 1:, None]).squeeze(-1)
            expected = expected[-sample.response_length :]
            actual = torch.tensor(sample.rollout_log_probs)
            assert actual.shape == expected.shape
            error = (actual - expected).abs().max().item()
            assert error < 0.06, (rollout_id, error)
            max_error = max(max_error, error)
    if previous.exists():
        initial = AutoModelForCausalLM.from_pretrained(args.hf_checkpoint, torch_dtype=torch.float32)
        experts = {name: param for name, param in model.named_parameters() if ".experts." in name}
        original = dict(initial.named_parameters())
        assert experts and any(not torch.equal(param, original[name]) for name, param in experts.items())
    with (directory / "verified.jsonl").open("a") as stream:
        stream.write(json.dumps({"rollout": rollout_id, "checkpoint": checkpoint.name, "max_error": max_error}) + "\n")
    return False


def execute(directory, *, pd=False, offload_rollout=False):
    import ray

    from slime.utils.p2p.mooncake import MooncakeTransport

    # Fail before starting expensive model processes on a runner without RDMA.
    transport = MooncakeTransport("127.0.0.1", device="cuda:0")
    transport.reset()
    directory.mkdir(parents=True, exist_ok=True)
    model = prepare(directory)
    # All endpoints are local to this test; cluster control must bypass proxies.
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    # This test owns a private Ray instance; do not stop a developer's cluster.
    num_gpus = 6 if pd else 4
    context = ray.init(address="local", num_gpus=num_gpus, num_cpus=24, include_dashboard=False)
    env = dict(os.environ)
    env.update(
        RAY_ADDRESS=context.address_info["address"],
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")]),
        TOKENIZERS_PARALLELISM="false",
    )
    arguments = [
        "--hf-checkpoint",
        str(model),
        "--load",
        str(model),
        "--start-rollout-id",
        "0",
        "--save",
        str(directory / "checkpoints"),
        "--save-interval",
        "1",
        "--save-hf",
        str(directory / "hf-{rollout_id}"),
        "--num-gpus-per-node",
        str(num_gpus),
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "2",
        "--rollout-num-gpus",
        "4" if pd else "2",
        "--rollout-num-gpus-per-engine",
        "2",
        "--tensor-model-parallel-size",
        "2",
        "--expert-model-parallel-size",
        "2",
        "--expert-tensor-parallel-size",
        "1",
        "--sequence-parallel",
        "--num-layers",
        "2",
        "--hidden-size",
        "256",
        "--ffn-hidden-size",
        "512",
        "--num-attention-heads",
        "4",
        "--num-query-groups",
        "2",
        "--kv-channels",
        "64",
        "--group-query-attention",
        "--qk-layernorm",
        "--disable-bias-linear",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        "10000",
        "--rotary-percent",
        "1.0",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--vocab-size",
        "256",
        "--num-experts",
        "8",
        "--moe-ffn-hidden-size",
        "256",
        "--moe-router-topk",
        "2",
        "--moe-layer-freq",
        "1",
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-grouped-gemm",
        "--moe-router-dtype",
        "fp32",
        "--moe-aux-loss-coeff",
        "0",
        "--moe-router-score-function",
        "softmax",
        "--attention-dropout",
        "0",
        "--hidden-dropout",
        "0",
        "--update-weight-transport",
        "p2p",
        "--update-weight-buffer-size",
        "1048576",
        "--update-weight-p2p-timeout",
        "120",
        "--sglang-ep-size",
        "2",
        "--sglang-mem-fraction-static",
        "0.3",
        "--sglang-disable-cuda-graph",
        "--sglang-disable-radix-cache",
        "--sglang-attention-backend",
        "triton",
        "--sglang-max-total-tokens",
        "4096",
        "--sglang-max-running-requests",
        "8",
        "--sglang-context-length",
        "256",
        "--prompt-data",
        str(directory / "prompts.jsonl"),
        "--input-key",
        "prompt",
        "--custom-rm-path",
        "test_p2p_qwen3_moe.reward",
        "--custom-rollout-log-function-path",
        "test_p2p_qwen3_moe.check_rollout",
        "--num-rollout",
        "3",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "2",
        "--rollout-max-response-len",
        "16",
        "--rollout-temperature",
        "1",
        "--rollout-top-p",
        "1",
        "--global-batch-size",
        "4",
        "--micro-batch-size",
        "1",
        "--seq-length",
        "256",
        "--advantage-estimator",
        "grpo",
        "--lr",
        "0.01",
        "--lr-decay-style",
        "constant",
        "--optimizer",
        "adam",
        "--weight-decay",
        "0.1",
        "--attention-backend",
        "flash",
        "--ci-test",
        "--ci-disable-kl-checker",
        "--ci-train-rollout-logprob-abs-diff-threshold",
        "0.06",
    ]
    try:
        if pd:
            import yaml

            config = directory / "sglang.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "sglang": [
                            {
                                "name": "default",
                                "server_groups": [
                                    {
                                        "worker_type": role,
                                        "num_gpus": 2,
                                        "num_gpus_per_engine": 2,
                                        "overrides": {"disaggregation_transfer_backend": "mooncake"},
                                    }
                                    for role in ("prefill", "decode")
                                ],
                            }
                        ]
                    }
                )
            )
            arguments.extend(["--sglang-config", str(config)])
        if offload_rollout:
            arguments.append("--offload-rollout")
        subprocess.run(
            [sys.executable, str(ROOT / "train.py"), *arguments], cwd=ROOT, env=env, check=True, timeout=900
        )
        records = [json.loads(line) for line in (directory / "verified.jsonl").read_text().splitlines()]
        assert [record["rollout"] for record in records] == [0, 1, 2]
        assert [record["checkpoint"] for record in records] == ["model", "hf-0", "hf-1"]
        print("P2P MoE e2e passed: three serving versions match independent HF checkpoints after real training")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--pd", action="store_true")
    parser.add_argument("--offload-rollout", action="store_true")
    options = parser.parse_args()
    if options.directory:
        execute(options.directory.resolve(), pd=options.pd, offload_rollout=options.offload_rollout)
    else:
        with tempfile.TemporaryDirectory(prefix="slime-p2p-moe-") as temporary:
            execute(Path(temporary), pd=options.pd, offload_rollout=options.offload_rollout)
