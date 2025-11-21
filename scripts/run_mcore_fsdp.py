from dataclasses import dataclass
from typing import Literal, Optional

import typer

import slime.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    train_backend: Literal["fsdp", "megatron"] = "fsdp"
    use_ref: bool = False
    colocate: bool = True
    model_name: str = "Qwen3-4B-Instruct-2507"
    num_gpus_per_node: Optional[int] = None
    hardware: Literal["H100", "GB300"] = "H100"
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    multi_eval: bool = False
    true_on_policy: bool = False
    dynamic_sampling: bool = False
    enable_eval: bool = True
    megatron_model_type: Optional[str] = None
    extra_args: str = ""

    def __post_init__(self):
        if self.train_backend == "megatron":
            self.megatron_model_type = {
                "Qwen3-4B": "qwen3-4B",
                "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
                "Qwen3-4B-Base": "qwen3-4B",
            }[self.model_name]

        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{args.model_name} " f"--local-dir /root/models/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.hf_download_dataset("zyzshishui0627/gpqa_diamond")
    U.hf_download_dataset("zyzshishui0627/IFBench")
    if args.train_backend == "megatron":
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst="/root/models",
        )


def execute(args: ScriptArgs):
    load_save_path = f"/root/shared_data/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint /root/models/{args.model_name} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    if args.train_backend == "megatron":
        # Megatron uses torch_dist checkpoint format for ref model
        if args.use_ref:
            ckpt_args += f"--ref-load /root/models/{args.model_name}_torch_dist "
    else:
        # FSDP now supports ref-load with HF checkpoint
        if args.use_ref:
            ckpt_args += f"--ref-load /root/models/{args.model_name} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type math "
        f"--num-rollout {10 if args.mode == 'debug_minimal' else 3000} "
        f"--rollout-batch-size {8 if args.mode == 'debug_minimal' else 64} "
        f"--n-samples-per-prompt {8 if args.mode == 'debug_minimal' else 16} "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
        "--rollout-temperature 0.8 "
        f"--global-batch-size {64 if args.mode == 'debug_minimal' else 1024} "
    )

    if args.dynamic_sampling and (args.mode != "debug_minimal"):
        rollout_args += (
            "--over-sampling-batch-size 128 "
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_max_response_len = 32768
        eval_args += "--eval-interval 20 "
        if args.multi_eval:
            eval_config_text = f"""
eval:
  defaults:
    max_response_len: {eval_max_response_len}
    top_p: 0.7
  datasets:
    - name: aime
      path: /root/datasets/aime-2024/aime-2024.jsonl
      rm_type: math
      n_samples_per_eval_prompt: 16
    - name: gpqa
      path: /root/datasets/gpqa_diamond/gpqa_eval.jsonl
      rm_type: gpqa
      n_samples_per_eval_prompt: 2
    - name: ifbench
      path: /root/datasets/IFBench/IFBench_eval.jsonl
      rm_type: ifbench
      n_samples_per_eval_prompt: 1
""".strip()
            eval_args += f"--eval-config {U.save_to_temp_file(eval_config_text, 'yaml')} "
        else:
            eval_args += (
                "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 16 "
                f"--eval-max-response-len {eval_max_response_len} "
                "--eval-top-p 0.7 "
            )

    perf_args = (
        "--use-dynamic-batch-size " f"--max-tokens-per-gpu {9216 if args.train_backend == 'megatron' else 32768} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if args.use_ref:
        grpo_args += "--use-kl-loss "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.7 if args.train_backend == 'megatron' else 0.75} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-chunked-prefill-size 4096 "
    )

    if args.train_backend == "fsdp":
        train_backend_args = (
            "--train-backend fsdp "
            f"--update-weight-buffer-size {512 * 1024 * 1024} "  # 512MB
            "--gradient-checkpointing "
            "--attn-implementation flash_attention_2 "
        )
        if args.true_on_policy:
            train_backend_args += "--sglang-attention-backend fa3 " "--attn-implementation flash_attention_3 "
    else:
        tp_size = 2 if args.num_gpus_per_node == 8 else 1
        cp_size = 4 if args.num_gpus_per_node == 8 else 1
        train_backend_args = (
            f"--tensor-model-parallel-size {tp_size} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            f"--context-parallel-size {cp_size} "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
        )

    if args.colocate:
        misc_args = (
            f"--actor-num-nodes {args.num_nodes} " f"--actor-num-gpus-per-node {args.num_gpus_per_node} " "--colocate "
        )
    else:
        actor_gpus = args.num_gpus_per_node // 2
        rollout_gpus = args.num_gpus_per_node // 2
        misc_args = (
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {actor_gpus} "
            f"--rollout-num-gpus {rollout_gpus} "
        )

    if args.train_backend == "fsdp":
        misc_args += """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """

    misc_args += "--use-fault-tolerance " f"--dump-details /root/shared_data/{args.run_id}/dump_details "

    misc_env_vars = {}

    true_on_policy_args = ""
    true_on_policy_envs = {}
    if args.true_on_policy:
        true_on_policy_args = (
            "--sglang-enable-deterministic-inference "
            "--sglang-rl-on-policy-target fsdp "
            "--deterministic-mode "
            "--true-on-policy-mode "
        )
        true_on_policy_envs = {
            "NCCL_ALGO": "allreduce:tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    backend_name = "fsdp" if args.train_backend == "fsdp" else "megatron"
    ref_name = "ref" if args.use_ref else "noref"
    colocate_name = "" if args.colocate else "-dist"
    wandb_group = f"qwen3-4B-{backend_name}-{ref_name}{colocate_name}"

    wandb_args = f"--use-wandb " f"--wandb-project slime-dev-megatron-fsdp " f"--wandb-group {wandb_group} "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{perf_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
        f"{wandb_args} "
        f"{args.extra_args} "
    )

    extra_env_vars = misc_env_vars
    if args.train_backend == "megatron":
        import os

        nvlink_count = os.popen('nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l').read().strip()
        has_nvlink = "1" if int(nvlink_count or "0") > 0 else "0"

        extra_env_vars |= {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": has_nvlink,
        }

    extra_env_vars |= true_on_policy_envs

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type if args.train_backend == "megatron" else None,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    """Main entry point for unified MCORE/FSDP training script.

    Examples:
        # FSDP colocated without ref
        python run_mcore_fsdp.py --train-backend fsdp --colocate --no-use-ref

        # FSDP distributed with ref
        python run_mcore_fsdp.py --train-backend fsdp --no-colocate --use-ref

        # Megatron colocated without ref
        python run_mcore_fsdp.py --train-backend megatron --colocate --no-use-ref

        # Megatron distributed with ref
        python run_mcore_fsdp.py --train-backend megatron --no-colocate --use-ref
    """
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
