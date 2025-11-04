import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))

import command_utils as U


@dataclass
class ScriptArgs:
    mode: Literal["normal", "debug_minimal"] = "normal"
    model_name: str = "Qwen3-4B-Instruct-2507"
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    hardware: Literal["H100"] = "H100"
    extra_args: str = ""
    multi_eval: bool = True
    true_on_policy: bool = False
    dynamic_sampling: bool = False
    enable_eval: bool = True


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{args.model_name} --local-dir /root/models/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.hf_download_dataset("zyzshishui0627/gpqa_diamond")
    U.hf_download_dataset("zyzshishui0627/IFBench")


def execute(args: ScriptArgs):
    run_id = U.create_run_id()

    load_save_path = f"/root/shared_data/{run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{args.model_name} "
        # "--ref-load /root/models/{args.model_name} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # By default it is thinking mode
        # """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 16 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 1024 "
        "--balance-data "
    )

    if args.dynamic_sampling and (args.true_on_policy != "debug_minimal"):
        rollout_args += (
            "--over-sampling-batch-size 64 "
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    # sometimes disable eval to speed up debugging
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

    perf_args = "--use-dynamic-batch-size " "--max-tokens-per-gpu 32768 "

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        # "--optimizer deepspeed_cpu_adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine 1 " f"--sglang-mem-fraction-static 0.75 " "--sglang-chunked-prefill-size 4096 "
    )

    fsdp_args = (
        "--train-backend fsdp "
        "--attn-implementation flash_attention_2 "
        "--gradient-checkpointing "
        f"--update-weights-bucket-size {512 * 1024 * 1024} "  # 512MB
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--offload-train-mode move "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{run_id}/dump_details "
    )

    true_on_policy_args = ""
    true_on_policy_envs = {}
    if args.true_on_policy:
        true_on_policy_args = (
            "--sglang-enable-deterministic-inference "
            "--sglang-rl-on-policy-target fsdp "
            "--sglang-attention-backend fa3 "
            "--attn-implementation flash_attention_3 "
            "--deterministic-mode "
            "--true-on-policy-mode "
        )
        true_on_policy_envs = {
            "NCCL_ALGO": "allreduce:tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fsdp_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=args.num_gpus_per_node,
        model_type=None,
        extra_env_vars={
            **true_on_policy_envs,
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
