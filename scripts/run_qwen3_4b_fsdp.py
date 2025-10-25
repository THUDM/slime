import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))

import command_utils as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-4B")
NUM_GPUS = 8


MODE = os.environ.get("SLIME_SCRIPT_MODE", "normal")
assert MODE in {"normal", "debug_minimal"}

ENABLE_TRUE_ON_POLICY = bool(int(os.environ.get("SLIME_SCRIPT_ENABLE_TRUE_ON_POLICY", "0")))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        # "--ref-load /root/models/{MODEL_NAME} "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if MODE == 'debug_minimal' else 8192} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    # when using tiny response len, cannot do dynamic sampling
    if MODE != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 64 "
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if MODE != "debug_minimal":
        eval_args += (
            "--eval-interval 20 "
            "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 0.7 "
        )

    perf_args = "--use-dynamic-batch-size " "--max-tokens-per-gpu 9216 "

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

    # TODO improve mem-frac
    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "

    fsdp_args = (
        "--train-backend fsdp "
        "--attn-implementation flash_attention_2 "
        "--gradient-checkpointing "
        f"--update-weights-bucket-size {512 * 1024 * 1024} "  # 512MB
    )

    misc_args = "--actor-num-nodes 1 " "--actor-num-gpus-per-node 8 " "--colocate "

    true_on_policy_args = ""
    true_on_policy_envs = {}
    if ENABLE_TRUE_ON_POLICY:
        true_on_policy_args = (
            "--sglang-enable-deterministic-inference "
            "--sglang-rl-on-policy-target fsdp "
            "--sglang-attention-backend fa3 "
            "--attn-implementation flash_attention_3 "
            "--deterministic-mode "
            "--true-on-policy-mode "
        )
        true_on_policy_envs = {
            # TODO note: "Ring" in original RL PR, "allreduce:tree" in SGLang
            # "NCCL_ALGO": "Ring",
            "NCCL_ALGO": "allreduce:tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{fsdp_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=NUM_GPUS,
        model_type=None,
        extra_env_vars={
            **true_on_policy_envs,
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
