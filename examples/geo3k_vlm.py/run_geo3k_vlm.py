import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))

import command_utils as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in {"Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "1"))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("hiyouga/geometry3k")
    # Rename the dataset directory and files to match expected structure
    U.exec_command("mv /root/datasets/geometry3k /root/datasets/geo3k")
    U.exec_command("mv /root/datasets/geo3k/data/train-00000-of-00001.parquet /root/datasets/geo3k/data/train.parquet")
    U.exec_command("mv /root/datasets/geo3k/data/test-00000-of-00001.parquet /root/datasets/geo3k/data/test.parquet")
    U.exec_command("mv /root/datasets/geo3k/data/validation-00000-of-00001.parquet /root/datasets/geo3k/data/val.parquet")



def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/geo3k/data/train.parquet "
        "--input-key problem "
        "--label-key answer "
        "--multimodal-keys '{\"image\": \"images\"}' "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type geo3k "
        f"--num-rollout 3000 "
        f"--rollout-batch-size 32 "
        f"--n-samples-per-prompt 8 "
        f"--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size 64 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data geo3k-test /root/datasets/geo3k/data/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

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
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-mem-fraction-static 0.4 "
        "--sglang-disable-cuda-graph "
    )

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/geo3k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate " "--train-backend fsdp "

    misc_args += (
        "--use-dynamic-batch-size "
        # TODO pick a good value
        "--max-tokens-per-gpu 2048 "
    )

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
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=NUM_GPUS,
        model_type=None,
        train_script="/root/slime/train.py",
        extra_env_vars={
            **true_on_policy_envs,
            "SGLANG_DUMPER_ENABLE": "0",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "0",
        },
    )


if __name__ == "__main__":
    prepare()
    execute()