import json
import os
import subprocess

import slime.utils.misc as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in {"Qwen2.5-VL-3B-Instruct", "Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "1"))
EXTERNAL_RAY = int(os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY", "0"))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")


def detect_nvlink():
    """Detect if NVLink is available on the system."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        nvlink_count = result.stdout.count("NVLink")
        has_nvlink = 1 if nvlink_count > 0 else 0
        print(f"HAS_NVLINK: {has_nvlink} (detected {nvlink_count} NVLink references)")
        return has_nvlink
    except Exception as e:
        print(f"Failed to detect NVLink: {e}")
        return 0


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    dataset_name = "chenhegu/geo3k_imgurl"
    _, partial_name = dataset_name.split("/")
    U.exec_command(f"hf download --repo-type dataset {dataset_name} --local-dir /root/datasets/{partial_name}")


def execute():
    # Detect NVLink for optimized NCCL settings
    has_nvlink = detect_nvlink()

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/geo3k_imgurl/train.parquet "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 512 "
    )

    eval_args = (
        # "--eval-interval 20 "
        "--eval-prompt-data geo3k-test /root/datasets/geo3k_imgurl/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
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
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [1, 2, 4, 8] + list(range(16, 257, 8))))} "
    )

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
    )

    wandb_args = (
        "--use-wandb "
        "--wandb-project geo3k-vlm "
        "--wandb-group geo3k-vlm "
        "--wandb-key ${WANDB_API_KEY} "
        "--disable-wandb-random-suffix "
    )

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate "

    # misc_args += (
    #     "--use-dynamic-batch-size "
    #     # TODO pick a good value
    #     "--max-tokens-per-gpu 2048 "
    # )

    # true_on_policy_args = (
    #     "--sglang-enable-deterministic-inference "
    #     "--sglang-rl-on-policy-target fsdp "
    #     "--deterministic-mode "
    #     "--true-on-policy-mode "
    # )
    # true_on_policy_envs = {
    #     # TODO note: "Ring" in original RL PR, "allreduce:tree" in SGLang
    #     # "NCCL_ALGO": "Ring",
    #     "NCCL_ALGO": "allreduce:tree",
    #     "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    #     "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{fsdp_args} "
        f"{eval_args} "
        f"{misc_args} "
        f"{wandb_args} "
        # f"{true_on_policy_args} "
    )

    # Kill existing processes
    U.exec_command(
        "pkill -9 sglang; "
        "sleep 3; "
        f"{'' if EXTERNAL_RAY else 'ray stop --force; '}"
        f"{'' if EXTERNAL_RAY else 'pkill -9 ray; '}"
        "pkill -9 slime; "
        "sleep 3; "
        f"{'' if EXTERNAL_RAY else 'pkill -9 ray; '}"
        "pkill -9 slime; "
        "pkill -9 redis; "
        "true; "
    )

    if not EXTERNAL_RAY:
        # Start Ray
        U.exec_command(
            f"export PYTHONBUFFERED=16 && "
            f"ray start --head --node-ip-address {MASTER_ADDR} --num-gpus {NUM_GPUS} "
            f"--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
        )

    # Prepare runtime environment
    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": str(has_nvlink),
                # **true_on_policy_envs,
                # "SGLANG_DUMPER_ENABLE": "0",
                # "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "0",
            }
        }
    )

    # Submit Ray job
    U.exec_command(
        f"export no_proxy=127.0.0.1 && export PYTHONBUFFERED=16 && "
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json='{runtime_env_json}' "
        f"-- python3 /root/slime/train.py "
        f"{train_args}"
    )


if __name__ == "__main__":
    prepare()
    execute()
