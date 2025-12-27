import os

import slime.utils.misc as U
from huggingface_hub import snapshot_download
from slime.utils.external_utils.command_utils import execute_train, get_default_wandb_args

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in {"Qwen2.5-VL-3B-Instruct", "Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "1"))
EXTERNAL_RAY = int(os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY", "0"))

DATA_ROOT = os.environ.get("SLIME_SCRIPT_SOKOBAN_DATA_ROOT", "/root/datasets/sokoban_processed")
DATASET_ID = os.environ.get("SLIME_SCRIPT_SOKOBAN_DATASET_ID", "VeraIsHere/sokoban_processed")
TRAIN_DATA_PATH = os.environ.get("SLIME_SCRIPT_SOKOBAN_TRAIN_PATH", f"{DATA_ROOT}/train.parquet")
IMAGES_DIR = os.environ.get("SLIME_SCRIPT_SOKOBAN_IMAGES_DIR", f"{DATA_ROOT}/images")
DOWNLOAD_WORKERS = int(os.environ.get("SLIME_SCRIPT_SOKOBAN_DOWNLOAD_WORKERS", "1"))
DOWNLOAD_ALLOW_PATTERNS = ["train.parquet", "test.parquet", "README.md"]


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=DATA_ROOT,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=DOWNLOAD_WORKERS,
        allow_patterns=DOWNLOAD_ALLOW_PATTERNS,
    )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        "--use-wandb "
        "--wandb-project slime-dev "
        "--wandb-group vlm_multi_turn "
        "--wandb-key ${WANDB_API_KEY} "
    )

    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key prompt "
        "--metadata-key extra_info "
        '--multimodal-keys \'{"image": "images"}\' '
        "--apply-chat-template "
        "--custom-generate-function-path examples.vlm_multi_turn.rollout.generate "
        "--custom-rm-path examples.vlm_multi_turn.reward_sokoban.async_compute_reward "
        "--rollout-shuffle "
        "--num-rollout 800 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1536 "
        "--rollout-max-context-len 8192 "
        "--rollout-temperature 0.8 "
        "--rollout-top-p 0.9 "
        "--global-batch-size 16 "
    )

    custom_args = (
        "--custom-config-path examples/vlm_multi_turn/sokoban_vlm_multi_turn_config.yaml "
    )

    eval_args = (
        "--eval-interval 50 "
        f"--eval-prompt-data sokoban_eval {TRAIN_DATA_PATH}@[0:64] "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
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
        "--update-weight-buffer-size 536870912 "
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--colocate "
    )

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
        # f"{get_default_wandb_args(__file__)} "
    )

    execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=(
            {"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}
            if os.environ.get("WANDB_API_KEY")
            else {}
        ),
    )


if __name__ == "__main__":
    prepare()
    execute()
