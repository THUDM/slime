import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train

MODEL_NAME = "Qwen3.5-35B-A3B"

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))

DATASET_NAME = "VeraIsHere/geo3k_imgurl_processed"
DATA_ROOT = "/root/datasets/geo3k_imgurl_processed"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    data_missing = not os.path.exists(TRAIN_DATA_PATH)
    if data_missing:
        U.exec_command(f"hf download --repo-type dataset {DATASET_NAME} --local-dir {DATA_ROOT}")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found. Expected local dataset at {TRAIN_DATA_PATH}; ")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project slime-dev "
            "--wandb-group geo3k_vlm_multi_turn "
            f"--wandb-key '{wandb_api_key}' "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--rm-type math "
        "--apply-chat-template "
        "--custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate "
        "--custom-config-path examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
    )

    # eval_args = (
    #     "--eval-interval 20 "
    #     f"--eval-prompt-data geo3k_eval {TRAIN_DATA_PATH}@[0:64] "
    #     "--n-samples-per-eval-prompt 1 "
    #     "--eval-max-response-len 4096 "
    #     "--eval-top-k 1 "
    # )

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
        f"--rollout-num-gpus-per-engine {NUM_GPUS} "
        f"--sglang-ep-size {NUM_GPUS} "
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [1, 2, 4, 8] + list(range(16, 257, 8))))} "
    )

    backend_args = (
        "--train-backend megatron "
        f"--load /root/models/{MODEL_NAME} "
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {NUM_GPUS} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    misc_args = (
        "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " f"--rollout-num-gpus {NUM_GPUS} " "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{backend_args} "
        f"{misc_args} "
        f"{wandb_args} "
        # f"{get_default_wandb_args(__file__)} "
    )

    execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type="qwen3.5-35B-A3B-vl",
        extra_env_vars=({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
    )


if __name__ == "__main__":
    prepare()
    execute()
