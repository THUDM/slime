import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train_npu

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in {
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Thinking",
}

EXTERNAL_RAY = int(os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY", "0"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "fsdp").lower()
assert TRAIN_BACKEND in {"fsdp", "megatron"}

DATASET_NAME = "VeraIsHere/geo3k_imgurl_processed"
DATA_ROOT = "/root/datasets/geo3k_imgurl_processed"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")


def get_megatron_model_type(model_name: str) -> str:
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")


def execute():
    ckpt_args = f"--hf-checkpoint /home/data/{MODEL_NAME} "

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
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
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
        "--use-kl-loss "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "

        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [4, 8] + list(range(16, 257, 8))))} "
        "--sglang-device npu "
        "--sglang-disable-radix-cache "
        "--sglang-chunked-prefill-size 32768 "
        "--sglang-max-prefill-tokens 4000 "
        "--sglang-max-total-tokens 327680 "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load /home/data/{MODEL_NAME} "
        f"--ref-load /home/data/{MODEL_NAME} "
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--balance-data "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 " f"--actor-num-gpus-per-node 8 " f"--rollout-num-gpus 8 "
        "--no-gradient-accumulation-fusion "
        "--use-flash-attn "
    )

    if TRAIN_BACKEND == "megatron":
        backend_args = megatron_args
        megatron_model_type = get_megatron_model_type(MODEL_NAME)
        os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    else:
        exit()

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

    execute_train_npu(
        train_args=train_args,
        megatron_model_type=megatron_model_type,
        extra_env_vars=({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
    )


if __name__ == "__main__":
    execute()
