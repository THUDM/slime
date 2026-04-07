import os

import slime.utils.external_utils.command_utils as U


ENABLE_EVAL = U.get_bool_env_var("SLIME_TEST_ENABLE_EVAL", "1")

MODEL_NAME = "Qwen3.5-4B"
MODEL_TYPE = "qwen3.5-4B"
DATASET_NAME = "chenhegu/geo3k_imgurl"
DATASET_LOCAL_NAME = "geo3k_imgurl"
NUM_GPUS = 8
ACTOR_NUM_GPUS = 4
ROLLOUT_NUM_GPUS = 4
TORCH_DIST_CKPT = f"/root/{MODEL_NAME}_torch_dist"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=ACTOR_NUM_GPUS,
        hf_checkpoint=f"/root/models/{MODEL_NAME}",
    )
    U.hf_download_dataset(DATASET_NAME)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--load {TORCH_DIST_CKPT} "

    rollout_args = (
        f"--prompt-data /root/datasets/{DATASET_LOCAL_NAME}/train.parquet "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 4 "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        f"--eval-prompt-data geo3k /root/datasets/{DATASET_LOCAL_NAME}/test.parquet "
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
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-cuda-graph-max-bs 8 "
        "--sglang-attention-backend triton "
        "--sglang-mm-attention-backend triton_attn "
    )

    backend_args = (
        "--train-backend megatron "
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--qkv-format bshd "
        "--micro-batch-size 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS} " f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{backend_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
