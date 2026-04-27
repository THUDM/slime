import os

import slime.utils.external_utils.command_utils as U

TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

MODEL_NAME = "Qwen3-VL-2B-Thinking"
MODEL_TYPE = "qwen3-1.7B"
DATASET_NAME = "chenhegu/geo3k_imgurl"
DATASET_LOCAL_NAME = DATASET_NAME.split("/")[-1]
NUM_GPUS = 4


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"hf download --repo-type dataset {DATASET_NAME} --local-dir /root/datasets/{DATASET_LOCAL_NAME}")


def execute():
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--load /root/models/{MODEL_NAME} "

    rollout_args = (
        f"--prompt-data /root/datasets/{DATASET_LOCAL_NAME}/train.parquet "
        "--input-key problem "
        "--label-key answer "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 1 "
        "--rollout-batch-size 1 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 128 "
        "--rollout-temperature 0.0 "
        "--global-batch-size 2 "
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
        f"--sglang-mem-fraction-static {0.6 if TIGHT_DEVICE_MEMORY else 0.7} "
        "--sglang-cuda-graph-bs 1 2 4 "
        "--sglang-enable-metrics "
    )

    ci_args = "--ci-test "

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
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        '--multimodal-keys \'{"image": "images"}\' '
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
        f"{backend_args} "
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
