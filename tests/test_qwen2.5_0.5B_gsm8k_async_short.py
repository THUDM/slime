import os
import slime.utils.external_utils.command_utils as U

TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 8 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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
        f"--sglang-mem-fraction-static {0.55 if TIGHT_DEVICE_MEMORY else 0.65} "
        "--sglang-enable-metrics "
    )

    ci_args = "--ci-test "

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 3 "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{fault_tolerance_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    os.environ.pop("http_proxy")
    os.environ.pop("https_proxy")
    os.environ.pop("HTTP_PROXY")
    os.environ.pop("HTTPS_PROXY")
    execute()
