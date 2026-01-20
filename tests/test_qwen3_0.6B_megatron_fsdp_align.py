import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 4
CP_SIZE = 1
MEGATRON_TP_SIZE = 1
MEGATRON_PP_SIZE = 1


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/"

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
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
        "--rollout-num-gpus-per-engine 1 " "--sglang-chunked-prefill-size 4096 " "--sglang-mem-fraction-static 0.75 "
    )

    ci_args = "--ci-test "

    misc_args = "--actor-num-nodes 1 " "--colocate " f"--actor-num-gpus-per-node {NUM_GPUS} "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    debug_data_path = "test_rollout_data_megatron_fsdp_align.pt"
    grad_norm_path = "grad_norm_fsdp.pt"

    fsdp_args = (
        "--train-backend fsdp "
        "--attn-implementation flash_attention_2 "
        "--gradient-checkpointing "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    try:
        U.execute_train(
            train_args=train_args + (f"{fsdp_args}" f"--save-debug-rollout-data {debug_data_path} "),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=None,
        )

        U.execute_train(
            train_args=train_args
            + (
                f"{fsdp_args}"
                f"--load-debug-rollout-data {debug_data_path} "
                f"--ci-save-grad-norm {grad_norm_path} "
                "--debug-train-only "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=None,
        )

        U.execute_train(
            train_args=train_args
            + (
                f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
                f"--tensor-model-parallel-size {MEGATRON_TP_SIZE} "
                "--sequence-parallel "
                f"--pipeline-model-parallel-size {MEGATRON_PP_SIZE} "
                f"--context-parallel-size {CP_SIZE} "
                "--expert-model-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
                "--recompute-granularity full "
                "--recompute-method uniform "
                "--recompute-num-layers 1 "
                "--train-memory-margin-bytes 3221225472 "
                f"--load-debug-rollout-data {debug_data_path} "
                f"--ci-load-grad-norm {grad_norm_path} "
                "--attention-dropout 0.0 "
                "--hidden-dropout 0.0 "
                "--accumulate-allreduce-grads-in-fp32 "
                "--attention-softmax-in-fp32 "
                "--attention-backend flash "
                "--debug-train-only "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )

    finally:
        if os.path.exists(grad_norm_path):
            os.remove(grad_norm_path)
        if os.path.exists(debug_data_path):
            os.remove(debug_data_path)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
