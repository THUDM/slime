import os
import tempfile

import slime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3.6-35B-A3B"
MODEL_TYPE = "qwen3.5-35B-A3B"
NUM_GPUS = 8
TORCH_DIST_CKPT = f"/root/models/{MODEL_NAME}_torch_dist"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def execute():
    debug_data_path = os.environ.get("DEBUG_ROLLOUT_DATA") or tempfile.mktemp(
        prefix="qwen3_6_35b_a3b_pd_rollout_", suffix=".pt"
    )
    try:
        os.remove(debug_data_path)
    except FileNotFoundError:
        pass
    print(f"Saving debug rollout data to {debug_data_path}")

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load {TORCH_DIST_CKPT} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 16384 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 2 "
        "--eval-max-response-len 16384 "
        "--eval-temperature 0.6 "
        "--eval-top-p 0.95 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--max-tokens-per-gpu 8192 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
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
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.75 "
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 4 "
        "--sglang-ep-size 4 "
        "--sglang-enable-dp-lm-head "
        "--sglang-cuda-graph-bs 1 2 4 8 16 24 32 "
        "--sglang-max-running-requests 512 "
        "--prefill-num-servers 1 "
        "--sglang-disaggregation-transfer-backend mooncake "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 3 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 4 "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--sglang-enable-metrics "
    )

    misc_args = (
        "--ci-test "
        f"--save-debug-rollout-data {debug_data_path} "
        "--update-weight-buffer-size 2147483648 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type flex "
        "--moe-enable-deepep "
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
