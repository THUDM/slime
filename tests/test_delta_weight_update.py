"""E2E smoke test for disk-backed delta weight updates.

Runs a tiny Qwen3.5-0.8B job so the first weight update seeds the delta
snapshot and the post-train update publishes sparse delta files through
``update_weights_from_disk(load_format="delta", files=...)``.
"""

import os
import tempfile
from pathlib import Path

import slime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
NUM_GPUS = 4
TORCH_DIST_CKPT = f"/dev/shm/{MODEL_NAME}_torch_dist"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/dev/shm",
    )


def execute():
    with tempfile.TemporaryDirectory(prefix="slime_delta_weight_update_") as delta_dir:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load {TORCH_DIST_CKPT} "

        rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 1 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--over-sampling-batch-size 8 "
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
            "--global-batch-size 16 "
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
            # Nonzero entropy coef guarantees a nonzero gradient even when all
            # rewards in a group tie (advantages=0), so the delta sync writes
            # real sparse files instead of an empty no-op.
            "--entropy-coef 0.01 "
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
            "--rollout-num-gpus 3 "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-cuda-graph-max-bs 32 "
            "--sglang-enable-metrics "
        )

        delta_args = (
            "--update-weight-mode delta "
            "--update-weight-transport disk "
            "--update-weight-encoding deltas "
            f"--update-weight-delta-dir {delta_dir} "
            "--update-weight-delta-keep-files "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--loss-mask-type qwen3_5 "
            "--actor-num-nodes 1 "
            "--actor-num-gpus-per-node 1 "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{sglang_args} "
            f"{delta_args} "
            f"{ci_args} "
            f"{misc_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )

        delta_files = list(Path(delta_dir).glob("weight_v*/*.safetensors"))
        assert delta_files, f"No disk delta safetensors were written under {delta_dir}"


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
