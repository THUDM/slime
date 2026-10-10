"""
Debug rollout and replay test:
  Phase 1 – debug_rollout_only: launch sglang, generate rollout data for 2 steps,
            and save them to a temp directory.
  Phase 2 – load_debug_rollout_data (train only): skip sglang entirely, load the
            saved straw archives, and run 2 training steps without copying Samples.
  Phase 3 – replay exported .pt files to retain coverage of the legacy format.

Uses Qwen2.5-0.5B-Instruct (smallest supported model) with 8 GPUs.
"""

import os
import tempfile
from shlex import quote

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_ROLLOUT = 2


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _common_args():
    """Arguments shared by generation and replay."""

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--rollout-data-transport straw "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
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
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = "--advantage-estimator grpo " "--eps-clip 0.2 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    return f"{ckpt_args} " f"{rollout_args} " f"{optimizer_args} " f"{grpo_args} " f"{perf_args} " f"{misc_args} "


def execute_rollout_only(debug_data_dir: str):
    """Phase 1: rollout-only, save data."""

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 " "--sglang-cuda-graph-max-bs 16 "
    )

    phase1_args = (
        f"--rollout-data-dir {quote(os.path.join(debug_data_dir, 'rollout_queue'))} "
        f"{_common_args()} "
        f"{sglang_args} "
        "--debug-rollout-only "
        f"--save-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.straw.json "
    )

    print("=" * 60)
    print("Phase 1: debug-rollout-only (generate + save rollout data)")
    print("=" * 60)

    U.execute_train(
        train_args=phase1_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def execute_train_only(debug_data_dir: str, extension: str):
    """Train from a saved archive or its self-contained export."""

    phase2_args = (
        (f"--rollout-data-dir {quote(os.path.join(debug_data_dir, 'train_queue'))} " if extension == "pt" else "")
        + f"{_common_args()} "
        f"--load-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.{extension} "
        "--ci-test "
    )

    print("=" * 60)
    print(f"Replay {extension}: load-debug-rollout-data (train only)")
    print("=" * 60)

    U.execute_train(
        train_args=phase2_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def execute():
    with tempfile.TemporaryDirectory(prefix="slime_debug_rollout_") as debug_data_dir:
        print(f"Using temp dir for rollout data: {debug_data_dir}")
        execute_rollout_only(debug_data_dir)
        # Export before replay so both formats use exactly the same samples.
        from slime.data.archive import RolloutArchive

        for rollout_id in range(NUM_ROLLOUT):
            with RolloutArchive(f"{debug_data_dir}/rollout_{rollout_id}.straw.json") as archive:
                archive.export_pt(f"{debug_data_dir}/rollout_{rollout_id}.pt")
        execute_train_only(debug_data_dir, "straw.json")
        execute_train_only(debug_data_dir, "pt")

    print("=" * 60)
    print("Generation and both replay formats completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
