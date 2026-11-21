"""MoE/R3 recovery with stateless Adam: OOM, manager loss, resharding and route replay."""

import argparse
import runpy
from pathlib import Path

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
NUM_GPUS = 8


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute(directory=None, model_path=None, dataset_path=None, pd=False):
    suite = runpy.run_path(str(Path(__file__).with_name("test_qwen2.5_0.5B_training_recovery.py")))
    suite["execute"](
        directory=directory,
        model_path=model_path or f"/root/models/{MODEL_NAME}",
        dataset_path=dataset_path,
        failure_rollout=1,
        save_interval=1,
        kill_manager=True,
        manager_crash_phase="training",
        fault_tolerance=False,
        model_type="qwen3-30B-A3B",
        train_gpus=4,
        rollout_gpus=4,
        pd=pd,
        extra_args=(
            "--rollout-num-gpus-per-engine 2 --expert-model-parallel-size 4 "
            # Exercise recovery without Adam moments; scheduler/RNG still resume
            # with the model, and accepted R3 routes must replay byte-for-byte.
            "--use-stateless-adam --no-save-optim "
            "--use-rollout-routing-replay --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer --recompute-granularity full --recompute-method uniform "
            "--recompute-num-layers 1 --sglang-server-concurrency 32 "
            # Parallelize the CPU master-parameter updates for this full-size model.
            '--train-env-vars \'{"OMP_NUM_THREADS":"8"}\' '
            # Keep communication scratch bounded across repeated training calls.
            "--overlap-grad-reduce --ddp-bucket-size 40000000 "
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--directory")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--pd", action="store_true")
    args = parser.parse_args()
    if not args.no_prepare:
        prepare()
    execute(args.directory, args.model_path, args.dataset_path, args.pd)
