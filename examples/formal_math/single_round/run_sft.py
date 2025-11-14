import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "tests"))

import command_utils as U

dataset_transform_id = os.environ["SLIME_DATASET_TRANSFORM_ID"]

MODEL_NAME, MODEL_TYPE = "Qwen3-8B-Base", "qwen3-8B"

NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.convert_checkpoint(model_name=MODEL_NAME, model_type=MODEL_TYPE, num_gpus=NUM_GPUS)


def execute():
    run_id = U.create_run_id()

    load_save_path = f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}_{run_id}/"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 1000 "
    )

    sft_args = (
        "--rollout-function-path slime.rollout.sft_rollout.generate_rollout "
        f"--prompt-data /root/datasets/formal_math_single_round/{dataset_transform_id}/leanabell.parquet "
        "--input-key messages "
        "--rollout-shuffle "
        # NOTE temporarily only 1 epoch to speed up
        "--num-epoch 1 "
        "--rollout-batch-size 128 "
        "--global-batch-size 128 "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
    )

    perf_args = (
        # TP1 + no-cpu-adam + expendable segments => NCCL error (is it oom?) when saving ckpt
        f"--tensor-model-parallel-size {os.environ.get('ARG_TP_SIZE', '2')} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # --micro-batch-size 1
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {os.environ.get('ARG_LR', '2e-4')} "
        "--lr-decay-style cosine "
        "--min-lr 1e-6 "
        "--lr-warmup-fraction 0.1 "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.95 "
    )

    if bool(int(os.environ.get("ARG_CPU_ADAM", "0"))):
        optimizer_args += (
            # https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/optimizer_cpu_offload.html
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} "
        f"{sft_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__, run_id=run_id)} "
        f"{perf_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus=NUM_GPUS,
        model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
