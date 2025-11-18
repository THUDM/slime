import datetime
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))

import command_utils as U

mode = os.environ.get("SLIME_SCRIPT_MODE", "8xh100")

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"

match mode:
    case "8xh100":
        num_gpus_for_convert = num_gpus = 8
    case "4xgb300":
        num_gpus_for_convert = num_gpus = 4
    case "8xgb300":
        num_gpus_for_convert = 4
        num_gpus = 8
    case "32xgb300":
        num_gpus_for_convert = 4
        num_gpus = 32
    case _:
        raise NotImplementedError(f"{mode=}")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        model_type=MODEL_TYPE,
        num_gpus=num_gpus_for_convert,
        # To support multi-node training, for simplicity, we put model into shared folder
        dir_dst="/root/models",
    )


# TODO improve layering: split algorithm vs infra
def execute():
    load_save_path = (
        f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}__{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/"
    )
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 "
        "--eval-top-p 0.7 "
    )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 20480 "
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

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--colocate "
    )

    match mode:
        case "8xh100":
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 8 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-bs 1 2 4 8 " + " ".join(str(x) for x in range(16, 257, 8)) + " "
            )
            optimizer_args += (
                "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )
            misc_args += "--actor-num-gpus-per-node 8 " "--actor-num-nodes 1 "
        case "4xgb300":
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 4 "
                "--sglang-ep-size 4 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-bs 1 2 4 8 " + " ".join(str(x) for x in range(16, 513, 8)) + " "
            )
            misc_args += "--actor-num-gpus-per-node 4 " "--actor-num-nodes 1 " "--num-gpus-per-node 4"
        case "8xgb300":
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 4 "
                "--sglang-ep-size 4 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-bs 1 2 4 8 " + " ".join(str(x) for x in range(16, 513, 8)) + " "
            )
            misc_args += "--actor-num-gpus-per-node 4 " "--actor-num-nodes 2 " "--num-gpus-per-node 4"
        case "32xgb300":
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 4 "
                "--sglang-ep-size 4 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-bs 1 2 4 8 " + " ".join(str(x) for x in range(16, 513, 8)) + " "
            )
            misc_args += "--actor-num-gpus-per-node 4 " "--actor-num-nodes 8 " "--num-gpus-per-node 4"
        case _:
            raise NotImplementedError(f"{mode=}")

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
        num_gpus=num_gpus,
        model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute()
