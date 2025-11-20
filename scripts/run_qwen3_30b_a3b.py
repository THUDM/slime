from dataclasses import dataclass
from typing import Literal, Optional

import typer

import slime.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-30B-A3B"
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: Optional[int] = None
    hardware: Literal["H100", "GB300"] = "H100"
    enable_eval: bool = True
    extra_args: str = ""

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{args.model_name} --local-dir /root/models/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        # To support multi-node training, for simplicity, we put model into shared folder
        dir_dst="/root/models",
    )


# TODO improve layering: split algorithm vs infra
def execute(args: ScriptArgs):
    load_save_path = f"/root/shared_data/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{args.model_name}/ "
        f"--ref-load /root/models/{args.model_name}_torch_dist "
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
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
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
        "--max-tokens-per-gpu 32768 "
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
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
    )

    match (args.hardware, args.num_nodes):
        case ("H100", 1):
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
                "--sglang-cuda-graph-max-bs 512 "
            )
            optimizer_args += (
                "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )
            misc_args += "--actor-num-gpus-per-node 8 " "--actor-num-nodes 1 "
        case ("GB300", 1):
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
                # "--sglang-ep-size 4 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-max-bs 512 "
            )
            misc_args += "--actor-num-gpus-per-node 4 " "--actor-num-nodes 1 " "--num-gpus-per-node 4"
        case ("GB300", 2):
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
                # "--sglang-ep-size 4 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-max-bs 512 "
            )
            misc_args += "--actor-num-gpus-per-node 4 " "--actor-num-nodes 2 " "--num-gpus-per-node 4"
        case ("GB300", 4):
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
                # "--sglang-ep-size 4 "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-max-bs 512 "
            )
            misc_args += "--actor-num-gpus-per-node 4 " "--actor-num-nodes 8 " "--num-gpus-per-node 4"
        case _:
            raise NotImplementedError

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
