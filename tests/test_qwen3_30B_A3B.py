import os

import command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command("hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B")
    U.exec_command("hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/models/Qwen3-30B-A3B-FP8")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k")
    U.exec_command("hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024")

    U.convert_checkpoint(model_name=MODEL_NAME, model_type=MODEL_TYPE)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}-FP8 " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if TIGHT_HOST_MEMORY else 16384} "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        f"{'' if TIGHT_HOST_MEMORY else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-tis "
        "--use-routing-replay "
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
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-moe-a2a-backend deepep "
        "--sglang-deepep-mode auto "
        "--sglang-max-running-requests 512 "
        "--sglang-disable-radix-cache "
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
        "--moe-token-dispatcher-type flex "
        "--moe-enable-deepep "
        "--ci-test "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
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
        num_gpus=8,
        model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    # TODO also use typer
    prepare()
    execute()
