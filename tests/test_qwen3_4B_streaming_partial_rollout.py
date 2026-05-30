"""CI smoke test for the streaming sglang rollout path.

Wires ``slime.rollout.sglang_streaming_rollout.generate_streaming`` in as the
per-sample generate function, with ``--over-sampling-batch-size`` >
``--rollout-batch-size`` and ``--partial-rollout`` enabled so the rollout
loop *must* abort in-flight requests every step — exercising the streaming
abort path (partial state should already be on the sample when the SSE is
cut, then the partial groups get recycled into the data buffer).

Uses Qwen3-4B (vs the 0.5B in other short tests) so responses on dapo-math
are long enough to actually trigger mid-stream aborts, and good enough to
produce non-zero rewards.
"""

import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        # Streaming generate at the per-sample level — the outer rollout
        # loop is still the stock sglang one (semaphore, abort orchestration).
        "--custom-generate-function-path slime.rollout.sglang_streaming_rollout.generate_streaming "
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        # Over-sample 2x so half of every rollout's in-flight groups must
        # be aborted, then partial-rollout recycles them.
        "--over-sampling-batch-size 8 "
        "--partial-rollout "
        "--mask-offpolicy-in-partial-rollout "
        "--n-samples-per-prompt 4 "
        # Long enough that aborts cut samples mid-generation rather than
        # everything finishing first; Qwen3-4B also generates real reasoning
        # on dapo-math at this length, so the reward signal is non-trivial.
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
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
        "--rollout-num-gpus-per-engine 2 "
        "--rollout-num-gpus 8 "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-cuda-graph-max-bs 32 "
        "--sglang-max-running-requests 512 "
        "--sglang-enable-metrics "
    )

    ci_args = "--ci-test "

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

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    execute()
