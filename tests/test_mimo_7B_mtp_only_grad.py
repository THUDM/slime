"""End-to-end test for MTP-only gradient verification.

This test verifies that when MTP training is enabled and all outputs are truncated
(due to very short max response length), only MTP parameters receive non-zero
gradients while all other model parameters have zero gradients.

This validates that the MTP loss computation correctly isolates gradient flow
to only the MTP layers when the main model loss is zero (due to truncation).
"""

import os

import slime.utils.external_utils.command_utils as U


MODEL_NAME = "MiMo-7B-RL"
MODEL_TYPE = "mimo-7B-rl"
NUM_GPUS = 8


def prepare():
    """Download model and convert checkpoint with MTP layers."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download XiaomiMiMo/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")

    # Convert checkpoint with MTP layers enabled
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        extra_args=" --mtp-num-layers 1",
        dir_dst="/root/models",
    )


def execute():
    """Run training with MTP enabled and very short output length to cause truncation."""
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "

    # Use very short rollout-max-response-len to ensure all outputs are truncated
    # This should result in zero loss for the main model, leaving only MTP loss
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        # Very short max response length to cause all outputs to be truncated
        "--rollout-max-response-len 128 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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
        "--sglang-enable-metrics "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
    )

    # Enable MTP training with loss scaling
    mtp_args = "--mtp-num-layers 1 " "--enable-mtp-training " "--mtp-loss-scaling-factor 0.2 "

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        # MTP grad check is automatically triggered when ci_test and enable_mtp_training are both set
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

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{mtp_args} "
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
    # Remove proxy settings that might interfere with local operations
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(key, None)
    execute()
