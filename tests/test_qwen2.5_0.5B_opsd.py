import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    # OPSD loads the teacher (frozen init policy) directly into Megatron, so we need a
    # Megatron (torch_dist) checkpoint for both the student load and the teacher.
    U.convert_checkpoint(MODEL_NAME, MODEL_TYPE, NUM_GPUS)


def execute():
    torch_dist_ckpt = f"/root/{MODEL_NAME}_torch_dist"

    # Teacher (frozen init policy) and student both start from the converted checkpoint.
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load {torch_dist_ckpt}/ "
        f"--load {torch_dist_ckpt}/ "
    )

    # Smoke test: the dapo-math dataset has no full reference-solution field, so we use the
    # `label` field as the privileged information just to exercise the OPSD data/forward path.
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "  # OPSD requires context-parallel-size 1
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    # On-Policy Self-Distillation (OPSD): single model is both student and teacher; the
    # frozen teacher is conditioned on privileged info and the loss is a full-vocab JSD.
    opsd_args = (
        "--use-opd "
        "--opd-type self "
        f"--opd-teacher-load {torch_dist_ckpt}/ "
        "--opsd-beta 0.5 "
        "--opsd-jsd-clip 0.05 "
        "--opsd-privileged-info-key label "
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
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-cuda-graph-max-bs 16 "
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
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{opsd_args} "
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
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
