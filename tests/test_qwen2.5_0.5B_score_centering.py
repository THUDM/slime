"""Megatron + SGLang smoke test with score centering enabled."""

import os
import tempfile
from pathlib import Path

import torch

import slime.utils.external_utils.command_utils as U
from slime.utils.score_centering import validate_sampler_topk
from slime.utils.types import Sample

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    with tempfile.TemporaryDirectory(prefix="slime-score-centering-") as directory:
        train_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} --ref-load /root/models/{MODEL_NAME} "
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages --label-key label --apply-chat-template --rm-type math "
            "--num-rollout 2 --rollout-batch-size 2 --n-samples-per-prompt 4 "
            "--rollout-max-response-len 256 --rollout-temperature 0.8 --rollout-top-p 1 --rollout-top-k -1 "
            "--global-batch-size 8 --use-score-centering --score-centering-top-k 128 "
            "--pg-loss-type reinforce --advantage-estimator grpo --disable-grpo-std-normalization "
            "--calculate-per-token-loss --entropy-coef 0 --kl-coef 0 "
            "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 "
            "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size --max-tokens-per-gpu 4096 --log-probs-chunk-size 128 "
            "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 --sglang-cuda-graph-max-bs 8 "
            "--attention-dropout 0 --hidden-dropout 0 --attention-backend flash "
            "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
            "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate --ci-test "
            f"--save-debug-rollout-data {directory}/rollout_{{rollout_id}}.pt "
            f"--ci-save-grad-norm {directory}/grad_{{rollout_id}}_{{step_id}}.pt "
        )
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
            extra_env_vars={"SGLANG_RETURN_ORIGINAL_LOGPROB": "0"},
        )
        for rollout_id in range(2):
            data = torch.load(Path(directory) / f"rollout_{rollout_id}.pt", weights_only=False)
            assert data["samples"]
            for item in data["samples"]:
                sample = Sample.from_dict(item)
                validate_sampler_topk(sample, 128)
                assert len(sample.rollout_log_probs) == sample.response_length
            grad = torch.load(Path(directory) / f"grad_{rollout_id}_0.pt", weights_only=False)
            assert torch.isfinite(torch.as_tensor(grad)).all()


if __name__ == "__main__":
    prepare()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    execute()
