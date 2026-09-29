"""Megatron + SGLang smoke test for FlashREINFORCE on the fully-async rollout path.

One rollout per prompt, the rollout-batch-mean advantage, the binary-KL trust region and
evaluation while trajectories are in flight (evaluation runs on the fully-async worker's
event loop). Batches are formed like molt's asynchronous trainer: a 12-rollout pool, two
batches formed ahead, and each pass over the 20 prompts drained into a 4-rollout batch.
Whether a trajectory actually straddles a weight update depends on timing, so the test
checks that per-token rollout data stays aligned rather than requiring it.
"""

import os
import tempfile
from pathlib import Path

import torch

import slime.utils.external_utils.command_utils as U
from slime.utils.types import Sample

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
NUM_ROLLOUT = 4
NUM_PROMPTS = 20
BATCH_SIZES = [8, 8, 4, 8]  # 20 prompts = 8 + 8 + a 4-rollout tail, then the next pass


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    with tempfile.TemporaryDirectory(prefix="slime-flash-reinforce-") as directory:
        # GSM8K gives the 0.5B model nonzero rewards, so the centered advantages carry a gradient.
        train_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} --ref-load /root/models/{MODEL_NAME} "
            "--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async "
            "--fully-async-pool-size 12 --fully-async-max-queued-batches 2 --fully-async-drain-each-epoch "
            "--sglang-server-concurrency 4 "
            f"--prompt-data /root/datasets/gsm8k/train.parquet@[0:{NUM_PROMPTS}] "
            "--input-key messages --label-key label --apply-chat-template --rollout-shuffle --rm-type math "
            f"--num-rollout {NUM_ROLLOUT} --rollout-batch-size 8 --n-samples-per-prompt 1 --num-steps-per-rollout 1 "
            "--rollout-max-response-len 1024 --rollout-temperature 1.0 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet@[0:32] "
            "--n-samples-per-eval-prompt 1 --eval-max-response-len 1024 --eval-interval 2 "
            "--advantage-estimator flash_reinforce --use-tis "
            "--custom-tis-function-path slime.backends.megatron_utils.loss.binary_kl_trust_region_function "
            "--tis-binary-kl-threshold 5e-3 --kl-coef 0 --entropy-coef 0 "
            "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --apply-wd-to-all-params "
            "--adam-beta1 0.9 --adam-beta2 0.95 "
            "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
            "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.65 --sglang-cuda-graph-max-bs 16 "
            "--attention-dropout 0 --hidden-dropout 0 --attention-backend flash "
            "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
            "--actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 3 --ci-test "
            f"--save-debug-rollout-data {directory}/rollout_{{rollout_id}}.pt "
            f"--ci-save-grad-norm {directory}/grad_{{rollout_id}}_{{step_id}}.pt "
            f"{U.get_default_wandb_args(__file__)} "
        )
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )
        multi_version, rewards, prompts = 0, [], []
        for rollout_id in range(NUM_ROLLOUT):
            data = torch.load(Path(directory) / f"rollout_{rollout_id}.pt", weights_only=False)
            assert len(data["samples"]) == BATCH_SIZES[rollout_id]
            for item in data["samples"]:
                sample = Sample.from_dict(item)
                prompts.append(str(sample.prompt))
                assert len(sample.rollout_log_probs) == sample.response_length
                assert sample.loss_mask is None or len(sample.loss_mask) == sample.response_length
                versions = [int(version) for version in sample.weight_versions]
                assert versions and versions == sorted(versions)
                multi_version += len(set(versions)) > 1
                rewards.append(sample.reward)
            grad = torch.load(Path(directory) / f"grad_{rollout_id}_0.pt", weights_only=False)
            assert torch.isfinite(torch.as_tensor(grad)).all()
        # The first pass trains every prompt once before the next pass starts.
        assert len(set(prompts[:NUM_PROMPTS])) == NUM_PROMPTS
        print(f"mean reward {sum(rewards) / len(rewards):.3f}; samples across a weight update: {multi_version}")


if __name__ == "__main__":
    prepare()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    execute()
