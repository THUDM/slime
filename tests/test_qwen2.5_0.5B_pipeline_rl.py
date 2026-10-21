"""PipelineRL e2e: real training plus live sequences spanning weight updates.

Train-step hooks start long streaming requests on every rollout engine and
check continuity after a weight update. A second set of probes verifies either
continued generation or the periodic full refresh, depending on the interval.
This catches abort pauses, KV flushes, and disk reload's implicit flush.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from shlex import quote

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def train_args(
    directory,
    *,
    transport="nccl",
    flush_cache_interval=0,
    actor_nodes=1,
    actor_gpus=1,
    rollout_gpus=3,
    engine_gpus=1,
    model_path=None,
    dataset_path=None,
):
    model_path = model_path or f"/root/models/{MODEL_NAME}"
    dataset_path = dataset_path or "/root/datasets/gsm8k/train.parquet"
    args = (
        f"--hf-checkpoint {quote(model_path)} --ref-load {quote(model_path)} "
        f"--flush-cache-interval {flush_cache_interval} --rollout-data-transport straw --rollout-queue-online-gc "
        f"--rollout-data-dir {quote(str(directory / 'queue'))} "
        f"--prompt-data {quote(dataset_path)} --input-key messages --label-key label --apply-chat-template "
        "--custom-rm-path pipeline_rl_test_helpers.reward --num-rollout 3 --rollout-batch-size 4 --n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 --rollout-temperature 0.8 --global-batch-size 16 --balance-data "
        "--sglang-server-concurrency 8 --sglang-mem-fraction-static 0.65 --sglang-cuda-graph-max-bs 16 "
        f"--actor-num-nodes {actor_nodes} --actor-num-gpus-per-node {actor_gpus} "
        f"--rollout-num-gpus {rollout_gpus} --rollout-num-gpus-per-engine {engine_gpus} "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
        "--advantage-estimator grpo --use-rollout-logprobs --eps-clip 0.2 "
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        "--custom-megatron-before-train-step-hook-path pipeline_rl_test_helpers.before_train_step "
        f"--save {quote(str(directory / 'checkpoint'))} --save-interval 3 --ci-test --ci-disable-kl-checker "
    )
    if transport == "disk":
        args += f"--update-weight-transport disk --update-weight-disk-dir {quote(str(directory / 'weights'))} "
    return args


def verify(directory, expected_engines, flush_cache_interval=0):
    results = json.loads((directory / "probes.json").read_text())
    assert results["weights_changed"]
    records = results["continuous"]
    assert len(records) == expected_engines
    for record in records:
        assert record["before"]["version"] == "1", record
        assert record["after"]["version"] == "2", record
        assert record["after"]["tokens"] > record["before"]["tokens"] + 16, record
    records = results["next_update"]
    assert len(records) == expected_engines
    for record in records:
        assert record["before"]["version"] == "2", record
        if flush_cache_interval == 2:
            assert record["after"]["finish_reason"]["type"] == "abort", record
            assert record["serving_version"] == "3", record
        else:
            assert record["after"]["version"] == "3", record
            assert record["after"]["tokens"] > record["before"]["tokens"] + 16, record
    assert (directory / "checkpoint" / "rollout" / "committed_2.json").exists()
    print(f"PipelineRL verified on {expected_engines} engines: interval={flush_cache_interval}; checkpoint committed.")


def execute(transport, flush_cache_interval):
    with tempfile.TemporaryDirectory(prefix="slime_pipeline_rl_") as tmp:
        directory = Path(tmp)
        U.execute_train(
            train_args=train_args(directory, transport=transport, flush_cache_interval=flush_cache_interval),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
            extra_env_vars={
                "PYTHONPATH": f"{U.repo_base_dir / 'tests'}:{U.repo_base_dir}:/root/Megatron-LM/",
                "SLIME_PIPELINE_RL_PROBE_FILE": str(directory / "probes.json"),
                "NO_PROXY": "*",
                "no_proxy": "*",
                **{
                    key: ""
                    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy")
                },
            },
        )
        verify(directory, expected_engines=3, flush_cache_interval=flush_cache_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["nccl", "disk"], default="nccl")
    parser.add_argument("--flush-cache-interval", type=int, choices=[-1, 0, 2], default=0)
    options = parser.parse_args()
    prepare()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    execute(options.transport, options.flush_cache_interval)
