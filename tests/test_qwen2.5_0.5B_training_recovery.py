"""Restart Megatron with TP=2 after a real OOM, retaining the SGLang cluster."""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from shlex import quote
from types import SimpleNamespace

import pytest
import ray
import requests
from ray.job_submission import JobStatus, JobSubmissionClient

import slime.utils.external_utils.command_utils as U
from slime.ray.training_recovery import RECOVERY_NAMESPACE, training_session_name

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
REPO = Path(__file__).resolve().parents[1]


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def train_args(directory, mode, tp_size, model_path, dataset_path, save_interval, fault_tolerance):
    arguments = (
        f"--hf-checkpoint {quote(model_path)} --ref-load {quote(model_path)} "
        "--rollout-health-check-first-wait 600 "
        f"--rollout-data-transport {'straw' if mode == 'straw' else 'object-store'} "
        "--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async "
        f"--prompt-data {quote(dataset_path)} --input-key messages --label-key label --apply-chat-template "
        "--custom-rm-path training_recovery_test_helpers.reward "
        "--num-rollout 3 --rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16 "
        "--rollout-max-response-len 128 --rollout-temperature 0.8 "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 "
        f"--tensor-model-parallel-size {tp_size} --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
        "--sglang-server-concurrency 8 --sglang-mem-fraction-static 0.65 --sglang-cuda-graph-max-bs 16 "
        "--advantage-estimator grpo --use-rollout-logprobs --eps-clip 0.2 "
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        "--custom-megatron-before-train-step-hook-path training_recovery_test_helpers.before_train_step "
        f"--save {quote(str(directory / 'checkpoint'))} --save-interval {save_interval} "
    )
    if fault_tolerance:
        arguments += "--use-fault-tolerance "
    if tp_size > 1:
        arguments += "--sequence-parallel "
    if mode == "straw":
        arguments += f"--rollout-data-dir {quote(str(directory / 'queue'))} --rollout-queue-online-gc "
    else:
        arguments += f"--save-debug-rollout-data {quote(str(directory / 'rollout_{rollout_id}.pt'))} "
    return arguments


def wait_job(client, submission_id, path):
    # Large models also save and reload checkpoints on shared storage.
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        status = client.get_job_status(submission_id)
        if status.is_terminal():
            path.write_text(client.get_job_logs(submission_id))
            return status
        print(f"{submission_id}: {status}", flush=True)
        time.sleep(10)
    raise TimeoutError(f"Training job did not finish: {submission_id}")


def verify(directory, mode, failure_rollout, save_interval, kill_manager, wedge_engine=False, train_gpus=2):
    before = json.loads((directory / "before.json").read_text())
    after = json.loads((directory / "after.json").read_text())
    assert before["trainer"]["job_id"] != after["trainer"]["job_id"]
    assert before["trainer"]["pid"] != after["trainer"]["pid"]
    assert before["trainer"]["tp_size"] == 1 and after["trainer"]["tp_size"] == 2
    for name in (
        "serving_pid",
        "router_pids",
        "engines",
        "engine_actor_ids",
        "router",
        "controller_id",
        "rollout_pg_id",
        "sample_digest",
        "route_digest",
        "sample_indices",
    ):
        if wedge_engine and name in {"engines", "engine_actor_ids"}:
            # Reattachment retires only the peer that cannot fence its old RPCs.
            assert before[name][0] != after[name][0]
            assert before[name][1:] == after[name][1:]
        else:
            assert before[name] == after[name], (name, before[name], after[name])
    assert (before["manager_pid"] != after["manager_pid"]) == kill_manager
    assert before["parallel"]["dp_size"] == train_gpus and after["parallel"]["dp_size"] == train_gpus // 2
    restore_start = failure_rollout if save_interval == 1 else 0
    assert after["start_rollout_id"] == restore_start
    for rollout_id in range(restore_start, failure_rollout + 1):
        original = json.loads((directory / f"original_{rollout_id}.json").read_text())
        replayed = json.loads((directory / f"replayed_{rollout_id}.json").read_text())
        assert original["sample_digest"] == replayed["sample_digest"]
        assert original["route_digest"] == replayed["route_digest"]
        assert original["sample_indices"] == replayed["sample_indices"]
    assert before["scheduler_num_steps"] == after["scheduler_num_steps"] == 16 * failure_rollout
    assert all(
        int(new) > int(old) for old, new in zip(before["weight_versions"], after["weight_versions"], strict=True)
    )
    failed_log = (directory / "failed.log").read_text()
    assert "CUDA out of memory" in failed_log or "torch.OutOfMemoryError" in failed_log
    resumed_log = (directory / "resumed.log").read_text()
    gradients = [float(value) for value in re.findall(r"'train/grad_norm': ([^,}]+)", resumed_log)]
    # Include NaN/Inf in parsing and require every resumed step: a later failed
    # update must not be hidden by an earlier finite gradient in the same log.
    assert len(gradients) == 3 - restore_start and all(0 < value < float("inf") for value in gradients), gradients
    if mode == "straw":
        committed = json.loads((directory / "checkpoint/rollout/committed_2.json").read_text())
        assert committed["weight_version"] == int(after["weight_versions"][0]) + 2 - failure_rollout
    else:
        assert (directory / "checkpoint/iter_0000002").exists()
        continued = json.loads((directory / "continued_2.json").read_text())
        previous_indices = {
            index
            for rollout_id in range(failure_rollout + 1)
            for index in json.loads((directory / f"original_{rollout_id}.json").read_text())["sample_indices"]
        }
        assert previous_indices.isdisjoint(continued["sample_indices"])
    assert json.loads((directory / "serving_after_failure.json").read_text())["text"]
    assert not list(directory.glob("*.train-recovery.pt"))
    print("Trainer recovery verified: serving processes retained, data replayed with new DP, real training completed.")


def execute(
    mode="straw",
    failure_rollout=0,
    directory=None,
    model_path=None,
    dataset_path=None,
    role_config=False,
    save_interval=None,
    kill_manager=False,
    fault_tolerance=True,
    manager_crash_phase="after-failure",
    weight_sync="nccl",
    wedge_engine=False,
    pd=False,
    model_type=MODEL_TYPE,
    train_gpus=2,
    rollout_gpus=2,
    extra_args="",
):
    directory = Path(directory or tempfile.mkdtemp(prefix="slime_training_recovery_"))
    directory.mkdir(parents=True, exist_ok=True)
    save_interval = save_interval or (1 if failure_rollout else 3)
    # prepare() may need the download proxy; the driver and Ray head must then
    # contact cluster services directly, before any job runtime_env is applied.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    external_ray = os.environ.get("SLIME_SCRIPT_EXTERNAL_RAY") == "1"
    if not external_ray:
        subprocess.run(
            [
                "ray",
                "start",
                "--head",
                "--node-ip-address",
                "127.0.0.1",
                "--num-gpus",
                str(train_gpus + rollout_gpus),
                "--disable-usage-stats",
            ],
            check=True,
        )
    client = JobSubmissionClient(os.environ.get("SLIME_TEST_RAY_DASHBOARD", "http://127.0.0.1:8265"))
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{REPO}/tests:{REPO}:/root/Megatron-LM",
            "SLIME_RECOVERY_TEST_DIR": str(directory),
            "SLIME_RECOVERY_TEST_FAILURE_ROLLOUT": str(failure_rollout),
            "SLIME_RECOVERY_TEST_KILL_MANAGER": str(int(kill_manager and manager_crash_phase == "training")),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "RAY_USE_UVLOOP": "0",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            **{
                name: os.environ[name]
                for name in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_NVLS_ENABLE")
                if name in os.environ
            },
            **{
                name: ""
                for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
            },
        }
    }
    model_path = model_path or f"/root/models/{MODEL_NAME}"
    dataset_path = dataset_path or "/root/datasets/gsm8k/train.parquet"
    session_args = SimpleNamespace(
        rollout_data_transport="straw" if mode == "straw" else "object-store",
        rollout_data_dir=str(directory / "queue"),
        rollout_queue_run_id="rollout",
        save_debug_rollout_data=str(directory / "rollout_{rollout_id}.pt"),
    )
    manager = None
    try:
        for tp_size in (1, 2):
            cluster_address = os.environ.get("SLIME_TEST_RAY_ADDRESS", "127.0.0.1:6379")
            command = (
                f"cd {quote(str(REPO))} && source scripts/models/{model_type}.sh "
                f'&& export RAY_ADDRESS={quote(cluster_address)} && python train.py "${{MODEL_ARGS[@]}}" '
            )
            command += train_args(directory, mode, tp_size, model_path, dataset_path, save_interval, fault_tolerance)
            command += f"--actor-num-gpus-per-node {train_gpus} --rollout-num-gpus {rollout_gpus} " + extra_args + " "
            if pd:
                config = directory / "serving.json"
                config.write_text(
                    json.dumps(
                        {
                            "sglang": [
                                {
                                    "name": "default",
                                    "server_groups": [
                                        {
                                            "worker_type": side,
                                            "num_gpus": rollout_gpus // 2,
                                            "num_gpus_per_engine": rollout_gpus // 2,
                                            "overrides": {"disaggregation_transfer_backend": "mooncake"},
                                        }
                                        for side in ("prefill", "decode")
                                    ],
                                }
                            ]
                        }
                    )
                )
                command += f"--sglang-config {quote(str(config))} "

            if weight_sync == "disk-delta":
                command += (
                    "--update-weight-mode delta --update-weight-transport disk "
                    f"--update-weight-disk-dir {quote(str(directory / 'weights'))} "
                    f"--update-weight-local-checkpoint-dir {quote('/tmp/slime-delta-' + directory.name)} "
                )
            if wedge_engine:
                command += "--rollout-health-check-timeout 5 "
            if role_config:
                path = directory / f"megatron_tp{tp_size}.json"
                path.write_text(
                    json.dumps(
                        {
                            "megatron": [
                                {
                                    "name": "actor",
                                    "role": "actor",
                                    "overrides": {
                                        "load": model_path,
                                        "save": str(directory / "checkpoint"),
                                        "tensor_model_parallel_size": tp_size,
                                        "sequence_parallel": tp_size > 1,
                                        "update_weight_start_version": 0,
                                    },
                                }
                            ]
                        }
                    )
                )
                command += "--megatron-config-path " + quote(str(path))
            submission_id = client.submit_job(entrypoint="bash -c " + quote(command), runtime_env=runtime_env)
            (directory / f"submission_tp{tp_size}.json").write_text(
                json.dumps(
                    {"submission_id": submission_id, "entrypoint": command, "runtime_env": runtime_env}, indent=2
                )
            )
            status = wait_job(client, submission_id, directory / ("failed.log" if tp_size == 1 else "resumed.log"))
            expected = JobStatus.FAILED if tp_size == 1 else JobStatus.SUCCEEDED
            assert status == expected, (status, directory)
            if tp_size == 1:
                # Fail at the actual injection boundary, rather than accepting
                # an unrelated initialization failure as the expected OOM.
                assert (directory / "before.json").exists(), directory / "failed.log"
                ray.init(
                    address=os.environ.get("SLIME_TEST_RAY_ADDRESS", "127.0.0.1:6379"),
                    namespace=RECOVERY_NAMESPACE,
                    ignore_reinit_error=True,
                )
                serving = ray.get_actor(training_session_name(session_args) + ":serving", namespace=RECOVERY_NAMESPACE)
                assert ray.get(serving.get_updatable_engines_and_lock.remote())[0]
                if kill_manager and manager_crash_phase == "training":
                    try:
                        manager = ray.get_actor(training_session_name(session_args), namespace=RECOVERY_NAMESPACE)
                    except ValueError:
                        pass
                    else:
                        with pytest.raises(ray.exceptions.RayActorError):
                            ray.get(manager.get_updatable_engines_and_lock.remote())
                else:
                    manager = ray.get_actor(training_session_name(session_args), namespace=RECOVERY_NAMESPACE)
                    assert ray.get(manager.get_updatable_engines_and_lock.remote())[0]
                before = json.loads((directory / "before.json").read_text())
                if kill_manager and manager_crash_phase == "after-failure":
                    ray.kill(manager, no_restart=True)
                    assert ray.get(serving.get_updatable_engines_and_lock.remote())[0]
                with requests.Session() as http:
                    http.trust_env = False
                    response = http.post(
                        f"http://{before['router'][0]}:{before['router'][1]}/generate",
                        json={
                            "text": "The result of 1 + 1 is",
                            "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    (directory / "serving_after_failure.json").write_text(json.dumps(response.json()))
                if wedge_engine:
                    # Queue an RPC that never reaches reset/shutdown. The next
                    # attachment must time out, kill this actor and its SGLang
                    # children, then allocate a healthy replacement on its GPUs.
                    engine = ray.get(serving.get_updatable_engines_and_lock.remote())[0][0]
                    started = directory / "wedged_engine.pid"

                    def block(actor, started=started):
                        started.write_text(str(os.getpid()))
                        time.sleep(300)

                    engine.__ray_call__.remote(block)
                    deadline = time.monotonic() + 10
                    while not started.exists():
                        assert time.monotonic() < deadline
                        time.sleep(0.1)

            else:
                before = json.loads((directory / "before.json").read_text())
                # Successful completion must release retained serving resources,
                # without depending on the runner's final ray stop cleanup.
                with requests.Session() as http:
                    http.trust_env = False
                    for engine in before["engines"]:
                        deadline = time.monotonic() + 20
                        while True:
                            try:
                                http.get(engine["url"] + "/health", timeout=2)
                            except requests.ConnectionError:
                                break
                            assert time.monotonic() < deadline, engine
                            time.sleep(1)
        verify(directory, mode, failure_rollout, save_interval, kill_manager, wedge_engine, train_gpus)
    finally:
        if not ray.is_initialized():
            ray.init(address=os.environ.get("SLIME_TEST_RAY_ADDRESS", "127.0.0.1:6379"), namespace=RECOVERY_NAMESPACE)
        for suffix in ("", ":serving"):
            try:
                retained = ray.get_actor(training_session_name(session_args) + suffix, namespace=RECOVERY_NAMESPACE)
            except ValueError:
                continue
            try:
                ray.get(retained.dispose.remote(), timeout=60)
            except ray.exceptions.RayActorError:
                pass
            finally:
                ray.kill(retained, no_restart=True)
        ray.shutdown()
        if not external_ray:
            subprocess.run(["ray", "stop", "--force"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["straw", "debug"], default="straw")
    parser.add_argument("--failure-rollout", type=int, choices=[0, 1], default=0)
    parser.add_argument("--directory")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--role-config", action="store_true")
    parser.add_argument("--save-interval", type=int, choices=[1, 3])
    parser.add_argument("--kill-manager", action="store_true")
    parser.add_argument("--no-fault-tolerance", action="store_true")
    parser.add_argument("--manager-crash-phase", choices=["after-failure", "training"], default="after-failure")
    parser.add_argument("--weight-sync", choices=["nccl", "disk-delta"], default="nccl")
    parser.add_argument("--wedge-engine", action="store_true")
    parser.add_argument("--pd", action="store_true")
    cli = parser.parse_args()
    if not cli.no_prepare:
        prepare()
    execute(
        cli.mode,
        cli.failure_rollout,
        cli.directory,
        cli.model_path,
        cli.dataset_path,
        cli.role_config,
        cli.save_interval,
        cli.kill_manager,
        not cli.no_fault_tolerance,
        cli.manager_crash_phase,
        cli.weight_sync,
        cli.wedge_engine,
        cli.pd,
    )
