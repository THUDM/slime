"""Real process death and restart of fully async rollout on a local straw pool.

Two logical Ray nodes run the production queue and generation actors. Only
inference/reward are deterministic CPU fixtures; no model or GPU is required.
The supervisor kills the entire first job without close/save, then starts a
fresh driver and cluster against its persisted queue.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import pytest
import torch
from test_distributed_rollout import _generate_locally, _rollout_args

from slime.rollout.base_types import iter_samples
from slime.utils.types import Sample

NUM_GPUS = 0
TENSOR_FIELDS = ("rollout_routed_experts", "rollout_topk_token_ids", "rollout_topk_log_probs")


def _tensor(value):
    return value.load() if hasattr(value, "load") else torch.as_tensor(value)


def _prefix(sample):
    return {
        "index": sample.index,
        "tokens": list(sample.tokens),
        "log_probs": list(sample.rollout_log_probs),
        **{key: _tensor(getattr(sample, key)).tolist() for key in TENSOR_FIELDS},
    }


async def _generate_with_interrupt(args, sample, sampling_params):
    # One group completes, one persists a prefix, all other work stays in flight.
    # Gates are task identities, so node/actor scheduling cannot move the boundary.
    if sample.group_index > 1:
        await asyncio.Event().wait()
    if args.test_recovery_phase == "interrupt":
        if sample.response_length:
            await asyncio.Event().wait()
        sample = await _generate_locally(args, sample, sampling_params)
        sample.metadata["generation_pid"] = os.getpid()
        if sample.group_index == 0:
            sample.tokens = sample.tokens[:2]
            sample.response_length = 1
            sample.response = "partial"
            sample.loss_mask = [1]
            sample.rollout_log_probs = sample.rollout_log_probs[:1]
            for key in TENSOR_FIELDS:
                setattr(sample, key, getattr(sample, key)[:1])
            sample.status = Sample.Status.ABORTED
        return sample

    assert sample.group_index == 0, "An accepted group was generated again instead of replaying its receipt"
    assert sample.response_length == 1, "Restart discarded the durable prefix"
    sample.metadata["resumed_prefix"] = _prefix(sample)
    sample.metadata["resumed_lease"] = dict(sample._queue_lease)
    sample.tokens.append(3)
    sample.response_length += 1
    sample.response = "answer"
    sample.loss_mask.append(1)
    sample.rollout_log_probs.append(-0.7)
    rows = (
        torch.full((1, 1, 1), sample.index % 8, dtype=torch.uint8),
        torch.tensor([[3, 5]], dtype=torch.int32),
        torch.tensor([[-0.7, -2.5]], dtype=torch.float32),
    )
    for key, row in zip(TENSOR_FIELDS, rows, strict=True):
        setattr(sample, key, torch.cat((_tensor(getattr(sample, key)), row)))
    sample.status = Sample.Status.COMPLETED
    return sample


async def _reward_once(args, sample):
    assert sample.metadata.get("reward_calls", 0) == 0, "A completed sample was scored twice"
    sample.metadata["reward_calls"] = 1
    return float(sample.index % 2)


def _run_job(directory, phase, online_gc):
    import ray
    from ray.cluster_utils import Cluster
    from straw.protocol import Lease, RecordSetRef
    from straw.reporting import write_report

    from slime.data.queue_data_source import QueueDataSource
    from slime.data.transport import DiskPayloadRef, load_rollout_samples
    from slime.rollout.fully_async_rollout import generate_rollout_fully_async

    root = Path(directory)
    args = _rollout_args(root)
    args.rollout_batch_size = 2
    args.sglang_server_concurrency = 4
    args.rollout_queue_segment_mib = 1
    args.rollout_queue_online_gc = online_gc
    from slime.data.checkpoint import RestorePlan

    restore_plan = RestorePlan(mode="resume" if phase == "resume" else "new")
    args.custom_generate_function_path = "test_straw_fully_async_recovery._generate_with_interrupt"
    args.custom_rm_path = "test_straw_fully_async_recovery._reward_once"
    args.group_rm = False
    args.test_recovery_phase = phase
    cluster = Cluster()
    source = None
    try:
        for _ in range(2):
            cluster.add_node(num_cpus=2, num_gpus=0, object_store_memory=128 * 1024**2, include_dashboard=False)
        ray.init(address=cluster.address, runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})
        source = QueueDataSource(args, restore_plan=restore_plan)
        controller = source.controller
        if phase == "interrupt":
            failure = []

            def generate():
                try:
                    generate_rollout_fully_async(args, 0, source)
                except BaseException as error:
                    failure.append(error)

            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if failure:
                    raise failure[0]
                statuses = ray.get([controller.status.remote(f"prompt:{i}") for i in range(2)], timeout=10)
                if all(statuses) and statuses[0]["state"] == "leased" and statuses[1]["state"] == "completed":
                    partial = DiskPayloadRef(
                        RecordSetRef.from_dict(statuses[0]["spec"]["input_ref"]), args.rollout_data_dir
                    ).load()
                    if all(sample.response_length == 1 for sample in partial):
                        receipt = ray.get(controller.result.remote(Lease(**statuses[1]["lease"])), timeout=10)
                        accepted = DiskPayloadRef(receipt.result_ref, args.rollout_data_dir).load()
                        write_report(
                            root / "interrupted.json",
                            {
                                "prefixes": [_prefix(sample) for sample in partial],
                                "old_lease": statuses[0]["lease"],
                                "receipt": asdict(receipt),
                                "worker_pids": sorted(
                                    {sample.metadata["generation_pid"] for sample in [*partial, *accepted]}
                                ),
                            },
                        )
                        # No graceful save/close is allowed before the parent kills us.
                        threading.Event().wait()
                time.sleep(0.05)
            raise TimeoutError("Fully async did not persist both a partial and an accepted group")

        before = json.loads((root / "interrupted.json").read_text())
        assert ray.get(controller.heartbeat.remote([Lease(**before["old_lease"])])) == ["StaleAttempt"]
        if online_gc:
            ray.get(controller._collect_storage.remote())
        result = generate_rollout_fully_async(args, 0, source)
        samples = list(iter_samples(load_rollout_samples(result.samples)))
        assert sorted(sample.index for sample in samples) == [0, 1, 2, 3]
        assert all(sample.response_length == 2 and sample.metadata["reward_calls"] == 1 for sample in samples)
        for sample in samples[:2]:
            assert sample.metadata["resumed_prefix"] == before["prefixes"][sample.index]
            lease = sample.metadata["resumed_lease"]
            assert lease["coordinator_epoch"] != before["old_lease"]["coordinator_epoch"]
            assert lease["attempt_id"] != before["old_lease"]["attempt_id"]
            assert sample.loss_mask == [0, 1]
            actual = _prefix(sample)
            for key in ("tokens", "log_probs", *TENSOR_FIELDS):
                old = before["prefixes"][sample.index][key]
                assert actual[key][: len(old)] == old
        # Query receipts without republishing the startup recovery snapshot.
        # Consecutive positions also detect duplicate acceptance during restart.
        statuses = ray.get([controller.status.remote(f"prompt:{i}") for i in range(2)])
        assert all(status["state"] == "completed" for status in statuses)
        history = [
            asdict(receipt)
            for receipt in ray.get([controller.result.remote(Lease(**status["lease"])) for status in statuses])
        ]
        assert {item["task_id"] for item in history} == {"prompt:0", "prompt:1"}
        assert sorted(item["position"] for item in history) == [0, 1]
        assert ray.get(controller.metrics.remote())["tasks"]["completed"] == 2
        assert next(item for item in history if item["task_id"] == "prompt:1") == before["receipt"]
        write_report(
            root / "resumed.json", {"samples": len(samples), "accepted_groups": len(history), "online_gc": online_gc}
        )
    finally:
        if source is not None:
            source.close()
        ray.shutdown()
        cluster.shutdown()


def _kill_job(process):
    """Freeze and kill only this subprocess tree, including its local Ray nodes."""
    if process.poll() is not None:
        return set()
    owner = psutil.Process(process.pid)
    owned = {owner}
    # Stop parents before enumerating again so Ray cannot fork new workers
    # between our ownership snapshot and SIGKILL. psutil checks PID reuse.
    while True:
        for child in owned:
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
        try:
            discovered = set(owner.children(recursive=True)) - owned
        except psutil.NoSuchProcess:
            discovered = set()
        if not discovered:
            break
        owned.update(discovered)
    for child in owned:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=20)
    deadline = time.monotonic() + 20
    while True:
        alive = []
        for child in owned:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.append(child)
            except psutil.NoSuchProcess:
                pass
        if not alive:
            break
        assert time.monotonic() < deadline, "A process from the old job is still alive"
        time.sleep(0.05)
    return {child.pid for child in owned}


@pytest.mark.parametrize("online_gc", [False, True], ids=["retain", "online-gc"])
def test_fully_async_sigkill_and_restart_preserves_results_and_prefixes(tmp_path, online_gc):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    code = "from test_straw_fully_async_recovery import _run_job; import sys; _run_job(sys.argv[1], sys.argv[2], sys.argv[3] == 'True')"
    for phase in ("interrupt", "resume"):
        log_path = tmp_path / f"{phase}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", code, str(tmp_path), phase, str(online_gc)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                if phase == "interrupt":
                    deadline = time.monotonic() + 180
                    while not (tmp_path / "interrupted.json").exists():
                        assert process.poll() is None, log_path.read_text()[-12000:]
                        assert time.monotonic() < deadline, log_path.read_text()[-12000:]
                        time.sleep(0.1)
                    before = json.loads((tmp_path / "interrupted.json").read_text())
                    assert len(before["worker_pids"]) == 2
                    assert not list((tmp_path / "checkpoint").rglob("queue_state_*.json"))
                    killed = _kill_job(process)
                    assert set(before["worker_pids"]) <= killed
                    assert process.returncode == -signal.SIGKILL
                else:
                    assert process.wait(timeout=180) == 0, log_path.read_text()[-12000:]
            finally:
                _kill_job(process)
    assert json.loads((tmp_path / "resumed.json").read_text()) == {
        "samples": 4,
        "accepted_groups": 2,
        "online_gc": online_gc,
    }
    assert sum(path.is_file() for path in (tmp_path / "queue").rglob("*")) < 64
    assert sum(path.stat().st_size for path in (tmp_path / "queue").rglob("*") if path.is_file()) < 8 * 1024**2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
