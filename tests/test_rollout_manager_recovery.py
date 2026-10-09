"""Lose a manager or RPC reply at persistence boundaries and consume its replay.

Two local Ray nodes run the production manager, distributed generators and Straw
controller. Inference and the optimizer use CPU fixtures; this does not exercise
SGLang or Megatron GPU recovery.
"""

import json
import os
import signal
import sys
from pathlib import Path

import pytest
import ray
import torch
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from test_distributed_rollout import _rollout_args

from slime.data.checkpoint import RestorePlan
from slime.data.queue_data_source import create_queue_controller
from slime.data.transport import load_rollout_samples
from slime.ray.rollout import RolloutManager
from slime.ray.serving import ServingCluster, ServingDeployment
from slime.rollout.base_types import iter_samples
from slime.utils.data import process_rollout_data

NUM_GPUS = 0


@ray.remote(num_cpus=0)
class _ServingFixture:
    """Keep the production queue alive without starting GPU inference servers."""

    def __init__(self, args):
        self.controller = create_queue_controller(args)

    def deployment(self, reused):
        return ServingDeployment(
            placements={"rollout": None},
            servers={},
            controller=self.controller,
            restore_plan=RestorePlan(),
            routers={},
            engine_lock=None,
            reused=reused,
        )

    def identity(self):
        return os.getpid(), self.controller._actor_id.hex()

    def health_monitoring_resume(self):
        return {}

    def finish_rollout(self):
        return {}

    def get_weight_version(self, *, allow_inconsistent=False):
        return 1

    def dispose(self):
        ray.get(self.controller.close.remote())
        ray.kill(self.controller, no_restart=True)


def _post_process_once(args, samples):
    # A successfully journaled conversion must never invoke the hook again.
    indices = sorted(sample.index for sample in samples)
    path = Path(args.rollout_data_dir).parent / f"converted_{indices[0]}.json"
    with path.open("x") as stream:
        json.dump(indices, stream)
    rewards = [sample.reward for sample in samples]
    return rewards, [3 * reward for reward in rewards]


def _install_crash(manager, phase):
    from straw.reporting import write_report

    def crash(raw):
        samples = list(iter_samples(load_rollout_samples(raw)))
        write_report(
            Path(manager.args.rollout_data_dir).parent / "crash.json",
            {
                "manager_pid": os.getpid(),
                "samples": [
                    {"index": sample.index, "tokens": sample.tokens, "reward": sample.reward} for sample in samples
                ],
            },
        )
        if phase == "conversion_reply_lost":
            raise TimeoutError("Conversion committed but its RPC reply was lost")
        os.kill(os.getpid(), signal.SIGKILL)

    if phase == "raw_accepted":
        original = manager._get_rollout_data

        def accept(*args, **kwargs):
            original(*args, **kwargs)
            crash(manager.batch_builder.raw_ref)

        manager._get_rollout_data = accept
    elif phase in {"conversion_accepted", "conversion_reply_lost"}:
        original = manager.batch_builder.publish_converted

        def publish(data):
            original(data)
            manager.batch_builder.publish_converted = original
            crash(manager.batch_builder.raw_ref)

        manager.batch_builder.publish_converted = publish
    else:
        original = manager.recovery.remember_converted

        def remember(rollout_id, data, batch_id):
            original(rollout_id, data, batch_id)
            crash(manager.recovery.batches[rollout_id].raw)

        manager.recovery.remember_converted = remember


def _consume_batch(refs, rank):
    data = process_rollout_data(refs, rank, len(refs))
    weight = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([weight], lr=0.01)
    inputs = torch.tensor([float(tokens.sum()) for tokens in data["tokens"]])
    targets = torch.tensor(data["local_raw_reward"]) + 1
    loss = ((inputs * weight - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and weight.item() > 0
    return {
        "node": ray.get_runtime_context().get_node_id(),
        "samples": [
            {"index": index, "tokens": tokens.tolist(), "reward": reward}
            for index, tokens, reward in zip(data["sample_indices"], data["tokens"], data["rewards"], strict=True)
        ],
    }


@pytest.fixture(scope="module")
def cluster():
    cluster = Cluster()
    try:
        for _ in range(2):
            cluster.add_node(num_cpus=3, num_gpus=0, object_store_memory=128 * 1024**2, include_dashboard=False)
        ray.init(
            address=cluster.address,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join(
                        [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]
                    ),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            },
        )
        yield sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
    finally:
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.parametrize("online_gc", [False, True], ids=["retain", "online-gc"])
@pytest.mark.parametrize("phase", ["raw_accepted", "conversion_accepted", "converted", "conversion_reply_lost"])
def test_manager_failure_replays_accepted_batch_with_new_dp(tmp_path, cluster, phase, online_gc):
    args = _rollout_args(tmp_path)
    args.rollout_queue_run_id = "manager-recovery"
    args.rollout_queue_online_gc = online_gc
    args.rollout_queue_segment_mib = 1
    args.start_rollout_id = 0
    args.save_debug_rollout_data = None
    args.custom_reward_post_process_path = "test_rollout_manager_recovery._post_process_once"
    parallel = dict(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    serving = _ServingFixture.remote(args)
    managers = []
    try:
        identity = ray.get(serving.identity.remote(), timeout=60)
        deployment = ray.get(serving.deployment.remote(False), timeout=60)
        manager = RolloutManager.options(num_cpus=1).remote(args, None, serving=serving, deployment=deployment)
        managers.append(manager)
        ray.get(manager.attach_training.remote(args, deployment), timeout=60)
        ray.get(manager.set_train_parallel_config.remote(parallel), timeout=60)
        ray.get(manager.load.remote(-1), timeout=60)
        ray.get(manager.__ray_call__.remote(_install_crash, phase), timeout=60)
        failure = ray.exceptions.RayTaskError if phase == "conversion_reply_lost" else ray.exceptions.RayActorError
        with pytest.raises(failure):
            ray.get(manager.generate.remote(0), timeout=90)
        before = json.loads((tmp_path / "crash.json").read_text())
        assert len(before["samples"]) == args.global_batch_size

        deployment = ray.get(serving.deployment.remote(True), timeout=60)
        if phase != "conversion_reply_lost":
            manager = RolloutManager.options(num_cpus=1).remote(args, None, serving=serving, deployment=deployment)
            managers.append(manager)
        resumed = ray.get(manager.attach_training.remote(args, deployment), timeout=60)
        assert resumed.checkpoint.start_rollout_id == 0
        ray.get(manager.set_train_parallel_config.remote({**parallel, "dp_size": 2}), timeout=60)
        ray.get(manager.load.remote(-1), timeout=60)
        refs = ray.get(manager.generate.remote(0), timeout=90)
        assert len(refs) == 2
        results = ray.get(
            [
                ray.remote(_consume_batch)
                .options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False))
                .remote(refs, rank)
                for rank, node in enumerate(cluster)
            ],
            timeout=60,
        )
        assert len({result["node"] for result in results}) == 2
        expected = [{**sample, "reward": 3 * sample["reward"]} for sample in before["samples"]]
        actual = [sample for result in results for sample in result["samples"]]
        assert sorted(actual, key=lambda sample: sample["index"]) == sorted(
            expected, key=lambda sample: sample["index"]
        )
        assert ray.get(serving.identity.remote(), timeout=30) == identity
        same_manager = (
            ray.get(manager.__ray_call__.remote(lambda self: os.getpid()), timeout=30) == before["manager_pid"]
        )
        assert same_manager == (phase == "conversion_reply_lost")
        assert len(list(tmp_path.glob("converted_*.json"))) == 1
        ray.get(manager.training_completed.remote(0), timeout=60)

        # The retained controller must advance past the replayed selection.
        refs = ray.get(manager.generate.remote(1), timeout=90)
        next_samples = [sample for rank in range(2) for sample in _consume_batch(refs, rank)["samples"]]
        assert {sample["index"] for sample in actual}.isdisjoint(sample["index"] for sample in next_samples)
        ray.get(manager.training_completed.remote(1), timeout=60)
        ray.get(manager.dispose.remote(), timeout=60)
    finally:
        for manager in managers:
            ray.kill(manager, no_restart=True)
        ray.kill(serving, no_restart=True)


@ray.remote(num_cpus=0)
class _ResetPeer:
    def __init__(self, wedged, blocked_peer=None):
        self.wedged = wedged
        self.blocked_peer = blocked_peer

    def pid(self):
        return os.getpid()

    def get_url(self):
        return "http://engine"

    def reset_weights_update_groups(self):
        if self.blocked_peer is not None:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(self.blocked_peer.pid.remote(), timeout=1)
        if self.wedged:
            import time

            time.sleep(300)

    def shutdown(self):
        pass


@pytest.mark.parametrize("block_actor", [False, True])
def test_serving_reset_retires_blocked_actor_and_keeps_healthy_peer(cluster, monkeypatch, tmp_path, block_actor):
    import time
    from types import SimpleNamespace
    from slime.backends.sglang_utils import engine_group

    blocked = _ResetPeer.remote(True)
    healthy = _ResetPeer.remote(False, blocked if block_actor else None)
    pids = ray.get([blocked.pid.remote(), healthy.pid.remote()], timeout=30)
    if block_actor:
        marker = tmp_path / "blocked"

        def block(actor):
            marker.touch()
            time.sleep(300)

        blocked.__ray_call__.remote(block)
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
    removed = []
    monkeypatch.setattr(engine_group, "unregister_worker", lambda router, url, **kw: removed.append(url))
    group = engine_group.ServerGroup(
        args=SimpleNamespace(num_gpus_per_node=1),
        pg=None,
        all_engines=[blocked, healthy],
        num_gpus_per_engine=1,
        num_new_engines=0,
        router_ip="router",
        router_port=1,
        engine_urls={0: "http://blocked", 1: "http://healthy"},
    )
    try:
        start = time.monotonic()
        engine_group.reset_weights_update_groups([group], timeout=0.5)
        assert time.monotonic() - start < 5
        assert group.all_engines == [None, healthy]
        assert removed == ["http://blocked"]
        assert ray.get(healthy.pid.remote(), timeout=5) == pids[1]
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(blocked.pid.remote(), timeout=5)
    finally:
        ray.kill(blocked, no_restart=True)
        ray.kill(healthy, no_restart=True)


@ray.remote(num_cpus=0)
class _AttemptManager:
    def __init__(self, blocked):
        self.blocked = blocked

    def ready(self):
        return True

    def detach_training(self):
        if self.blocked:
            import time

            time.sleep(300)
        raise RuntimeError("consumer pause failed")


@ray.remote(num_cpus=0)
class _ResetBarrier:
    """Model NCCL group destruction: every peer must enter before any can leave."""

    def __init__(self):
        self.entered = set()

    def enter(self, name):
        self.entered.add(name)

    def ready(self):
        return len(self.entered) == 2


@ray.remote(num_cpus=0)
class _PDResetPeer:
    def __init__(self, barrier, name):
        self.barrier, self.name = barrier, name

    def reset_weights_update_groups(self):
        import time

        ray.get(self.barrier.enter.remote(self.name))
        while not ray.get(self.barrier.ready.remote()):
            time.sleep(0.01)

    def ready(self):
        return True

    def get_url(self):
        return self.name


def test_prefill_and_decode_enter_group_reset_together(cluster):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from slime.backends.sglang_utils.engine_group import reset_weights_update_groups

    barrier = _ResetBarrier.remote()
    peers = [_PDResetPeer.remote(barrier, name) for name in ("prefill", "decode")]
    try:
        ray.get([peer.ready.remote() for peer in peers], timeout=30)
        groups = [SimpleNamespace(all_engines=[peer], nodes_per_engine=1, retire_engine=Mock()) for peer in peers]
        reset_weights_update_groups(groups, timeout=2)
        for group in groups:
            group.retire_engine.assert_not_called()
        assert ray.get(barrier.ready.remote(), timeout=5)
    finally:
        for actor in [*peers, barrier]:
            ray.kill(actor, no_restart=True)


@ray.remote(num_cpus=0)
class _AttemptServing(ServingCluster.__ray_metadata__.modified_class):
    """Use production detach/fencing with a CPU trainer and no inference GPUs."""

    def __init__(self, trainer, job_id):
        from types import SimpleNamespace

        self.args = SimpleNamespace(rollout_cleanup_timeout=2)
        self.training_actors = {"actor": [trainer]}
        self.driver_job_id = job_id
        self._health_monitors = []

    def state(self):
        return os.getpid(), self.driver_job_id, self.training_actors


@pytest.mark.parametrize("blocked", [False, True], ids=["pause-error", "wedged-manager"])
def test_driver_detach_releases_real_trainer_when_manager_cannot_pause(cluster, blocked):
    import time
    from types import SimpleNamespace

    from slime.ray.placement_group import RolloutStartup

    job_id = ray.get_runtime_context().get_job_id()
    trainer = _ResetPeer.remote(False)
    manager = _AttemptManager.remote(blocked)
    serving = _AttemptServing.remote(trainer, job_id)
    try:
        identity = ray.get(serving.state.remote(), timeout=30)[0]
        ray.get([manager.ready.remote(), trainer.pid.remote()], timeout=30)
        startup = RolloutStartup(SimpleNamespace(rollout_cleanup_timeout=2), manager=manager, serving=serving)
        start = time.monotonic()
        startup.close(failed=True)
        assert time.monotonic() - start < 5
        assert ray.get(serving.state.remote(), timeout=5) == (identity, None, {"actor": []})
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(trainer.pid.remote(), timeout=5)
        # Repeat a lost-reply cleanup and reject a stale driver's later request.
        ray.get(serving.detach_training.remote(job_id), timeout=5)
        ray.get(serving.__ray_call__.remote(lambda self: setattr(self, "driver_job_id", "successor")), timeout=5)
        with pytest.raises(ray.exceptions.RayTaskError, match="owning driver"):
            ray.get(serving.detach_training.remote(job_id), timeout=5)
    finally:
        for actor in (manager, serving, trainer):
            ray.kill(actor, no_restart=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
