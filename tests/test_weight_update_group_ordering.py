import threading
from datetime import timedelta
from types import SimpleNamespace

import pytest

from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as update_module


NUM_GPUS = 0


class _OppositeArrivalScheduler:
    """Model two serial engine actors choosing opposite first requests."""

    def __init__(self):
        self._condition = threading.Condition()
        self._groups = []
        self._ordered_phase_started = False
        self._release_deadlock = threading.Event()
        self.cycle_observed = False

    @property
    def groups(self):
        with self._condition:
            return list(self._groups)

    def ordered_phase_started(self):
        with self._condition:
            self._ordered_phase_started = True
            self._condition.notify_all()

    def connect(self, _args, group_name, _engines, engine_gpu_counts=None):
        del engine_gpu_counts
        with self._condition:
            self._groups.append(group_name)
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._ordered_phase_started or len(self._groups) == 2)
            if not self._ordered_phase_started and len(self._groups) == 2:
                # Engine 0 starts PP0 while engine 1 starts PP1. Both serial
                # actors are now blocked in different process-group joins.
                self.cycle_observed = True

        if self.cycle_observed:
            self._release_deadlock.wait()
        return object()

    def release(self):
        self._release_deadlock.set()


def test_pp_weight_update_groups_are_connected_in_global_order(monkeypatch):
    scheduler = _OppositeArrivalScheduler()
    train_barrier = threading.Barrier(2)
    rank_context = threading.local()

    monkeypatch.setattr(
        update_module.mpu,
        "get_data_parallel_rank",
        lambda with_context_parallel=True: 0,
    )
    monkeypatch.setattr(update_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        update_module.mpu,
        "get_pipeline_model_parallel_rank",
        lambda: rank_context.pp_rank,
    )
    monkeypatch.setattr(update_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(update_module, "connect_rollout_engines_from_distributed", scheduler.connect)
    monkeypatch.setattr(update_module, "get_gloo_group", lambda: object())

    def all_reduce(failed, *, op, group):
        del failed, op
        del group
        scheduler.ordered_phase_started()
        train_barrier.wait(timeout=2)

    monkeypatch.setattr(update_module.dist, "all_reduce", all_reduce)

    updaters = []
    for _ in range(2):
        updater = update_module.UpdateWeightFromDistributed.__new__(update_module.UpdateWeightFromDistributed)
        updater.args = SimpleNamespace(rollout_num_gpus_per_engine=1)
        updater._model_update_groups = None
        updaters.append(updater)

    errors = []

    def connect(pp_rank):
        rank_context.pp_rank = pp_rank
        try:
            updaters[pp_rank].connect_rollout_engines(
                rollout_engines=("engine-0", "engine-1"),
                rollout_engine_lock=object(),
                engine_gpu_counts=(1, 1),
            )
        except BaseException as exc:  # surface worker-thread failures in the test thread
            errors.append(exc)

    threads = [threading.Thread(target=connect, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    stuck = [thread.is_alive() for thread in threads]
    scheduler.release()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert not any(stuck), "opposite engine arrival order formed a PP0/PP1 connection cycle"
    assert not scheduler.cycle_observed
    assert scheduler.groups == ["slime-pp_0", "slime-pp_1"]


class _RemoteMethod:
    def __init__(self, ref):
        self.ref = ref

    def remote(self, **_kwargs):
        return self.ref


class _Engine:
    def __init__(self, ref):
        self.init_weights_update_group = _RemoteMethod(ref)


class _Lock:
    def __init__(self):
        self.acquire = _RemoteMethod(True)
        self.release = _RemoteMethod(None)


def test_failed_group_join_cancels_refs_and_kills_associated_engines(monkeypatch):
    engines = [_Engine("ref-0"), _Engine("ref-1")]
    cancelled = []
    killed = []

    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def bind(self, _address):
            return None

        def getsockname(self):
            return "127.0.0.1", 12345

    monkeypatch.setattr(update_module.socket, "socket", _Socket)
    monkeypatch.setattr(update_module.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(update_module.accelerator, "weight_update_backend", lambda: "nccl")

    def fail_group_creation(**kwargs):
        assert kwargs["timeout"] == timedelta(seconds=0.25)
        raise RuntimeError("injected group-join failure")

    monkeypatch.setattr(update_module, "init_process_group", fail_group_creation)
    monkeypatch.setattr(update_module.ray, "cancel", lambda ref, force: cancelled.append((ref, force)))
    monkeypatch.setattr(
        update_module.ray,
        "kill",
        lambda engine, no_restart: killed.append((engine, no_restart)),
    )

    args = SimpleNamespace(
        rollout_num_gpus_per_engine=1,
        update_weight_group_timeout_seconds=0.25,
    )
    with pytest.raises(RuntimeError, match="injected group-join failure"):
        update_module.connect_rollout_engines_from_distributed(
            args,
            "slime-pp_0",
            engines,
            engine_gpu_counts=(1, 1),
        )

    assert cancelled == [("ref-0", False), ("ref-1", False)]
    assert killed == [(engines[0], True), (engines[1], True)]


def test_transfer_failure_releases_rollout_engine_lock(monkeypatch):
    released = []
    lock = _Lock()
    lock.release.remote = lambda: released.append(True)

    def get(refs, **_kwargs):
        if refs == "transfer-ref":
            raise RuntimeError("injected transfer failure")
        return refs

    monkeypatch.setattr(update_module.ray, "get", get)
    monkeypatch.setattr(
        update_module,
        "update_weights_from_distributed",
        lambda *_args, **_kwargs: "transfer-ref",
    )

    updater = update_module.UpdateWeightFromDistributed.__new__(update_module.UpdateWeightFromDistributed)
    updater.rollout_engine_lock = lock
    updater.rollout_engines = ()
    updater._group_name = "slime-pp_0"
    updater._model_update_groups = object()
    updater.weight_version = 1

    with pytest.raises(RuntimeError, match="injected transfer failure"):
        updater._update_bucket_weights_from_distributed([("weight", object())])

    assert released == [True]


def test_failed_reconnect_does_not_retain_destroyed_group(monkeypatch):
    old_group = object()
    disconnected = []

    monkeypatch.setattr(
        update_module.mpu,
        "get_data_parallel_rank",
        lambda with_context_parallel=True: 0,
    )
    monkeypatch.setattr(update_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(update_module.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(update_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(update_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(update_module.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        update_module,
        "disconnect_rollout_engines_from_distributed",
        lambda name, group, engines: disconnected.append((name, group, engines)),
    )

    def fail_connect(*_args, **_kwargs):
        raise RuntimeError("injected reconnect failure")

    monkeypatch.setattr(update_module, "connect_rollout_engines_from_distributed", fail_connect)

    updater = update_module.UpdateWeightFromDistributed.__new__(update_module.UpdateWeightFromDistributed)
    updater.args = SimpleNamespace(rollout_num_gpus_per_engine=1)
    updater._model_update_groups = old_group

    with pytest.raises(RuntimeError, match="injected reconnect failure"):
        updater.connect_rollout_engines(
            rollout_engines=("engine-0",),
            rollout_engine_lock=object(),
            engine_gpu_counts=(1,),
        )

    assert disconnected == [("slime-pp_0", old_group, ("engine-0",))]
    assert updater._model_update_groups is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
