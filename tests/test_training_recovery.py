"""Trainer restart boundaries and parallelism-independent rollout retention."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.data.batch_builder import BatchBuilder
from slime.data.checkpoint import RestorePlan
from slime.data.transport import pack_rollout_payload, rollout_store
from slime.ray.serving import ServingCluster
from slime.ray.training_recovery import (
    TrainingRecovery,
    retained_rollout_configuration,
    training_recovery_enabled,
    training_session_name,
)
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        use_fault_tolerance=True,
        rollout_data_transport="straw",
        rollout_data_dir=str(tmp_path / "pool"),
        rollout_queue_run_id="test",
        save_debug_rollout_data=None,
        load_debug_rollout_data=None,
        debug_train_only=False,
        debug_rollout_only=False,
        load="initial-model",
        save=str(tmp_path / "model"),
        start_rollout_id=0,
        ckpt_step=None,
        finetune=True,
        no_load_optim=True,
        no_load_rng=True,
        hf_checkpoint="model",
        global_batch_size=4,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        balance_data=False,
        balance_by_flops=False,
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
    )


@pytest.mark.parametrize(
    "transport,dump,enabled",
    [
        ("straw", None, True),
        ("object-store", "rollout_{rollout_id}.pt", True),
        ("object-store", None, False),
        ("nixl", "rollout_{rollout_id}.pt", True),
    ],
)
def test_replay_persistence_is_independent_of_health_checks(args, transport, dump, enabled):
    args.rollout_data_transport, args.save_debug_rollout_data = transport, dump
    assert training_recovery_enabled(args) == enabled
    args.use_fault_tolerance = False
    assert training_recovery_enabled(args) == enabled


@pytest.mark.parametrize("mode", ["debug_train_only", "debug_rollout_only", "load_debug_rollout_data"])
def test_debug_only_modes_do_not_replay_training(args, mode):
    setattr(args, mode, True)
    assert not training_recovery_enabled(args)


def test_session_identity_survives_checkpoint_branch_and_parallelism_changes(args):
    name = training_session_name(args)
    args.save += "/branches/new"
    args.tensor_model_parallel_size = 8
    args.actor_num_gpus_per_node = 16
    assert training_session_name(args) == name
    args.rollout_queue_run_id = "another-run"
    assert training_session_name(args) != name


def test_configuration_allows_trainer_parallelism_and_memory_changes(args):
    serving = object.__new__(ServingCluster.__ray_metadata__.modified_class)
    serving.configuration = retained_rollout_configuration(args)
    serving.driver_job_id = None
    updated = copy.copy(args)
    updated.tensor_model_parallel_size = 4
    updated.context_parallel_size = 2
    updated.micro_batch_size = 2
    updated.max_tokens_per_gpu = 1024
    serving.validate_attachment(updated)
    updated.global_batch_size = 8
    with pytest.raises(ValueError, match="global_batch_size"):
        serving.validate_attachment(updated)


def test_role_yaml_cannot_override_retained_resume_boundary(args):
    recovery = TrainingRecovery(args, RestorePlan())
    actor = copy.copy(args)
    critic = copy.copy(args)
    actor.save += "/actor"
    critic.save += "/critic"
    critic.load = "critic-initial-model"
    assert recovery.resume_role("actor", actor)["load"] == args.load
    assert recovery.resume_role("critic", critic)["load"] == critic.load
    recovery.initial_load_completed(0)
    recovery.checkpoint_committed(2)
    for role, initial in (("actor", actor), ("critic", critic)):
        edited_yaml = copy.copy(initial)
        edited_yaml.load = "wrong-initial-checkpoint"
        edited_yaml.save = "wrong-new-save-path"
        edited_yaml.tensor_model_parallel_size = 4
        resumed = recovery.resume_role(role, edited_yaml)
        assert resumed["load"] == resumed["save"] == initial.save
        assert resumed["ckpt_step"] == 2 and resumed["start_rollout_id"] == 3
        assert not resumed["finetune"] and not resumed["no_load_optim"] and not resumed["no_load_rng"]
        assert "tensor_model_parallel_size" not in resumed


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_megatron_init_receives_retained_role_checkpoint(args, monkeypatch, role):
    from unittest.mock import Mock

    from slime.ray import actor_group

    args.train_env_vars = {}
    args.offload_train = args.use_routing_replay = False
    args.ckpt_format = "torch_dist"
    args.no_save_optim = args.no_save_rng = False
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.resume_role(role, args)
    recovery.initial_load_completed(0)
    recovery.checkpoint_committed(2)
    expected_load = args.save
    recovery.args = copy.copy(args)
    recovery.args.update_weight_start_version = 12
    args.load = "initial-checkpoint-from-yaml"
    args.save = "new-path-from-yaml"
    args.update_weight_start_version = 0
    args.ckpt_fully_parallel_save = args.dist_ckpt_optim_fully_reshardable = False
    worker, manager, remote_class = Mock(), Mock(), Mock()
    worker.get_master_addr_and_port.remote.return_value = ("127.0.0.1", 12345)
    worker.init.remote.return_value = 3
    remote_class.options.return_value = remote_class
    remote_class.remote.return_value = worker
    manager.register_training_actors.remote.side_effect = lambda role, actors, config: recovery.resume_role(
        role, config
    )
    monkeypatch.setattr(actor_group.ray, "remote", lambda **kw: lambda cls: remote_class)
    monkeypatch.setattr(actor_group.ray, "get", lambda value: value)
    group = actor_group.RayTrainGroup(args, 1, 1, pg=(object(), [0], [0]), role=role, actor_cls=object)
    assert group.create(rollout_manager=manager) == [3]
    configuration = worker.init.remote.call_args.args[0]
    assert configuration.load == configuration.save == expected_load
    assert configuration.ckpt_step == 2 and configuration.start_rollout_id == 3
    assert configuration.update_weight_start_version == group._disk_weight_version == 12
    assert configuration.ckpt_fully_parallel_save and configuration.dist_ckpt_optim_fully_reshardable
    assert not configuration.no_load_optim and not configuration.no_load_rng


@pytest.mark.parametrize("previous", [None, {"JobID": "old", "IsDead": False}])
def test_live_or_unknown_previous_driver_cannot_be_replaced(args, monkeypatch, previous):
    serving = object.__new__(ServingCluster.__ray_metadata__.modified_class)
    serving.configuration = retained_rollout_configuration(args)
    serving.driver_job_id = "old"
    monkeypatch.setattr("ray._private.state.jobs", lambda: [] if previous is None else [previous])
    with pytest.raises(RuntimeError, match="still owned"):
        serving.validate_attachment(args)


def test_dead_driver_can_be_replaced(args, monkeypatch):
    serving = object.__new__(ServingCluster.__ray_metadata__.modified_class)
    serving.configuration = retained_rollout_configuration(args)
    serving.driver_job_id = "old"
    monkeypatch.setattr("ray._private.state.jobs", lambda: [{"JobID": "old", "IsDead": True}])
    serving.validate_attachment(args)


@pytest.mark.parametrize("was_paused", [None, False, True])
@pytest.mark.parametrize("weight_sync_fails", [False, True])
def test_initial_weight_sync_controls_retained_producer_resume(args, monkeypatch, was_paused, weight_sync_fails):
    import runpy
    import sys
    from unittest.mock import Mock

    from slime.ray import rollout

    # Argument parsing is outside this CPU test and imports SGLang server args.
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", SimpleNamespace(parse_args=None))
    train = runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))["train"]
    manager = object.__new__(rollout.RolloutManager.__ray_metadata__.modified_class)
    manager._recovery_admission_was_paused = was_paused
    producer = Mock()
    manager.data_source = SimpleNamespace(consumers={"fully_async": producer})
    endpoint, actor = Mock(), Mock()
    endpoint.training_ready.remote.side_effect = manager.training_ready

    def publish_weights():
        producer.resume.assert_not_called()
        if weight_sync_fails:
            raise RuntimeError("weight sync failed")

    actor.update_weights.side_effect = publish_weights
    monkeypatch.setitem(train.__globals__, "create_training_models", lambda *a: (actor, None))
    monkeypatch.setattr(rollout.ray, "get", lambda value: value)
    args.release_train = args.offload_rollout = args.check_weight_update_equal = False
    args.num_rollout, args.eval_interval = 0, None
    if weight_sync_fails:
        with pytest.raises(RuntimeError, match="weight sync failed"):
            train(args, {}, endpoint, 1, RestorePlan())
        endpoint.training_ready.remote.assert_not_called()
        endpoint.dispose.remote.assert_not_called()
        assert manager._recovery_admission_was_paused is was_paused
    else:
        train(args, {}, endpoint, 1, RestorePlan())
        endpoint.training_ready.remote.assert_called_once_with()
        assert manager._recovery_admission_was_paused is None
        manager.training_ready()
    assert producer.resume.call_count == int(not weight_sync_fails and was_paused is False)


@pytest.mark.parametrize("external_ray,live_head", [(False, True), (False, False), (True, True)])
@pytest.mark.parametrize(
    "options",
    [
        "--rollout-data-transport=straw",
        "--num-rollout 3",
        "--use-fault-tolerance --save-debug-rollout-data 'dir with spaces/rollout_{rollout_id}.pt'",
    ],
)
def test_launcher_preserves_serving_and_reuses_ray(monkeypatch, external_ray, live_head, options):
    from unittest.mock import Mock

    from slime.utils.external_utils import command_utils

    commands = []
    status = Mock(return_value=SimpleNamespace(returncode=0 if live_head else 1))
    monkeypatch.setattr(command_utils, "exec_command", commands.append)
    monkeypatch.setattr(command_utils.subprocess, "run", status)
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)
    monkeypatch.setenv("SLIME_SCRIPT_EXTERNAL_RAY", str(int(external_ray)))
    monkeypatch.setenv("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    command_utils.execute_train(options, num_gpus_per_node=4, megatron_model_type=None)
    assert not any("pkill" in command or "ray stop" in command for command in commands)
    assert any("ray start" in command for command in commands) == (not external_ray and not live_head)
    assert options in commands[-1]
    assert status.call_count == int(not external_ray)


def test_checkpoint_restores_serving_version_after_manual_restart(args, tmp_path):
    import json

    from slime.data.checkpoint import commit_checkpoint, resolve_checkpoint

    root = tmp_path / "checkpoint"
    model = root / "iter_0000002"
    model.mkdir(parents=True)
    (model / "weights.pt").write_bytes(b"model-and-optimizer")
    (root / "latest_checkpointed_iteration.txt").write_text("2")
    (root / "rollout").mkdir()
    for name in ("queue_state", "builder_state"):
        (root / "rollout" / f"{name}_2.json").write_text(json.dumps({"test": name}))
    args.save = str(root)
    commit_checkpoint(args, 2, model_args=[args], weight_version=7)
    args.load, args.save = str(root), str(tmp_path / "new")
    args.ckpt_step, args.start_rollout_id = 2, None
    args.finetune = args.no_load_optim = args.no_load_rng = False
    resolve_checkpoint(args)
    assert args.start_rollout_id == 3
    assert args.update_weight_start_version == 7


def global_train_data():
    return {
        "tokens": [[1, 2, 3], [1, 4], [1, 5, 6, 7], [1, 8]],
        "sample_indices": list(range(4)),
        "rollout_ids": list(range(4)),
        "response_lengths": [2, 1, 3, 1],
        "rewards": [0.0, 1.0, 0.0, 1.0],
        "raw_reward": [0.0, 1.0, 0.0, 1.0],
        "loss_masks": [[1, 1], [1], [1, 1, 1], [1]],
    }


def test_retained_straw_batch_reshards_without_reusing_old_plan(args, monkeypatch):
    from slime.data import batch_builder

    monkeypatch.setattr(batch_builder.ray, "put", lambda value: value)
    recovery = TrainingRecovery(args, RestorePlan())
    samples = [Sample(index=i, tokens=tokens) for i, tokens in enumerate(global_train_data()["tokens"])]
    raw = pack_rollout_payload(samples, args, 0)
    recovery.remember_raw(0, raw)
    recovery.remember_converted(0, global_train_data(), "old-dp-batch")
    store, _, lock = rollout_store(args)
    with lock:
        store.release_publications([raw.manifest])
    builder = BatchBuilder(args)
    builder.batch_id = "old-dp-batch"
    builder._commit_ready = lambda *a: pytest.fail("Replay must not overwrite the existing immutable queue batch")
    for dp_size in (2, 1, 4):
        builder.train_parallel_config = dict(
            dp_size=dp_size, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1
        )
        refs = builder.split_by_dp(recovery.load_converted(0), publish_batch=False)
        recovery.retain_replay_shards(0, builder.replay_refs)
        assert len(refs) == dp_size
        shards = [ref.inner.load() for ref in refs]
        assert sorted(index for shard in shards for index in shard["sample_indices"]) == list(range(4))
        assert all(shard["global_batch_sizes"] == [4] for shard in shards)
        assert [sample.tokens for sample in recovery.load_raw(0)] == global_train_data()["tokens"]
        recovery.release_replay_shards(0)
    recovery.release_batches()


def test_debug_batch_replay_keeps_converted_rewards_and_original_dump(args, tmp_path):
    from slime.observability.rollout_data_utils import save_debug_rollout_data

    args.rollout_data_transport = "object-store"
    args.save_debug_rollout_data = str(tmp_path / "rollout_{rollout_id}.pt")
    samples = [Sample(index=0, tokens=[1, 2], reward=3.0)]
    save_debug_rollout_data(args.save_debug_rollout_data, samples, rollout_id=0, evaluation=False)
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.remember_raw(0, args.save_debug_rollout_data)
    recovery.remember_converted(0, {"rewards": torch.tensor([7.0])}, None)
    assert recovery.load_raw(0)[0].reward == 3.0
    assert recovery.load_converted(0)["rewards"].tolist() == [7.0]
    recovery.release_batches()
    assert (tmp_path / "rollout_0.pt").exists()
    assert not (tmp_path / "rollout_0.pt.train-recovery.pt").exists()


def test_only_committed_model_boundary_releases_replay_batches(args):
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.initial_load_completed(0)
    for step in range(3):
        recovery.remember_raw(step, pack_rollout_payload([], args, step))
        recovery.remember_converted(step, global_train_data(), f"batch-{step}")
    # Runtime completion alone has no model/optimizer checkpoint to resume.
    assert recovery.resume_configuration["start_rollout_id"] == 0
    recovery.checkpoint_committed(1)
    assert set(recovery.batches) == {2}
    assert recovery.resume_configuration["start_rollout_id"] == 2
    assert recovery.resume_configuration["load"] == args.save
    assert recovery.resume_configuration["ckpt_step"] == 1
    assert not recovery.resume_configuration["no_load_optim"]
    assert not recovery.resume_configuration["no_load_rng"]
    recovery.release_batches()


@pytest.mark.parametrize("transport", ["straw", "object-store"])
def test_new_manager_restores_journal_without_health_checks(args, tmp_path, transport):
    args.use_fault_tolerance = False
    args.rollout_data_transport = transport
    args.save_debug_rollout_data = str(tmp_path / "rollout_{rollout_id}.pt")
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.resume_role("actor", args)
    recovery.initial_load_completed(0)
    recovery.checkpoint_committed(1)
    raw = pack_rollout_payload([], args, 2) if transport == "straw" else args.save_debug_rollout_data
    source_state = {"sample_offset": 12, "metadata": {"custom": 1}}
    recovery.remember_raw(2, raw, source_state=source_state)
    # Manager death immediately after raw acceptance must restore both the
    # advanced source cursor and the batch that owns those consumed samples.
    accepted = TrainingRecovery(args, RestorePlan())
    assert accepted.source_state == source_state and 2 in accepted.batches
    assert accepted.batches[2].converted is None
    recovery.remember_converted(2, global_train_data(), "batch-2")
    if transport == "straw":
        store, _, lock = rollout_store(args)
        with lock:
            store.release_publications([raw.manifest])
            store.seal()
            store.collect_garbage()

    rebuilt = TrainingRecovery(args, RestorePlan())
    assert rebuilt.incarnation == recovery.incarnation
    assert rebuilt.loaded and rebuilt.checkpoint_step == 1
    assert rebuilt.resume_configuration["start_rollout_id"] == 2
    assert rebuilt.source_state == recovery.source_state
    assert rebuilt.load_converted(2) == global_train_data()
    assert rebuilt.resume_role("actor", args)["load"] == args.save
    rebuilt.checkpoint_committed(2)
    assert not TrainingRecovery(args, RestorePlan()).batches


def test_manager_restart_before_initial_model_load(args):
    args.start_rollout_id = None
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.source_state = {"reader_generation": "old", "metadata": {}}
    recovery.persist()

    restarted = TrainingRecovery(args, RestorePlan())
    restarted.reconcile_collection(None, "branch")
    # The model checkpoint, loaded later, still determines where training starts.
    restarted.initial_load_completed(7)
    assert restarted.resume_configuration["start_rollout_id"] == 7
    assert not restarted.batches


def test_source_snapshot_restores_cursor_and_buffer(args):
    from slime.data.data_source import RolloutDataSourceWithBuffer

    source = object.__new__(RolloutDataSourceWithBuffer)
    source.args, source.dataset = args, None
    args.rollout_shuffle = False
    source.sample_offset, source.epoch_id = 12, 2
    source.sample_group_index, source.sample_index = 12, 48
    source.metadata = {"custom": "value"}
    source.buffer = [[Sample(index=47, tokens=[1, 2])]]
    state = source.state_dict()
    source.buffer.clear()
    rebuilt = object.__new__(RolloutDataSourceWithBuffer)
    rebuilt.args, rebuilt.dataset = args, None
    rebuilt.load_state_dict(state)
    assert rebuilt.sample_offset == 12 and rebuilt.sample_index == 48
    assert rebuilt.buffer[0][0].index == 47
    assert rebuilt.metadata == {"custom": "value"}


def test_plain_serving_identity_does_not_depend_on_driver_or_health_checks(args):
    args.rollout_data_transport = "object-store"
    args.save = args.save_debug_rollout_data = None
    name = training_session_name(args)
    args.use_fault_tolerance = False
    assert training_session_name(args) == name
    args.hf_checkpoint = "another-model"
    assert training_session_name(args) != name


def test_new_serving_owner_uses_cold_checkpoint_handoff(args):
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.initial_load_completed(0)
    recovery.remember_raw(0, pack_rollout_payload([], args, 0))
    recovery.remember_converted(0, global_train_data(), "batch-0")
    rebuilt = TrainingRecovery(args, RestorePlan(mode="snapshot"), retained_serving=False)
    assert not rebuilt.loaded and not rebuilt.batches
    assert rebuilt.source_state is None
    assert rebuilt.restore_plan.mode == "snapshot"
    assert rebuilt.incarnation != recovery.incarnation


def test_committed_checkpoint_survives_lost_manager_notification(args, tmp_path):
    import json

    from slime.data.checkpoint import commit_checkpoint

    args.save = str(tmp_path / "checkpoint")
    recovery = TrainingRecovery(args, RestorePlan())
    recovery.initial_load_completed(0)
    recovery.remember_raw(2, pack_rollout_payload([], args, 2))
    root = Path(args.save)
    (root / "iter_0000002").mkdir(parents=True)
    (root / "iter_0000002/weights.pt").write_bytes(b"optimizer-and-model")
    (root / "rollout").mkdir()
    for name in ("queue_state", "builder_state"):
        (root / "rollout" / f"{name}_2.json").write_text(json.dumps({"version": 1}))
    commit_checkpoint(args, 2, model_args=[args])
    rebuilt = TrainingRecovery(args, RestorePlan())
    assert rebuilt.checkpoint_step is None and 2 in rebuilt.batches
    rebuilt.reconcile_checkpoint()
    assert rebuilt.checkpoint_step == 2 and not rebuilt.batches
    assert rebuilt.resume_configuration["start_rollout_id"] == 3


def test_replay_completes_interrupted_dp_publication(args, monkeypatch):
    from dataclasses import asdict
    from unittest.mock import Mock

    from slime.data import batch_builder

    monkeypatch.setattr(batch_builder.ray, "put", lambda value: value)
    monkeypatch.setattr(batch_builder.ray, "get", lambda value: value)
    plan = pack_rollout_payload({"digest": "old-plan"}, args, 0)
    controller = Mock()
    controller.batch.remote.return_value = {"plan_ref": asdict(plan.manifest), "ready": False}
    controller.ready_batch.remote.return_value = pack_rollout_payload({}, args, 0).manifest
    builder = BatchBuilder(args, controller=controller)
    builder.train_parallel_config = dict(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    ranks = builder.replay_converted(global_train_data(), "batch-0")
    assert ranks[0].inner.load()["sample_indices"] == [0, 1, 2, 3]
    assert ranks[0].inner.plan_digest == "old-plan"
    controller.ready_batch.remote.assert_called_once()
    assert not builder.replay_refs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
