import importlib
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def disk_delta_updater_cls(monkeypatch):
    ray = types.ModuleType("ray")
    ray.get = lambda refs: refs
    ray_actor = types.ModuleType("ray.actor")
    ray_actor.ActorHandle = object
    ray.actor = ray_actor
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor)

    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = types.SimpleNamespace()
    megatron.core = megatron_core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    base_module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    base_module = types.ModuleType(base_module_name)
    base_module.UpdateWeightFromDistributed = type("UpdateWeightFromDistributed", (), {})
    monkeypatch.setitem(sys.modules, base_module_name, base_module)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_disk_delta"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)

    def save_actor_weights(_args, output_dir, model, **_kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "weights.json").write_text(json.dumps(model))

    monkeypatch.setattr(module, "save_hf_model_to_path", save_actor_weights)
    yield module.UpdateWeightFromDiskDelta
    sys.modules.pop(module_name, None)


def _make_updater(updater_cls, tmp_path, *, finetune):
    class DiskDeltaHarness(updater_cls):
        def __init__(self):
            self.args = Namespace(finetune=finetune)
            self.weight_version = 0
            self._baseline_captured = False
            self._snapshot = {}
            self.actor_weights = {"model.weight": 41}
            self.engine_weights = {"model.weight": 7}
            self.hf_weights = dict(self.engine_weights)
            self.delta_dir = str(tmp_path)
            self.model = self.actor_weights
            self.model_name = "test-model"
            self.quantization_config = None
            self.rollout_engines = []
            self._post_write_hook = None
            self.events = []

        def _capture_baseline(self):
            self._snapshot = dict(self.hf_weights)

        def _capture_snapshot(self, checkpoint):
            self.events.append("snapshot")
            self._snapshot = json.loads((Path(checkpoint) / "weights.json").read_text())

        def _reload_engines(self):
            self.events.append("reload")
            version_dir = tmp_path / f"weight_v{self.weight_version:06d}"
            self.engine_weights = json.loads((version_dir / "weights.json").read_text())

        def _publish(self):
            raise AssertionError("the first sync must not publish a delta")

        def _record_metrics(self):
            raise AssertionError("the first sync has no delta metrics")

    return DiskDeltaHarness()


@pytest.mark.unit
def test_resume_first_sync_publishes_current_actor_as_full_baseline(disk_delta_updater_cls, tmp_path):
    updater = _make_updater(disk_delta_updater_cls, tmp_path, finetune=False)

    updater.update_weights()

    assert updater.engine_weights == updater.actor_weights
    assert updater._snapshot == updater.actor_weights
    assert updater.weight_version == 1


@pytest.mark.unit
def test_fresh_finetune_first_sync_only_seeds_baseline(disk_delta_updater_cls, tmp_path):
    updater = _make_updater(disk_delta_updater_cls, tmp_path, finetune=True)

    updater.update_weights()

    assert updater.engine_weights == updater.hf_weights
    assert updater._snapshot == updater.hf_weights
    assert updater.weight_version == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_resume_first_sync_preserves_checkpoint_save_error(disk_delta_updater_cls, tmp_path, monkeypatch):
    error = RuntimeError("checkpoint save failed")
    module = sys.modules[disk_delta_updater_cls.__module__]

    def fail_save(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(module, "save_hf_model_to_path", fail_save)
    updater = _make_updater(disk_delta_updater_cls, tmp_path, finetune=False)

    with pytest.raises(RuntimeError) as raised:
        updater.update_weights()

    assert raised.value is error


@pytest.mark.unit
def test_resume_commits_full_baseline_before_snapshot_and_reload(disk_delta_updater_cls, tmp_path):
    updater = _make_updater(disk_delta_updater_cls, tmp_path, finetune=False)
    updater._post_write_hook = lambda *_args: updater.events.append("hook")

    updater.update_weights()

    assert updater.events == ["hook", "snapshot", "reload"]
