"""Actor state at the initial rollout weight sync."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_actor_module():
    if "sglang_router" not in sys.modules:
        stub = types.ModuleType("sglang_router")
        stub.__version__ = "0.2.3"
        sys.modules["sglang_router"] = stub
    try:
        from slime.backends.megatron_utils import actor as actor_module
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing == "megatron" or missing.startswith("megatron."):
            pytest.skip(f"Megatron is not installed: {missing}")
        raise
    return actor_module


class _FakeBackuper:
    def __init__(self, tags):
        self._tags = set(tags)
        self.restored: list[str] = []

    @property
    def backup_tags(self):
        return self._tags

    def restore(self, tag):
        self.restored.append(tag)


def _bare_actor(actor_module, active_tag, tags=("actor", "ref")):
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.weights_backuper = _FakeBackuper(tags)
    actor._active_model_tag = active_tag
    return actor


@pytest.mark.unit
@pytest.mark.parametrize("active_tag", ["ref", "teacher", "old_actor"])
def test_restore_actor_after_loads_switches_back_from_auxiliary_weights(active_tag):
    actor_module = _load_actor_module()
    actor = _bare_actor(
        actor_module,
        active_tag=active_tag,
        tags=("actor", active_tag),
    )

    actor._restore_actor_after_loads()

    assert actor._active_model_tag == "actor"
    assert actor.weights_backuper.restored == ["actor"]


@pytest.mark.unit
def test_restore_actor_after_loads_is_a_no_op_when_the_actor_is_active():
    actor_module = _load_actor_module()
    actor = _bare_actor(actor_module, active_tag="actor")

    actor._restore_actor_after_loads()

    assert actor.weights_backuper.restored == []


@pytest.mark.unit
def test_update_weights_refuses_to_broadcast_a_non_actor_model(monkeypatch):
    actor_module = _load_actor_module()
    actor = _bare_actor(actor_module, active_tag="ref")
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        use_fault_tolerance=False,
        offload_train=False,
        use_critic=False,
        colocate=False,
        ci_test=False,
        keep_old_actor=False,
        update_weights_interval=1,
    )
    actor.rollout_manager = SimpleNamespace(
        get_updatable_engines_and_lock=SimpleNamespace(remote=lambda: ([object()], None, 0, [1], [0], [None]))
    )
    pushed = []
    actor.weight_updater = SimpleNamespace(update_weights=lambda: pushed.append("pushed"))
    monkeypatch.setattr(actor_module.ray, "get", lambda value: value)
    monkeypatch.setattr(actor_module, "print_memory", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="requires actor weights"):
        actor.update_weights()
    assert pushed == []

    actor._active_model_tag = "actor"
    actor.update_weights()
    assert pushed == ["pushed"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
