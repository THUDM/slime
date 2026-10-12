"""Native model initialization and ordinary slime resume choose distinct loaders."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from slime_plugins.models import prism

NUM_GPUS = 0


def _checkpoint_module(monkeypatch, args):
    events = []
    training = types.ModuleType("megatron.training.checkpointing")
    training.load_checkpoint = lambda **_kwargs: events.append("resume") or (12, 100)
    training.save_checkpoint = lambda *_args, **_kwargs: None
    globals_module = types.ModuleType("megatron.training.global_vars")
    globals_module.get_args = lambda: args
    monkeypatch.setitem(sys.modules, training.__name__, training)
    monkeypatch.setitem(sys.modules, globals_module.__name__, globals_module)
    monkeypatch.setitem(sys.modules, "torch.distributed._shard.sharding_spec", None)
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/checkpoint.py"
    spec = importlib.util.spec_from_file_location("test_prism_checkpoint_dispatch_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "_load_checkpoint_hf", lambda **kwargs: events.append(("hf", kwargs["load_path"])) or (0, 0)
    )
    monkeypatch.setattr(
        prism,
        "load_prism_pretrained",
        lambda _models, selected_args: events.append(
            (
                "native",
                selected_args.prism_pretrained_checkpoint,
                getattr(selected_args, "prism_pretrained_step", None),
            )
        )
        or (0, 0),
    )
    return module, events


def _args(load=None, native="initial-native"):
    return types.SimpleNamespace(
        load=load,
        prism_pretrained_checkpoint=native,
        fp16=False,
        bf16=True,
        use_stateless_adam=False,
        ckpt_step=None,
        finetune=False,
    )


def _native_run(tmp_path):
    root = tmp_path / "native" / "checkpoints"
    iteration = root / "iter_0000020"
    (iteration / "model").mkdir(parents=True)
    (iteration / "model" / ".metadata").write_bytes(b"DCP")
    (iteration / "checkpoint.json").write_text('{"schema_version":2,"iteration":20}')
    (root / "latest_checkpointed_iteration.txt").write_text("20")
    return root.parent, root, iteration


def test_native_flag_initializes_with_no_load_path_and_refreshes_optimizer(monkeypatch):
    args = _args()
    module, events = _checkpoint_module(monkeypatch, args)
    optimizer = types.SimpleNamespace(reload_model_params=lambda: events.append("refresh-main-params"))
    assert module.load_checkpoint([object()], optimizer, None, {}) == (0, 0)
    assert events == [("native", "initial-native", None), "refresh-main-params"]
    assert args.load is None


@pytest.mark.parametrize("direct_iteration", [False, True])
def test_standard_resume_has_precedence_over_native_initializer(monkeypatch, tmp_path, direct_iteration):
    root = tmp_path / "rl"
    iteration = root / "iter_0000012"
    iteration.mkdir(parents=True)
    (iteration / ".metadata").write_bytes(b"DCP")
    (root / "latest_checkpointed_iteration.txt").write_text("12")
    args = _args(str(iteration if direct_iteration else root))
    module, events = _checkpoint_module(monkeypatch, args)
    assert module.load_checkpoint([object()], None, None, {}) == (12, 100)
    assert events == ["resume"]


@pytest.mark.parametrize("source", ["run", "root", "iteration"])
def test_split_native_layout_is_model_initialization_not_resume(monkeypatch, tmp_path, source):
    run, root, iteration = _native_run(tmp_path)
    path = {"run": run, "root": root, "iteration": iteration}[source]
    args = _args(str(path), native="another-initializer")
    module, events = _checkpoint_module(monkeypatch, args)
    assert module._is_native_model_checkpoint(path)
    assert not module._is_megatron_checkpoint(path)
    assert module.load_checkpoint([object()], None, None, {}) == (0, 0)
    assert events == [("native", str(path), None)]
    assert args.prism_pretrained_checkpoint == "another-initializer"


def test_explicit_native_iteration_rejects_conflicting_step(monkeypatch, tmp_path):
    _, _, iteration = _native_run(tmp_path)
    args = _args(str(iteration))
    args.ckpt_step = 19
    module, events = _checkpoint_module(monkeypatch, args)
    with pytest.raises(ValueError, match="explicit native checkpoint"):
        module.load_checkpoint([object()], None, None, {})
    assert events == []
    args.ckpt_step = 20
    assert module.load_checkpoint([object()], None, None, {}) == (0, 0)
    assert events == [("native", str(iteration), None)]


def test_explicit_reference_hf_load_is_not_overridden_by_actor_initializer(monkeypatch, tmp_path):
    hf = tmp_path / "reference-hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    args = _args(str(hf))
    args.finetune = True
    module, events = _checkpoint_module(monkeypatch, args)
    assert module.load_checkpoint([object()], None, None, {}) == (0, 0)
    assert events == [("hf", str(hf))]


def test_missing_explicit_resume_does_not_silently_reinitialize(monkeypatch, tmp_path):
    module, events = _checkpoint_module(monkeypatch, _args(str(tmp_path / "missing")))
    with pytest.raises(AssertionError, match="does not exist"):
        module.load_checkpoint([object()], None, None, {})
    assert events == []


def test_no_source_is_a_clear_initialization_error(monkeypatch):
    module, events = _checkpoint_module(monkeypatch, _args(native=None))
    assert not module._is_megatron_checkpoint(None)
    with pytest.raises(ValueError, match="requires --load"):
        module.load_checkpoint([object()], None, None, {})
    assert events == []
