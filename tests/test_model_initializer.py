"""Explicit model-only initialization and standard Megatron/HF checkpoint loading."""

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

NUM_GPUS = 0
pytestmark = pytest.mark.unit
_CALLBACK_IDS = itertools.count()


def _args(load=None, source="opaque://model-source", hook=None, **kwargs):
    return SimpleNamespace(
        load=load,
        pretrained_checkpoint=source,
        custom_megatron_model_initializer_path=hook,
        fp16=kwargs.pop("fp16", False),
        bf16=kwargs.pop("bf16", True),
        use_stateless_adam=False,
        ckpt_step=kwargs.pop("ckpt_step", None),
        finetune=kwargs.pop("finetune", False),
        **kwargs,
    )


def _callback(monkeypatch, function):
    # load_function caches resolved functions; each case gets a distinct path.
    name = f"_test_model_initializer_callback_{next(_CALLBACK_IDS)}"
    module = ModuleType(name)
    module.initialize = function
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}.initialize"


def _checkpoint_module(monkeypatch, args, *, resume_result=(12, 100)):
    events = []
    checkpointing = ModuleType("megatron.training.checkpointing")

    def load(**kwargs):
        events.append(("resume", kwargs))
        return resume_result

    checkpointing.load_checkpoint = load
    checkpointing.save_checkpoint = lambda *args, **kwargs: None
    globals_module = ModuleType("megatron.training.global_vars")
    globals_module.get_args = lambda: args
    monkeypatch.setitem(sys.modules, checkpointing.__name__, checkpointing)
    monkeypatch.setitem(sys.modules, globals_module.__name__, globals_module)
    # Skip checkpoint.py's Torch shard monkey patch to avoid changing global Torch classes.
    monkeypatch.setitem(sys.modules, "torch.distributed._shard.sharding_spec", None)
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/checkpoint.py"
    spec = importlib.util.spec_from_file_location("test_model_initializer_checkpoint_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def load_hf(**kwargs):
        events.append(("hf", kwargs))
        return (0, 0)

    monkeypatch.setattr(module, "_load_checkpoint_hf", load_hf)
    return module, events


@pytest.mark.parametrize(
    ("fp16", "bf16", "dtype", "reload_expected"),
    [(True, False, torch.float16, True), (False, True, torch.bfloat16, True), (False, False, torch.float32, False)],
    ids=["fp16-master-sync", "bf16-master-sync", "fp32"],
)
def test_initializer_receives_models_and_args_and_refreshes_master_params(
    monkeypatch, fp16, bf16, dtype, reload_expected
):
    calls = []
    model = SimpleNamespace(weight=torch.zeros(2, dtype=dtype))
    models = [model]
    master = torch.zeros(2, dtype=torch.float32)

    def initialize(selected_models, selected_args):
        assert selected_models is models and selected_args is args
        assert selected_args.pretrained_checkpoint == "opaque://model-source"
        calls.append("initialize")
        selected_models[0].weight.fill_(7.25)

    def reload():
        calls.append("reload-master")
        master.copy_(model.weight.float())

    args = _args(hook=_callback(monkeypatch, initialize), fp16=fp16, bf16=bf16)
    module, events = _checkpoint_module(monkeypatch, args)
    optimizer = SimpleNamespace(reload_model_params=reload)
    assert module.load_checkpoint(models, optimizer, object(), {}) == (0, 0)
    assert events == []
    assert calls == (["initialize", "reload-master"] if reload_expected else ["initialize"])
    assert torch.equal(master, torch.full_like(master, 7.25) if reload_expected else torch.zeros_like(master))
    assert args.load is None


def test_initializer_runs_without_an_optimizer(monkeypatch):
    calls = []
    args = _args(hook=_callback(monkeypatch, lambda models, args: calls.append((models, args))))
    module, events = _checkpoint_module(monkeypatch, args)
    models = [object()]
    assert module.load_checkpoint(models, None, None, {}) == (0, 0)
    assert calls == [(models, args)] and events == []


@pytest.mark.parametrize("result", [False, 0, (0, 0), {}])
def test_initializer_must_return_none_before_master_params_are_refreshed(monkeypatch, result):
    args = _args(hook=_callback(monkeypatch, lambda models, args: result))
    module, events = _checkpoint_module(monkeypatch, args)
    optimizer = SimpleNamespace(reload_model_params=lambda: events.append("reload"))
    with pytest.raises((ValueError, TypeError), match="return None"):
        module.load_checkpoint([object()], optimizer, None, {})
    assert events == []


@pytest.mark.parametrize("error", [RuntimeError("model initialization failed"), FileNotFoundError("source missing")])
def test_initializer_preserves_callback_failure_and_does_not_refresh_optimizer(monkeypatch, error):
    def initialize(models, args):
        raise error

    args = _args(hook=_callback(monkeypatch, initialize))
    module, events = _checkpoint_module(monkeypatch, args)
    optimizer = SimpleNamespace(reload_model_params=lambda: events.append("reload"))
    with pytest.raises(type(error)) as raised:
        module.load_checkpoint([object()], optimizer, None, {})
    assert raised.value is error and events == []


@pytest.mark.parametrize("source", [None, ""])
def test_initializer_requires_source_before_hook_import(monkeypatch, source):
    args = _args(source=source, hook="missing_initializer_module.initialize")
    module, events = _checkpoint_module(monkeypatch, args)
    with pytest.raises(ValueError, match="pretrained"):
        module.load_checkpoint([object()], None, None, {})
    assert events == []


@pytest.mark.parametrize("source", [None, "model-source"])
def test_missing_initializer_hook_does_not_infer_a_model_loader(monkeypatch, source):
    module, events = _checkpoint_module(monkeypatch, _args(source=source))
    with pytest.raises(ValueError, match="initializer|initialization|--load"):
        module.load_checkpoint([object()], None, None, {})
    assert events == []


@pytest.mark.parametrize("bad_hook", ["module", "attribute", "not_callable"])
def test_bad_initializer_hook_errors_are_not_silently_ignored(monkeypatch, bad_hook):
    if bad_hook == "module":
        path, expected_error = "missing_initializer_module.initialize", ModuleNotFoundError
    else:
        module_name = f"_test_bad_initializer_{next(_CALLBACK_IDS)}"
        callback_module = ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, callback_module)
        path = f"{module_name}.initialize"
        expected_error = AttributeError
        if bad_hook == "not_callable":
            callback_module.initialize = 42
            expected_error = TypeError
    module, events = _checkpoint_module(monkeypatch, _args(hook=path))
    with pytest.raises(expected_error):
        module.load_checkpoint([object()], None, None, {})
    assert events == []


@pytest.mark.parametrize("layout", ["root", "iteration", "native-looking"])
def test_explicit_resume_wins_without_resolving_initializer(monkeypatch, tmp_path, layout):
    root = tmp_path / "resume"
    iteration = root / "iter_0000012"
    iteration.mkdir(parents=True)
    (iteration / ".metadata").write_bytes(b"DCP")
    (root / "latest_checkpointed_iteration.txt").write_text("12")
    if layout == "native-looking":
        (iteration / "model").mkdir()
        (iteration / "checkpoint.json").write_text("This manifest must never be interpreted by core.")
    load = str(iteration if layout == "iteration" else root)
    args = _args(load=load, hook="missing_initializer_module.initialize", ckpt_step=12)
    module, events = _checkpoint_module(monkeypatch, args)
    models, optimizer, scheduler, context = [object()], object(), object(), object()
    assert module.load_checkpoint(models, optimizer, scheduler, context) == (12, 100)
    assert len(events) == 1 and events[0][0] == "resume"
    assert events[0][1] == {
        "ddp_model": models,
        "optimizer": optimizer,
        "opt_param_scheduler": scheduler,
        "checkpointing_context": context,
        "skip_load_to_model_and_opt": False,
    }
    assert args.load == load and args.pretrained_checkpoint == "opaque://model-source"
    assert args.exit_on_missing_checkpoint


def test_standard_resume_keeps_checkpoint_step_validation_including_zero(monkeypatch, tmp_path):
    root = tmp_path / "resume"
    root.mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("0")
    args = _args(load=str(root), hook="missing_initializer_module.initialize", ckpt_step=0)
    module, events = _checkpoint_module(monkeypatch, args, resume_result=(1, 100))
    with pytest.raises(ValueError, match="requires step 0"):
        module.load_checkpoint([object()], None, None, {})
    assert len(events) == 1 and events[0][0] == "resume"


def test_explicit_reference_hf_load_bypasses_actor_initializer(monkeypatch, tmp_path):
    root = tmp_path / "reference-hf"
    root.mkdir()
    (root / "config.json").write_text("{}")
    args = _args(load=str(root), source=None, hook="missing_initializer_module.initialize", finetune=True)
    module, events = _checkpoint_module(monkeypatch, args)
    models, optimizer = [object()], object()
    assert module.load_checkpoint(models, optimizer, None, {}) == (0, 0)
    assert len(events) == 1 and events[0][0] == "hf"
    assert events[0][1] == {"ddp_model": models, "optimizer": optimizer, "args": args, "load_path": str(root)}


@pytest.mark.parametrize("layout", ["missing", "empty", "unknown"])
def test_invalid_explicit_load_cannot_fall_back_to_initializer(monkeypatch, tmp_path, layout):
    root = tmp_path / layout
    if layout in {"empty", "unknown"}:
        root.mkdir()
    if layout == "unknown":
        (root / "weights.bin").write_bytes(b"unknown checkpoint format")
    args = _args(load=str(root), hook="missing_initializer_module.initialize")
    module, events = _checkpoint_module(monkeypatch, args)
    expected_error = ValueError if layout == "unknown" else AssertionError
    expected_message = "Explicit --load" if layout == "unknown" else "does not exist|empty"
    with pytest.raises(expected_error, match=expected_message):
        module.load_checkpoint([object()], None, None, {})
    assert events == []
