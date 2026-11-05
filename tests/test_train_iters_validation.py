"""CPU tests for the train_iters > 0 validation in get_optimizer_param_scheduler."""

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

NUM_GPUS = 0


class _RecordingScheduler:
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.kwargs = kwargs


def _load_model_module(monkeypatch):
    def _module(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    identity = lambda value, *args, **kwargs: value  # noqa: E731
    modules = {
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core", mpu=types.SimpleNamespace()),
        "megatron.core.distributed": _module(
            "megatron.core.distributed",
            DistributedDataParallel=object,
            finalize_model_grads=identity,
        ),
        "megatron.core.enums": _module(
            "megatron.core.enums", ModelType=types.SimpleNamespace(encoder_or_decoder=None)
        ),
        "megatron.core.models": _module("megatron.core.models"),
        "megatron.core.models.gpt": _module("megatron.core.models.gpt", GPTModel=object),
        "megatron.core.optimizer": _module(
            "megatron.core.optimizer", OptimizerConfig=object, get_megatron_optimizer=identity
        ),
        "megatron.core.optimizer.optimizer": _module(
            "megatron.core.optimizer.optimizer", MegatronOptimizer=object
        ),
        "megatron.core.optimizer_param_scheduler": _module(
            "megatron.core.optimizer_param_scheduler", OptimizerParamScheduler=_RecordingScheduler
        ),
        "megatron.core.pipeline_parallel": _module(
            "megatron.core.pipeline_parallel", get_forward_backward_func=identity
        ),
        "megatron.core.pipeline_parallel.utils": _module(
            "megatron.core.pipeline_parallel.utils", unwrap_model=identity
        ),
        "megatron.core.utils": _module(
            "megatron.core.utils", get_model_config=identity, unwrap_model=identity
        ),
        "megatron.training": _module("megatron.training"),
        "megatron.training.global_vars": _module("megatron.training.global_vars", get_args=identity),
        "megatron.training.training": _module("megatron.training.training", get_model=identity),
        "slime.observability": _module(
            "slime.observability",
            logging_utils=types.SimpleNamespace(),
            train_metric_utils=types.SimpleNamespace(),
        ),
        "slime.utils.memory_utils": _module("slime.utils.memory_utils", clear_memory=identity),
        "slime.backends.megatron_utils": _module("slime.backends.megatron_utils"),
        "slime.backends.megatron_utils.checkpoint": _module(
            "slime.backends.megatron_utils.checkpoint",
            load_checkpoint=identity,
            save_checkpoint=identity,
        ),
        "slime.backends.megatron_utils.data": _module(
            "slime.backends.megatron_utils.data", DataIterator=object, get_batch=identity
        ),
        "slime.backends.megatron_utils.loss": _module(
            "slime.backends.megatron_utils.loss",
            ROLLOUT_TOP_P_TOKEN_KEYS=(),
            get_rollout_top_p_logprob_kwargs=identity,
            loss_function=identity,
        ),
        "slime.backends.megatron_utils.model_provider": _module(
            "slime.backends.megatron_utils.model_provider", get_model_provider_func=identity
        ),
        "slime.backends.megatron_utils.stateless_adam": _module(
            "slime.backends.megatron_utils.stateless_adam", StatelessAdam=object
        ),
    }
    modules["slime.backends.megatron_utils"].__path__ = []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "model.py"
    module_name = "slime.backends.megatron_utils.model"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _scheduler_args(**overrides):
    values = dict(
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=64,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1e-6,
        min_lr=0.0,
        lr_decay_style="constant",
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.unit
def test_zero_train_iters_raises_value_error_naming_inputs(monkeypatch):
    model = _load_model_module(monkeypatch)
    # 2 * 8 * 8 = 128 < 256 -> train_iters floors to 0.
    args = _scheduler_args(global_batch_size=256)

    with pytest.raises(ValueError, match=r"train_iters is 0") as excinfo:
        model.get_optimizer_param_scheduler(args, optimizer=object())

    message = str(excinfo.value)
    assert "num_rollout (2)" in message
    assert "rollout_batch_size (8)" in message
    assert "n_samples_per_prompt (8)" in message
    assert "global_batch_size (256)" in message
    assert "128" in message


@pytest.mark.unit
def test_valid_config_builds_scheduler(monkeypatch):
    model = _load_model_module(monkeypatch)
    # 2 * 8 * 8 = 128 >= 64 -> train_iters = 2.
    args = _scheduler_args(global_batch_size=64)

    scheduler = model.get_optimizer_param_scheduler(args, optimizer=object())

    assert args.train_iters == 2
    assert scheduler.kwargs["lr_decay_steps"] == 2 * 64
    assert scheduler.kwargs["wd_incr_steps"] == 2 * 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
