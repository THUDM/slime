"""CPU regression test for eval-only mode (``--num-rollout 0``).

With ``num_rollout == 0`` the estimated ``train_iters`` is 0, so ``lr_decay_steps``
is 0 and Megatron's ``OptimizerParamScheduler`` aborts on ``assert lr_decay_steps > 0``.
Megatron is stubbed because it isn't installed on the CPU CI runner.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

NUM_GPUS = 0


class _RecordingScheduler:
    """Stub for OptimizerParamScheduler that keeps Megatron's lr_decay_steps assertion."""

    def __init__(self, optimizer, **kwargs):
        assert kwargs["lr_decay_steps"] > 0


def _register(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


def _load_model_module(monkeypatch):
    """Import slime.backends.megatron_utils.model with Megatron stubbed out."""
    s = object  # placeholder for symbols that are imported but unused here

    _register(monkeypatch, "megatron")
    _register(monkeypatch, "megatron.core", mpu=types.ModuleType("megatron.core.mpu"))
    _register(monkeypatch, "megatron.core.mpu")
    _register(monkeypatch, "megatron.core.distributed", DistributedDataParallel=s, finalize_model_grads=s)
    _register(monkeypatch, "megatron.core.enums", ModelType=s)
    _register(monkeypatch, "megatron.core.models", gpt=types.ModuleType("megatron.core.models.gpt"))
    _register(monkeypatch, "megatron.core.models.gpt", GPTModel=s)
    _register(monkeypatch, "megatron.core.optimizer", OptimizerConfig=s, get_megatron_optimizer=s)
    _register(monkeypatch, "megatron.core.optimizer.optimizer", MegatronOptimizer=s)
    _register(monkeypatch, "megatron.core.optimizer_param_scheduler", OptimizerParamScheduler=_RecordingScheduler)
    _register(monkeypatch, "megatron.core.pipeline_parallel", get_forward_backward_func=s)
    _register(monkeypatch, "megatron.core.pipeline_parallel.utils", unwrap_model=s)
    _register(monkeypatch, "megatron.core.utils", get_model_config=s, unwrap_model=s)
    _register(monkeypatch, "megatron.training")
    _register(monkeypatch, "megatron.training.global_vars", get_args=s)
    _register(monkeypatch, "megatron.training.training", get_model=s)

    _register(monkeypatch, "slime.backends.megatron_utils.checkpoint", load_checkpoint=s, save_checkpoint=s)
    _register(monkeypatch, "slime.backends.megatron_utils.cp_utils", reduce_train_step_metrics=s)
    _register(monkeypatch, "slime.backends.megatron_utils.data", DataIterator=s, get_batch=s)
    _register(
        monkeypatch,
        "slime.backends.megatron_utils.loss",
        ROLLOUT_TOP_P_TOKEN_KEYS=(),
        get_rollout_top_p_logprob_kwargs=s,
        loss_function=s,
    )
    _register(monkeypatch, "slime.backends.megatron_utils.model_provider", get_model_provider_func=s)
    # slime.utils.logging_utils pulls wandb/tensorboard and memory_utils pulls
    # psutil; none are installed on the CPU CI runner, so stub them too.
    _register(monkeypatch, "slime.utils.logging_utils")
    _register(monkeypatch, "slime.utils.memory_utils", clear_memory=s)

    sys.modules.pop("slime.backends.megatron_utils.model", None)
    return importlib.import_module("slime.backends.megatron_utils.model")


def _make_args(**overrides):
    args = SimpleNamespace(
        num_rollout=4,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=16,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1e-6,
        min_lr=0.0,
        lr_decay_style="constant",
        start_weight_decay=0.0,
        end_weight_decay=0.0,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    args.__dict__.update(overrides)
    return args


@pytest.mark.unit
def test_eval_only_num_rollout_zero_does_not_crash(monkeypatch):
    model = _load_model_module(monkeypatch)
    args = _make_args(num_rollout=0)
    model.get_optimizer_param_scheduler(args, optimizer=object())  # would assert without the clamp
    assert args.train_iters == 1


@pytest.mark.unit
def test_clamp_is_a_noop_for_normal_training(monkeypatch):
    model = _load_model_module(monkeypatch)
    args = _make_args(num_rollout=4, rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=16)
    model.get_optimizer_param_scheduler(args, optimizer=object())
    assert args.train_iters == 16  # 4 * 8 * 8 // 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
