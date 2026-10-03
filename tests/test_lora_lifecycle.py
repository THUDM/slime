"""CPU regressions for LoRA's actor and training-loop integration.

Execute the production methods with CPU collaborators, avoiding imports of Ray,
CUDA kernels and the Megatron pipeline scheduler. No method body is rewritten.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _lora_fakes import (
    FakeColumnParallelLinear,
    FakeGPTModel,
    FakeRowParallelLinear,
    make_config,
    make_lora_args,
    patch_named_tensor_iterator,
)

from slime.backends.megatron_utils.lora import (
    LoRAInjectionError,
    inject_lora,
    iter_lora_modules,
    layers,
    save_lora_adapter,
)

NUM_GPUS = 0
BACKEND = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils"


def _production_function(filename, name, **globals_):
    tree = ast.parse((BACKEND / filename).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[])
    namespace = {
        "__package__": "slime.backends.megatron_utils",
        "torch": torch,
        "logger": logging.getLogger(__name__),
        **globals_,
    }
    exec(compile(module, str(BACKEND / filename), "exec"), namespace)
    return namespace[name]


@pytest.mark.unit
@pytest.mark.parametrize("distributed_optimizer", [False, True])
@pytest.mark.parametrize("finetune", [False, True])
def test_actor_load_respects_resume_and_refreshes_masters(tmp_path, monkeypatch, distributed_optimizer, finetune):
    patch_named_tensor_iterator(monkeypatch)
    args = make_lora_args(lora_rank=4)
    config = make_config(lora_rank=4)
    model = FakeGPTModel(num_layers=1).to(torch.bfloat16)
    inject_lora(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    masters = [p.detach().float().clone().requires_grad_() for p in params]
    inner = torch.optim.SGD(masters, lr=0.1)
    with torch.no_grad():
        for p in params:
            p.fill_(2)
    path = save_lora_adapter(args, [model], tmp_path / "adapter", config)
    with torch.no_grad():
        for p in params:
            p.zero_()
        for master in masters:
            master.zero_()

    def reload_model_params():
        with torch.no_grad():
            for master, param in zip(masters, params, strict=True):
                master.copy_(param)

    optimizer = SimpleNamespace(optimizer=inner, reload_model_params=reload_model_params)
    if distributed_optimizer:
        optimizer.model_param_gbuf_map = {p: None for p in params}
    else:
        optimizer.float16_groups = [params]
        optimizer.fp32_from_float16_groups = [masters]
    args.lora_load = str(path)
    args.finetune = finetune
    args._lora_reports = [True]
    actor = SimpleNamespace(args=args, model=[model], optimizer=optimizer)
    _production_function("actor.py", "_init_lora")(actor)
    for master, param in zip(masters, params, strict=True):
        torch.testing.assert_close(master, param.float(), rtol=0, atol=0)
        master.grad = torch.ones_like(master)
    inner.step()
    expected = 1.9 if finetune else -0.1
    assert all(torch.allclose(master, torch.full_like(master, expected)) for master in masters)


@pytest.mark.unit
def test_release_train_recreation_keeps_resumed_adapter(monkeypatch):
    import slime.backends.megatron_utils.lora as lora

    args = make_lora_args(lora_load="initial-adapter", finetune=True)
    args._lora_reports = [True]
    args.save = "training-checkpoint"
    args.no_save_optim = False
    state = {"adapter": "initial"}
    monkeypatch.setattr(lora, "load_lora_adapter", lambda *a: state.update(adapter="initial"))
    init = _production_function("actor.py", "_init_lora")
    actor = SimpleNamespace(args=args, model=state, optimizer=None)
    init(actor)
    state["adapter"] = "trained"

    # Execute the actual release-train save transition used before actor recreation.
    group = SimpleNamespace(args=args, _actor_handlers=[], _release_train_enabled=lambda: True)
    save = _production_function("../../ray/actor_group.py", "save_model", ray=SimpleNamespace(get=lambda x: x))
    save(group, rollout_id=0)
    assert args.load == args.save and not args.finetune
    assert args.lora_load == "initial-adapter"
    # The full checkpoint loader has restored the trained adapter at this point.
    init(SimpleNamespace(args=args, model=state, optimizer=None))
    assert state["adapter"] == "trained"


@pytest.mark.unit
@pytest.mark.parametrize("projection", ["linear_fc1", "linear_fc2"])
def test_deepgemm_moe_rejects_adapter_before_installing_forward(projection):
    original_forward = object()
    module = SimpleNamespace(forward=original_forward)
    setattr(module, projection, SimpleNamespace(lora_A=object()))
    wrap = _production_function("alignment/deepgemm_moe_forward.py", "_wrap_te_grouped_mlp")
    with pytest.raises(RuntimeError, match="routed-expert LoRA"):
        wrap(module, "decoder.layers.0.mlp.experts")
    assert module.forward is original_forward


@pytest.mark.unit
def test_deepgemm_moe_allows_unadapted_experts():
    # An already installed, unadapted layer remains valid (e.g. attention-only LoRA).
    module = SimpleNamespace(_slime_deepgemm_moe_forward_wrapped=True)
    wrap = _production_function("alignment/deepgemm_moe_forward.py", "_wrap_te_grouped_mlp")
    assert wrap(module, "decoder.layers.0.mlp.experts") is False


@pytest.mark.unit
@pytest.mark.parametrize("backup_fails", [False, True])
def test_reference_backup_restores_the_live_adapter(backup_fails):
    model = FakeGPTModel(num_layers=1)
    inject_lora(model, make_config(lora_rank=4))
    adapters = [m for _, m in iter_lora_modules(model)]
    with torch.no_grad():
        for m in adapters:
            m.lora_B.fill_(0.5)
    before = [(m.lora_A.clone(), m.lora_B.clone()) for m in adapters]

    def backup(tag):
        assert tag == "ref"
        assert all(torch.count_nonzero(m.lora_B) == 0 for m in adapters)
        if backup_fails:
            raise RuntimeError("backup failed")

    actor = SimpleNamespace(model=[model], weights_backuper=SimpleNamespace(backup=backup))
    method = _production_function("actor.py", "_backup_lora_reference")
    if backup_fails:
        with pytest.raises(RuntimeError, match="backup failed"):
            method(actor)
    else:
        method(actor)
    for m, (a, b) in zip(adapters, before, strict=True):
        torch.testing.assert_close(m.lora_A, a, rtol=0, atol=0)
        torch.testing.assert_close(m.lora_B, b, rtol=0, atol=0)


class _RuntimeModel(FakeGPTModel):
    def forward(self, input_ids, **kwargs):
        return super().forward(input_ids)

    def zero_grad_buffer(self):
        pass


@pytest.mark.unit
@pytest.mark.parametrize("signal", [0.0, 1.0])
def test_training_step_preserves_zero_lr_warmup_and_zero_gradient_semantics(signal):
    model = _RuntimeModel(num_layers=1)
    inject_lora(model, make_config(lora_rank=4))
    args = make_lora_args(
        custom_megatron_before_train_step_hook_path=None,
        seq_length=4,
        micro_batch_size=1,
        decoder_seq_length=None,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        enable_mtp_training=False,
        save_debug_train_data=None,
        data_pad_size_multiplier=1,
        allgather_cp=False,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    inner = torch.optim.Adam(params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(inner, lambda step: min(step / 2, 1.0))
    steps = []

    def optimizer_step():
        steps.append(inner.param_groups[0]["lr"])
        inner.step()
        return True, 1.0, 0

    def forward_backward_func(forward_step_func, **kwargs):
        output, loss_fn = forward_step_func(None, model)
        loss_fn(output).backward()
        return []

    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4]]),
        "multimodal_train_inputs": None,
        "packed_seq_params": None,
        "full_loss_masks": None,
    }
    step = _production_function(
        "model.py",
        "train_one_step",
        get_args=lambda: args,
        math=math,
        os=os,
        partial=partial,
        mpu=SimpleNamespace(is_pipeline_last_stage=lambda **_: False),
        _wrap_forward_step_with_microbatch_pbar=lambda f, _: f,
        _with_rollout_top_p_token_keys=lambda args, keys: keys,
        get_batch=lambda *a, **kw: batch,
        loss_function=lambda args, batch, microbatches, batch_size, output: output.square().sum() * signal,
        get_forward_backward_func=lambda: forward_backward_func,
    )
    optimizer = SimpleNamespace(zero_grad=inner.zero_grad, step=optimizer_step, param_groups=inner.param_groups)
    schedule = SimpleNamespace(step=lambda increment: scheduler.step())
    before = [p.clone() for p in params]
    step(args, 0, 0, [], [model], optimizer, schedule, 1, 1)
    assert steps == [0.0]
    assert scheduler.last_epoch == 1
    for old, p in zip(before, params, strict=True):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
    step(args, 0, 1, [], [model], optimizer, schedule, 1, 1)
    assert steps[1] > 0
    assert scheduler.last_epoch == 2
    assert any(not torch.equal(p, old) for p, old in zip(params, before, strict=True)) == bool(signal)


@pytest.mark.unit
@pytest.mark.parametrize("parallel_mode", ["column", "row", "replicated"])
def test_sharded_checkpoint_preserves_base_and_assigns_adapter_axes(monkeypatch, parallel_mode):
    from slime.backends.megatron_utils.lora.layers import attach_lora_adapter, describe_lora_target

    group, dp_group = object(), object()
    factory = {"column": FakeColumnParallelLinear, "row": FakeRowParallelLinear, "replicated": torch.nn.Linear}
    module = factory[parallel_mode](4, 4)
    module._tp_group = group
    sentinel = object()
    module.sharded_state_dict = lambda **_: {"layer.weight": sentinel, "layer.lora_A": "wrong replica"}
    monkeypatch.setattr(layers, "tensor_model_parallel_world_size", lambda: 2)
    config = make_config(lora_rank=4, lora_allow_replicated_modules=["projection"])
    if parallel_mode == "replicated":
        monkeypatch.setattr(layers, "tensor_model_parallel_group", lambda: group)
        monkeypatch.setattr(torch.distributed, "get_global_rank", lambda *a: 0)
        monkeypatch.setattr(layers, "_broadcast_parameter", lambda *a, **kw: None)
    attach_lora_adapter(module, describe_lora_target("projection", module, config), config)
    utils = types.ModuleType("megatron.core.transformer.utils")
    utils.ensure_metadata_has_dp_cp_group = lambda metadata: metadata

    def sharded(state, prefix, axes, offsets, *, tp_group, dp_cp_group):
        assert tp_group is group and dp_cp_group is dp_group
        return {prefix + name: (tensor, axes.get(name), offsets) for name, tensor in state.items()}

    utils.make_sharded_tensors_for_checkpoint = sharded
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    offsets = ((0, 2, 4),)
    state = module.sharded_state_dict(prefix="layer.", sharded_offsets=offsets, metadata={"dp_cp_group": dp_group})
    assert state["layer.weight"] is sentinel
    expected_axes = {"column": (1, 0), "row": (1, 1), "replicated": (None, None)}[parallel_mode]
    for name, axis in zip(("lora_A", "lora_B"), expected_axes, strict=True):
        tensor, actual_axis, actual_offsets = state["layer." + name]
        assert tensor is getattr(module, name)
        assert actual_axis == axis and actual_offsets == offsets


@pytest.mark.unit
def test_disconnected_adapter_is_rejected_without_gradient_norm_scans():
    from slime.backends.megatron_utils.lora import assert_lora_backward, prepare_lora_backward

    model = FakeGPTModel(num_layers=1)
    inject_lora(model, make_config())
    prepare_lora_backward(model)
    with pytest.raises(LoRAInjectionError, match="did not reach"):
        assert_lora_backward(model)


@pytest.mark.unit
@pytest.mark.parametrize("finetune", [False, True])
def test_megatron_lora_preflight_failure_never_calls_loader(monkeypatch, tmp_path, finetune):
    from slime.backends.megatron_utils.lora import state

    args = SimpleNamespace(use_lora=True, finetune=finetune, load=str(tmp_path))
    model_module = types.ModuleType("slime.backends.megatron_utils.model")
    model_module.get_load_checkpoint_path_by_args = lambda args: tmp_path / "iter_0000001"
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)

    def reject(path, model):
        raise state.LoRACheckpointError("missing adapter")

    def unexpected_loader(**kwargs):
        pytest.fail("Megatron loader must not run after failed adapter validation")

    monkeypatch.setattr(state, "validate_megatron_lora_checkpoint", reject)
    load = _production_function(
        "checkpoint.py",
        "load_checkpoint",
        get_args=lambda: args,
        Path=Path,
        _is_dir_nonempty=lambda path: True,
        _is_megatron_checkpoint=lambda path: True,
        _load_checkpoint_megatron=unexpected_loader,
    )
    with pytest.raises(state.LoRACheckpointError, match="missing adapter"):
        load([], None, None, {})


@pytest.mark.unit
@pytest.mark.parametrize("use_lora, megatron_checkpoint", [(False, True), (True, False)])
def test_checkpoint_other_loading_paths_are_unchanged(monkeypatch, tmp_path, use_lora, megatron_checkpoint):
    from slime.backends.megatron_utils.lora import state

    args = SimpleNamespace(use_lora=use_lora, load=str(tmp_path))

    def unexpected_preflight(*args):
        pytest.fail("LoRA Megatron preflight must not run on this path")

    monkeypatch.setattr(state, "validate_megatron_lora_checkpoint", unexpected_preflight)
    expected = (7, 0) if megatron_checkpoint else (0, 0)
    load = _production_function(
        "checkpoint.py",
        "load_checkpoint",
        get_args=lambda: args,
        Path=Path,
        _is_dir_nonempty=lambda path: True,
        _is_megatron_checkpoint=lambda path: megatron_checkpoint,
        _load_checkpoint_megatron=lambda **kwargs: expected,
        _load_checkpoint_hf=lambda **kwargs: expected,
    )
    assert load([], None, None, {}) == expected


@pytest.mark.unit
def test_megatron_lora_preflight_success_calls_loader(monkeypatch, tmp_path):
    from slime.backends.megatron_utils.lora import state

    args = SimpleNamespace(use_lora=True, load=str(tmp_path))
    model = [object()]
    events = []
    checkpoint_path = tmp_path / "iter_0000001"
    model_module = types.ModuleType("slime.backends.megatron_utils.model")
    model_module.get_load_checkpoint_path_by_args = lambda args: checkpoint_path
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)

    def validate(path, chunks):
        assert path == checkpoint_path and chunks is model
        events.append("validate")

    def loader(**kwargs):
        assert kwargs["ddp_model"] is model
        events.append("load")
        return (7, 0)

    monkeypatch.setattr(state, "validate_megatron_lora_checkpoint", validate)
    load = _production_function(
        "checkpoint.py",
        "load_checkpoint",
        get_args=lambda: args,
        Path=Path,
        _is_dir_nonempty=lambda path: True,
        _is_megatron_checkpoint=lambda path: True,
        _load_checkpoint_megatron=loader,
    )
    assert load(model, None, None, {}) == (7, 0)
    assert events == ["validate", "load"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
