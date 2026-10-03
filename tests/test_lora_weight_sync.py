"""Merged-LoRA rollout weight-synchronisation tests with a mock SGLang engine.

The invariant under test is the one the whole design rests on: what the rollout
engine executes must be exactly ``W_base + (alpha / r) * B @ A`` for the adapted
tensors and exactly ``W_base`` for everything else, with the raw adapter factors
never appearing in the full-weight payload.
"""

from __future__ import annotations

import _lora_fakes  # noqa: F401  (installs the megatron stub before slime imports)
import pytest
import torch
import torch.nn.functional as F
from _lora_fakes import (
    FakeGPTModel,
    FakeMoEModel,
    make_config,
    make_lora_args,
    patch_named_tensor_iterator,
)

from slime.backends.megatron_utils.lora.merge import merged_named_params  # noqa: E402
from slime.backends.megatron_utils.lora import (  # noqa: E402
    EffectiveWeightMapping,
    build_merge_plan,
    inject_lora,
    iter_lora_modules,
    lora_local_delta,
)

NUM_GPUS = 0

RANK = 8


class MockSGLangEngine:
    """Accepts standard HF-style full weights and can run a forward pass with them.

    Checks the merged tensor payload using an in-process consumer. Failure is
    injected before this mock writes anything; these tests do not establish
    rollback or version publication guarantees for real NCCL, disk, or IPC updates.
    """

    def __init__(self) -> None:
        self.weights: dict[str, torch.Tensor] = {}
        self.weight_version = 0
        self.update_calls = 0
        self.fail_next = False

    def update_weights(self, named_tensors, weight_version: int) -> None:
        self.update_calls += 1
        offenders = [n for n, _ in named_tensors if "lora_A" in n or "lora_B" in n]
        assert not offenders, f"raw LoRA factors reached the engine: {offenders}"
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated engine failure")
        for name, tensor in named_tensors:
            self.weights[name] = tensor.detach().clone()
        # Only reached when the update succeeded.
        self.weight_version = weight_version

    def linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights[name])


@pytest.fixture()
def lora_setup(monkeypatch):
    patch_named_tensor_iterator(monkeypatch)
    torch.manual_seed(23)
    model = FakeGPTModel(num_layers=1, hidden=16, ffn=32)
    config = make_config(lora_rank=RANK, lora_alpha=float(2 * RANK))
    inject_lora(model, config)
    args = make_lora_args(lora_rank=RANK, lora_alpha=float(2 * RANK))
    return args, [model], config


def _sync(args, model, engine: MockSGLangEngine, version: int) -> None:
    engine.update_weights(list(merged_named_params(args, model)), weight_version=version)


NUM_LOCAL_EXPERTS = 3


@pytest.fixture()
def moe_lora_setup(monkeypatch):
    """A MoE model with LoRA on attention, shared experts *and* routed experts."""
    patch_named_tensor_iterator(monkeypatch)
    torch.manual_seed(23)
    model = FakeMoEModel(num_layers=1, hidden=16, ffn=32, num_local_experts=NUM_LOCAL_EXPERTS)
    overrides = {
        "lora_rank": RANK,
        "lora_alpha": float(2 * RANK),
        "lora_target_preset": "moe_language_all",
    }
    config = make_config(**overrides)
    inject_lora(model, config)
    with torch.no_grad():
        # lora_B is zero-initialised; a non-zero adapter is what makes the
        # merged weights differ from the base and the tests meaningful.
        for _, module in iter_lora_modules(model):
            module.lora_B.normal_(0.0, 0.1)
    return make_lora_args(**overrides), [model], config


# --------------------------------------------------------------------------- #
# routed MoE experts
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_routed_expert_adapter_is_shared_by_every_local_expert(moe_lora_setup):
    """One adapter, N base weights: the plan must fan out to every ``weight{i}``."""
    args, model, _ = moe_lora_setup
    plan = build_merge_plan(args, model)

    experts_prefix = "module.module.decoder.layers.0.mlp.experts"
    for projection in ("linear_fc1", "linear_fc2"):
        keys = [f"{experts_prefix}.{projection}.weight{i}" for i in range(NUM_LOCAL_EXPERTS)]
        assert all(key in plan for key in keys), f"missing routed-expert weights for {projection}"
        # All of them resolve to the same shared adapter module.
        assert len({id(plan[key]) for key in keys}) == 1


@pytest.mark.unit
def test_every_routed_expert_weight_receives_the_same_delta(moe_lora_setup):
    args, model, _ = moe_lora_setup
    effective = dict(merged_named_params(args, model))

    for name, module in iter_lora_modules(model[0]):
        spec = module._slime_lora_spec
        if not spec.is_expert:
            continue
        delta = lora_local_delta(module)
        for weight_name in spec.base_weight_names:
            base = getattr(module, weight_name).detach()
            expected = base + delta.to(base.dtype)
            torch.testing.assert_close(
                effective[f"module.module.{name}.{weight_name}"], expected, rtol=1e-6, atol=1e-7
            )


@pytest.mark.unit
def test_merged_moe_weights_reproduce_the_training_forward_pass(moe_lora_setup):
    """The core train/rollout consistency property, including routed experts.

    Running the base model with the merged weights must reproduce the adapter
    forward pass bit-for-bit; otherwise the rollout policy silently diverges from
    the policy the gradients were computed for.
    """
    import copy

    args, model, _ = moe_lora_setup
    tokens = torch.randint(0, 32, (2, 4))
    with torch.no_grad():
        adapter_output = model[0](tokens)

    merged = copy.deepcopy(model[0])
    with torch.no_grad():
        for _, module in iter_lora_modules(merged):
            spec = module._slime_lora_spec
            delta = lora_local_delta(module)
            for weight_name in spec.base_weight_names:
                getattr(module, weight_name).add_(delta.to(getattr(module, weight_name).dtype))
            # Disable the adapter so the delta is applied exactly once.
            module._slime_lora_hook_handle.remove()
        merged_output = merged(tokens)

    torch.testing.assert_close(adapter_output, merged_output, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_moe_sync_payload_never_contains_adapter_factors(moe_lora_setup):
    args, model, _ = moe_lora_setup
    engine = MockSGLangEngine()
    # MockSGLangEngine.update_weights asserts no lora_A/lora_B reaches it.
    _sync(args, model, engine, version=1)

    assert engine.weight_version == 1
    assert "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0" in engine.weights
    assert not any("lora" in name for name in engine.weights)


# --------------------------------------------------------------------------- #
# merge plan / effective iterator
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_merge_plan_covers_every_injected_module(lora_setup):
    args, model, _ = lora_setup
    plan = build_merge_plan(args, model)
    assert len(plan) == len(list(iter_lora_modules(model)))
    assert all(name.endswith(".weight") for name in plan)
    assert "module.module.decoder.layers.0.mlp.linear_fc1.weight" in plan


@pytest.mark.unit
def test_effective_iterator_drops_adapters_and_keeps_every_base_tensor(lora_setup):
    args, model, _ = lora_setup
    from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers

    raw = dict(named_params_and_buffers(args, model, convert_to_global_name=True))
    effective = dict(merged_named_params(args, model))

    assert not any(name.endswith((".lora_A", ".lora_B")) for name in effective)
    assert set(effective) == {name for name in raw if not name.endswith((".lora_A", ".lora_B"))}
    for name, tensor in effective.items():
        assert tensor.shape == raw[name].shape
        assert tensor.dtype == raw[name].dtype


@pytest.mark.unit
def test_effective_iterator_returns_base_plus_delta_for_targets(lora_setup):
    args, model, config = lora_setup
    with torch.no_grad():
        for _, module in iter_lora_modules(model[0]):
            module.lora_B.normal_(0.0, 0.1)

    effective = dict(merged_named_params(args, model))
    for name, module in iter_lora_modules(model[0]):
        key = f"module.module.{name}.weight"
        expected = module.weight.detach() + lora_local_delta(module).to(module.weight.dtype)
        torch.testing.assert_close(effective[key], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.unit
def test_effective_iterator_leaves_non_target_weights_untouched(lora_setup):
    args, model, _ = lora_setup
    with torch.no_grad():
        for _, module in iter_lora_modules(model[0]):
            module.lora_B.normal_(0.0, 0.1)

    effective = dict(merged_named_params(args, model))
    for name in ("module.module.output_layer.weight", "module.module.embedding.word_embeddings.weight"):
        torch.testing.assert_close(
            effective[name], dict(model[0].named_parameters())[name.removeprefix("module.module.")].detach()
        )


@pytest.mark.unit
def test_effective_iterator_is_identity_when_lora_is_disabled(monkeypatch):
    """USE_LORA=0 regression: the sync payload must be byte-identical to before."""
    from slime.backends.megatron_utils.update_weight import common

    torch.manual_seed(2)
    model = FakeGPTModel(num_layers=1, hidden=8, ffn=16)
    args = make_lora_args(use_lora=False)
    monkeypatch.setattr(
        common,
        "named_params_and_buffers",
        lambda a, m, convert_to_global_name=True: iter([(n, p) for chunk in m for n, p in chunk.named_parameters()]),
    )
    payload = list(common.effective_named_params_and_buffers(args, [model]))
    assert len(payload) == len(list(model.named_parameters()))
    for name, tensor in payload:
        assert tensor is dict(model.named_parameters())[name], "no copy may be introduced when LoRA is off"


# --------------------------------------------------------------------------- #
# lazy mapping (colocated tensor / IPC and --save-hf)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_lazy_mapping_hides_adapters_and_merges_on_access(lora_setup):
    args, model, _ = lora_setup
    from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers

    with torch.no_grad():
        for _, module in iter_lora_modules(model[0]):
            module.lora_B.normal_(0.0, 0.1)

    backing = {
        name: tensor.detach().clone()
        for name, tensor in named_params_and_buffers(args, model, convert_to_global_name=True)
    }
    mapping = EffectiveWeightMapping(args, model, backing)

    assert not any(name.endswith((".lora_A", ".lora_B")) for name in mapping)
    assert len(mapping) == len([n for n in backing if not n.endswith((".lora_A", ".lora_B"))])

    key = "module.module.decoder.layers.0.mlp.linear_fc1.weight"
    module = dict(iter_lora_modules(model[0]))["decoder.layers.0.mlp.linear_fc1"]
    torch.testing.assert_close(
        mapping[key], backing[key] + lora_local_delta(module).to(backing[key].dtype), rtol=1e-6, atol=1e-7
    )
    # The backing snapshot itself is never modified.
    torch.testing.assert_close(backing[key], module.weight.detach())


@pytest.mark.unit
def test_lazy_mapping_does_not_materialise_everything_up_front(lora_setup):
    args, model, _ = lora_setup
    calls = {"count": 0}
    import slime.backends.megatron_utils.lora.merge as merge_module

    original = merge_module.merge_into

    def counting_merge(base, module, **kwargs):
        calls["count"] += 1
        return original(base, module, **kwargs)

    merge_module.merge_into = counting_merge
    try:
        from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers

        backing = dict(named_params_and_buffers(args, model, convert_to_global_name=True))
        mapping = EffectiveWeightMapping(args, model, backing)
        assert calls["count"] == 0, "constructing the mapping must not merge anything"
        _ = mapping["module.module.decoder.layers.0.mlp.linear_fc1.weight"]
        assert calls["count"] == 1
    finally:
        merge_module.merge_into = original


@pytest.mark.unit
def test_lazy_mapping_merges_one_coherent_host_snapshot(lora_setup):
    """The sync payload must not read a live adapter after actor backup/offload."""
    args, model, _ = lora_setup
    from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers

    backing = {
        name: tensor.detach().clone()
        for name, tensor in named_params_and_buffers(args, model, convert_to_global_name=True)
    }
    target = "module.module.decoder.layers.0.mlp.linear_fc1"
    backing[f"{target}.lora_B"].fill_(0.125)
    mapping = EffectiveWeightMapping(args, model, backing)

    # Deliberately diverge the live module after the snapshot was taken.
    module = dict(iter_lora_modules(model[0]))["decoder.layers.0.mlp.linear_fc1"]
    with torch.no_grad():
        module.lora_A.fill_(7.0)
        module.lora_B.fill_(9.0)

    expected_delta = lora_setup[2].scale * (
        backing[f"{target}.lora_B"].float() @ backing[f"{target}.lora_A"].float()
    )
    torch.testing.assert_close(
        mapping[f"{target}.weight"],
        backing[f"{target}.weight"] + expected_delta.to(backing[f"{target}.weight"].dtype),
    )


# --------------------------------------------------------------------------- #
# mock engine synchronisation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_lora_factors_never_reach_the_full_weight_payload(lora_setup):
    args, model, _ = lora_setup
    engine = MockSGLangEngine()
    _sync(args, model, engine, version=1)
    assert engine.weights
    assert not any("lora_A" in name or "lora_B" in name for name in engine.weights)


@pytest.mark.unit
def test_engine_matches_the_trainer_policy_after_sync(lora_setup):
    args, model, config = lora_setup
    with torch.no_grad():
        for _, module in iter_lora_modules(model[0]):
            module.lora_B.normal_(0.0, 0.1)

    engine = MockSGLangEngine()
    _sync(args, model, engine, version=1)

    module = dict(iter_lora_modules(model[0]))["decoder.layers.0.mlp.linear_fc1"]
    x = torch.randn(3, 16)
    trainer_output, _ = module(x)
    engine_output = engine.linear("module.module.decoder.layers.0.mlp.linear_fc1.weight", x)
    torch.testing.assert_close(engine_output, trainer_output, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_optimizer_step_changes_target_weights_but_not_the_rest(lora_setup):
    args, model, _ = lora_setup
    engine = MockSGLangEngine()
    _sync(args, model, engine, version=1)
    before = {name: tensor.clone() for name, tensor in engine.weights.items()}

    optimizer = torch.optim.Adam([p for p in model[0].parameters() if p.requires_grad], lr=1e-2)
    model[0](torch.randint(0, 32, (2, 4))).pow(2).mean().backward()
    optimizer.step()

    _sync(args, model, engine, version=2)

    target_keys = {f"module.module.{name}.weight" for name, _ in iter_lora_modules(model[0])}
    changed = {name for name, tensor in engine.weights.items() if not torch.equal(tensor, before[name])}
    assert changed, "the merged payload did not change after a LoRA update"
    assert changed <= target_keys, f"non-target weights changed: {sorted(changed - target_keys)}"
    assert "module.module.output_layer.weight" not in changed
    assert "module.module.visual.blocks.0.qkv.weight" not in changed


@pytest.mark.unit
def test_repeated_sync_without_training_is_idempotent(lora_setup):
    """Guards against the delta being accumulated into the base on every sync."""
    args, model, _ = lora_setup
    with torch.no_grad():
        for _, module in iter_lora_modules(model[0]):
            module.lora_B.normal_(0.0, 0.1)

    engine = MockSGLangEngine()
    _sync(args, model, engine, version=1)
    first = {name: tensor.clone() for name, tensor in engine.weights.items()}
    _sync(args, model, engine, version=2)
    _sync(args, model, engine, version=3)
    for name, tensor in engine.weights.items():
        torch.testing.assert_close(tensor, first[name], rtol=0, atol=0)


@pytest.mark.unit
def test_failed_sync_does_not_publish_a_new_policy_version(lora_setup):
    args, model, _ = lora_setup
    engine = MockSGLangEngine()
    _sync(args, model, engine, version=1)
    assert engine.weight_version == 1

    engine.fail_next = True
    with pytest.raises(RuntimeError, match="simulated engine failure"):
        _sync(args, model, engine, version=2)
    assert engine.weight_version == 1, "a failed update must not advance the policy version"

    _sync(args, model, engine, version=2)
    assert engine.weight_version == 2


@pytest.mark.unit
def test_multiple_engines_receive_identical_weights(lora_setup):
    args, model, _ = lora_setup
    with torch.no_grad():
        for _, module in iter_lora_modules(model[0]):
            module.lora_B.normal_(0.0, 0.1)

    engines = [MockSGLangEngine() for _ in range(3)]
    payload = list(merged_named_params(args, model))
    for engine in engines:
        engine.update_weights(payload, weight_version=4)

    reference = engines[0].weights
    for engine in engines[1:]:
        assert set(engine.weights) == set(reference)
        for name, tensor in engine.weights.items():
            torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)
        assert engine.weight_version == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
