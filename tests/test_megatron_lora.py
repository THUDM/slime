"""LoRA layer / injection unit tests.

The CPU part (``NUM_GPUS = 0``) pins the mathematical contract at TP=1:
zero-initialised delta, the forward formula, gradient flow, optimizer membership,
non-destructive merging and every fail-fast path.

Two-process forward/backward parity lives in test_lora_parallel.py (CPU) and
real Megatron/Transformer Engine checkpoint parity in test_lora_parallel_gpu.py.
"""

from __future__ import annotations

from types import SimpleNamespace

# Import the fakes first so the megatron stub lands in sys.modules before slime's
# backend package is imported (see tests/_lora_fakes.py).
import _lora_fakes  # noqa: F401
import pytest
import torch
import torch.nn.functional as F
from _lora_fakes import (
    FakeColumnParallelLinear,
    FakeFusedLayerNormColumnLinear,
    FakeGPTModel,
    FakeMoEModel,
    FakeRowParallelLinear,
    FakeUnsupportedModule,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
    make_config,
)

from slime.backends.megatron_utils.lora import layers as lora_layers  # noqa: E402
from slime.backends.megatron_utils.lora import (  # noqa: E402
    LoRAConfigError,
    LoRAInjectionError,
    LoRAUnsupportedModuleError,
    assert_lora_gradients,
    assert_optimizer_holds_only_lora,
    describe_lora_target,
    inject_lora,
    is_lora_module,
    is_lora_param_name,
    iter_lora_modules,
    lora_local_delta,
    match_target_modules,
    prepare_lora_backward,
)

NUM_GPUS = 0

HIDDEN = 16
FFN = 32
RANK = 8


def _small_config(**overrides):
    return make_config(lora_rank=RANK, lora_alpha=float(2 * RANK), **overrides)


def _model_and_config(**overrides):
    torch.manual_seed(1234)
    model = FakeGPTModel(num_layers=2, hidden=HIDDEN, ffn=FFN)
    return model, _small_config(**overrides)


@pytest.mark.unit
def test_cpu_lora_parameter_uses_cuda_staging_for_nccl_broadcast(monkeypatch):
    moves: list[str] = []
    broadcasts: list[tuple[str, int, object]] = []

    class FakeTensor:
        def __init__(self, device: str):
            self.device = torch.device(device)
            self.copied = False

        def detach(self):
            return self

        def to(self, device):
            moves.append(str(device))
            return FakeTensor(str(device))

        def copy_(self, _other):
            self.copied = True

    parameter = FakeTensor("cpu")
    group = object()
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "nccl")
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src, group: broadcasts.append((str(tensor.device), src, group)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    lora_layers._broadcast_parameter(parameter, src=7, group=group)

    assert moves == ["cuda:3", "cpu"]
    assert broadcasts == [("cuda:3", 7, group)]
    assert parameter.copied is True


# --------------------------------------------------------------------------- #
# target matching
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_preset_matches_attention_and_mlp_projections_only():
    model, config = _model_and_config()
    matched = dict(match_target_modules(model, config))
    assert set(matched) == {
        "decoder.layers.0.self_attention.linear_qkv",
        "decoder.layers.0.self_attention.linear_proj",
        "decoder.layers.0.mlp.linear_fc1",
        "decoder.layers.0.mlp.linear_fc2",
        "decoder.layers.1.self_attention.linear_qkv",
        "decoder.layers.1.self_attention.linear_proj",
        "decoder.layers.1.mlp.linear_fc1",
        "decoder.layers.1.mlp.linear_fc2",
    }


@pytest.mark.unit
def test_zero_matches_fails_fast_instead_of_training_without_adapters():
    model, config = _model_and_config(lora_target_modules=[r"this\.module\.does\.not\.exist"])
    with pytest.raises(LoRAConfigError, match="matched no module"):
        inject_lora(model, config)


@pytest.mark.unit
def test_everything_excluded_fails_fast_and_reports_the_exclusions():
    model, config = _model_and_config(
        lora_target_modules=[r"mlp\.linear_fc1$"], lora_exclude_modules=[r"mlp\.linear_fc1$"]
    )
    with pytest.raises(LoRAConfigError) as excinfo:
        inject_lora(model, config)
    assert "excluded by the exclude list" in str(excinfo.value)


@pytest.mark.unit
def test_overlapping_target_regexes_inject_each_module_once():
    model, config = _model_and_config(lora_target_modules=[r"mlp\.linear_fc1$", r"linear_fc1$"])
    report = inject_lora(model, config)
    assert report.matched_module_count == len(model.decoder.layers)
    assert len(list(iter_lora_modules(model))) == report.matched_module_count


@pytest.mark.unit
def test_double_injection_is_rejected():
    model, config = _model_and_config()
    inject_lora(model, config)
    with pytest.raises(LoRAInjectionError, match="twice"):
        inject_lora(model, config)


@pytest.mark.unit
def test_unsupported_module_fails_fast_with_diagnostic_context():
    module = FakeUnsupportedModule()
    with pytest.raises(LoRAUnsupportedModuleError) as excinfo:
        describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, _small_config())
    message = str(excinfo.value)
    assert "no 2-D `weight`" in message
    assert "module type" in message and "parallel rank" in message and "lora config" in message


@pytest.mark.unit
def test_non_grouped_routed_expert_module_is_rejected():
    """Only Transformer Engine's GroupedLinear (--moe-grouped-gemm) is supported.

    A routed-expert module without ``weight0`` is some other expert
    implementation whose packing slime has not been verified against.
    """
    module = FakeColumnParallelLinear(HIDDEN, FFN)
    with pytest.raises(LoRAUnsupportedModuleError, match="no `weight0` parameter"):
        describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", module, _small_config())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cls", "suffix", "expected_mode"),
    [
        (TEColumnParallelGroupedLinear, "linear_fc1", "expert_column"),
        (TERowParallelGroupedLinear, "linear_fc2", "expert_row"),
    ],
)
def test_grouped_routed_experts_are_classified_as_expert_parallel(cls, suffix, expected_mode):
    num_local_experts = 3
    module = cls(num_local_experts, HIDDEN, FFN)
    spec = describe_lora_target(f"decoder.layers.0.mlp.experts.{suffix}", module, _small_config())

    assert spec.parallel_mode == expected_mode
    assert spec.is_expert
    assert spec.num_local_experts == num_local_experts
    # One shared adapter, but a delta applied to every packed expert weight.
    assert spec.base_weight_names == tuple(f"weight{i}" for i in range(num_local_experts))


@pytest.mark.unit
def test_routed_expert_column_adapter_keeps_lora_a_replicated():
    """The MoE dispatcher gathers along tokens, so A sees the full hidden width."""
    module = TEColumnParallelGroupedLinear(2, HIDDEN, FFN)
    spec = describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", module, _small_config())
    assert spec.lora_in_features == HIDDEN
    assert not spec.a_is_tensor_parallel


@pytest.mark.unit
def test_grouped_experts_with_mismatched_shapes_are_rejected():
    module = TEColumnParallelGroupedLinear(2, HIDDEN, FFN)
    with torch.no_grad():
        module.weight1 = torch.nn.Parameter(torch.randn(FFN + 8, HIDDEN))
    with pytest.raises(LoRAUnsupportedModuleError, match="differing weight shapes"):
        describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", module, _small_config())


@pytest.mark.unit
def test_shared_experts_linear_is_accepted():
    """``shared_experts`` are ordinary dense TP linears: no grouped GEMM, no expert parallelism."""
    module = FakeColumnParallelLinear(HIDDEN, FFN)
    spec = describe_lora_target("decoder.layers.0.mlp.shared_experts.linear_fc1", module, _small_config())
    assert spec.parallel_mode == "column"
    assert not spec.is_expert


@pytest.mark.unit
def test_replicated_linear_is_rejected_when_tensor_parallel(monkeypatch):
    """This is exactly why vision LoRA is unsupported: replicated grads would need syncing."""
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(LoRAUnsupportedModuleError, match="replicated"):
        describe_lora_target("visual.blocks.0.attn.qkv", torch.nn.Linear(HIDDEN, HIDDEN), _small_config())


@pytest.mark.unit
def test_explicitly_allowed_replicated_linear_is_supported_when_tensor_parallel(monkeypatch):
    """Replicated linears need explicit opt-in regardless of the selected preset."""
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: 2)
    module = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    spec = describe_lora_target(
        "decoder.layers.0.self_attention.linear_attn.in_proj_qkv",
        module,
        _small_config(
            lora_allow_replicated_modules=[
                r"(?:^|\.)decoder\.layers\.\d+\.self_attention\.linear_attn\."
                r"(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$"
            ]
        ),
    )
    assert spec.parallel_mode == "replicated"
    assert spec.tp_size == 2
    assert spec.lora_in_features == HIDDEN
    assert spec.lora_out_features == HIDDEN


@pytest.mark.unit
def test_arbitrary_replicated_language_linear_is_still_rejected(monkeypatch):
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(LoRAUnsupportedModuleError, match="replicated"):
        describe_lora_target(
            "decoder.layers.0.some_new_block.proj",
            torch.nn.Linear(HIDDEN, HIDDEN),
            _small_config(),
        )


@pytest.mark.unit
def test_row_parallel_requires_rank_divisible_by_tensor_parallel_size(monkeypatch):
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: 4)
    module = FakeRowParallelLinear(FFN, HIDDEN)
    with pytest.raises(LoRAUnsupportedModuleError, match="divisible"):
        describe_lora_target("decoder.layers.0.mlp.linear_fc2", module, make_config(lora_rank=6, lora_alpha=12.0))


@pytest.mark.unit
def test_fused_layernorm_with_unknown_normalisation_is_rejected():
    module = FakeFusedLayerNormColumnLinear(HIDDEN, FFN, normalization="MysteryNorm")
    with pytest.raises(LoRAUnsupportedModuleError, match="normalisation"):
        describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, _small_config())


# --------------------------------------------------------------------------- #
# parameter naming and shapes
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_base_parameter_names_are_unchanged_by_injection():
    """The Megatron->HF converters and delta sync are name-driven; renaming breaks them."""
    model, config = _model_and_config()
    before = {name for name, _ in model.named_parameters()}
    inject_lora(model, config)
    after = {name for name, _ in model.named_parameters()}

    assert before <= after, "injection must not rename or drop any base parameter"
    added = after - before
    assert added == {
        f"{module}.{suffix}"
        for module in [
            "decoder.layers.0.self_attention.linear_qkv",
            "decoder.layers.0.self_attention.linear_proj",
            "decoder.layers.0.mlp.linear_fc1",
            "decoder.layers.0.mlp.linear_fc2",
            "decoder.layers.1.self_attention.linear_qkv",
            "decoder.layers.1.self_attention.linear_proj",
            "decoder.layers.1.mlp.linear_fc1",
            "decoder.layers.1.mlp.linear_fc2",
        ]
        for suffix in ("lora_A", "lora_B")
    }
    assert "decoder.layers.0.self_attention.linear_qkv.weight" in after
    assert not any("base_layer" in name for name in after)


@pytest.mark.unit
def test_adapter_shapes_follow_the_parallel_mode():
    model, config = _model_and_config()
    inject_lora(model, config)
    modules = dict(iter_lora_modules(model))

    column = modules["decoder.layers.0.mlp.linear_fc1"]
    assert tuple(column.lora_A.shape) == (RANK, HIDDEN)
    assert tuple(column.lora_B.shape) == (FFN, RANK)

    row = modules["decoder.layers.0.mlp.linear_fc2"]
    assert tuple(row.lora_A.shape) == (RANK, FFN)
    assert tuple(row.lora_B.shape) == (HIDDEN, RANK)


@pytest.mark.unit
def test_report_counts_are_consistent():
    model, config = _model_and_config()
    report = inject_lora(model, config)
    assert report.matched_module_count == 8
    assert report.trainable_parameters == report.expected_trainable_parameters()
    assert report.frozen_parameters > 0
    assert 0.0 < report.trainable_ratio < 1.0
    metrics = report.numeric_metrics()
    assert all(isinstance(value, float) for value in metrics.values()), "logger must only receive scalars"
    assert metrics["lora/scale"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# routed MoE experts, end to end
# --------------------------------------------------------------------------- #


def _moe_model_and_config(num_local_experts: int = 3):
    torch.manual_seed(11)
    model = FakeMoEModel(num_layers=1, hidden=HIDDEN, ffn=FFN, num_local_experts=num_local_experts)
    config = make_config(
        lora_rank=RANK, lora_alpha=float(2 * RANK), lora_target_preset="moe_language_all"
    )
    return model, config


@pytest.mark.unit
def test_injection_covers_attention_shared_and_routed_experts():
    model, config = _moe_model_and_config()
    report = inject_lora(model, config)

    modes = {spec.module_name: spec.parallel_mode for spec in report.specs}
    assert modes["decoder.layers.0.mlp.experts.linear_fc1"] == "expert_column"
    assert modes["decoder.layers.0.mlp.experts.linear_fc2"] == "expert_row"
    assert modes["decoder.layers.0.mlp.shared_experts.linear_fc1"] == "column"
    assert modes["decoder.layers.0.self_attention.linear_qkv"] == "column"
    # The router must never be adapted: changing it changes expert assignment.
    assert not any("router" in name for name in modes)
    assert report.trainable_parameters == report.expected_trainable_parameters()


@pytest.mark.unit
def test_routed_expert_adapter_is_shared_not_per_expert():
    """Parameter count must not scale with the number of experts."""
    few, many = _moe_model_and_config(2), _moe_model_and_config(8)
    report_few = inject_lora(few[0], few[1])
    report_many = inject_lora(many[0], many[1])
    assert report_few.trainable_parameters == report_many.trainable_parameters


@pytest.mark.unit
def test_moe_initial_logits_match_the_base_model_bitwise():
    """``lora_B`` starts at zero, so the routed-expert delta must vanish too."""
    torch.manual_seed(11)
    model = FakeMoEModel(num_layers=1, hidden=HIDDEN, ffn=FFN, num_local_experts=3)
    tokens = torch.randint(0, 32, (2, 4))
    with torch.no_grad():
        baseline = model(tokens).clone()

    inject_lora(model, make_config(lora_rank=RANK, lora_target_preset="moe_language_all"))
    with torch.no_grad():
        assert torch.equal(model(tokens), baseline)


@pytest.mark.unit
def test_routed_expert_base_weights_are_frozen_and_adapters_train():
    model, config = _moe_model_and_config()
    inject_lora(model, config)
    experts = model.decoder.layers[0].mlp.experts

    for projection in (experts.linear_fc1, experts.linear_fc2):
        for index in range(projection.num_gemms):
            assert not getattr(projection, f"weight{index}").requires_grad
        assert projection.lora_A.requires_grad and projection.lora_B.requires_grad

    model(torch.randint(0, 32, (2, 4))).sum().backward()
    for projection in (experts.linear_fc1, experts.linear_fc2):
        assert projection.lora_B.grad is not None and projection.lora_B.grad.norm() > 0


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_initial_delta_is_exactly_zero():
    model, config = _model_and_config()
    inject_lora(model, config)
    for _, module in iter_lora_modules(model):
        assert torch.equal(module.lora_B, torch.zeros_like(module.lora_B))
        assert torch.equal(lora_local_delta(module), torch.zeros_like(module.weight, dtype=torch.float32))


@pytest.mark.unit
def test_initial_logits_match_the_base_model_bitwise():
    torch.manual_seed(7)
    model = FakeGPTModel(num_layers=2, hidden=HIDDEN, ffn=FFN)
    tokens = torch.randint(0, 32, (2, 5))
    with torch.no_grad():
        baseline = model(tokens).clone()

    inject_lora(model, _small_config())
    with torch.no_grad():
        after = model(tokens)
    assert torch.equal(baseline, after)


@pytest.mark.unit
@pytest.mark.parametrize("factory", ["column", "row"])
def test_forward_matches_the_lora_formula(factory):
    torch.manual_seed(3)
    module = (
        FakeColumnParallelLinear(HIDDEN, FFN) if factory == "column" else FakeRowParallelLinear(HIDDEN, FFN)
    )
    base_weight = module.weight.detach().clone()
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    with torch.no_grad():
        module.lora_B.normal_(0.0, 0.1)

    x = torch.randn(2, 3, HIDDEN)
    expected = F.linear(x, base_weight) + config.scale * F.linear(F.linear(x, module.lora_A), module.lora_B)
    got, _ = module(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_sequence_parallel_column_gathers_tokens_before_hidden_reduce(monkeypatch):
    """TP ranks hold different tokens, so hidden reduction must see gathered sequences."""
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_rank", lambda: 0)
    calls: list[tuple[str, tuple[int, ...]]] = []

    def gather(x):
        calls.append(("gather", tuple(x.shape)))
        return torch.cat((x, x + 1.0), dim=0)

    def reduce(x, _group):
        calls.append(("reduce", tuple(x.shape)))
        return x

    monkeypatch.setattr(lora_layers, "_gather_along_sequence", gather)
    monkeypatch.setattr(lora_layers, "symmetric_all_reduce", reduce)
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_group", lambda: object())

    module = FakeColumnParallelLinear(HIDDEN, FFN, sequence_parallel=True)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.self_attention.linear_qkv", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    delta = lora_layers.compute_lora_delta(module, torch.randn(3, 1, HIDDEN))

    assert calls == [("gather", (3, 1, HIDDEN)), ("reduce", (6, 1, RANK))]
    assert delta.shape == (6, 1, FFN)


@pytest.mark.unit
def test_fused_layernorm_lora_consumes_the_normalised_input():
    """A LoRA on TELayerNormColumnParallelLinear must see layer_norm(x), not x."""
    torch.manual_seed(5)
    module = FakeFusedLayerNormColumnLinear(HIDDEN, FFN)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.self_attention.linear_qkv", module, config)
    assert spec.fused_layernorm is True
    lora_layers.attach_lora_adapter(module, spec, config)
    with torch.no_grad():
        module.lora_B.normal_(0.0, 0.1)

    x = torch.randn(2, 3, HIDDEN) * 3.0 + 1.0
    normalized = module._normalize(x)
    expected = F.linear(normalized, module.weight) + config.scale * F.linear(
        F.linear(normalized, module.lora_A), module.lora_B
    )
    got, _ = module(x)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    # Using the raw input would give a materially different answer.
    wrong = F.linear(normalized, module.weight) + config.scale * F.linear(F.linear(x, module.lora_A), module.lora_B)
    assert not torch.allclose(got, wrong, rtol=1e-3, atol=1e-4)


@pytest.mark.unit
def test_merged_weight_reproduces_the_adapter_forward():
    torch.manual_seed(11)
    module = FakeColumnParallelLinear(HIDDEN, FFN)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    with torch.no_grad():
        module.lora_B.normal_(0.0, 0.1)

    x = torch.randn(4, HIDDEN)
    adapter_output, _ = module(x)
    merged = module.weight.detach() + lora_local_delta(module).to(module.weight.dtype)
    torch.testing.assert_close(F.linear(x, merged), adapter_output, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_merge_is_non_destructive_and_not_cumulative():
    from slime.backends.megatron_utils.lora import merge_into

    torch.manual_seed(13)
    module = FakeRowParallelLinear(FFN, HIDDEN)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.linear_fc2", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    with torch.no_grad():
        module.lora_B.normal_(0.0, 0.1)

    base_snapshot = module.weight.detach().clone()
    first = merge_into(module.weight.detach(), module)
    second = merge_into(module.weight.detach(), module)

    assert torch.equal(module.weight.detach(), base_snapshot), "merging must never touch the frozen base"
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.data_ptr() != module.weight.data_ptr()
    assert first.dtype == module.weight.dtype and first.shape == module.weight.shape


@pytest.mark.unit
def test_merge_preserves_megatron_distribution_attributes():
    from slime.backends.megatron_utils.lora import merge_into

    module = FakeColumnParallelLinear(HIDDEN, FFN)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    base = module.weight.detach()
    base.tensor_model_parallel = True
    base.partition_dim = 0
    base.partition_stride = 1
    base.parallel_mode = "column"

    merged = merge_into(base, module)

    assert merged.tensor_model_parallel is True
    assert merged.partition_dim == 0
    assert merged.partition_stride == 1
    assert merged.parallel_mode == "column"


# --------------------------------------------------------------------------- #
# gradients and optimizer
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_only_adapters_are_trainable_and_receive_gradients():
    model, config = _model_and_config()
    inject_lora(model, config)

    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable and all(name.endswith((".lora_A", ".lora_B")) for name in trainable)

    model(torch.randint(0, 32, (2, 4))).sum().backward()
    for name, param in model.named_parameters():
        if name.endswith((".lora_A", ".lora_B")):
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all()
        else:
            assert param.grad is None, f"frozen parameter {name} received a gradient"


@pytest.mark.unit
def test_lora_a_receives_gradient_even_though_b_starts_at_zero():
    """B=0 zeroes dL/dA only if the graph is wrong; the loss must still reach A via B's update."""
    model, config = _model_and_config()
    inject_lora(model, config)
    module = dict(iter_lora_modules(model))["decoder.layers.0.mlp.linear_fc1"]
    model(torch.randint(0, 32, (2, 4))).sum().backward()
    # dL/dB is non-zero at initialisation; dL/dA is zero *because* B == 0, which is the
    # documented LoRA behaviour and resolves after the first optimizer step.
    assert module.lora_B.grad.abs().sum() > 0
    with torch.no_grad():
        module.lora_B.normal_(0.0, 0.1)
    model.zero_grad(set_to_none=True)
    model(torch.randint(0, 32, (2, 4))).sum().backward()
    assert module.lora_A.grad.abs().sum() > 0


@pytest.mark.unit
def test_optimizer_holds_only_lora_parameters():
    model, config = _model_and_config()
    report = inject_lora(model, config)

    # This mirrors Megatron's _get_param_groups(), which skips requires_grad=False.
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-3)
    assert assert_optimizer_holds_only_lora(optimizer, model) == report.matched_module_count * 2

    optimizer_numel = sum(param.numel() for group in optimizer.param_groups for param in group["params"])
    assert optimizer_numel == report.trainable_parameters


@pytest.mark.unit
def test_optimizer_including_the_base_is_detected():
    model, config = _model_and_config()
    inject_lora(model, config)
    bad_optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    with pytest.raises(LoRAInjectionError, match="must never enter the optimizer"):
        assert_optimizer_holds_only_lora(bad_optimizer, model)


@pytest.mark.unit
def test_distributed_optimizer_accepts_dp_local_lora_shards():
    model, config = _model_and_config()
    inject_lora(model, config)
    lora_parameters = [param for name, param in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))]
    local_model_parameters = lora_parameters[::4]
    local_main_parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in local_model_parameters]
    distributed_optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"params": local_main_parameters}]),
        model_param_gbuf_map={param: object() for param in local_model_parameters},
    )
    optimizer = SimpleNamespace(chained_optimizers=[distributed_optimizer])

    assert assert_optimizer_holds_only_lora(optimizer, model) == len(local_main_parameters)


@pytest.mark.unit
def test_distributed_optimizer_mapping_rejects_a_frozen_base_parameter():
    model, config = _model_and_config()
    inject_lora(model, config)
    base_parameter = next(param for name, param in model.named_parameters() if not is_lora_param_name(name))
    distributed_optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"params": [torch.nn.Parameter(torch.zeros(1))]}]),
        model_param_gbuf_map={base_parameter: object()},
    )
    optimizer = SimpleNamespace(chained_optimizers=[distributed_optimizer])

    with pytest.raises(LoRAInjectionError, match="must never enter the optimizer"):
        assert_optimizer_holds_only_lora(optimizer, model)


@pytest.mark.unit
def test_one_optimizer_step_changes_the_adapter_and_leaves_the_base_untouched():
    model, config = _model_and_config()
    inject_lora(model, config)
    base_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not name.endswith((".lora_A", ".lora_B"))
    }
    adapter_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.endswith((".lora_A", ".lora_B"))
    }

    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-2)
    tokens = torch.randint(0, 32, (2, 4))
    model(tokens).pow(2).mean().backward()
    optimizer.step()

    for name, param in model.named_parameters():
        if name in base_snapshot:
            assert torch.equal(param.detach(), base_snapshot[name]), f"base parameter {name} changed"
    current = dict(model.named_parameters())
    changed = [name for name, snapshot in adapter_snapshot.items() if not torch.equal(current[name], snapshot)]
    assert changed, "no adapter parameter changed after an optimizer step"

    # After the step the effective policy must actually differ from the base model.
    module = dict(iter_lora_modules(model))["decoder.layers.0.mlp.linear_fc1"]
    assert lora_local_delta(module).abs().sum() > 0


@pytest.mark.unit
def test_gradient_sanity_check_detects_a_trainable_base_parameter():
    model, config = _model_and_config()
    inject_lora(model, config)
    model(torch.randint(0, 32, (2, 4))).sum().backward()
    stats = assert_lora_gradients([model], require_nonzero=True)
    assert stats.gradient_norm > 0
    assert stats.b_gradient_norm > 0
    assert stats.parameter_count_with_grad == stats.parameter_count

    leaked = model.output_layer.weight
    leaked.requires_grad = True
    leaked.grad = torch.zeros_like(leaked)
    with pytest.raises(LoRAInjectionError, match="frozen parameters with gradient"):
        assert_lora_gradients([model])


@pytest.mark.unit
def test_gradient_sanity_check_rejects_all_zero_adapter_gradients():
    model, config = _model_and_config()
    inject_lora(model, config)
    for name, param in model.named_parameters():
        if is_lora_param_name(name):
            param.grad = torch.zeros_like(param)

    with pytest.raises(LoRAInjectionError, match="all-zero adapter gradient"):
        assert_lora_gradients(model, require_nonzero=True)


@pytest.mark.unit
def test_gradient_sanity_check_prefers_megatron_main_grad():
    model, config = _model_and_config()
    inject_lora(model, config)
    for name, param in model.named_parameters():
        if is_lora_param_name(name):
            param.grad = torch.zeros_like(param)
            param.main_grad = torch.ones_like(param)

    stats = assert_lora_gradients(model, require_nonzero=True)
    assert stats.gradient_norm > 0


@pytest.mark.unit
def test_megatron_main_grad_bridge_accumulates_lora_backward():
    model, config = _model_and_config()
    inject_lora(model, config)
    adapter_parameters = [
        param for name, param in model.named_parameters() if is_lora_param_name(name)
    ]
    for param in adapter_parameters:
        param.main_grad = torch.zeros_like(param, dtype=torch.float32)
        param.grad_added_to_main_grad = False

    assert prepare_lora_backward(model) == len(adapter_parameters)
    model(torch.randint(0, 32, (2, 4))).sum().backward()

    stats = assert_lora_gradients(model, require_nonzero=True)
    assert stats.parameter_count_seen_in_backward == stats.parameter_count
    assert stats.b_gradient_norm > 0
    assert all(param.grad_added_to_main_grad for param in adapter_parameters)


@pytest.mark.unit
def test_prepare_lora_backward_installs_hooks_on_replaced_runtime_parameters():
    model, config = _model_and_config()
    inject_lora(model, config)
    modules = dict(iter_lora_modules(model))
    replaced_count = 0
    for module in modules.values():
        for name in ("lora_A", "lora_B"):
            original = getattr(module, name)
            replacement = torch.nn.Parameter(original.detach().clone())
            replacement.main_grad = torch.zeros_like(replacement, dtype=torch.float32)
            replacement.grad_added_to_main_grad = False
            setattr(module, name, replacement)
            replaced_count += 1

    assert prepare_lora_backward(model) == replaced_count
    assert prepare_lora_backward(model) == 0
    model(torch.randint(0, 32, (2, 4))).sum().backward()

    stats = assert_lora_gradients(model, require_nonzero=True)
    assert stats.parameter_count_seen_in_backward == stats.parameter_count
    assert stats.b_gradient_norm > 0


@pytest.mark.unit
def test_main_grad_bridge_leaves_plain_pytorch_grad_unchanged():
    model, config = _model_and_config()
    inject_lora(model, config)
    prepare_lora_backward(model)
    model(torch.randint(0, 32, (2, 4))).sum().backward()

    stats = assert_lora_gradients(model, require_nonzero=True)
    assert stats.parameter_count_seen_in_backward == stats.parameter_count
    assert all(
        param.grad is not None
        for name, param in model.named_parameters()
        if is_lora_param_name(name)
    )


@pytest.mark.unit
def test_zero_training_signal_is_distinguished_from_a_disconnected_adapter():
    model, config = _model_and_config()
    inject_lora(model, config)
    prepare_lora_backward(model)
    (model(torch.randint(0, 32, (2, 4))).sum() * 0.0).backward()

    stats = assert_lora_gradients(model)
    assert stats.parameter_count_seen_in_backward == stats.parameter_count
    assert stats.gradient_norm == 0.0


@pytest.mark.unit
def test_is_lora_module_only_flags_injected_modules():
    model, config = _model_and_config()
    inject_lora(model, config)
    assert is_lora_module(model.decoder.layers[0].mlp.linear_fc1)
    assert not is_lora_module(model.output_layer)
    assert not is_lora_module(model.embedding)


@pytest.mark.unit
def test_instance_level_forward_attribute_does_not_disable_the_adapter():
    """The original failure mode: an instance ``forward`` shadowed the class override.

    Transformer Engine wrappers, CUDA-graph capture and several Hugging Face
    modules install ``self.forward = ...``. ``_call_impl`` then never reaches a
    class-level override, so the base GEMM ran alone and no adapter parameter
    ever saw a gradient (``reached parameters: 0/N``). Forward hooks are invoked
    by ``_call_impl`` itself, so they survive the shadowing.
    """
    module = FakeColumnParallelLinear(HIDDEN, FFN)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    with torch.no_grad():
        module.lora_B.fill_(0.1)

    bound_forward = module.forward
    module.forward = bound_forward  # instance attribute shadows the class method

    x = torch.randn(4, 1, HIDDEN)
    output = module(x)[0]
    expected = bound_forward(x)[0] + lora_layers.compute_lora_delta(module, x)
    torch.testing.assert_close(output, expected)

    diagnostics = lora_layers.lora_forward_diagnostics(module)
    assert diagnostics["modules_executed"] == 1
    assert diagnostics["modules_without_hook"] == []
    assert diagnostics["modules_with_instance_forward"] == [""]


@pytest.mark.unit
def test_adapter_output_reaches_lora_parameters_in_backward():
    module = FakeColumnParallelLinear(HIDDEN, FFN)
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.linear_fc1", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    with torch.no_grad():
        module.lora_B.fill_(0.1)

    module(torch.randn(4, 1, HIDDEN))[0].sum().backward()
    assert module.lora_A.grad is not None and module.lora_A.grad.abs().sum() > 0
    assert module.lora_B.grad is not None and module.lora_B.grad.abs().sum() > 0


@pytest.mark.unit
def test_frozen_embedding_output_is_promoted_so_recompute_can_reconnect_the_graph():
    """Regression: activation recomputation silently detached the whole decoder.

    ``megatron.core.tensor_parallel.random.CheckpointFunction`` runs the layer
    under ``no_grad`` and only re-runs it from its ``backward``. Parameters are
    not inputs of that Function, so with a frozen embedding the decoder input did
    not require grad, the checkpoint output had no ``grad_fn``, and backward never
    happened at all -- producing ``reached parameters: 0/N`` with every adapter
    forward recorded under ``no_grad``.
    """
    model = FakeGPTModel()
    inject_lora(model, _small_config())

    assert not model.embedding.word_embeddings.weight.requires_grad
    hidden = model.embedding(torch.zeros(2, 4, dtype=torch.long))
    assert hidden.requires_grad


@pytest.mark.unit
def test_embedding_promotion_is_skipped_under_no_grad():
    """The ``forward_only`` log-prob passes must stay allocation-free."""
    model = FakeGPTModel()
    inject_lora(model, _small_config())

    with torch.no_grad():
        hidden = model.embedding(torch.zeros(2, 4, dtype=torch.long))
    assert not hidden.requires_grad


@pytest.mark.unit
@pytest.mark.parametrize("pre_process", [False, True])
def test_embedding_check_uses_the_virtual_chunk_pre_process_flag(pre_process, monkeypatch):
    model, config = _model_and_config()
    del model.embedding
    model.pre_process = pre_process
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "is_pipeline_first_stage", lambda **_: True, raising=False)
    if pre_process:
        with pytest.raises(LoRAInjectionError, match="embedding"):
            inject_lora(model, config)
    else:
        assert inject_lora(model, config).embedding_modules_hooked == 0


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_optimizer_maps_master_parameters_to_lora(dtype):
    model, config = _model_and_config()
    model.to(dtype)
    inject_lora(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    masters = [p.detach().float().clone() for p in params]
    optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"params": list(masters)}]),
        float16_groups=[params],
        fp32_from_float16_groups=[masters],
    )
    assert assert_optimizer_holds_only_lora(optimizer, model) == len(params)
    optimizer.optimizer.param_groups[0]["params"].append(model.output_layer.weight)
    with pytest.raises(LoRAInjectionError, match="must never enter"):
        assert_optimizer_holds_only_lora(optimizer, model)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_optimizer_still_rejects_missing_adapters(dtype):
    model, config = _model_and_config()
    model.to(dtype)
    inject_lora(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    masters = [p.detach().float().clone() for p in params]
    optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"params": masters[:-1]}]),
        float16_groups=[params],
        fp32_from_float16_groups=[masters],
    )
    with pytest.raises(LoRAInjectionError, match="missing LoRA"):
        assert_optimizer_holds_only_lora(optimizer, model)


@pytest.mark.unit
@pytest.mark.parametrize("tp_size, etp_size, tp_rank, etp_rank", [(1, 2, 0, 0), (3, 2, 0, 0), (2, 2, 0, 1)])
def test_routed_experts_reject_unaligned_checkpoint_coordinates(monkeypatch, tp_size, etp_size, tp_rank, etp_rank):
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(lora_layers, "expert_tensor_parallel_world_size", lambda: etp_size)
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(lora_layers, "expert_tensor_parallel_rank", lambda: etp_rank)
    module = TEColumnParallelGroupedLinear(2, HIDDEN, FFN)
    with pytest.raises(LoRAUnsupportedModuleError, match="expert-TP to divide TP"):
        describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", module, _small_config())


@pytest.mark.unit
@pytest.mark.parametrize("tp_size, etp_size, tp_rank", [(1, 1, 0), (4, 1, 3), (4, 2, 3), (2, 2, 1)])
def test_routed_experts_accept_unambiguous_checkpoint_coordinates(monkeypatch, tp_size, etp_size, tp_rank):
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(lora_layers, "expert_tensor_parallel_world_size", lambda: etp_size)
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(lora_layers, "expert_tensor_parallel_rank", lambda: tp_rank % etp_size)
    module = TEColumnParallelGroupedLinear(2, HIDDEN, FFN)
    spec = describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", module, _small_config())
    assert spec.tp_size == etp_size
    assert spec.tp_rank == tp_rank % etp_size


@pytest.mark.unit
@pytest.mark.parametrize(
    "tp_size, etp_size, ep_size, base_allreduce, expected",
    [(1, 1, 1, True, True), (2, 1, 1, True, False), (2, 2, 2, False, False), (1, 1, 1, False, False)],
)
def test_expert_adapter_uses_correct_ddp_topology(monkeypatch, tp_size, etp_size, ep_size, base_allreduce, expected):
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(lora_layers, "tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_layers, "expert_tensor_parallel_world_size", lambda: etp_size)
    monkeypatch.setattr(lora_layers, "expert_tensor_parallel_rank", lambda: 0)
    monkeypatch.setattr(lora_layers, "expert_model_parallel_world_size", lambda: ep_size)
    monkeypatch.setattr(lora_layers, "_broadcast_replicated_factors", lambda *args: None)
    module = TEColumnParallelGroupedLinear(2, HIDDEN, FFN)
    module.weight0.allreduce = base_allreduce
    module.weight1.allreduce = base_allreduce
    config = _small_config()
    spec = describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", module, config)
    lora_layers.attach_lora_adapter(module, spec, config)
    assert module.lora_A.allreduce is expected
    assert module.lora_B.allreduce is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
