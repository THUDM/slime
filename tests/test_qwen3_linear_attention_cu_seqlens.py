from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

NUM_GPUS = 0

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_megatron_stubs() -> None:
    if "megatron" in sys.modules:
        return

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    inference_mod = types.ModuleType("megatron.core.inference")
    inference_contexts_mod = types.ModuleType("megatron.core.inference.contexts")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_module_mod = types.ModuleType("megatron.core.transformer.module")
    spec_utils_mod = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_block_mod = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")

    @dataclass
    class PackedSeqParams:
        cu_seqlens_q: torch.Tensor | None = None
        cu_seqlens_kv: torch.Tensor | None = None
        max_seqlen_q: int | None = None
        max_seqlen_kv: int | None = None
        qkv_format: str | None = None

    class MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class ModuleSpec:
        def __init__(self, module=None, params=None):
            self.module = module
            self.params = params or {}

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    tensor_parallel_stub = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )

    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    inference_contexts_mod.BaseInferenceContext = type("BaseInferenceContext", (), {})
    packed_seq_mod.PackedSeqParams = PackedSeqParams
    transformer_module_mod.MegatronModule = MegatronModule
    spec_utils_mod.ModuleSpec = ModuleSpec
    transformer_block_mod.get_num_layers_to_build = lambda *args, **kwargs: 0
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0

    core_mod.mpu = mpu_stub
    core_mod.tensor_parallel = tensor_parallel_stub

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["megatron.core.inference"] = inference_mod
    sys.modules["megatron.core.inference.contexts"] = inference_contexts_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod
    sys.modules["megatron.core.transformer"] = transformer_mod
    sys.modules["megatron.core.transformer.module"] = transformer_module_mod
    sys.modules["megatron.core.transformer.spec_utils"] = spec_utils_mod
    sys.modules["megatron.core.transformer.transformer_block"] = transformer_block_mod
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_mod


class FakeShortConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, cu_seqlens=None, **kwargs):
        return x, None


class FakeFusedRMSNormGated(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, z):
        return x


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.float32,
    )


def load_module(module_name: str):
    install_megatron_stubs()
    sys.modules.pop("slime_plugins.models.hf_attention", None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "class_name", "args", "expected_backend"),
    [
        ("slime_plugins.models.qwen3_5", "Qwen3_5GatedDeltaNet", None, "fla"),
        (
            "slime_plugins.models.qwen3_5",
            "Qwen3_5GatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
        ("slime_plugins.models.qwen3_next", "Qwen3NextGatedDeltaNet", None, "fla"),
        (
            "slime_plugins.models.qwen3_next",
            "Qwen3NextGatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
    ],
)
def test_linear_attention_forwards_cu_seqlens_to_chunk_kernel(
    monkeypatch,
    module_name: str,
    class_name: str,
    args,
    expected_backend: str,
):
    module = load_module(module_name)

    monkeypatch.setattr(module.accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(module, "ShortConvolution", FakeShortConvolution, raising=False)
    monkeypatch.setattr(module, "FusedRMSNormGated", FakeFusedRMSNormGated, raising=False)

    chunk_calls = []
    selected_backends = []

    def fake_chunk_gated_delta_rule(
        q,
        k,
        v,
        *,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens=None,
        **kwargs,
    ):
        chunk_calls.append(cu_seqlens.clone() if cu_seqlens is not None else None)
        assert q.shape[0] == 1
        assert cu_seqlens is not None
        return torch.zeros_like(v), None

    def fake_get_chunk_gated_delta_rule(backend):
        selected_backends.append(backend)
        return fake_chunk_gated_delta_rule

    monkeypatch.setattr(module, "get_chunk_gated_delta_rule", fake_get_chunk_gated_delta_rule)

    layer = getattr(module, class_name)(make_config(), layer_idx=0, args=args)
    hidden_states = torch.randn(1, 7, 32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    output = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert selected_backends == [expected_backend]
    assert layer.gdn_backend == expected_backend
    assert output.shape == hidden_states.shape
    assert len(chunk_calls) == 1
    assert torch.equal(chunk_calls[0], cu_seqlens)


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("cached_offsets", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_cp_wrapper_preserves_packed_outputs_and_gradients(monkeypatch, cp_size, cached_offsets, sequence_parallel):
    module = load_module("slime_plugins.models.hf_attention")
    lengths = [8, 16]
    offsets = (0, 8, 24)

    class Attention(module.HuggingfaceAttention):
        def __init__(self):
            nn.Module.__init__(self)
            self.args = SimpleNamespace(sequence_parallel=sequence_parallel)
            self.weight = nn.Parameter(torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64))

        def hf_forward(self, hidden_states, packed_seq_params):
            # A causal operation with independent sequence boundaries exercises
            # ordering as well as gradient contributions from later CP chunks.
            return torch.cat(
                [(seq * self.weight).tanh().cumsum(1) for seq in hidden_states.split(lengths, dim=1)], dim=1
            )

    def local_indices(rank):
        indices = []
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            chunks = torch.arange(start, end).chunk(2 * cp_size)
            indices.extend([chunks[rank], chunks[-rank - 1]])
        return torch.cat(indices)

    torch.manual_seed(17)
    full = torch.randn(24, 1, 3, dtype=torch.float64, requires_grad=True)
    baseline = full.detach().clone().requires_grad_()
    attention = Attention()
    weight = attention.weight.detach().clone().requires_grad_()
    expected = torch.cat([(seq * weight).tanh().cumsum(0) for seq in baseline.split(lengths)], dim=0)
    gathered = tuple(full[local_indices(rank)] for rank in range(cp_size))
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(module.dist_nn, "all_gather", lambda value, group: gathered)

    def sp_gather(value, *, tensor_parallel_output_grad, group):
        assert tensor_parallel_output_grad is False
        return value

    monkeypatch.setattr(module.tensor_parallel, "gather_from_sequence_parallel_region", sp_gather)
    monkeypatch.setattr(module.tensor_parallel, "scatter_to_sequence_parallel_region", lambda value, group: value)
    # A meta tensor cannot supply host scalars: the cached path must never read
    # or convert the GPU-style offset tensor while rearranging hidden states.
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor(offsets, device="meta" if cached_offsets else "cpu"),
        cu_seqlens_q_cpu=offsets if cached_offsets else None,
    )
    losses = []
    for rank in range(cp_size):
        monkeypatch.setattr(module.mpu, "get_context_parallel_rank", lambda r=rank: r)
        output, bias = attention(gathered[rank], None, packed_seq_params=packed)
        torch.testing.assert_close(output, expected[local_indices(rank)])
        assert bias is None
        losses.append(output.square().sum())
    sum(losses).backward()
    expected.square().sum().backward()
    torch.testing.assert_close(full.grad, baseline.grad)
    torch.testing.assert_close(attention.weight.grad, weight.grad)


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("lengths", [(5, 8), (8, 24)])
def test_batch_keeps_host_offsets_aligned_with_padded_device_offsets(monkeypatch, cp_size, allgather_cp, lengths):
    install_megatron_stubs()
    from slime.backends.megatron_utils import data

    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(data.accelerator, "device", lambda: "cpu")
    tokens = [torch.arange(length) for length in lengths]
    batch = {
        "tokens": tokens,
        "loss_masks": [torch.ones(length - 2) for length in lengths],
        "total_lengths": list(lengths),
        "response_lengths": [length - 2 for length in lengths],
    }
    result = data.get_batch(SimpleNamespace(get_next=lambda keys: batch.copy()), list(batch), 4, allgather_cp)
    packed = result["packed_seq_params"]
    assert packed.cu_seqlens_q_cpu == tuple(packed.cu_seqlens_q.tolist())
    assert packed.cu_seqlens_q_cpu[-1] == result["tokens"].numel() * cp_size
    assert packed.max_seqlen_q == max(torch.diff(packed.cu_seqlens_q).tolist())
    assert packed.max_seqlen_kv == packed.max_seqlen_q
    assert result["full_loss_masks"].shape == result["tokens"].shape
    if allgather_cp:
        assert packed.cu_seqlens_q_cpu[:3] == (0, lengths[0], sum(lengths))
    else:
        multiple = 2 * cp_size if cp_size > 1 else 1
        padded_lengths = [(length + multiple - 1) // multiple * multiple for length in lengths]
        assert packed.cu_seqlens_q_cpu[:3] == (0, padded_lengths[0], sum(padded_lengths))


def _cp_wrapper_collective_worker(rank, world_size, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    module = load_module("slime_plugins.models.hf_attention")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        module.mpu.get_context_parallel_world_size = lambda: world_size
        module.mpu.get_context_parallel_rank = lambda: rank
        module.mpu.get_context_parallel_group = lambda: dist.group.WORLD

        class Attention(module.HuggingfaceAttention):
            def __init__(self):
                nn.Module.__init__(self)
                self.args = SimpleNamespace(sequence_parallel=False)
                self.weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

            def hf_forward(self, hidden_states, packed_seq_params):
                return torch.cat(
                    [(seq * self.weight).tanh().cumsum(1) for seq in hidden_states.split([8, 16], dim=1)], dim=1
                )

        indices = torch.cat(
            [
                torch.arange(start, end).reshape(2 * world_size, -1)[[rank, 2 * world_size - rank - 1]].flatten()
                for start, end in [(0, 8), (8, 24)]
            ]
        )
        full = torch.linspace(-1.0, 1.0, 24 * 3, dtype=torch.float64).reshape(24, 1, 3).requires_grad_()
        weight = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        expected = torch.cat([(seq * weight).tanh().cumsum(0) for seq in full.split([8, 16])])
        local = full.detach()[indices].clone().requires_grad_()
        attention = Attention()
        packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 8, 24]), cu_seqlens_q_cpu=(0, 8, 24))
        output, _ = attention(local, None, packed_seq_params=packed)
        torch.testing.assert_close(output, expected[indices])
        output.square().sum().backward()
        expected.square().sum().backward()
        dist.all_reduce(attention.weight.grad)
        torch.testing.assert_close(local.grad, full.grad[indices])
        torch.testing.assert_close(attention.weight.grad, weight.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_wrapper_gradients_with_real_collectives(tmp_path, cp_size):
    torch.multiprocessing.spawn(
        _cp_wrapper_collective_worker,
        args=(cp_size, (tmp_path / "rendezvous").as_uri()),
        nprocs=cp_size,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
