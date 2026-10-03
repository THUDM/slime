"""Shared fixtures for the LoRA test suite.

CPU tests use lightweight parallel-state substitutes, including in training images.
Run each CPU test file in its own process, as in CI, to isolate these substitutes.
The GPU worker in test_lora_parallel_gpu.py sets SLIME_LORA_MULTI_GPU_TEST=1
before importing this helper and must import real Megatron successfully. The
substitutes do not validate Transformer Engine, DDP, or rollout-engine integration.
"""

from __future__ import annotations

import os
import sys
import types

# --- Stub the megatron namespace (must run before any slime backend import) ---
_fake_mpu = types.ModuleType("megatron.core.mpu")
_fake_mpu.get_tensor_model_parallel_world_size = lambda: 1
_fake_mpu.get_tensor_model_parallel_rank = lambda: 0
_fake_mpu.get_pipeline_model_parallel_world_size = lambda: 1
_fake_mpu.get_pipeline_model_parallel_rank = lambda: 0
_fake_mpu.get_expert_model_parallel_world_size = lambda: 1
_fake_mpu.get_expert_model_parallel_rank = lambda: 0
_fake_mpu.get_context_parallel_world_size = lambda: 1
_fake_mpu.get_context_parallel_rank = lambda: 0
_fake_mpu.get_data_parallel_rank = lambda with_context_parallel=False: 0

_fake_core = types.ModuleType("megatron.core")
_fake_core.mpu = _fake_mpu
_fake_megatron = types.ModuleType("megatron")
_fake_megatron.core = _fake_core

if os.environ.get("SLIME_LORA_MULTI_GPU_TEST", "0") == "1":
    import megatron.core  # noqa: F401
else:
    sys.modules["megatron"] = _fake_megatron
    sys.modules["megatron.core"] = _fake_core
    sys.modules["megatron.core.mpu"] = _fake_mpu

    transformer = types.ModuleType("megatron.core.transformer")
    transformer_layer = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer.get_transformer_layer_offset = lambda config, vp_stage=None: getattr(config, "layer_offset", 0)
    sys.modules["megatron.core.transformer"] = transformer
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

class FakeColumnParallelLinear(torch.nn.Module):
    """Stands in for ``linear_qkv`` / ``linear_fc1``: weight is ``[out_local, in_full]``."""

    def __init__(self, in_features: int, out_features: int, *, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.parallel_mode = "column"
        self.gather_output = False
        self.sequence_parallel = sequence_parallel
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, input_: torch.Tensor):
        return F.linear(input_, self.weight), None


class FakeRowParallelLinear(torch.nn.Module):
    """Stands in for ``linear_proj`` / ``linear_fc2``: weight is ``[out_full, in_local]``."""

    def __init__(self, in_features: int, out_features: int, *, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.parallel_mode = "row"
        self.input_is_parallel = True
        self.sequence_parallel = sequence_parallel
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, input_: torch.Tensor):
        return F.linear(input_, self.weight), None


class FakeFusedLayerNormColumnLinear(FakeColumnParallelLinear):
    """Stands in for ``TELayerNormColumnParallelLinear``: the GEMM input is normalised."""

    def __init__(self, in_features: int, out_features: int, *, normalization: str = "RMSNorm") -> None:
        super().__init__(in_features, out_features)
        self.normalization = normalization
        self.eps = 1e-6
        self.zero_centered_gamma = False
        self.layer_norm_weight = torch.nn.Parameter(torch.ones(in_features))
        self.layer_norm_bias = None

    def _normalize(self, input_: torch.Tensor) -> torch.Tensor:
        variance = input_.float().pow(2).mean(dim=-1, keepdim=True)
        return (input_.float() * torch.rsqrt(variance + self.eps) * self.layer_norm_weight.float()).to(input_.dtype)

    def forward(self, input_: torch.Tensor):
        return F.linear(self._normalize(input_), self.weight), None


class FakeUnsupportedModule(torch.nn.Module):
    """A matched module that LoRA must refuse instead of silently skipping."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4))


class TEColumnParallelGroupedLinear(torch.nn.Module):
    """Stands in for TE's grouped ``mlp.experts.linear_fc1``.

    The class *name* matters: ``TEGroupedLinear`` sets ``parallel_mode=None`` on
    itself, so slime classifies routed experts by walking the MRO. Weights are
    packed as ``weight0 .. weight{num_experts-1}``, each ``[ffn_local, hidden]``.
    """

    def __init__(self, num_local_experts: int, hidden: int, ffn: int) -> None:
        super().__init__()
        self.num_gemms = num_local_experts
        # TEGroupedLinear disables TE's own TP, so parallel_mode really is None.
        self.parallel_mode = None
        for index in range(num_local_experts):
            self.register_parameter(
                f"weight{index}", torch.nn.Parameter(torch.randn(ffn, hidden) * 0.02)
            )

    def _weights(self):
        return [getattr(self, f"weight{index}") for index in range(self.num_gemms)]

    def forward(self, x: torch.Tensor, m_splits: list[int]):
        outputs = [F.linear(chunk, weight) for chunk, weight in zip(torch.split(x, m_splits), self._weights(), strict=True)]
        return torch.cat(outputs, dim=0), None


class TERowParallelGroupedLinear(TEColumnParallelGroupedLinear):
    """Stands in for TE's grouped ``mlp.experts.linear_fc2``."""


class FakeGroupedExperts(torch.nn.Module):
    """Mirrors ``mlp.experts`` of a Megatron MoE layer using grouped GEMM."""

    def __init__(self, num_local_experts: int, hidden: int, ffn: int) -> None:
        super().__init__()
        self.linear_fc1 = TEColumnParallelGroupedLinear(num_local_experts, hidden, ffn)
        self.linear_fc2 = TERowParallelGroupedLinear(num_local_experts, ffn, hidden)

    def forward(self, x: torch.Tensor, m_splits: list[int]):
        hidden, _ = self.linear_fc1(x, m_splits)
        out, _ = self.linear_fc2(torch.nn.functional.gelu(hidden), m_splits)
        return out


class FakeMoELayer(torch.nn.Module):
    """A MoE decoder layer with both routed and shared experts.

    The routed path deliberately keeps the toy "dispatch" trivial (every token
    goes to every local expert in order) because the adapter is shared across
    experts and therefore independent of the routing decision.
    """

    def __init__(self, hidden: int, ffn: int, num_local_experts: int = 2) -> None:
        super().__init__()
        self.num_local_experts = num_local_experts
        self.self_attention = FakeAttention(hidden)
        self.mlp = torch.nn.Module()
        self.mlp.experts = FakeGroupedExperts(num_local_experts, hidden, ffn)
        self.mlp.shared_experts = FakeMLP(hidden, ffn)
        self.mlp.router = torch.nn.Linear(hidden, num_local_experts, bias=False)
        self.add_module("mlp", self.mlp)

    def forward(self, x):
        x = x + self.self_attention(x)
        tokens = x.reshape(-1, x.shape[-1])
        # Split the permuted tokens evenly over the local experts.
        per_expert = tokens.shape[0] // self.num_local_experts
        m_splits = [per_expert] * self.num_local_experts
        m_splits[-1] += tokens.shape[0] - per_expert * self.num_local_experts
        routed = self.mlp.experts(tokens, m_splits).reshape(x.shape)
        return routed + self.mlp.shared_experts(x)


class FakeMoEModel(torch.nn.Module):
    """Minimal MoE stand-in whose module names match a real mcore MoE GPTModel."""

    def __init__(self, num_layers: int = 2, hidden: int = 16, ffn: int = 32, num_local_experts: int = 2) -> None:
        super().__init__()
        self.embedding = FakeLanguageModelEmbedding(32, hidden)
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList(
            [FakeMoELayer(hidden, ffn, num_local_experts) for _ in range(num_layers)]
        )
        self.add_module("decoder", self.decoder)
        self.output_layer = torch.nn.Linear(hidden, 32, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.decoder.layers:
            x = layer(x)
        return self.output_layer(x)


class FakeAttention(torch.nn.Module):
    def __init__(self, hidden: int, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.linear_qkv = FakeColumnParallelLinear(hidden, 3 * hidden, sequence_parallel=sequence_parallel)
        self.linear_proj = FakeRowParallelLinear(hidden, hidden, sequence_parallel=sequence_parallel)

    def forward(self, x):
        qkv, _ = self.linear_qkv(x)
        out, _ = self.linear_proj(qkv[..., : x.shape[-1]])
        return out


class FakeMLP(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.linear_fc1 = FakeColumnParallelLinear(hidden, ffn, sequence_parallel=sequence_parallel)
        self.linear_fc2 = FakeRowParallelLinear(ffn, hidden, sequence_parallel=sequence_parallel)

    def forward(self, x):
        h, _ = self.linear_fc1(x)
        out, _ = self.linear_fc2(torch.nn.functional.gelu(h))
        return out


class FakeLayer(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.self_attention = FakeAttention(hidden, sequence_parallel)
        self.mlp = FakeMLP(hidden, ffn, sequence_parallel)

    def forward(self, x):
        return self.mlp(x + self.self_attention(x))


class FakeDecoder(torch.nn.Module):
    def __init__(self, num_layers: int, hidden: int, ffn: int, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(hidden, ffn, sequence_parallel) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeVisionBlock(torch.nn.Module):
    """Plain replicated HF-style linears, mirroring Qwen3.5-VL's ``visual.`` tower."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(hidden, 3 * hidden)
        self.proj = torch.nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.proj(self.qkv(x)[..., : x.shape[-1]])


class FakeLanguageModelEmbedding(torch.nn.Module):
    """Mirrors Megatron's ``LanguageModelEmbedding``, which owns ``word_embeddings``."""

    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab, hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(tokens)


class FakeGPTModel(torch.nn.Module):
    """Minimal stand-in for the Megatron ``GPTModel`` used by Qwen3.5-VL.

    Module names deliberately match the real ones (``decoder.layers.N.*``,
    ``embedding``, ``output_layer``, ``visual.blocks.N.*``) so the LoRA target
    presets are exercised against realistic paths.
    """

    def __init__(self, num_layers: int = 2, hidden: int = 16, ffn: int = 32, *, with_vision: bool = True) -> None:
        super().__init__()
        self.embedding = FakeLanguageModelEmbedding(32, hidden)
        self.decoder = FakeDecoder(num_layers, hidden, ffn)
        self.output_layer = torch.nn.Linear(hidden, 32, bias=False)
        if with_vision:
            self.visual = torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([FakeVisionBlock(hidden)])})

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.decoder(self.embedding(tokens)))


def make_lora_args(**overrides):
    """Namespace with every LoRA argument at its CLI default, plus overrides."""
    import argparse

    from slime.utils.lora_config import add_lora_arguments

    parser = argparse.ArgumentParser()
    add_lora_arguments(parser)
    args = parser.parse_args([])
    args.use_lora = True
    args.train_backend = "megatron"
    args.lr = 1e-6
    args.weight_decay = 0.1
    args.hf_checkpoint = None
    args.num_experts = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_config(**overrides):
    from slime.utils.lora_config import LoRAConfig

    return LoRAConfig.from_args(make_lora_args(**overrides))


def patch_named_tensor_iterator(monkeypatch):
    """Substitute the parallel-state-dependent canonical iterator for CPU tests.

    The production implementation lives in
    ``slime.backends.megatron_utils.update_weight.common.named_params_and_buffers``,
    which needs a real Megatron parallel state. For the CPU tests the canonical
    name is simply ``module.module.<qualified name>``, matching the prefix that
    the production iterator produces for a DDP-wrapped chunk.
    """
    from slime.backends.megatron_utils.update_weight import common

    def fake_iter(args, model, convert_to_global_name=True):
        for chunk in model:
            for name, param in chunk.named_parameters():
                yield (f"module.module.{name}" if convert_to_global_name else name), param

    monkeypatch.setattr(common, "named_params_and_buffers", fake_iter)
    return fake_iter
