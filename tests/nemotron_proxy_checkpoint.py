"""A small random-weight Nemotron 3.5 Super VL checkpoint for single-node tests.

The real checkpoint (88 layers, 512 experts, 124B parameters) needs 4 nodes and a
235 GB conversion, so the end-to-end test runs a proxy that keeps the structure and
shrinks the depth and the expert count:

* the layer pattern is a prefix of the real one, so it still has Mamba2, MoE and
  attention layers (``MEMEMEM*`` by default);
* ``n_routed_experts`` and ``num_experts_per_tok`` are reduced, so the experts divide
  across the 8 GPUs of one node;
* everything else is the real model: hidden size, head and Mamba dimensions, the
  RADIO tower (SGLang takes its depth from the model name, so it cannot be cut), the
  projector, the MTP weights and the tokenizer / processor files.

Tensor names and shapes are exactly the real checkpoint's, so slime's loaders, its
weight-sync converters and SGLang read the proxy as they read the real model. The
weights are random (seeded): the proxy checks that training and rollout agree, not
that the model is accurate.

``tensor_specs`` derives every name and shape from ``config.json`` alone.
``tests/test_nemotron35_super_vl_proxy_e2e.py`` checks it against the real
checkpoint's totals, and against its safetensors headers when they are available.
"""

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

BLOCK_TYPES = {"M": "mamba", "E": "moe", "*": "attention"}


class TensorSpec(NamedTuple):
    shape: tuple
    dtype: str  # "BF16", "F32" or "I64", as in the safetensors header


@dataclass(frozen=True)
class ProxyShape:
    """What the proxy shrinks. Everything not listed here keeps the real size."""

    pattern: str = "MEMEMEM*"  # prefix of the real pattern: Mamba2, MoE and attention
    num_experts: int = 32  # routed experts; must divide by the EP size (8 on one node)
    topk: int = 4  # experts per token; at most num_experts

    def layer_types(self) -> list[str]:
        return [BLOCK_TYPES[symbol] for symbol in self.pattern]


# Frozen, so sharing one instance as a default argument is safe (ruff B008).
DEFAULT_SHAPE = ProxyShape()


def proxy_config(config: dict, shape: ProxyShape) -> dict:
    """The real ``config.json`` with the language model cut down to ``shape``."""
    config = json.loads(json.dumps(config))
    llm = config["llm_config"]
    if shape.topk > shape.num_experts:
        raise ValueError(f"topk {shape.topk} exceeds the {shape.num_experts} experts")
    llm["layers_block_type"] = shape.layer_types()
    llm["n_routed_experts"] = shape.num_experts
    llm["num_experts_per_tok"] = shape.topk
    return config


def _attention(prefix: str, llm: dict, specs: dict) -> None:
    hidden, heads, kv, head_dim = (
        llm[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )
    specs[f"{prefix}.mixer.q_proj.weight"] = TensorSpec((heads * head_dim, hidden), "BF16")
    specs[f"{prefix}.mixer.k_proj.weight"] = TensorSpec((kv * head_dim, hidden), "BF16")
    specs[f"{prefix}.mixer.v_proj.weight"] = TensorSpec((kv * head_dim, hidden), "BF16")
    specs[f"{prefix}.mixer.o_proj.weight"] = TensorSpec((hidden, heads * head_dim), "BF16")
    specs[f"{prefix}.norm.weight"] = TensorSpec((hidden,), "BF16")


def _mamba(prefix: str, llm: dict, specs: dict) -> None:
    hidden = llm["hidden_size"]
    heads, head_dim, groups, state = (
        llm[k] for k in ("mamba_num_heads", "mamba_head_dim", "n_groups", "ssm_state_size")
    )
    d_inner = heads * head_dim
    conv_dim = d_inner + 2 * groups * state
    specs[f"{prefix}.mixer.in_proj.weight"] = TensorSpec((2 * d_inner + 2 * groups * state + heads, hidden), "BF16")
    specs[f"{prefix}.mixer.conv1d.weight"] = TensorSpec((conv_dim, 1, llm["conv_kernel"]), "BF16")
    specs[f"{prefix}.mixer.conv1d.bias"] = TensorSpec((conv_dim,), "BF16")
    specs[f"{prefix}.mixer.A_log"] = TensorSpec((heads,), "BF16")
    specs[f"{prefix}.mixer.D"] = TensorSpec((heads,), "BF16")
    specs[f"{prefix}.mixer.dt_bias"] = TensorSpec((heads,), "BF16")
    specs[f"{prefix}.mixer.norm.weight"] = TensorSpec((d_inner,), "BF16")
    specs[f"{prefix}.mixer.out_proj.weight"] = TensorSpec((hidden, d_inner), "BF16")
    specs[f"{prefix}.norm.weight"] = TensorSpec((hidden,), "BF16")


def _moe(prefix: str, llm: dict, specs: dict) -> None:
    hidden, latent, ffn = llm["hidden_size"], llm["moe_latent_size"], llm["moe_intermediate_size"]
    shared, experts = llm["moe_shared_expert_intermediate_size"], llm["n_routed_experts"]
    for expert in range(experts):
        specs[f"{prefix}.mixer.experts.{expert}.up_proj.weight"] = TensorSpec((ffn, latent), "BF16")
        specs[f"{prefix}.mixer.experts.{expert}.down_proj.weight"] = TensorSpec((latent, ffn), "BF16")
    specs[f"{prefix}.mixer.fc1_latent_proj.weight"] = TensorSpec((latent, hidden), "BF16")
    specs[f"{prefix}.mixer.fc2_latent_proj.weight"] = TensorSpec((hidden, latent), "BF16")
    specs[f"{prefix}.mixer.gate.weight"] = TensorSpec((experts, hidden), "BF16")
    specs[f"{prefix}.mixer.gate.e_score_correction_bias"] = TensorSpec((experts,), "F32")
    specs[f"{prefix}.mixer.shared_experts.up_proj.weight"] = TensorSpec((shared, hidden), "BF16")
    specs[f"{prefix}.mixer.shared_experts.down_proj.weight"] = TensorSpec((hidden, shared), "BF16")
    specs[f"{prefix}.norm.weight"] = TensorSpec((hidden,), "BF16")


_LAYERS = {"mamba": _mamba, "attention": _attention, "moe": _moe}


def tensor_specs(config: dict) -> dict[str, TensorSpec]:
    """Every tensor of the checkpoint, by name, derived from ``config.json``."""
    llm, vision = config["llm_config"], config["vision_config"]
    hidden, vocab = llm["hidden_size"], llm["vocab_size"]
    specs: dict[str, TensorSpec] = {}

    specs["language_model.backbone.embeddings.weight"] = TensorSpec((vocab, hidden), "BF16")
    for index, block in enumerate(llm["layers_block_type"]):
        _LAYERS[block](f"language_model.backbone.layers.{index}", llm, specs)
    specs["language_model.backbone.norm_f.weight"] = TensorSpec((hidden,), "BF16")
    specs["language_model.lm_head.weight"] = TensorSpec((vocab, hidden), "BF16")

    # Multi-token prediction: one head with the sub-layers of mtp_layers_block_type.
    # The first carries the embedding / hidden-state merge, the last the final norm.
    if llm["num_nextn_predict_layers"] != 1:
        raise ValueError("only one MTP head is modelled")
    mtp_blocks = llm["mtp_layers_block_type"]
    for index, block in enumerate(mtp_blocks):
        prefix = f"language_model.mtp.layers.{index}"
        _LAYERS[block](prefix, llm, specs)
        if index == 0:
            specs[f"{prefix}.eh_proj.weight"] = TensorSpec((hidden, 2 * hidden), "BF16")
            specs[f"{prefix}.enorm.weight"] = TensorSpec((hidden,), "BF16")
            specs[f"{prefix}.hnorm.weight"] = TensorSpec((hidden,), "BF16")
        if index == len(mtp_blocks) - 1:
            specs[f"{prefix}.final_layernorm.weight"] = TensorSpec((hidden,), "BF16")

    # RADIO tower.
    width, patch = vision["hidden_size"], vision["patch_size"]
    mlp = int(width * vision["mlp_ratio"])
    tower = "vision_model.radio_model.model"
    specs["vision_model.radio_model.input_conditioner.norm_mean"] = TensorSpec((3, 1, 1), "F32")
    specs["vision_model.radio_model.input_conditioner.norm_std"] = TensorSpec((3, 1, 1), "F32")
    specs["vision_model.summary_idxs"] = TensorSpec((len(vision["summary_idxs"]),), "I64")
    positions = (vision["max_img_size"] // patch) ** 2
    specs[f"{tower}.patch_generator.embedder.weight"] = TensorSpec((width, vision["num_channels"] * patch**2), "BF16")
    specs[f"{tower}.patch_generator.video_embedder.weight"] = TensorSpec(
        (width, vision["video_temporal_patch_size"] * vision["num_channels"] * patch**2), "BF16"
    )
    specs[f"{tower}.patch_generator.pos_embed"] = TensorSpec((1, positions, width), "BF16")
    specs[f"{tower}.patch_generator.cls_token.token"] = TensorSpec(
        (vision["num_cls_tokens"] + vision["num_registers"], width), "BF16"
    )
    for block in range(vision["num_hidden_layers"]):
        prefix = f"{tower}.blocks.{block}"
        specs[f"{prefix}.attn.qkv.weight"] = TensorSpec((3 * width, width), "BF16")
        specs[f"{prefix}.attn.qkv.bias"] = TensorSpec((3 * width,), "BF16")
        specs[f"{prefix}.attn.proj.weight"] = TensorSpec((width, width), "BF16")
        specs[f"{prefix}.attn.proj.bias"] = TensorSpec((width,), "BF16")
        specs[f"{prefix}.mlp.fc1.weight"] = TensorSpec((mlp, width), "BF16")
        specs[f"{prefix}.mlp.fc1.bias"] = TensorSpec((mlp,), "BF16")
        specs[f"{prefix}.mlp.fc2.weight"] = TensorSpec((width, mlp), "BF16")
        specs[f"{prefix}.mlp.fc2.bias"] = TensorSpec((width,), "BF16")
        for norm in ("norm1", "norm2"):
            specs[f"{prefix}.{norm}.weight"] = TensorSpec((width,), "BF16")
            specs[f"{prefix}.{norm}.bias"] = TensorSpec((width,), "BF16")

    # Projector. The final LayerNorm exists only when the language model has MTP.
    if llm["num_nextn_predict_layers"] > 0:
        specs["vision_projector.vision_final_layernorm.weight"] = TensorSpec((config["vit_hidden_size"],), "BF16")
        specs["vision_projector.vision_final_layernorm.bias"] = TensorSpec((config["vit_hidden_size"],), "BF16")
    shuffled = config["vit_hidden_size"] * int(1 / config["downsample_ratio"]) ** 2
    specs["mlp1.0.weight"] = TensorSpec((shuffled,), "BF16")
    specs["mlp1.1.weight"] = TensorSpec((config["projector_hidden_size"], shuffled), "BF16")
    specs["mlp1.3.weight"] = TensorSpec((hidden, config["projector_hidden_size"]), "BF16")
    return specs


def numel(spec: TensorSpec) -> int:
    return math.prod(spec.shape)


_BYTES = {"BF16": 2, "F32": 4, "I64": 8}


def _init(name: str, spec: TensorSpec, config: dict, generator):
    """Random weights with sane statistics: norms at 1, Mamba parameters as in the HF init."""
    import torch

    dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}[spec.dtype]
    if name.endswith("summary_idxs"):
        return torch.tensor(config["vision_config"]["summary_idxs"], dtype=dtype)
    if name.endswith("input_conditioner.norm_mean"):
        return torch.tensor(config["norm_mean"], dtype=dtype).view(3, 1, 1)
    if name.endswith("input_conditioner.norm_std"):
        return torch.tensor(config["norm_std"], dtype=dtype).view(3, 1, 1)
    if name.endswith("e_score_correction_bias") or name.endswith(".bias"):
        return torch.zeros(spec.shape, dtype=dtype)
    # RMSNorm / LayerNorm scales: the decoder norms, the vision tower's norm1 / norm2 and the
    # projector's mlp1.0 (the norm in front of its two linears).
    if (
        name.endswith(("norm.weight", "norm_f.weight", "layernorm.weight", "norm1.weight", "norm2.weight"))
        or name == "mlp1.0.weight"
    ):
        return torch.ones(spec.shape, dtype=dtype)
    if name.endswith("mixer.D"):
        return torch.ones(spec.shape, dtype=dtype)
    if name.endswith("mixer.A_log"):
        return torch.log(torch.empty(spec.shape).uniform_(1, 16, generator=generator)).to(dtype)
    if name.endswith("mixer.dt_bias"):
        dt = torch.exp(torch.empty(spec.shape).uniform_(math.log(1e-3), math.log(1e-1), generator=generator))
        return (dt + torch.log(-torch.expm1(-dt))).to(dtype)
    return (torch.randn(spec.shape, generator=generator) * 0.02).to(dtype)


def _stamp(config: dict, shape: ProxyShape, seed: int) -> str:
    payload = json.dumps({"shape": asdict(shape), "seed": seed, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_proxy_checkpoint(
    files_dir: str | Path,
    out_dir: str | Path,
    shape: ProxyShape = DEFAULT_SHAPE,
    seed: int = 0,
    shard_bytes: int = 2 * 1024**3,
) -> Path:
    """Write the proxy checkpoint to ``out_dir`` and return it.

    ``files_dir`` holds the real checkpoint's non-weight files (``config.json``, the
    remote-code ``*.py``, tokenizer and processor files). A previous run with the same
    shape and seed is reused.
    """
    import torch
    from safetensors.torch import save_file

    files_dir, out_dir = Path(files_dir), Path(out_dir)
    config = proxy_config(json.loads((files_dir / "config.json").read_text()), shape)
    stamp = _stamp(config, shape, seed)
    stamp_file = out_dir / "proxy_stamp.txt"
    if stamp_file.exists() and stamp_file.read_text() == stamp:
        return out_dir

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for path in files_dir.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(path, out_dir / path.name)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    generator = torch.Generator().manual_seed(seed)
    specs = tensor_specs(config)
    weight_map, shard, shard_size, shard_index, total = {}, {}, 0, 0, 0

    def flush():
        nonlocal shard, shard_size, shard_index
        if not shard:
            return
        name = f"model-{shard_index:05d}.safetensors"
        save_file(shard, out_dir / name, metadata={"format": "pt"})
        weight_map.update({tensor: name for tensor in shard})
        shard, shard_size, shard_index = {}, 0, shard_index + 1

    for name, spec in specs.items():
        size = numel(spec) * _BYTES[spec.dtype]
        if shard and shard_size + size > shard_bytes:
            flush()
        shard[name] = _init(name, spec, config, generator)
        shard_size += size
        total += size
    flush()
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    stamp_file.write_text(stamp)
    return out_dir
