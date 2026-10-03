"""Configuration, presets and validation for slime's Megatron LoRA support.

This module intentionally avoids importing torch, Megatron or Transformer Engine so
that ``slime/utils/arguments.py`` can build and validate the CLI before any backend
is loaded, and so that the configuration unit tests run outside the training image.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

logger = logging.getLogger(__name__)

LORA_A_NAME: Final[str] = "lora_A"
LORA_B_NAME: Final[str] = "lora_B"

ADAPTER_CONFIG_FILE: Final[str] = "adapter_config.json"
ADAPTER_INDEX_FILE: Final[str] = "adapter_model.safetensors.index.json"
ADAPTER_TRAINING_STATE_FILE: Final[str] = "training_state.json"
ADAPTER_FORMAT_VERSION: Final[int] = 1

LoRABias = Literal["none"]
LoRARolloutSyncMode = Literal["merged", "native_adapter"]

SUPPORTED_LORA_BIAS: Final[tuple[str, ...]] = ("none",)
SUPPORTED_ROLLOUT_SYNC_MODES: Final[tuple[str, ...]] = ("merged", "native_adapter")

DEFAULT_LORA_TARGET_PRESET: Final[str] = "dense_language"

# Standard Megatron attention/MLP projections for dense decoder layers. MoE and
# hybrid attention have separate presets below; MLA is not supported yet.
_DENSE_ATTENTION_TARGETS: Final[tuple[str, ...]] = (
    r"(?:^|\.)decoder\.layers\.\d+\.self_attention\.linear_qkv$",
    r"(?:^|\.)decoder\.layers\.\d+\.self_attention\.linear_proj$",
)

_DENSE_MLP_TARGETS: Final[tuple[str, ...]] = (
    r"(?:^|\.)decoder\.layers\.\d+\.mlp\.linear_fc1$",
    r"(?:^|\.)decoder\.layers\.\d+\.mlp\.linear_fc2$",
)

_HYBRID_ATTENTION_TARGETS: Final[tuple[str, ...]] = _DENSE_ATTENTION_TARGETS + (
    # Hybrid attention with linear-attention / gated-delta-net projections.
    r"(?:^|\.)decoder\.layers\.\d+\.self_attention\.linear_attn\."
    r"(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
)

# MoE shared experts are ordinary dense TP linears and behave exactly like a dense MLP.
_MOE_SHARED_MLP_TARGETS: Final[tuple[str, ...]] = (
    r"(?:^|\.)decoder\.layers\.\d+\.mlp\.shared_experts\.linear_fc1$",
    r"(?:^|\.)decoder\.layers\.\d+\.mlp\.shared_experts\.linear_fc2$",
)

# Routed experts (grouped GEMM). One adapter is shared by every expert of a layer,
# sharing both factors A and B. Per-expert adapters are not implemented: they
# multiply the trainable parameters by ``num_experts`` and make the merge depend on
# the expert-parallel placement.
_MOE_ROUTED_EXPERT_TARGETS: Final[tuple[str, ...]] = (
    r"(?:^|\.)decoder\.layers\.\d+\.mlp\.experts\.linear_fc1$",
    r"(?:^|\.)decoder\.layers\.\d+\.mlp\.experts\.linear_fc2$",
)

LORA_TARGET_PRESETS: Final[dict[str, tuple[str, ...]]] = {
    "dense_attention": _DENSE_ATTENTION_TARGETS,
    "dense_mlp": _DENSE_MLP_TARGETS,
    "dense_language": _DENSE_ATTENTION_TARGETS + _DENSE_MLP_TARGETS,
    "hybrid_attention": _HYBRID_ATTENTION_TARGETS,
    "hybrid_language": _HYBRID_ATTENTION_TARGETS + _DENSE_MLP_TARGETS,
    "moe_attention": _DENSE_ATTENTION_TARGETS,
    "moe_shared_mlp": _MOE_SHARED_MLP_TARGETS,
    "moe_routed_experts": _MOE_ROUTED_EXPERT_TARGETS,
    "moe_mlp": _MOE_SHARED_MLP_TARGETS + _MOE_ROUTED_EXPERT_TARGETS,
    "moe_language": _DENSE_ATTENTION_TARGETS + _MOE_SHARED_MLP_TARGETS,
    "moe_language_all": _DENSE_ATTENTION_TARGETS + _MOE_SHARED_MLP_TARGETS + _MOE_ROUTED_EXPERT_TARGETS,
}

# Always excluded unless the user overrides ``--lora-exclude-modules`` entirely.
DEFAULT_LORA_EXCLUDE_MODULES: Final[tuple[str, ...]] = (
    r"(?:^|\.)embedding(?:\.|$)",
    r"(?:^|\.)output_layer(?:\.|$)",
    r"(?:^|\.)visual(?:\.|$)",
)


class LoRAConfigError(ValueError):
    """Raised for invalid or unsupported LoRA configuration."""


@dataclass(frozen=True)
class LoRAConfig:
    """Fully resolved, immutable LoRA configuration."""

    rank: int
    alpha: float
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    exclude_modules: tuple[str, ...]
    preset: str | None
    learning_rate: float | None
    weight_decay: float
    load_path: str | None
    save_path: str | None
    rollout_sync_mode: str
    save_merged_hf: str | None
    base_model_path: str | None
    base_model_config_hash: str | None
    model_type: str | None
    allow_replicated_modules: tuple[str, ...]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_args(cls, args: Namespace) -> LoRAConfig:
        if not getattr(args, "use_lora", False):
            raise LoRAConfigError("LoRAConfig.from_args() called while --use-lora is disabled")

        preset, targets = _resolve_targets(args)
        excludes = getattr(args, "lora_exclude_modules", None) or DEFAULT_LORA_EXCLUDE_MODULES
        hf_checkpoint = getattr(args, "hf_checkpoint", None)
        allow_replicated = getattr(args, "lora_allow_replicated_modules", None)

        return cls(
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
            bias=str(args.lora_bias),
            target_modules=tuple(targets),
            exclude_modules=tuple(excludes),
            preset=preset,
            learning_rate=(
                float(args.lora_learning_rate) if getattr(args, "lora_learning_rate", None) is not None else None
            ),
            weight_decay=float(getattr(args, "lora_weight_decay", 0.0) or 0.0),
            load_path=getattr(args, "lora_load", None),
            save_path=getattr(args, "save_lora", None),
            rollout_sync_mode=str(getattr(args, "lora_rollout_sync_mode", "merged")),
            save_merged_hf=getattr(args, "lora_save_merged_hf", None),
            base_model_path=hf_checkpoint,
            base_model_config_hash=compute_base_model_config_hash(hf_checkpoint),
            model_type=read_hf_model_type(hf_checkpoint),
            allow_replicated_modules=tuple(allow_replicated or ()),
        )

    def compile_targets(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern) for pattern in self.target_modules]

    def compile_excludes(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern) for pattern in self.exclude_modules]

    def compile_allow_replicated(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern) for pattern in self.allow_replicated_modules]

    def matches(self, module_name: str) -> bool:
        if any(pattern.search(module_name) for pattern in self.compile_excludes()):
            return False
        return any(pattern.search(module_name) for pattern in self.compile_targets())

    def to_metadata(self) -> dict[str, Any]:
        """Adapter-config payload; parallel sizes and provenance are filled by the caller."""
        return {
            "format_version": ADAPTER_FORMAT_VERSION,
            "base_model_path_or_id": self.base_model_path,
            "base_model_config_hash": self.base_model_config_hash,
            "model_type": self.model_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_modules": list(self.target_modules),
            "excluded_modules": list(self.exclude_modules),
        }


def _resolve_targets(args: Namespace) -> tuple[str | None, tuple[str, ...]]:
    explicit = getattr(args, "lora_target_modules", None)
    preset = getattr(args, "lora_target_preset", None)

    if explicit and preset:
        raise LoRAConfigError(
            "Specify only ONE of --lora-target-modules or --lora-target-preset "
            f"(got targets={list(explicit)!r} and preset={preset!r})"
        )

    if explicit:
        return None, tuple(explicit)

    if preset is None:
        preset = DEFAULT_LORA_TARGET_PRESET
        logger.info("Neither --lora-target-modules nor --lora-target-preset given; using preset %r", preset)

    if preset not in LORA_TARGET_PRESETS:
        raise LoRAConfigError(
            f"Unknown --lora-target-preset {preset!r}. Available presets: {sorted(LORA_TARGET_PRESETS)}"
        )
    return preset, LORA_TARGET_PRESETS[preset]


def read_hf_model_type(hf_checkpoint: str | Path | None) -> str | None:
    config = _read_hf_config(hf_checkpoint)
    if config is None:
        return None
    model_type = config.get("model_type")
    return str(model_type) if model_type is not None else None


def compute_base_model_config_hash(hf_checkpoint: str | Path | None) -> str | None:
    """Stable hash of the base model's HF config, used to reject mismatched adapters."""
    config = _read_hf_config(hf_checkpoint)
    if config is None:
        return None
    # Drop keys that legitimately differ between otherwise-identical base checkpoints.
    volatile = {"_name_or_path", "transformers_version", "torch_dtype", "quantization_config"}
    canonical = {key: value for key, value in sorted(config.items()) if key not in volatile}
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_hf_config(hf_checkpoint: str | Path | None) -> dict[str, Any] | None:
    if not hf_checkpoint:
        return None
    path = Path(hf_checkpoint) / "config.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return config if isinstance(config, dict) else None


def is_lora_param_name(name: str) -> bool:
    """True for parameter names created by slime's LoRA injection."""
    return name.endswith(f".{LORA_A_NAME}") or name.endswith(f".{LORA_B_NAME}")


def add_lora_arguments(parser) -> None:
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=False,
        help="Enable LoRA (PEFT) fine-tuning: freeze the base model and only train low-rank adapters. "
        "Unrelated to architectural low-rank parameters such as --q-lora-rank.",
    )
    parser.add_argument(
        "--lora-debug",
        action="store_true",
        help="Compute per-step LoRA gradient diagnostics (adds tensor scans and device synchronization).",
    )
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank r (> 0).")
    parser.add_argument("--lora-alpha", type=float, default=128.0, help="LoRA alpha; effective scale is alpha / rank.")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="Dropout applied to the LoRA input. Must stay 0.0 for on-policy RL, otherwise the training "
        "forward pass no longer matches the rollout forward pass.",
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default="none",
        choices=list(SUPPORTED_LORA_BIAS),
        help="LoRA bias handling. Only 'none' is supported.",
    )
    parser.add_argument(
        "--lora-target-preset",
        type=str,
        default=None,
        help=f"Named set of target-module regexes. Available: {sorted(LORA_TARGET_PRESETS)}. "
        f"Defaults to {DEFAULT_LORA_TARGET_PRESET!r}, which matches the attention/MLP projections present in "
        "standard dense mcore GPT decoder layers (not MLA or MoE experts). "
        "Mutually exclusive with --lora-target-modules.",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="*",
        default=None,
        help="Explicit regexes matched with re.search against module names of the unwrapped Megatron model. "
        "Mutually exclusive with --lora-target-preset.",
    )
    parser.add_argument(
        "--lora-exclude-modules",
        type=str,
        nargs="*",
        default=None,
        help=f"Regexes removed from the matched set. Defaults to {list(DEFAULT_LORA_EXCLUDE_MODULES)}.",
    )
    parser.add_argument(
        "--lora-allow-replicated-modules",
        type=str,
        nargs="*",
        default=None,
        help="Regexes (matched against module names) of TP-replicated linear layers that are safe to adapt "
        "despite not being TP-sharded, because the model guarantees every TP replica sees identical inputs "
        "and gradients (e.g. modules that internally gather the full sequence before their GEMM). LoRA "
        "otherwise refuses to inject into replicated modules under tensor parallelism, since an unaccounted "
        "cross-TP gradient sync would silently desynchronise the replicas. Defaults to no exceptions; "
        "applies to both presets and explicit targets.",
    )
    parser.add_argument(
        "--lora-learning-rate",
        type=float,
        default=None,
        help="Learning rate for LoRA parameters. Falls back to --lr when unset.",
    )
    parser.add_argument(
        "--lora-weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for LoRA parameters. Unlike --lora-learning-rate this always overrides "
        "--weight-decay, so set this one rather than --weight-decay when --use-lora is on.",
    )
    parser.add_argument(
        "--lora-load",
        type=str,
        default=None,
        help="Adapter-only checkpoint for a new run (--finetune); ignored during full checkpoint resume.",
    )
    parser.add_argument(
        "--save-lora",
        type=str,
        default=None,
        help="Directory template for adapter-only checkpoints; supports {rollout_id}. "
        "Requires --save and --save-interval (or --release-train); the full checkpoint is also saved.",
    )
    parser.add_argument(
        "--lora-rollout-sync-mode",
        type=str,
        default="merged",
        choices=list(SUPPORTED_ROLLOUT_SYNC_MODES),
        help="How the rollout engines receive the LoRA policy. 'merged' materialises "
        "W_base + alpha/r * B @ A per tensor at the sync boundary and reuses the standard full-weight path.",
    )
    parser.add_argument(
        "--lora-save-merged-hf",
        type=str,
        default=None,
        help="Optional directory template for a merged (LoRA-free) HF checkpoint; supports {rollout_id}.",
    )


def validate_lora_args(args: Namespace) -> None:
    """Fail fast on invalid or unsupported LoRA configuration.

    Called from ``slime_validate_args``. It is a no-op when LoRA is disabled so that
    full fine-tuning behaviour is unchanged.
    """
    if not getattr(args, "use_lora", False):
        return

    if getattr(args, "train_backend", "megatron") != "megatron":
        raise LoRAConfigError(
            f"--use-lora is only implemented for --train-backend megatron, got {args.train_backend!r}"
        )

    if getattr(args, "multi_latent_attention", False):
        raise LoRAConfigError("--use-lora with MLA models is not supported yet")

    if args.lora_rank <= 0:
        raise LoRAConfigError(f"--lora-rank must be > 0, got {args.lora_rank}")
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        raise LoRAConfigError(f"--lora-alpha must be > 0 and finite, got {args.lora_alpha}")
    lora_lr = getattr(args, "lora_learning_rate", None)
    if lora_lr is not None and (not math.isfinite(lora_lr) or lora_lr <= 0):
        raise LoRAConfigError(f"--lora-learning-rate must be > 0 and finite, got {lora_lr}")
    lora_wd = args.lora_weight_decay
    if not math.isfinite(lora_wd) or lora_wd < 0:
        raise LoRAConfigError(f"--lora-weight-decay must be >= 0 and finite, got {lora_wd}")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise LoRAConfigError(f"--lora-dropout must be in [0.0, 1.0), got {args.lora_dropout}")
    if args.lora_dropout != 0.0:
        raise LoRAConfigError(
            f"--lora-dropout={args.lora_dropout} is rejected: slime trains on-policy, so a stochastic LoRA "
            "forward pass would make the training log-probs disagree with the rollout policy that produced "
            "them. Use --lora-dropout 0.0."
        )
    if args.lora_bias not in SUPPORTED_LORA_BIAS:
        raise LoRAConfigError(f"--lora-bias={args.lora_bias!r} is not supported; only {SUPPORTED_LORA_BIAS}")
    if args.lora_rollout_sync_mode not in SUPPORTED_ROLLOUT_SYNC_MODES:
        raise LoRAConfigError(f"--lora-rollout-sync-mode={args.lora_rollout_sync_mode!r} is unknown")
    if args.lora_rollout_sync_mode == "native_adapter":
        raise LoRAConfigError(
            "--lora-rollout-sync-mode=native_adapter is not implemented: slime's SGLang integration exposes "
            "no verified adapter load/update/unload API with atomic multi-rank version switching. "
            "Use --lora-rollout-sync-mode=merged."
        )

    if getattr(args, "only_train_params_name_list", None) or getattr(args, "freeze_params_name_list", None):
        raise LoRAConfigError(
            "--use-lora cannot be combined with --only-train-params-name-list / --freeze-params-name-list: "
            "LoRA already defines exactly which parameters are trainable. Drop the freeze lists."
        )
    if getattr(args, "use_critic", False) or getattr(args, "advantage_estimator", None) == "ppo":
        raise LoRAConfigError("--use-lora with a critic is not supported yet (the critic head is not adapter-based)")
    if getattr(args, "use_opd", False) or getattr(args, "opd_teacher_load", None):
        raise LoRAConfigError(
            "--use-lora with on-policy distillation is not supported yet: the teacher snapshot/load "
            "path does not have independent adapter bookkeeping. Disable --use-opd and --opd-teacher-load."
        )
    if getattr(args, "keep_old_actor", False):
        raise LoRAConfigError(
            "--use-lora with --keep-old-actor is not supported yet: the old-actor queue would need per-tag "
            "adapter bookkeeping that slime does not implement."
        )
    if getattr(args, "ref_update_interval", None) is not None:
        raise LoRAConfigError(
            "--use-lora with --ref-update-interval is not supported yet: in LoRA mode the reference policy is "
            "the frozen base model (a zero adapter) and is never refreshed."
        )
    if getattr(args, "ref_load", None):
        raise LoRAConfigError(
            "--use-lora ignores --ref-load: the reference policy is the frozen base model. Remove --ref-load."
        )

    # Resolve targets eagerly so preset/regex errors surface during argument validation.
    config = LoRAConfig.from_args(args)

    compiled_targets: list[re.Pattern[str]] = []
    for pattern in config.target_modules:
        try:
            compiled_targets.append(re.compile(pattern))
        except re.error as exc:
            raise LoRAConfigError(f"Invalid LoRA module regex {pattern!r}: {exc}") from exc
    for pattern in config.exclude_modules:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise LoRAConfigError(f"Invalid LoRA module regex {pattern!r}: {exc}") from exc

    # Apply user regexes to representative real module names. Searching the
    # regex *source* for another regex is incorrect for escaped patterns such as
    # ``model\.visual\..*`` and previously let explicit vision targets through.
    vision_module_names = (
        "visual",
        "visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.qkv",
        "module.module.visual.blocks.0.attn.qkv",
    )
    for source, pattern in zip(config.target_modules, compiled_targets, strict=True):
        if any(pattern.search(module_name) for module_name in vision_module_names):
            raise LoRAConfigError(
                f"LoRA target {source!r} points at the vision tower. Vision LoRA is not supported; "
                "the vision encoder stays frozen."
            )

    if config.learning_rate is None:
        logger.info("--lora-learning-rate not set; LoRA parameters use the global --lr=%s", getattr(args, "lr", None))


def validate_lora_checkpoint_args(args: Namespace) -> None:
    """Validate checkpoint options after loading and saving modes have been resolved."""
    if not getattr(args, "use_lora", False):
        return

    load_path = getattr(args, "lora_load", None)
    if load_path is not None and args.finetune:
        path = Path(load_path)
        if not path.is_dir():
            raise LoRAConfigError(f"--lora-load={load_path!r} is not an existing directory")
        if not (path / ADAPTER_CONFIG_FILE).is_file():
            raise LoRAConfigError(f"--lora-load={load_path!r} does not contain {ADAPTER_CONFIG_FILE}")

    if getattr(args, "save_lora", None) is not None or getattr(args, "lora_save_merged_hf", None) is not None:
        if getattr(args, "save", None) is None:
            raise LoRAConfigError("LoRA export requires --save; it runs alongside full checkpoint saving")
        interval = getattr(args, "save_interval", None)
        if not getattr(args, "release_train", False) and (interval is None or interval <= 0):
            raise LoRAConfigError("LoRA export requires a positive --save-interval or --release-train")
