"""LoRA (PEFT) support for slime's Megatron training backend.

Disabled by default: without ``--use-lora`` nothing in this package is imported by
the training path, and model structure, parameter names, optimizer behaviour,
checkpoints, HF export and weight synchronisation are unchanged.
"""

from slime.utils.lora_config import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_TRAINING_STATE_FILE,
    LORA_TARGET_PRESETS,
    LoRAConfig,
    LoRAConfigError,
    add_lora_arguments,
    is_lora_param_name,
)

from .inject import (
    assert_lora_backward,
    assert_lora_gradients,
    assert_optimizer_holds_only_lora,
    inject_lora,
    iter_lora_modules,
    log_injection_report,
    match_target_modules,
    prepare_lora_backward,
)
from .layers import (
    LoRAInjectionError,
    LoRAUnsupportedModuleError,
    describe_lora_target,
    is_lora_module,
    lora_forward_diagnostics,
    lora_local_delta,
    reset_lora_forward_counters,
)
from .merge import EffectiveWeightMapping, build_merge_plan, merge_into
from .state import (
    LoRACheckpointError,
    load_lora_adapter,
    lora_state_dict,
    read_adapter_metadata,
    save_lora_adapter,
    validate_adapter_metadata,
)

__all__ = [
    "ADAPTER_CONFIG_FILE",
    "ADAPTER_TRAINING_STATE_FILE",
    "LORA_TARGET_PRESETS",
    "EffectiveWeightMapping",
    "LoRACheckpointError",
    "LoRAConfig",
    "LoRAConfigError",
    "LoRAInjectionError",
    "LoRAUnsupportedModuleError",
    "add_lora_arguments",
    "assert_lora_backward",
    "assert_lora_gradients",
    "assert_optimizer_holds_only_lora",
    "build_merge_plan",
    "describe_lora_target",
    "inject_lora",
    "is_lora_module",
    "is_lora_param_name",
    "iter_lora_modules",
    "load_lora_adapter",
    "log_injection_report",
    "lora_forward_diagnostics",
    "lora_local_delta",
    "lora_state_dict",
    "match_target_modules",
    "merge_into",
    "prepare_lora_backward",
    "read_adapter_metadata",
    "reset_lora_forward_counters",
    "save_lora_adapter",
    "validate_adapter_metadata",
]
