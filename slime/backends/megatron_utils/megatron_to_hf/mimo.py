import re
from .qwen2 import convert_qwen2_to_hf


def convert_mimo_to_hf(args, name, param):
    """
    Convert MiMo model parameters from Megatron to HuggingFace format.

    MiMo extends Qwen2 with MTP (Multi-Token Prediction) layers.
    """

    if "mtp" in name:
        return convert_mimo_mtp_param(args, name, param)

    return convert_qwen2_to_hf(args, name, param)


def convert_mimo_mtp_param(args, name, param):
    """
    Convert MTP layer parameters from Megatron to HuggingFace format.

    MTP layers in MiMo contain:
    - LayerNorms (token_layernorm, hidden_layernorm, final_layernorm)
    - Input projection (input_proj)
    - Self attention (reuses Qwen2 attention structure)
    - MLP (reuses Qwen2 MLP structure)

    Based on MimoBridge._convert_mtp_param logic (reverse mapping)
    """
    mtp_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_pattern, name)

    if not match:
        raise ValueError(f"Invalid MTP parameter name: {name}")

    layer_idx, component = match.groups()

    # Direct mappings for MTP-specific components (Megatron -> HF)
    # Based on MimoBridge direct_name_mapping (reversed)
    direct_mappings = {
        "enorm.weight": f"model.mtp_layers.{layer_idx}.token_layernorm.weight",
        "hnorm.weight": f"model.mtp_layers.{layer_idx}.hidden_layernorm.weight",
        "eh_proj.weight": f"model.mtp_layers.{layer_idx}.input_proj.weight",
        "final_layernorm.weight": f"model.mtp_layers.{layer_idx}.final_layernorm.weight",
    }

    # Check direct mappings first
    if component in direct_mappings:
        return [(direct_mappings[component], param)]

    # Handle transformer_layer components
    if component.startswith("transformer_layer."):
        # Remove "transformer_layer." prefix
        transformer_component = component[len("transformer_layer.") :]

        # Create proxy name for reusing existing Qwen2 conversion functions
        proxy_name = f"module.module.decoder.layers.{layer_idx}.{transformer_component}"

        # Use existing convert_qwen2_to_hf function for transformer components
        results = convert_qwen2_to_hf(args, proxy_name, param)

        # Replace model.layers with mtp_layers in results
        converted_results = []
        for hf_name, hf_param in results:
            # Replace model.layers.{idx} with mtp_layers.{idx}
            hf_name = hf_name.replace(f"model.layers.{layer_idx}", f"model.mtp_layers.{layer_idx}")
            converted_results.append((hf_name, hf_param))

        return converted_results

    raise ValueError(f"Unknown MTP component: {component} in {name}")
