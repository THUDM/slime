import re

from ..hf_to_megatron.common import strip_mcore_wrappers
from .deepseekv3 import convert_deepseekv3_to_hf


def convert_glm5_next_to_hf(args, name, param):
    name = strip_mcore_wrappers(name)
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer, rest = match.groups()
        if rest.startswith("self_attention."):
            return [(f"model.language_model.layers.{layer}.self_attn.{rest.removeprefix('self_attention.')}", param)]
        if rest.startswith("hc_"):
            return [(f"model.language_model.layers.{layer}.{rest}", param)]
    return [
        (key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key, value)
        for key, value in convert_deepseekv3_to_hf(args, f"module.module.{name}", param)
    ]
