import functools

import torch.nn.functional as F
from mbridge.core import register_model
from mbridge.models import Gemma3Bridge


# Gemma uses GeGLU (GELU with tanh approximation + gated linear unit), not SwiGLU.
# The upstream mbridge base config hardcodes F.silu which is incorrect for all Gemma models.
# See: https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/
_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@register_model("gemma3")
class Gemma3BridgeFix(Gemma3Bridge):
    """Override Gemma3Bridge to fix activation: GeGLU instead of SwiGLU."""

    def _build_config(self):
        config = super()._build_config()
        config.activation_func = _gelu_tanh
        config.bias_activation_fusion = False
        return config
