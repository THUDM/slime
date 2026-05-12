import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def install_bridge_stubs():
    """Install minimal stubs so the gemma3 plugin can be imported without real dependencies."""
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod

    mbridge_mod = types.ModuleType("mbridge")
    mbridge_core_mod = types.ModuleType("mbridge.core")
    mbridge_models_mod = types.ModuleType("mbridge.models")

    def register_model(_names):
        def decorator(cls):
            return cls
        return decorator

    class Gemma3Bridge:
        """Stub that mimics the upstream mbridge Gemma3Bridge."""

        def _build_config(self):
            # Simulates upstream returning a config with wrong activation
            return types.SimpleNamespace(
                activation_func=torch.nn.functional.silu,
                bias_activation_fusion=True,
            )

    mbridge_core_mod.register_model = register_model
    mbridge_models_mod.Gemma3Bridge = Gemma3Bridge

    sys.modules["mbridge"] = mbridge_mod
    sys.modules["mbridge.core"] = mbridge_core_mod
    sys.modules["mbridge.models"] = mbridge_models_mod


def load_gemma3_module():
    install_bridge_stubs()
    module_path = Path(__file__).resolve().parents[1] / "slime_plugins" / "mbridge" / "gemma3.py"
    module_name = "test_gemma3_bridge_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_gemma3_bridge_uses_gelu_tanh_not_silu():
    """Gemma models use GeGLU (gelu_tanh + GLU), not SwiGLU (silu + GLU).

    See: https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/
    """
    module = load_gemma3_module()
    bridge = module.Gemma3BridgeFix.__new__(module.Gemma3BridgeFix)
    config = bridge._build_config()

    activation = config.activation_func

    # Verify it's gelu with tanh approximation, not silu
    x = torch.tensor([1.0, -1.0, 0.5])
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    actual = activation(x)
    assert torch.equal(actual, expected), f"Expected gelu_tanh output, got {actual}"

    # Verify it's NOT silu
    silu_out = torch.nn.functional.silu(x)
    assert not torch.equal(actual, silu_out), "Activation should not be silu"


@pytest.mark.unit
def test_gemma3_bridge_disables_bias_activation_fusion():
    """GeGLU has no fused kernel in Megatron, so bias_activation_fusion must be False."""
    module = load_gemma3_module()
    bridge = module.Gemma3BridgeFix.__new__(module.Gemma3BridgeFix)
    config = bridge._build_config()

    assert config.bias_activation_fusion is False
