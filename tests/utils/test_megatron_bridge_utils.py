import types

import pytest

from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_hf_config_for_megatron_bridge


@pytest.mark.unit
def test_patch_hf_config_adds_rope_theta_from_rope_parameters():
    hf_config = types.SimpleNamespace(rope_parameters={"rope_theta": 1000000})

    patched_config = patch_hf_config_for_megatron_bridge(hf_config)

    assert patched_config is hf_config
    assert hf_config.rope_theta == 1000000


@pytest.mark.unit
def test_patch_hf_config_does_not_override_existing_rope_theta():
    hf_config = types.SimpleNamespace(rope_theta=500000, rope_parameters={"rope_theta": 1000000})

    patch_hf_config_for_megatron_bridge(hf_config)

    assert hf_config.rope_theta == 500000


@pytest.mark.unit
def test_patch_hf_config_handles_nested_text_config():
    text_config = types.SimpleNamespace(rope_parameters={"rope_theta": 10000})
    hf_config = types.SimpleNamespace(text_config=text_config)

    patch_hf_config_for_megatron_bridge(hf_config)

    assert text_config.rope_theta == 10000


@pytest.mark.unit
def test_patch_auto_bridge_hf_config_patches_hf_pretrained():
    hf_config = types.SimpleNamespace(rope_parameters={"rope_theta": 12345})
    bridge = types.SimpleNamespace(hf_pretrained=hf_config)

    patched_bridge = patch_auto_bridge_hf_config(bridge)

    assert patched_bridge is bridge
    assert bridge.hf_pretrained.rope_theta == 12345
