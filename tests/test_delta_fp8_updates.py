import sys
import types

import torch


def _stub_module(name: str, *, package: bool = False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module


# Stub megatron and sglang imports to avoid heavy dependencies in unit tests.
megatron = _stub_module("megatron", package=True)
megatron_core = _stub_module("megatron.core", package=True)
megatron_core.mpu = types.SimpleNamespace()
megatron_core_transformer = _stub_module("megatron.core.transformer", package=True)
megatron_core_transformer_layer = _stub_module("megatron.core.transformer.transformer_layer")
megatron_core_transformer_layer.get_transformer_layer_offset = lambda *_args, **_kwargs: 0

sglang = _stub_module("sglang", package=True)
sglang_srt = _stub_module("sglang.srt", package=True)
sglang_srt_server_args = _stub_module("sglang.srt.server_args")
sglang_srt_utils = _stub_module("sglang.srt.utils", package=True)
sglang_srt_utils_patch = _stub_module("sglang.srt.utils.patch_torch")
sglang_srt_patch = _stub_module("sglang.srt.patch_torch")
sglang_srt_weight_sync = _stub_module("sglang.srt.weight_sync", package=True)
sglang_srt_weight_sync_tb = _stub_module("sglang.srt.weight_sync.tensor_bucket")
sglang_srt_model_executor = _stub_module("sglang.srt.model_executor", package=True)
sglang_srt_model_runner = _stub_module("sglang.srt.model_executor.model_runner")

sglang_srt_server_args.ServerArgs = object
sglang_srt_utils.MultiprocessingSerializer = types.SimpleNamespace(
    serialize=lambda obj, output_str=True: obj,
    deserialize=lambda obj: obj,
)
sglang_srt_utils.kill_process_tree = lambda _pid: None
sglang_srt_utils_patch.monkey_patch_torch_reductions = lambda: None
sglang_srt_patch.monkey_patch_torch_reductions = lambda: None
sglang_srt_weight_sync_tb.FlattenedTensorBucket = types.SimpleNamespace(supports_multi_dtypes=True)
sglang_srt_model_runner.FlattenedTensorBucket = types.SimpleNamespace(supports_multi_dtypes=True)

from slime.backends.megatron_utils.update_weight.update_weight_from_tensor import UpdateWeightFromTensor
from slime.backends.sglang_utils.sglang_engine import SGLangEngine


def _build_updater(block_size=(2, 2), threshold=0.5):
    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater._delta_cache = {}
    updater._delta_threshold = threshold
    updater._delta_block_size = block_size
    return updater


def test_fp8_delta_payload_block_change():
    updater = _build_updater()
    weight_name = "layer.weight"
    scale_name = "layer.weight_scale_inv"

    prev_qweight = torch.zeros((4, 4), dtype=torch.float16)
    prev_scale = torch.zeros((2, 2), dtype=torch.float32)
    updater._delta_cache[weight_name] = {
        "qweight": prev_qweight.clone(),
        "scale": prev_scale.clone(),
        "scale_name": scale_name,
        "block_size": (2, 2),
    }

    curr_qweight = prev_qweight.clone()
    curr_qweight[0:2, 2:4] = 1
    curr_scale = prev_scale.clone()
    curr_scale[0, 1] = 2.0

    payload, _qweight, _scale = updater._build_fp8_delta_payload(
        weight_name, curr_qweight, scale_name, curr_scale
    )
    assert payload is not None
    assert payload.get("full_update") is None
    assert payload["block_indices"] == [1]
    assert payload["block_shapes"] == [(2, 2)]
    assert len(payload["qweight_blocks"]) == 1
    assert payload["scale_blocks"].numel() == 1
    updater._update_delta_cache(weight_name, _qweight, scale_name, _scale)

    empty_payload, *_ = updater._build_fp8_delta_payload(
        weight_name, curr_qweight, scale_name, curr_scale
    )
    assert empty_payload == {"empty": True}


def test_apply_fp8_delta_reconstructs_full_tensor():
    engine = SGLangEngine.__new__(SGLangEngine)
    weight_name = "layer.weight"
    scale_name = "layer.weight_scale_inv"
    prev_qweight = torch.zeros((4, 4), dtype=torch.float16)
    prev_scale = torch.zeros((2, 2), dtype=torch.float32)
    engine._delta_weight_cache = [
        {
            weight_name: {
                "qweight": prev_qweight.clone(),
                "scale": prev_scale.clone(),
                "block_size": (2, 2),
                "scale_name": scale_name,
            }
        }
    ]

    delta_payload = {
        "weight_name": weight_name,
        "scale_name": scale_name,
        "full_shape": (4, 4),
        "scale_shape": (2, 2),
        "block_size": (2, 2),
        "block_indices": [1],
        "block_shapes": [(2, 2)],
        "qweight_blocks": [torch.ones((2, 2), dtype=torch.float16)],
        "scale_blocks": torch.tensor([2.0], dtype=torch.float32),
    }

    updated = engine._apply_fp8_delta(engine._delta_weight_cache[0], delta_payload)
    assert updated is not None
    _, qweight_full, _, scale_full = updated
    assert torch.all(qweight_full[0:2, 2:4] == 1)
    assert scale_full[0, 1].item() == 2.0


def test_delta_update_invokes_sglang_request():
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine._delta_weight_cache = [
        {
            "layer.weight": {
                "qweight": torch.zeros((2, 2), dtype=torch.float16),
                "scale": torch.zeros((1, 1), dtype=torch.float32),
                "block_size": (2, 2),
                "scale_name": "layer.weight_scale_inv",
            }
        }
    ]

    captured = []

    def _fake_make_request(endpoint, payload):
        captured.append((endpoint, payload))
        return {}

    engine._make_request = _fake_make_request
    engine._serialize_named_tensors_by_dtype = lambda named_tensors: {"dtype": f"serialized-{len(named_tensors)}"}

    delta_payload = {
        "deltas": [
            {
                "weight_name": "layer.weight",
                "scale_name": "layer.weight_scale_inv",
                "full_shape": (2, 2),
                "scale_shape": (1, 1),
                "block_size": (2, 2),
                "block_indices": [0],
                "block_shapes": [(2, 2)],
                "qweight_blocks": [torch.ones((2, 2), dtype=torch.float16)],
                "scale_blocks": torch.tensor([3.0], dtype=torch.float32),
            }
        ]
    }

    engine.update_weights_from_tensor_delta([delta_payload], weight_version="7")
    assert captured
    endpoint, payload = captured[0]
    assert endpoint == "update_weights_from_tensor"
    assert payload["serialized_named_tensors"] == ["serialized-2"]
    assert payload["weight_version"] == "7"
