"""CPU staging and serialization contracts for Megatron checkpoint workers."""

import ast
import importlib.util
import math
import pickle
import sys
import types
from contextlib import nullcontext
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import pytest
import torch
from torch.multiprocessing import reductions

NUM_GPUS = 0


@pytest.mark.parametrize("norm_type", ["float", "scalar_tensor", "vector_tensor"])
@pytest.mark.parametrize("finite", [False, True])
def test_optimizer_step_returns_scalar_norm_and_skips_invalid_gradients(norm_type, finite):
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/model.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_step"
    )
    value = 3.25 if finite else float("nan")
    norm = value if norm_type == "float" else torch.tensor([value] if norm_type == "vector_tensor" else value)
    args = types.SimpleNamespace(
        custom_megatron_before_train_step_hook_path=None,
        check_for_nan_in_loss_and_grad=False,
        ci_test=False,
        enable_mtp_training=False,
        seq_length=64,
        micro_batch_size=1,
        decoder_seq_length=None,
        calculate_per_token_loss=False,
    )
    updates, samples = [], []

    def step():
        updates.append(True)
        return True, norm, 0

    optimizer = types.SimpleNamespace(
        zero_grad=lambda: None, prepare_grads=lambda: False, get_grad_norm=lambda: norm, step=step
    )
    namespace = {
        "get_args": lambda: args,
        "torch": torch,
        "math": math,
        "get_forward_backward_func": lambda: lambda **kwargs: [],
        "compact_actor_logits": lambda enabled: nullcontext(),
        "can_compact_actor_logits": lambda args: False,
        "_wrap_forward_step_with_microbatch_pbar": lambda forward, progress: forward,
        "mpu": types.SimpleNamespace(
            is_pipeline_last_stage=lambda **kwargs: True,
            get_context_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda **kwargs: None,
        ),
        "train_metric_utils": types.SimpleNamespace(reduce_train_step_metrics=lambda *args, **kwargs: {"loss": 1.0}),
    }
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    losses, grad_norm = namespace["train_one_step"](
        args,
        0,
        0,
        None,
        [types.SimpleNamespace(zero_grad_buffer=lambda: None)],
        optimizer,
        types.SimpleNamespace(step=lambda *, increment: samples.append(increment)),
        1,
        16,
    )
    assert losses == {"loss": 1.0} and isinstance(grad_norm, float)
    assert grad_norm == value if finite else math.isnan(grad_norm)
    assert updates == ([True] if finite else [])
    assert samples == ([16] if finite else [])


def load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("remainders", [False, True])
def test_precision_aware_master_parameters_roundtrip_as_fp32(monkeypatch, remainders):
    class FusedAdam:
        store_param_remainders = remainders

    fake_te = types.ModuleType("transformer_engine.pytorch.optimizers")
    fake_te.FusedAdam = FusedAdam
    monkeypatch.setitem(sys.modules, fake_te.__name__, fake_te)
    extensions = load_module("checkpoint_te_extensions", "transformer_engine.py")
    # Signed values, signed zero, and both sides of BF16 rounding boundaries.
    bits = torch.tensor([0x3F801234, 0x3F808000, -1082097664, -1082104047, 0, -2147483648], dtype=torch.int32)
    expected = bits.view(torch.float32)
    param = torch.nn.Parameter(expected.to(torch.bfloat16))
    adam = FusedAdam()
    adam.param_groups = [{"params": [param]}]
    states = {"param": expected.clone(), "exp_avg": torch.arange(6, dtype=torch.float32), "exp_avg_sq": torch.ones(6)}

    def get_states(model_param):
        assert model_param is param
        return states.copy()

    def set_states(model_param, incoming):
        assert model_param is param
        assert incoming["param"].dtype == (torch.int16 if remainders else torch.float32)
        states.update(incoming)

    optimizer = types.SimpleNamespace(
        optimizer=adam,
        model_param_group_index_map={param: (0, 0)},
        _get_main_param_and_optimizer_states=get_states,
        _set_main_param_and_optimizer_states=set_states,
    )
    extensions.patch_precision_aware_optimizer_checkpointing(optimizer)
    optimizer._set_main_param_and_optimizer_states(param, states.copy())
    if remainders:
        assert states["param"].dtype == torch.int16
    for _ in range(2):
        checkpoint = pickle.loads(pickle.dumps(optimizer._get_main_param_and_optimizer_states(param)))
        assert checkpoint["param"].dtype == torch.float32
        assert torch.equal(checkpoint["param"].view(torch.int32), bits)
        torch.testing.assert_close(checkpoint["exp_avg"], torch.arange(6, dtype=torch.float32), rtol=0, atol=0)
        optimizer._set_main_param_and_optimizer_states(param, checkpoint)
    if not remainders:
        assert optimizer._get_main_param_and_optimizer_states is get_states
        assert optimizer._set_main_param_and_optimizer_states is set_states


@pytest.mark.parametrize("legacy_sglang", [False, True])
def test_checkpoint_cpu_tensors_serialize_after_weight_sync(monkeypatch, legacy_sglang):
    monkeypatch.setattr(ForkingPickler, "_extra_reducers", ForkingPickler._extra_reducers.copy())
    monkeypatch.setattr(reductions, "reduce_tensor", reductions.reduce_tensor)
    monkeypatch.delattr(reductions, "_reduce_tensor_original", raising=False)
    native_reduce = reductions.reduce_tensor
    cuda_result = ("rebuild_with_device_uuid", ("GPU-uuid",))

    def cuda_reduce(tensor):
        if tensor.device.type == "cpu":
            # Reproduce SGLang's assumption that all reducer tuples contain
            # CUDA device metadata. CPU tensors have a shorter tuple.
            _, args = native_reduce(tensor)
            args[6]
        return cuda_result

    def patch_reductions():
        if not hasattr(reductions, "_reduce_tensor_original"):
            monkeypatch.setattr(reductions, "_reduce_tensor_original", native_reduce, raising=False)
            reductions.reduce_tensor = cuda_reduce
            reductions.init_reductions()

    modules = {
        "sglang.srt.utils": {"MultiprocessingSerializer": object},
        "sglang.srt.weight_sync.tensor_bucket": {"FlattenedTensorBucket": object},
        "sglang.srt.layers.quantization.fp8_utils": {"quant_weight_ue8m0": None, "transform_scale_ue8m0": None},
        "sglang.srt.model_loader.utils": {"should_deepgemm_weight_requant_ue8m0": None},
        "sglang.srt.patch_torch" if legacy_sglang else "sglang.srt.utils.patch_torch": {
            "monkey_patch_torch_reductions": patch_reductions
        },
    }
    if legacy_sglang:
        monkeypatch.setitem(sys.modules, "sglang.srt.utils.patch_torch", None)
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    adapter = load_module("checkpoint_sglang_adapter", "sglang.py")
    tensor = torch.arange(16).reshape(4, 4)[:, ::2]
    for _ in range(2):
        adapter.monkey_patch_torch_reductions()
        restored = pickle.loads(ForkingPickler.dumps(tensor))
        torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
        fake_cuda = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        assert reductions.reduce_tensor(fake_cuda) == cuda_result


@pytest.mark.parametrize("empty", [False, True])
def test_async_writer_payload_is_cpu_only_and_hook_is_restored(monkeypatch, empty):
    class CudaAllocation:
        def __reduce__(self):
            raise RuntimeError("CUDA IPC is unavailable for this allocation")

    expected = torch.arange(8)

    class Writer:
        def get_save_function_and_args(self):
            if empty:
                return None, None, []
            return len, lambda: [expected], [0, [CudaAllocation()], None]

    modules = {
        "megatron.training.checkpointing": {"load_checkpoint": None, "save_checkpoint": None},
        "megatron.training.global_vars": {"get_args": None},
        "megatron.core.dist_checkpointing.strategies.filesystem_async": {"FileSystemWriterAsync": Writer},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    # Avoid applying the checkpoint module's unrelated sharding metadata hooks.
    monkeypatch.setitem(sys.modules, "torch.distributed._shard.sharding_spec", None)
    checkpoint = load_module("checkpoint_cpu_staging", "checkpoint.py")
    original = Writer.get_save_function_and_args
    with pytest.raises(ValueError, match="save failed"):
        with checkpoint._stage_async_save_on_cpu():
            fn, preload, args = Writer().get_save_function_and_args()
            assert preload is None
            if empty:
                assert fn is None and args == []
            else:
                restored = pickle.loads(pickle.dumps(args))
                torch.testing.assert_close(restored[1][0], expected, rtol=0, atol=0)
            raise ValueError("save failed")
    assert Writer.get_save_function_and_args is original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
