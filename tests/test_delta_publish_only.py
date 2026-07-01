from __future__ import annotations

import os
import sys
import types
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _install_import_stubs() -> None:
    if "safetensors.torch" not in sys.modules:
        try:
            import safetensors.torch  # noqa: F401
        except ImportError:
            safetensors = types.ModuleType("safetensors")
            safetensors_torch = types.ModuleType("safetensors.torch")
            safetensors_torch.save = lambda tensors, metadata=None: b""
            safetensors.torch = safetensors_torch
            sys.modules["safetensors"] = safetensors
            sys.modules["safetensors.torch"] = safetensors_torch

    if "ray" not in sys.modules:
        ray = types.ModuleType("ray")
        actor = types.ModuleType("ray.actor")

        class ActorHandle:
            pass

        class ObjectRef:
            pass

        actor.ActorHandle = ActorHandle
        ray.actor = actor
        ray.ObjectRef = ObjectRef
        ray.get = lambda refs: refs
        sys.modules["ray"] = ray
        sys.modules["ray.actor"] = actor

    if "megatron" not in sys.modules:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.mpu")
        mpu.get_data_parallel_rank = lambda with_context_parallel=False: 0
        mpu.get_tensor_model_parallel_rank = lambda: 0
        mpu.get_pipeline_model_parallel_rank = lambda: 0
        mpu.get_expert_model_parallel_world_size = lambda: 1
        mpu.get_expert_model_parallel_group = lambda: None
        mpu.get_expert_tensor_parallel_world_size = lambda: 1
        mpu.get_expert_tensor_parallel_group = lambda: None
        mpu.get_tensor_model_parallel_world_size = lambda: 1
        mpu.get_tensor_model_parallel_group = lambda: None
        mpu.get_expert_model_parallel_rank = lambda: 0
        transformer = types.ModuleType("megatron.core.transformer")
        transformer_layer = types.ModuleType("megatron.core.transformer.transformer_layer")
        transformer_layer.get_transformer_layer_offset = lambda config, *args, **kwargs: 0
        core.mpu = mpu
        core.transformer = transformer
        megatron.core = core
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = core
        sys.modules["megatron.core.mpu"] = mpu
        sys.modules["megatron.core.transformer"] = transformer
        sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer

    megatron_to_hf = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf.convert_to_hf = lambda args, model_name, name, param, quantization_config: [(name, param)]
    sys.modules.setdefault("slime.backends.megatron_utils.megatron_to_hf", megatron_to_hf)

    if "sglang" not in sys.modules:
        sglang = types.ModuleType("sglang")
        srt = types.ModuleType("sglang.srt")
        sys.modules["sglang"] = sglang
        sys.modules["sglang.srt"] = srt

    if "sglang.srt.layers.quantization.fp8_utils" not in sys.modules:
        fp8_utils = types.ModuleType("sglang.srt.layers.quantization.fp8_utils")
        fp8_utils.quant_weight_ue8m0 = None
        fp8_utils.transform_scale_ue8m0 = None
        sys.modules["sglang.srt.layers"] = types.ModuleType("sglang.srt.layers")
        sys.modules["sglang.srt.layers.quantization"] = types.ModuleType("sglang.srt.layers.quantization")
        sys.modules["sglang.srt.layers.quantization.fp8_utils"] = fp8_utils

    if "sglang.srt.model_loader.utils" not in sys.modules:
        model_loader_utils = types.ModuleType("sglang.srt.model_loader.utils")
        model_loader_utils.should_deepgemm_weight_requant_ue8m0 = None
        sys.modules["sglang.srt.model_loader"] = types.ModuleType("sglang.srt.model_loader")
        sys.modules["sglang.srt.model_loader.utils"] = model_loader_utils

    utils = sys.modules.get("sglang.srt.utils")
    if utils is None:
        utils = types.ModuleType("sglang.srt.utils")
        utils.__path__ = []
        sys.modules["sglang.srt.utils"] = utils
    utils.MultiprocessingSerializer = object

    patch_torch = types.ModuleType("sglang.srt.utils.patch_torch")
    patch_torch.monkey_patch_torch_reductions = lambda: None
    sys.modules.setdefault("sglang.srt.utils.patch_torch", patch_torch)
    sys.modules.setdefault("sglang.srt.patch_torch", patch_torch)

    if "sglang.srt.managers.io_struct" not in sys.modules:
        io_struct = types.ModuleType("sglang.srt.managers.io_struct")

        class DeltaEncoding(Enum):
            INDICES = "indices"
            DELTAS = "deltas"
            DELTAS_ZSTD = "deltas_zstd"

        @dataclass
        class DeltaParam:
            name: str
            dtype: str
            shape: list[int]
            pos_start: int
            pos_end: int
            pos_width: int
            val_start: int
            val_end: int

        @dataclass
        class DeltaSpec:
            encoding: DeltaEncoding
            params: list[DeltaParam]
            checksum: int

        io_struct.DeltaEncoding = DeltaEncoding
        io_struct.DeltaParam = DeltaParam
        io_struct.DeltaSpec = DeltaSpec
        sys.modules["sglang.srt.managers"] = types.ModuleType("sglang.srt.managers")
        sys.modules["sglang.srt.managers.io_struct"] = io_struct

    tensor_bucket = types.ModuleType("sglang.srt.weight_sync.tensor_bucket")
    tensor_bucket.FlattenedTensorBucket = object
    sys.modules.setdefault("sglang.srt.weight_sync", types.ModuleType("sglang.srt.weight_sync"))
    sys.modules.setdefault("sglang.srt.weight_sync.tensor_bucket", tensor_bucket)


_install_import_stubs()

from slime.backends.megatron_utils.update_weight import update_weight_from_distributed_delta as delta_mod  # noqa: E402


class _InlineFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _InlineExecutor:
    def submit(self, fn):
        return _InlineFuture(fn())


class _FakeWriter:
    def __init__(self):
        self.drain_calls = 0

    def drain(self):
        self.drain_calls += 1


class _RemoteMethod:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name

    def remote(self, **kwargs):
        self._owner.calls.append((self._name, kwargs))
        return f"{self._name}-ref"


class _FakeEngine:
    def __init__(self):
        self.calls = []
        self.update_weights_from_disk = _RemoteMethod(self, "update_weights_from_disk")
        self.set_weight_version = _RemoteMethod(self, "set_weight_version")
        self.continue_generation = _RemoteMethod(self, "continue_generation")


def _patch_single_rank_dist(monkeypatch):
    barrier_calls = []
    gathered = []

    monkeypatch.setattr(delta_mod, "get_gloo_group", lambda: None)
    monkeypatch.setattr(delta_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(delta_mod.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(delta_mod.dist, "barrier", lambda group=None: barrier_calls.append(group))

    def all_gather_object(outputs, value, group=None):
        gathered.append((list(value), group))
        outputs[0] = list(value)

    monkeypatch.setattr(delta_mod.dist, "all_gather_object", all_gather_object)
    return barrier_calls, gathered


def _make_publish_only_updater(tmp_path: Path, publish_hook, *, publish_wait: str = "next-sync"):
    updater = delta_mod.UpdateWeightFromDistributedDelta.__new__(delta_mod.UpdateWeightFromDistributedDelta)
    updater.args = Namespace(update_weight_delta_keep_files=True, update_weight_delta_publish_wait=publish_wait)
    updater.transport = "disk"
    updater._publish_only = True
    updater._publish_wait = publish_wait
    updater._pending_files = []
    updater._pending_publishes = []
    updater._published_any = False
    updater._pre_push_hook = None
    updater._publish_hook = publish_hook
    updater._rpc_executor = _InlineExecutor()
    updater.writer = _FakeWriter()
    updater.weight_version = 7
    updater._version_dir = os.path.join(tmp_path, "weight_v000007")
    os.makedirs(updater._version_dir, exist_ok=True)
    updater.rollout_engines = [_FakeEngine()]
    return updater


def test_publish_only_finalize_calls_publish_hook_without_engine_rpcs_or_cleanup(monkeypatch, tmp_path):
    _patch_single_rank_dist(monkeypatch)
    ray_get_calls = []
    monkeypatch.setattr(delta_mod.ray, "get", lambda refs: ray_get_calls.append(refs))
    monkeypatch.setattr(
        delta_mod.shutil, "rmtree", lambda *_args, **_kwargs: pytest.fail("publish-only must keep files")
    )

    hook_calls = []

    def publish_hook(args, version_dir, files, weight_version, engines):
        hook_calls.append((args, version_dir, files, weight_version, engines))
        return ["publish-ref"]

    updater = _make_publish_only_updater(tmp_path, publish_hook)
    updater._pending_files = ["rank0000_flush000000.safetensors"]

    updater._finalize_sync()

    assert updater.writer.drain_calls == 1
    assert updater._pending_files == []
    assert updater._published_any is True
    assert hook_calls == [
        (
            updater.args,
            updater._version_dir,
            ["rank0000_flush000000.safetensors"],
            "7",
            updater.rollout_engines,
        )
    ]
    assert updater.rollout_engines[0].calls == []
    # The publish stays in flight across the training step: finalize must not
    # await its refs; the next sync (or disconnect) drains it.
    assert len(updater._pending_publishes) == 1
    assert ray_get_calls == []
    assert os.path.isdir(updater._version_dir)

    updater._drain_pending_publishes()

    assert updater._pending_publishes == []
    assert ray_get_calls == [["publish-ref"]]


def test_publish_only_finalize_publishes_noop_version(monkeypatch, tmp_path):
    _patch_single_rank_dist(monkeypatch)
    ray_get_calls = []
    monkeypatch.setattr(delta_mod.ray, "get", lambda refs: ray_get_calls.append(refs))

    hook_calls = []

    def publish_hook(args, version_dir, files, weight_version, engines):
        hook_calls.append((version_dir, files, weight_version, engines))
        return None

    updater = _make_publish_only_updater(tmp_path, publish_hook)

    updater._finalize_sync()

    assert updater.writer.drain_calls == 1
    assert updater._published_any is True
    assert hook_calls == [(updater._version_dir, [], "7", updater.rollout_engines)]
    assert updater.rollout_engines[0].calls == []
    assert len(updater._pending_publishes) == 1

    updater._drain_pending_publishes()

    assert updater._pending_publishes == []
    assert ray_get_calls == []


def test_publish_only_sync_wait_drains_publish_before_return(monkeypatch, tmp_path):
    _patch_single_rank_dist(monkeypatch)
    ray_get_calls = []
    monkeypatch.setattr(delta_mod.ray, "get", lambda refs: ray_get_calls.append(refs))

    updater = _make_publish_only_updater(tmp_path, lambda *a: ["publish-ref"], publish_wait="sync")
    updater._pending_files = ["rank0000_flush000000.safetensors"]

    updater._finalize_sync()

    assert updater._pending_publishes == []
    assert ray_get_calls == [["publish-ref"]]


def test_disconnect_drains_pending_publish(monkeypatch, tmp_path):
    _patch_single_rank_dist(monkeypatch)
    ray_get_calls = []
    monkeypatch.setattr(delta_mod.ray, "get", lambda refs: ray_get_calls.append(refs))

    updater = _make_publish_only_updater(tmp_path, lambda *a: ["publish-ref"])
    updater._finalize_sync()
    assert len(updater._pending_publishes) == 1

    updater.disconnect_rollout_engines()

    assert updater._pending_publishes == []
    assert ray_get_calls == [["publish-ref"]]


def test_publish_only_flush_defers_publish_until_finalize(monkeypatch, tmp_path):
    barrier_calls, _gathered = _patch_single_rank_dist(monkeypatch)
    updater = _make_publish_only_updater(tmp_path, publish_hook=None)

    class FakeBucket:
        has_updates = True

    flush_calls = []
    monkeypatch.setattr(updater, "_flush_bucket", lambda bucket, pbar: flush_calls.append((bucket, pbar)))
    monkeypatch.setattr(
        updater,
        "_publish_batch",
        lambda: pytest.fail("publish-only should publish once from _finalize_sync"),
    )

    bucket = FakeBucket()
    updater._flush_and_publish(bucket, pbar=None)

    assert flush_calls == [(bucket, None)]
    assert len(barrier_calls) == 1
    assert updater._pending_publishes == []
