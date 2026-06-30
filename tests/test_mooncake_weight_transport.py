from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeTensor:
    def __init__(self, shape, dtype: str, element_size: int):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._element_size = element_size

    def numel(self):
        out = 1
        for dim in self.shape:
            out *= dim
        return out

    def element_size(self):
        return self._element_size


def load_mooncake_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "update_weight"
        / "update_weight_from_mooncake.py"
    )
    module_name = "test_update_weight_from_mooncake_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class RecordingClient:
    def __init__(self):
        self.close_count = 0
        self.calls = []

    def write_bucket(self, **kwargs):
        client = load_mooncake_module().FakeMooncakeTransferClient()
        manifests = client.write_bucket(**kwargs)
        self.calls.extend(client.calls)
        return manifests

    def close(self):
        self.close_count += 1


def ray_get(value):
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, Exception):
                raise item
            out.append(item)
        return out
    if isinstance(value, Exception):
        raise value
    return value


def patch_pp_source_rank(monkeypatch, module):
    monkeypatch.setattr(module, "ray", types.SimpleNamespace(get=ray_get))
    monkeypatch.setattr(module, "time", types.SimpleNamespace(sleep=lambda _seconds: None))
    monkeypatch.setattr(
        module.mpu,
        "get_data_parallel_rank",
        lambda with_context_parallel=True: 0,
    )
    monkeypatch.setattr(module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(module.mpu, "get_pipeline_model_parallel_rank", lambda: 0)


@pytest.mark.unit
def test_build_manifest_assigns_offsets_and_preserves_tensor_metadata():
    module = load_mooncake_module()

    tensors = [
        ("layer.0.weight", FakeTensor((2, 3), "bfloat16", 2)),
        ("layer.0.bias", FakeTensor((3,), "float32", 4)),
    ]
    descriptor = {
        "target_name": "127.0.0.1:18000",
        "buffers": [{"slot": 0, "addr": 4096, "capacity": 64}],
    }

    manifest = module.build_mooncake_manifest(
        descriptor=descriptor,
        named_tensors=tensors,
        weight_version=7,
        load_format=None,
        buffer_slot=0,
    )

    assert manifest["weight_version"] == "7"
    assert manifest["target_name"] == "127.0.0.1:18000"
    assert manifest["buffer_slot"] == 0
    assert manifest["remote_addr"] == 4096
    assert manifest["nbytes"] == 24
    assert manifest["tensors"] == [
        {
            "name": "layer.0.weight",
            "dtype": "bfloat16",
            "shape": [2, 3],
            "offset": 0,
            "nbytes": 12,
        },
        {
            "name": "layer.0.bias",
            "dtype": "float32",
            "shape": [3],
            "offset": 12,
            "nbytes": 12,
        },
    ]


@pytest.mark.unit
def test_build_manifest_preserves_cuda_receiver_buffer_device():
    module = load_mooncake_module()

    manifest = module.build_mooncake_manifest(
        descriptor={
            "target_name": "127.0.0.1:18000",
            "buffer_device": "cuda",
            "buffers": [{"slot": 0, "addr": 4096, "capacity": 64, "device": "cuda"}],
        },
        named_tensors=[("layer.weight", FakeTensor((4,), "float16", 2))],
        weight_version=1,
        load_format=None,
    )

    assert manifest["buffer_device"] == "cuda"


@pytest.mark.unit
def test_build_manifest_composes_target_name_from_server_name_and_rpc_port():
    module = load_mooncake_module()

    manifest = module.build_mooncake_manifest(
        descriptor={
            "server_name": "127.0.0.1",
            "rpc_port": 18000,
            "buffers": [{"slot": 0, "addr": 4096, "capacity": 64}],
        },
        named_tensors=[("layer.weight", FakeTensor((4,), "float16", 2))],
        weight_version=1,
        load_format=None,
    )

    assert manifest["target_name"] == "127.0.0.1:18000"


@pytest.mark.unit
def test_build_manifest_rejects_bucket_larger_than_receiver_buffer():
    module = load_mooncake_module()

    descriptor = {
        "target_name": "127.0.0.1:18000",
        "buffers": [{"slot": 0, "addr": 4096, "capacity": 8}],
    }

    with pytest.raises(ValueError, match="exceeds Mooncake receiver buffer"):
        module.build_mooncake_manifest(
            descriptor=descriptor,
            named_tensors=[("too.large", FakeTensor((3,), "float32", 4))],
            weight_version=1,
            load_format=None,
            buffer_slot=0,
        )


@pytest.mark.unit
def test_fake_mooncake_transfer_client_records_one_write_per_receiver():
    module = load_mooncake_module()

    client = module.FakeMooncakeTransferClient()
    tensors = [("layer.weight", FakeTensor((4,), "float16", 2))]
    descriptors = [
        {"target_name": "engine-a:18000", "buffers": [{"slot": 0, "addr": 1000, "capacity": 16}]},
        {"target_name": "engine-b:18001", "buffers": [{"slot": 0, "addr": 2000, "capacity": 16}]},
    ]

    manifests = client.write_bucket(
        receiver_descriptors=descriptors,
        named_tensors=tensors,
        weight_version=3,
        load_format=None,
        delta=None,
        buffer_slot=0,
    )

    assert [m["target_name"] for m in manifests] == ["engine-a:18000", "engine-b:18001"]
    assert [call.target_name for call in client.calls] == ["engine-a:18000", "engine-b:18001"]
    assert [call.remote_addr for call in client.calls] == [1000, 2000]
    assert all(call.nbytes == 8 for call in client.calls)
    assert all(call.tensor_names == ("layer.weight",) for call in client.calls)


@pytest.mark.unit
def test_mooncake_transfer_client_raises_when_unregister_fails(monkeypatch):
    module = load_mooncake_module()

    class FakeCudaTensor(FakeTensor):
        is_cuda = True
        device = "cuda:0"

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 1234

    class FakeEngine:
        def initialize(self, *_args):
            return 0

        def register_memory(self, *_args):
            return 0

        def unregister_memory(self, *_args):
            return -5

    engine_mod = types.ModuleType("mooncake.engine")
    engine_mod.TransferEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_mod)

    client = module.MooncakeTransferClient(
        local_server_name="trainer",
        metadata_server="P2PHANDSHAKE",
        protocol="tcp",
        device_name="",
    )
    monkeypatch.setattr(client, "_write_manifest", lambda _manifest, _named_tensors: None)

    with pytest.raises(RuntimeError, match="unregister_memory\\(1234\\)=-5"):
        client.write_bucket(
            receiver_descriptors=[
                {"target_name": "engine:18000", "buffers": [{"slot": 0, "addr": 1000, "capacity": 16}]}
            ],
            named_tensors=[("layer.weight", FakeCudaTensor((4,), "float16", 2))],
            weight_version=1,
            load_format=None,
            delta=None,
        )


@pytest.mark.unit
def test_mooncake_transfer_client_rejects_non_cuda_receiver_buffer(monkeypatch):
    module = load_mooncake_module()

    class FakeEngine:
        def initialize(self, *_args):
            return 0

        def register_memory(self, *_args):
            return 0

        def unregister_memory(self, *_args):
            return 0

    engine_mod = types.ModuleType("mooncake.engine")
    engine_mod.TransferEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_mod)

    client = module.MooncakeTransferClient(
        local_server_name="trainer",
        metadata_server="P2PHANDSHAKE",
        protocol="tcp",
        device_name="",
    )

    tensor = FakeTensor((4,), "float16", 2)
    with pytest.raises(RuntimeError, match="CUDA receiver buffers"):
        client.write_bucket(
            receiver_descriptors=[
                {
                    "target_name": "engine:18000",
                    "buffer_device": "cpu",
                    "buffers": [{"slot": 0, "addr": 1000, "capacity": 16, "device": "cpu"}],
                }
            ],
            named_tensors=[("layer.weight", tensor)],
            weight_version=1,
            load_format=None,
            delta=None,
        )


@pytest.mark.unit
def test_default_local_server_name_uses_host_ip_for_p2p_handshake(monkeypatch):
    module = load_mooncake_module()

    monkeypatch.setattr(module.socket, "gethostbyname", lambda hostname: f"ip-for-{hostname}")

    assert (
        module.default_mooncake_local_server_name(
            metadata_server="P2PHANDSHAKE",
            pp_rank=3,
            host_getter=lambda: "trainer-host",
        )
        == "ip-for-trainer-host"
    )


@pytest.mark.unit
def test_default_local_server_name_uses_stable_name_for_metadata_server():
    module = load_mooncake_module()

    assert (
        module.default_mooncake_local_server_name(
            metadata_server="127.0.0.1:2379",
            pp_rank=3,
            host_getter=lambda: "trainer-host",
        )
        == "slime-train-pp3"
    )


@pytest.mark.unit
def test_update_weight_from_mooncake_publishes_manifests_to_rollout_engines(monkeypatch):
    module = load_mooncake_module()

    acquire = types.SimpleNamespace(remote=lambda: True)
    release = types.SimpleNamespace(remote=lambda: None)
    lock = types.SimpleNamespace(acquire=acquire, release=release)
    published = []

    class _Engine:
        def __init__(self, descriptor):
            self.init_mooncake_weight_receiver = RemoteMethod(lambda _config: descriptor)
            self.update_weights_from_mooncake = RemoteMethod(lambda manifest: published.append(manifest))
            self.destroy_mooncake_weight_receiver = RemoteMethod(lambda: None)

    patch_pp_source_rank(monkeypatch, module)

    updater = module.UpdateWeightFromMooncake(
        args=types.SimpleNamespace(
            mooncake_metadata_server="P2PHANDSHAKE",
            mooncake_protocol="tcp",
            mooncake_device_name="",
            mooncake_buffer_size=16,
            mooncake_buffer_count=1,
        ),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
        transfer_client=module.FakeMooncakeTransferClient(),
    )
    updater.connect_rollout_engines(
        [
            _Engine({"target_name": "engine-a:18000", "buffers": [{"slot": 0, "addr": 1000, "capacity": 16}]}),
            _Engine({"target_name": "engine-b:18001", "buffers": [{"slot": 0, "addr": 2000, "capacity": 16}]}),
        ],
        lock,
    )

    updater._update_bucket_weights_from_mooncake(
        [("layer.weight", FakeTensor((4,), "float16", 2))],
        pbar=None,
    )

    assert [manifest["target_name"] for manifest in published] == ["engine-a:18000", "engine-b:18001"]
    assert updater.update_weight_metrics["mooncake/write_bucket_count"] == 1
    assert updater.update_weight_metrics["mooncake/write_bytes"] == 16


@pytest.mark.unit
def test_connect_reinitializes_existing_receivers_before_reconnect(monkeypatch):
    module = load_mooncake_module()
    patch_pp_source_rank(monkeypatch, module)

    class Engine:
        def __init__(self):
            self.init_count = 0
            self.destroy_count = 0
            self.init_mooncake_weight_receiver = RemoteMethod(self._init)
            self.destroy_mooncake_weight_receiver = RemoteMethod(self._destroy)

        def _init(self, _config):
            self.init_count += 1
            return {
                "target_name": f"engine:{18000 + self.init_count}",
                "buffers": [{"slot": 0, "addr": 1000, "capacity": 16}],
            }

        def _destroy(self):
            self.destroy_count += 1

    first_client = RecordingClient()
    second_client = RecordingClient()
    monkeypatch.setattr(
        module.MooncakeTransferClient,
        "from_args",
        classmethod(lambda _cls, _args, pp_rank: second_client),
    )
    engine = Engine()
    updater = module.UpdateWeightFromMooncake(
        args=types.SimpleNamespace(
            mooncake_metadata_server="P2PHANDSHAKE",
            mooncake_protocol="tcp",
            mooncake_device_name="",
            mooncake_buffer_size=16,
            mooncake_buffer_count=1,
        ),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
        transfer_client=first_client,
    )

    updater.connect_rollout_engines([engine], types.SimpleNamespace())
    updater.connect_rollout_engines([engine], types.SimpleNamespace())

    assert engine.init_count == 2
    assert engine.destroy_count == 1
    assert first_client.close_count == 1
    assert updater._transfer_client is second_client


@pytest.mark.unit
def test_connect_cleans_receiver_when_transfer_client_init_fails(monkeypatch):
    module = load_mooncake_module()
    patch_pp_source_rank(monkeypatch, module)

    class Engine:
        def __init__(self):
            self.destroy_count = 0
            self.init_mooncake_weight_receiver = RemoteMethod(
                lambda _config: {"target_name": "engine:18000", "buffers": [{"slot": 0, "addr": 1000, "capacity": 16}]}
            )
            self.destroy_mooncake_weight_receiver = RemoteMethod(self._destroy)

        def _destroy(self):
            self.destroy_count += 1

    monkeypatch.setattr(
        module.MooncakeTransferClient,
        "from_args",
        classmethod(lambda _cls, _args, pp_rank: (_ for _ in ()).throw(RuntimeError("client init failed"))),
    )
    engine = Engine()
    updater = module.UpdateWeightFromMooncake(
        args=types.SimpleNamespace(
            mooncake_metadata_server="P2PHANDSHAKE",
            mooncake_protocol="tcp",
            mooncake_device_name="",
            mooncake_buffer_size=16,
            mooncake_buffer_count=1,
        ),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
    )

    with pytest.raises(RuntimeError, match="client init failed"):
        updater.connect_rollout_engines([engine], types.SimpleNamespace())

    assert engine.destroy_count == 1
    assert updater._receiver_descriptors == []
    assert updater._connected is False


@pytest.mark.unit
def test_connect_rejects_multi_gpu_rollout_engine_until_tp_receiver_contract_exists(monkeypatch):
    module = load_mooncake_module()
    patch_pp_source_rank(monkeypatch, module)

    updater = module.UpdateWeightFromMooncake(
        args=types.SimpleNamespace(
            mooncake_metadata_server="P2PHANDSHAKE",
            mooncake_protocol="tcp",
            mooncake_device_name="",
            mooncake_buffer_size=16,
            mooncake_buffer_count=1,
            rollout_num_gpus_per_engine=2,
        ),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
        transfer_client=module.FakeMooncakeTransferClient(),
    )

    with pytest.raises(NotImplementedError, match="one GPU per rollout engine"):
        updater.connect_rollout_engines(
            rollout_engines=[types.SimpleNamespace()],
            rollout_engine_lock=types.SimpleNamespace(),
            engine_gpu_counts=[2],
        )


@pytest.mark.unit
def test_disconnect_closes_client_even_when_destroy_rpc_fails(monkeypatch):
    module = load_mooncake_module()
    patch_pp_source_rank(monkeypatch, module)
    client = RecordingClient()

    engine = types.SimpleNamespace(
        destroy_mooncake_weight_receiver=RemoteMethod(lambda: RuntimeError("destroy failed")),
    )
    updater = module.UpdateWeightFromMooncake(
        args=types.SimpleNamespace(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
        transfer_client=client,
    )
    updater._is_pp_src_rank = True
    updater.rollout_engines = [engine]
    updater._receiver_descriptors = [{"target_name": "engine:18000", "buffers": []}]
    updater._connected = True

    with pytest.raises(RuntimeError, match="destroy failed"):
        updater.disconnect_rollout_engines()

    assert client.close_count == 1
    assert updater._transfer_client is None
    assert updater._receiver_descriptors == []
    assert updater._connected is False


@pytest.mark.unit
def test_update_failure_still_releases_rollout_lock(monkeypatch):
    module = load_mooncake_module()
    patch_pp_source_rank(monkeypatch, module)

    class Lock:
        def __init__(self):
            self.release_count = 0
            self.acquire = RemoteMethod(lambda: True)
            self.release = RemoteMethod(self._release)

        def _release(self):
            self.release_count += 1

    lock = Lock()
    engine = types.SimpleNamespace(
        update_weights_from_mooncake=RemoteMethod(lambda _manifest: RuntimeError("publish failed")),
    )
    updater = module.UpdateWeightFromMooncake(
        args=types.SimpleNamespace(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
        transfer_client=module.FakeMooncakeTransferClient(),
    )
    updater._is_pp_src_rank = True
    updater.rollout_engine_lock = lock
    updater.rollout_engines = [engine]
    updater._receiver_descriptors = [
        {"target_name": "engine:18000", "buffers": [{"slot": 0, "addr": 1000, "capacity": 16}]}
    ]
    updater.weight_version = 1

    with pytest.raises(RuntimeError, match="publish failed"):
        updater._update_bucket_weights_from_mooncake([("layer.weight", FakeTensor((4,), "float16", 2))])

    assert lock.release_count == 1


@pytest.mark.unit
@pytest.mark.parametrize("patch_dir", ["latest", "v0.5.12.post1", "v0.5.9"])
def test_sglang_patch_exposes_mooncake_weight_receiver_contract(patch_dir):
    patch_path = Path(__file__).resolve().parents[1] / "docker" / "patch" / patch_dir / "sglang.patch"
    patch_text = patch_path.read_text()

    common_snippets = [
        "class InitMooncakeWeightReceiverReqInput(BaseReq):",
        "class UpdateWeightsFromMooncakeReqInput(BaseReq):",
        "class DestroyMooncakeWeightReceiverReqInput(BaseReq):",
        '@app.post("/init_mooncake_weight_receiver")',
        '@app.post("/update_weights_from_mooncake")',
        '@app.post("/destroy_mooncake_weight_receiver")',
        "self.tp_worker.init_mooncake_weight_receiver",
        "self.model_runner.init_mooncake_weight_receiver",
        "def update_weights_from_mooncake(",
        "def destroy_mooncake_weight_receiver(",
        "buffer_device: Optional[str] = None",
        '"buffer_device": result.buffer_device',
        '"device": buf.device.type',
        "cleanup_ok, cleanup_message = self.destroy_mooncake_weight_receiver()",
        'getattr(buf, "is_cuda", False)',
        'unsupported = "Mooncake weight receiver currently supports only dp_size=1."',
    ]
    version_snippets = {
        "latest": [
            '("init_mooncake_weight_receiver", InitMooncakeWeightReceiverReqOutput)',
            "self.weight_updater.init_mooncake_weight_receiver",
        ],
        "v0.5.12.post1": [
            '("init_mooncake_weight_receiver", InitMooncakeWeightReceiverReqOutput)',
            "self.init_mooncake_weight_receiver_communicator(obj)",
        ],
        "v0.5.9": [
            "self.init_mooncake_weight_receiver_communicator = _Communicator",
            "self.init_mooncake_weight_receiver_communicator.handle_recv",
        ],
    }
    required_snippets = common_snippets + version_snippets[patch_dir]
    missing = [snippet for snippet in required_snippets if snippet not in patch_text]

    assert missing == []
    assert "assert self.server_args.dp_size == 1" not in patch_text
    assert (
        'if ret != 0:\n+                    free_engine = getattr(engine, "freeEngine", None)\n'
        "+                    if callable(free_engine):\n+                        free_engine()\n+                    last_error ="
        in patch_text
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
