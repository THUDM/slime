from __future__ import annotations

import logging
import socket
import sys
import time
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import ray
    from megatron.core import mpu

    from .update_weight_from_distributed import UpdateWeightFromDistributed
except ImportError:  # pragma: no cover - keeps pure helpers importable in minimal unit environments.
    ray = None
    UpdateWeightFromDistributed = object

    class _MissingMPU:
        def get_data_parallel_rank(self, with_context_parallel: bool = True) -> int:
            raise RuntimeError("megatron.core.mpu is required for Mooncake weight updates.")

        def get_tensor_model_parallel_rank(self) -> int:
            raise RuntimeError("megatron.core.mpu is required for Mooncake weight updates.")

        def get_pipeline_model_parallel_rank(self) -> int:
            raise RuntimeError("megatron.core.mpu is required for Mooncake weight updates.")

    mpu = _MissingMPU()


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _find_buffer(descriptor: Mapping[str, Any], buffer_slot: int) -> Mapping[str, Any]:
    for buffer in descriptor.get("buffers", []):
        if int(buffer.get("slot", 0)) == buffer_slot:
            return buffer
    raise ValueError(f"Mooncake receiver descriptor has no buffer slot {buffer_slot}: {descriptor!r}")


def _format_target_name(server_name: str, rpc_port: int | str) -> str:
    if ":" in server_name and not server_name.startswith("["):
        return f"[{server_name}]:{rpc_port}"
    return f"{server_name}:{rpc_port}"


def _target_name_from_descriptor(descriptor: Mapping[str, Any]) -> str:
    target_name = descriptor.get("target_name")
    if target_name:
        return str(target_name)
    server_name = descriptor.get("server_name")
    rpc_port = descriptor.get("rpc_port")
    if server_name and rpc_port is not None:
        return _format_target_name(str(server_name), rpc_port)
    raise ValueError(
        "Mooncake receiver descriptor must include target_name, or both server_name and rpc_port: " f"{descriptor!r}"
    )


def _serialize_delta(delta: Any) -> dict[str, Any]:
    return {
        "encoding": delta.encoding.value,
        "params": [asdict(param) for param in delta.params],
        "checksum": delta.checksum,
    }


def _validate_single_gpu_rollout_engines(
    *,
    args: Namespace,
    engine_gpu_counts: Sequence[int] | None,
    rollout_engine_count: int,
) -> None:
    if engine_gpu_counts is None:
        engine_gpu_counts = [getattr(args, "rollout_num_gpus_per_engine", 1)] * rollout_engine_count
    if any(int(count) != 1 for count in engine_gpu_counts):
        raise NotImplementedError(
            "--update-weight-transport=mooncake currently supports only one GPU per rollout engine. "
            "Multi-TP rollout engines need one Mooncake receiver descriptor per TP rank."
        )


def default_mooncake_local_server_name(
    *,
    metadata_server: str,
    pp_rank: int,
    host_getter=socket.gethostname,
) -> str:
    if metadata_server == "P2PHANDSHAKE":
        hostname = host_getter()
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return hostname
    return f"slime-train-pp{pp_rank}"


def build_mooncake_manifest(
    *,
    descriptor: Mapping[str, Any],
    named_tensors: Sequence[tuple[str, Any]],
    weight_version: int,
    load_format: str | None,
    buffer_slot: int = 0,
    delta: Any | None = None,
) -> dict[str, Any]:
    buffer = _find_buffer(descriptor, buffer_slot)
    capacity = int(buffer["capacity"])
    remote_addr = int(buffer["addr"])
    buffer_device = str(buffer.get("device", descriptor.get("buffer_device", "cuda")))
    target_name = _target_name_from_descriptor(descriptor)

    offset = 0
    entries = []
    for name, tensor in named_tensors:
        nbytes = _tensor_nbytes(tensor)
        entries.append(
            {
                "name": name,
                "dtype": _dtype_name(tensor.dtype),
                "shape": list(tensor.shape),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        offset += nbytes

    if offset > capacity:
        raise ValueError(
            f"Weight bucket of {offset} bytes exceeds Mooncake receiver buffer slot "
            f"{buffer_slot} capacity {capacity} bytes for {target_name}."
        )

    manifest: dict[str, Any] = {
        "weight_version": str(weight_version),
        "target_name": target_name,
        "buffer_slot": buffer_slot,
        "remote_addr": remote_addr,
        "buffer_device": buffer_device,
        "nbytes": offset,
        "tensors": entries,
    }
    if load_format is not None:
        manifest["load_format"] = load_format
    if delta is not None:
        manifest["delta"] = _serialize_delta(delta)
    return manifest


@dataclass(frozen=True)
class MooncakeTransferCall:
    target_name: str
    remote_addr: int
    nbytes: int
    tensor_names: tuple[str, ...]


class FakeMooncakeTransferClient:
    def __init__(self) -> None:
        self.calls: list[MooncakeTransferCall] = []

    def write_bucket(
        self,
        *,
        receiver_descriptors: Sequence[Mapping[str, Any]],
        named_tensors: Sequence[tuple[str, Any]],
        weight_version: int,
        load_format: str | None,
        delta: Any | None,
        buffer_slot: int = 0,
    ) -> list[dict[str, Any]]:
        manifests = []
        for descriptor in receiver_descriptors:
            manifest = build_mooncake_manifest(
                descriptor=descriptor,
                named_tensors=named_tensors,
                weight_version=weight_version,
                load_format=load_format,
                buffer_slot=buffer_slot,
                delta=delta,
            )
            self.calls.append(
                MooncakeTransferCall(
                    target_name=manifest["target_name"],
                    remote_addr=manifest["remote_addr"],
                    nbytes=manifest["nbytes"],
                    tensor_names=tuple(entry["name"] for entry in manifest["tensors"]),
                )
            )
            manifests.append(manifest)
        return manifests

    def close(self) -> None:
        return None


class MooncakeTransferClient:
    def __init__(
        self,
        *,
        local_server_name: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> None:
        from mooncake.engine import TransferEngine

        self.engine = TransferEngine()
        ret = self.engine.initialize(local_server_name, metadata_server, protocol, device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed with code {ret}")
        self.local_server_name = local_server_name

    @classmethod
    def from_args(cls, args: Namespace, *, pp_rank: int) -> MooncakeTransferClient:
        return cls(
            local_server_name=default_mooncake_local_server_name(
                metadata_server=args.mooncake_metadata_server,
                pp_rank=pp_rank,
            ),
            metadata_server=args.mooncake_metadata_server,
            protocol=args.mooncake_protocol,
            device_name=args.mooncake_device_name or "",
        )

    def write_bucket(
        self,
        *,
        receiver_descriptors: Sequence[Mapping[str, Any]],
        named_tensors: Sequence[tuple[str, Any]],
        weight_version: int,
        load_format: str | None,
        delta: Any | None,
        buffer_slot: int = 0,
    ) -> list[dict[str, Any]]:
        manifests = [
            build_mooncake_manifest(
                descriptor=descriptor,
                named_tensors=named_tensors,
                weight_version=weight_version,
                load_format=load_format,
                buffer_slot=buffer_slot,
                delta=delta,
            )
            for descriptor in receiver_descriptors
        ]
        buffer_devices = {str(manifest.get("buffer_device", "cuda")) for manifest in manifests}
        if buffer_devices not in ({"cuda"}, set()):
            raise RuntimeError(f"Mooncake weight transport requires CUDA receiver buffers: {buffer_devices}")

        registered_ptrs: list[int] = []
        try:
            for _name, tensor in named_tensors:
                if not tensor.is_cuda:
                    raise TypeError("Mooncake GPU-direct weight transport requires CUDA tensors.")
                if not tensor.is_contiguous():
                    raise ValueError("Mooncake weight transport requires contiguous tensors.")
                ptr = int(tensor.data_ptr())
                ret = self.engine.register_memory(ptr, _tensor_nbytes(tensor))
                if ret != 0:
                    raise RuntimeError(f"Mooncake register_memory failed with code {ret}")
                registered_ptrs.append(ptr)

            for manifest in manifests:
                self._write_manifest(manifest, named_tensors)
            return manifests
        finally:
            unregister_errors = []
            for ptr in registered_ptrs:
                ret = self.engine.unregister_memory(ptr)
                if ret != 0:
                    unregister_errors.append(f"unregister_memory({ptr})={ret}")
            if unregister_errors:
                message = "Mooncake unregister_memory failed: " + "; ".join(unregister_errors)
                if sys.exc_info()[0] is None:
                    raise RuntimeError(message)
                logger.warning(message)

    def _write_manifest(
        self,
        manifest: Mapping[str, Any],
        named_tensors: Sequence[tuple[str, Any]],
    ) -> None:
        target_name = manifest["target_name"]
        remote_base = int(manifest["remote_addr"])
        tensor_by_name = dict(named_tensors)
        import torch

        stream_by_device: dict[Any, Any] = {}

        for entry in manifest["tensors"]:
            tensor = tensor_by_name[entry["name"]]
            device = tensor.device
            stream = stream_by_device.get(device)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                stream_by_device[device] = stream

            current_stream = torch.cuda.current_stream(device)
            event = current_stream.record_event()
            stream.wait_event(event)
            ret = self.engine.transfer_write_on_cuda(
                target_name,
                int(tensor.data_ptr()),
                remote_base + int(entry["offset"]),
                int(entry["nbytes"]),
                stream.cuda_stream,
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake transfer_write_on_cuda failed with code {ret}")

        for stream in stream_by_device.values():
            stream.synchronize()

    def close(self) -> None:
        free_engine = getattr(self.engine, "freeEngine", None)
        if callable(free_engine):
            free_engine()


class UpdateWeightFromMooncake(UpdateWeightFromDistributed):
    def __init__(
        self,
        args: Namespace,
        model: Sequence[Any],
        weights_getter,
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        transfer_client: MooncakeTransferClient | FakeMooncakeTransferClient | None = None,
    ) -> None:
        if UpdateWeightFromDistributed is object:
            self.args = args
            self.model = model
            self.model_name = model_name
            self.quantization_config = quantization_config
            self.weight_version = 0
            self.update_weight_metrics = {}
        else:
            super().__init__(
                args,
                model,
                weights_getter,
                model_name=model_name,
                quantization_config=quantization_config,
            )
        self._transfer_client = transfer_client
        self._receiver_descriptors: list[Mapping[str, Any]] = []
        self._buffer_slot = 0
        self._connected = False

    def _destroy_mooncake_receivers(self, rollout_engines: Sequence[Any] | None = None) -> Exception | None:
        if ray is None:
            return None
        engines = rollout_engines if rollout_engines is not None else getattr(self, "rollout_engines", None)
        if not getattr(self, "_is_pp_src_rank", False) or not engines:
            return None
        try:
            ray.get([engine.destroy_mooncake_weight_receiver.remote() for engine in engines])
        except Exception as exc:
            return exc
        return None

    def _close_transfer_client(self) -> Exception | None:
        if self._transfer_client is None:
            return None
        try:
            self._transfer_client.close()
        except Exception as exc:
            return exc
        finally:
            self._transfer_client = None
        return None

    def _reset_mooncake_state(self, *, raise_destroy_error: bool) -> None:
        destroy_error = self._destroy_mooncake_receivers()
        close_error = self._close_transfer_client()
        self._receiver_descriptors = []
        self._connected = False
        if raise_destroy_error and destroy_error is not None:
            raise destroy_error
        if raise_destroy_error and close_error is not None:
            raise close_error

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[Any],
        rollout_engine_lock: Any,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._group_name = f"slime-mooncake-pp_{pp_rank}"
        _validate_single_gpu_rollout_engines(
            args=self.args,
            engine_gpu_counts=engine_gpu_counts,
            rollout_engine_count=len(rollout_engines),
        )

        if not self._is_pp_src_rank:
            return

        if self._connected:
            self._reset_mooncake_state(raise_destroy_error=False)

        config = {
            "metadata_server": self.args.mooncake_metadata_server,
            "protocol": self.args.mooncake_protocol,
            "device_name": self.args.mooncake_device_name or "",
            "buffer_size": getattr(self.args, "mooncake_buffer_size", None) or self.args.update_weight_buffer_size,
            "buffer_count": self.args.mooncake_buffer_count,
            "rpc_port_base": getattr(self.args, "mooncake_rpc_port_base", None),
            "group_name": self._group_name,
        }
        try:
            self._receiver_descriptors = ray.get(
                [engine.init_mooncake_weight_receiver.remote(config) for engine in rollout_engines]
            )
            if self._transfer_client is None:
                self._transfer_client = MooncakeTransferClient.from_args(self.args, pp_rank=pp_rank)
            self._connected = True
        except Exception:
            self._destroy_mooncake_receivers(rollout_engines)
            self._close_transfer_client()
            self._receiver_descriptors = []
            raise

    def disconnect_rollout_engines(self) -> None:
        self._reset_mooncake_state(raise_destroy_error=True)

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, Any]],
        pbar=None,
        load_format: str | None = None,
        delta: Any | None = None,
    ) -> None:
        self._update_bucket_weights_from_mooncake(
            converted_named_tensors,
            pbar=pbar,
            load_format=load_format,
            delta=delta,
        )

    def _update_bucket_weights_from_mooncake(
        self,
        converted_named_tensors: list[tuple[str, Any]],
        pbar=None,
        load_format: str | None = None,
        delta: Any | None = None,
    ) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or not converted_named_tensors:
            converted_named_tensors.clear()
            return
        if self._transfer_client is None:
            raise RuntimeError("Mooncake transfer client is not initialized.")

        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            manifests = self._transfer_client.write_bucket(
                receiver_descriptors=self._receiver_descriptors,
                named_tensors=converted_named_tensors,
                weight_version=self.weight_version,
                load_format=load_format,
                delta=delta,
                buffer_slot=self._buffer_slot,
            )
            refs = [
                engine.update_weights_from_mooncake.remote(manifest)
                for engine, manifest in zip(self.rollout_engines, manifests, strict=True)
            ]
            ray.get(refs)
            total_bytes = sum(manifest["nbytes"] for manifest in manifests)
            self.update_weight_metrics["mooncake/write_bucket_count"] = (
                self.update_weight_metrics.get("mooncake/write_bucket_count", 0) + 1
            )
            self.update_weight_metrics["mooncake/write_bytes"] = (
                self.update_weight_metrics.get("mooncake/write_bytes", 0) + total_bytes
            )
            converted_named_tensors.clear()
            if pbar is not None:
                pbar.update(1)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
