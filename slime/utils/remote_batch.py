from __future__ import annotations

import ctypes
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import torch
from tensordict._td import TensorDict

ALLOWED_SETUP_METHODS = {"setup", "setup_dummy"}
_STORE_CACHE: dict[tuple[tuple[str, str], ...], Any] = {}
_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def normalize_store_init_kwargs(store_init_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if store_init_kwargs is None:
        raise ValueError("mooncake_dataproto requires --mooncake-dataproto-store-init-kwargs")
    if not store_init_kwargs:
        return {"setup_method": "setup"}
    setup_method = store_init_kwargs.get("setup_method", "setup")
    if setup_method not in ALLOWED_SETUP_METHODS:
        raise ValueError(f"unsupported Mooncake store setup_method {setup_method!r}; allowed: {sorted(ALLOWED_SETUP_METHODS)}")
    return dict(store_init_kwargs)


def create_mooncake_store(store_init_kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = normalize_store_init_kwargs(store_init_kwargs or {})
    setup_method = kwargs.get("setup_method", "setup")
    if setup_method == "setup_dummy":
        try:
            from mooncake.dataproto_transfer import InMemoryMooncakeStore
        except ImportError:
            pass
        else:
            return InMemoryMooncakeStore()

    from mooncake.store import MooncakeDistributedStore  # type: ignore

    store = MooncakeDistributedStore()
    setup_kwargs = {key: val for key, val in kwargs.items() if key != "setup_method"}
    setup = getattr(store, setup_method)
    try:
        ret = setup(**setup_kwargs)
    except TypeError:
        if setup_method != "setup":
            raise
        ret = setup(_env_store_config() | setup_kwargs)
    if ret != 0:
        raise RuntimeError(f"Mooncake store {setup_method} failed with retcode {ret}")
    return store


def get_cached_mooncake_store(store_init_kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = normalize_store_init_kwargs(store_init_kwargs)
    cache_key = tuple(sorted((key, repr(val)) for key, val in kwargs.items()))
    if cache_key not in _STORE_CACHE:
        _STORE_CACHE[cache_key] = create_mooncake_store(kwargs)
    return _STORE_CACHE[cache_key]


def remove_mooncake_keys(store: Any, keys: list[str]) -> None:
    errors = []
    for key in sorted(set(keys)):
        ret = store.remove(key, True)
        if ret != 0:
            errors.append((key, ret))
    if errors:
        raise RuntimeError(f"Mooncake key cleanup failed: {errors}")


def _env_store_config() -> dict[str, Any]:
    return {
        "local_hostname": os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
        "metadata_server": os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        "global_segment_size": int(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(16 * 1024 * 1024 * 1024))),
        "local_buffer_size": int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(16 * 1024 * 1024 * 1024))),
        "protocol": os.getenv("MOONCAKE_PROTOCOL", "rdma"),
        "rdma_devices": os.getenv("MOONCAKE_DEVICE", ""),
        "master_server_addr": os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051"),
    }


def _import_mooncake_helpers():
    try:
        from mooncake.dataproto_transfer import payload_to_buffer  # type: ignore
        from mooncake.remote_tensor_batch import (  # type: ignore
            RemoteTensorBatch,
            TensorFieldRef,
            normalize_dtype_name,
        )
    except ImportError as exc:
        raise ImportError("Mooncake remote tensor batch helpers are required for mooncake_dataproto transfer") from exc
    return RemoteTensorBatch, TensorFieldRef, normalize_dtype_name, payload_to_buffer


@dataclass
class MooncakeRemoteBatch:
    remote: Any
    store_init_kwargs: dict[str, Any] = field(default_factory=dict)
    keys_to_cleanup: tuple[str, ...] = ()
    use_reusable_buffer: bool = True

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        store: Any,
        prefix: str,
        store_init_kwargs: dict[str, Any] | None = None,
        use_hard_pin: bool = True,
        use_reusable_buffer: bool = True,
    ) -> MooncakeRemoteBatch:
        _validate_prefix(prefix)
        for name in tensors:
            _validate_field_name(name)
        RemoteTensorBatch, TensorFieldRef, normalize_dtype_name, payload_to_buffer = _import_mooncake_helpers()
        fields = {}
        batch_size = None
        config = _hard_pin_config(store) if use_hard_pin else None
        written_keys = []
        try:
            for name, tensor in tensors.items():
                cpu_tensor = tensor.detach().contiguous().cpu()
                if batch_size is None:
                    batch_size = int(cpu_tensor.shape[0])
                elif int(cpu_tensor.shape[0]) != batch_size:
                    raise ValueError(f"tensor {name} batch size {cpu_tensor.shape[0]} != {batch_size}")
                key = f"{prefix}/{name}"
                buffer, owner, _ = payload_to_buffer(cpu_tensor)
                ret = _pub_tensor_from(store, key, buffer, config)
                if ret != 0:
                    raise RuntimeError(f"Mooncake put failed for {key} with retcode {ret}")
                written_keys.append(key)
                fields[name] = TensorFieldRef(
                    key=key,
                    shape=tuple(cpu_tensor.shape),
                    dtype=normalize_dtype_name(cpu_tensor.dtype),
                    data_offset=0,
                )
                del owner
        except Exception:
            remove_mooncake_keys(store, written_keys)
            raise
        if batch_size is None:
            raise ValueError("MooncakeRemoteBatch.from_tensors requires non-empty tensors")
        return cls(
            remote=RemoteTensorBatch(fields=fields, batch_size=batch_size),
            store_init_kwargs=store_init_kwargs or {},
            keys_to_cleanup=tuple(written_keys),
            use_reusable_buffer=use_reusable_buffer,
        )

    def __len__(self) -> int:
        return len(self.remote)

    def keys(self) -> list[str]:
        return self.remote.keys()

    def materialize(self, fields: list[str] | None = None) -> TensorDict:
        store = get_cached_mooncake_store(self.store_init_kwargs)
        try:
            tensors = _materialize_remote_tensors(store, self.remote, fields, self.use_reusable_buffer)
            return TensorDict(source=tensors, batch_size=(len(self),))
        except Exception as exc:
            requested = self.remote.keys() if fields is None else fields
            raise RuntimeError(f"MooncakeRemoteBatch materialize failed for fields={list(requested)}") from exc

    def cleanup(self) -> None:
        if not self.keys_to_cleanup:
            return
        store = get_cached_mooncake_store(self.store_init_kwargs)
        remove_mooncake_keys(store, list(self.keys_to_cleanup))


def _materialize_remote_tensors(
    store: Any,
    remote: Any,
    fields: list[str] | None,
    use_reusable_buffer: bool,
) -> dict[str, torch.Tensor]:
    if use_reusable_buffer:
        return _materialize_remote_tensors_with_pool(store, remote, fields)
    return _materialize_remote_tensors_without_pool(store, remote, fields)


def _materialize_remote_tensors_without_pool(store: Any, remote: Any, fields: list[str] | None) -> dict[str, torch.Tensor]:
    requests = remote.read_requests(fields)
    regions = {request.name: _WritableRegion(bytearray(request.output_nbytes())) for request in requests}
    try:
        for request in requests:
            _materialize_request_into_region(store, request, regions[request.name], register=True)
        return {request.name: _region_to_tensor(regions[request.name], request) for request in requests}
    finally:
        for region in regions.values():
            region.close()


def _materialize_remote_tensors_with_pool(store: Any, remote: Any, fields: list[str] | None) -> dict[str, torch.Tensor]:
    from mooncake.dataproto_transfer import get_registered_buffer_pool

    pool = get_registered_buffer_pool(store)
    requests = remote.read_requests(fields)
    leases = {request.name: pool.acquire(request.output_nbytes()) for request in requests}
    try:
        for request in requests:
            _materialize_request_into_region(store, request, leases[request.name], register=False)
        return {request.name: _region_to_tensor(_lease_region(leases[request.name]), request) for request in requests}
    finally:
        for lease in leases.values():
            lease.release()


def _materialize_request_into_region(store: Any, request: Any, region: Any, register: bool) -> None:
    from mooncake.remote_tensor_batch import normalize_dtype_name

    required_size = request.output_nbytes()
    if required_size == 0:
        return
    if register:
        register_ret = store.register_buffer(region.ptr, region.size)
        if register_ret != 0:
            raise RuntimeError(f"register_buffer failed with retcode {register_ret}")
    try:
        if hasattr(store, "get_tensor_dim_selection_into"):
            ret = store.get_tensor_dim_selection_into(
                request.ref.key,
                region.ptr,
                required_size,
                list(request.ref.shape),
                normalize_dtype_name(request.ref.dtype),
                request.dim,
                request.store_selections(),
                request.ref.data_offset,
            )
            if ret < 0:
                raise RuntimeError(f"get_tensor_dim_selection_into failed with retcode {ret}")
        elif request.store_selections() or request.ref.data_offset:
            raise RuntimeError("store.get_tensor_dim_selection_into is required for selected remote tensors")
        else:
            ret = store.get_into(request.ref.key, region.ptr, required_size)
            if ret != required_size:
                raise RuntimeError(f"get_into failed for {request.ref.key}: expected {required_size}, got {ret}")
    finally:
        if register:
            unregister_ret = store.unregister_buffer(region.ptr)
            if unregister_ret != 0:
                raise RuntimeError(f"unregister_buffer failed with retcode {unregister_ret}")


def _lease_region(lease: Any) -> Any:
    return lease.view_region() if hasattr(lease, "view_region") else lease


def _region_to_tensor(region: Any, request: Any) -> torch.Tensor:
    dtype_name = str(request.ref.dtype).removeprefix("torch.").lower()
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unsupported Mooncake tensor dtype: {request.ref.dtype!r}")
    torch_dtype = getattr(torch, dtype_name)
    shape = request.output_shape()
    count = int(np.prod(shape, dtype=np.int64))
    return torch.frombuffer(region.buffer, dtype=torch_dtype, count=count).reshape(shape).clone()


def _validate_prefix(prefix: str) -> None:
    if not prefix or len(prefix) > 256 or ".." in prefix or any(ord(ch) < 32 for ch in prefix):
        raise ValueError(f"invalid Mooncake key prefix: {prefix!r}")


def _validate_field_name(name: str) -> None:
    if _FIELD_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid Mooncake tensor field name: {name!r}")


def _pub_tensor_from(store: Any, key: str, buffer: memoryview, config: Any) -> int:
    region = _WritableRegion(buffer)
    try:
        register_ret = store.register_buffer(region.ptr, region.size)
        if register_ret != 0:
            raise RuntimeError(f"register_buffer failed for Mooncake put_from key={key} retcode={register_ret}")
        try:
            return store.put_from(key=key, buffer_ptr=region.ptr, size=region.size, config=config)
        finally:
            unregister_ret = store.unregister_buffer(region.ptr)
            if unregister_ret != 0:
                raise RuntimeError(f"unregister_buffer failed for Mooncake put_from key={key} retcode={unregister_ret}")
    finally:
        region.close()


class _WritableRegion:
    def __init__(self, buffer: Any) -> None:
        self.buffer = buffer
        self.view = memoryview(buffer)
        if self.view.readonly:
            self.view.release()
            raise ValueError("buffer must be writable")
        if self.view.format != "B":
            cast_view = self.view.cast("B")
            self.view.release()
            self.view = cast_view
        self.c_buffer = (ctypes.c_ubyte * self.view.nbytes).from_buffer(self.view)
        self.ptr = ctypes.addressof(self.c_buffer)
        self.size = self.view.nbytes

    def close(self) -> None:
        self.c_buffer = None
        self.view.release()


def _hard_pin_config(store: Any) -> Any:
    try:
        from mooncake.store import ReplicateConfig  # type: ignore
    except ImportError as exc:
        raise ImportError("Mooncake ReplicateConfig is required for hard-pin transfer") from exc
    config = ReplicateConfig()
    config.preferred_segments = [store.get_hostname()]
    config.with_hard_pin = True
    return config
