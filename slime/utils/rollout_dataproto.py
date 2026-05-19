from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from tensordict._td import TensorDict
from torch.nn.utils.rnn import pad_sequence

from slime.utils.remote_batch import (
    MooncakeRemoteBatch,
    get_cached_mooncake_store,
    normalize_store_init_kwargs,
    remove_mooncake_keys,
)

REMOTE_TENSOR_KEYS = ("tokens", "loss_masks")
PARTITIONED_KEYS = (
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "round_number",
    "sample_indices",
    "rollout_log_probs",
    "rollout_routed_experts",
    "prompt",
    "teacher_log_probs",
)
GLOBAL_KEYS = ("raw_reward", "total_lengths")


class RemoteBatch(Protocol):
    def __len__(self) -> int: ...

    def keys(self) -> list[str]: ...

    def materialize(self, fields: list[str] | None = None) -> TensorDict: ...


@dataclass
class DataProto:
    batch: TensorDict | None = None
    non_tensor_batch: dict[str, np.ndarray] = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)
    remote_batch: RemoteBatch | None = None

    def __post_init__(self):
        self.check_consistency()

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch:
            return len(next(iter(self.non_tensor_batch.values())))
        if self.remote_batch is not None:
            return len(self.remote_batch)
        return 0

    def __getstate__(self):
        buffer = io.BytesIO()
        batch = self.batch.contiguous().consolidate() if self.batch is not None else None
        torch.save(batch, buffer)
        return buffer.getvalue(), self.non_tensor_batch, self.meta_info, self.remote_batch

    def __setstate__(self, data):
        batch_bytes, self.non_tensor_batch, self.meta_info, self.remote_batch = data
        self.batch = torch.load(io.BytesIO(batch_bytes), weights_only=False, map_location="cpu")

    def check_consistency(self):
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"
        if self.non_tensor_batch:
            batch_size = len(self)
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), f"non_tensor_batch[{key}] must be np.ndarray, got {type(val)}"
                assert val.shape[0] == batch_size, f"key {key} length {val.shape[0]} != batch size {batch_size}"
        if self.batch is not None and self.remote_batch is not None:
            assert len(self.batch) == len(self.remote_batch), "local and remote batch sizes must match"
        if self.non_tensor_batch and self.remote_batch is not None:
            assert len(next(iter(self.non_tensor_batch.values()))) == len(self.remote_batch)

    @classmethod
    def from_dict(
        cls,
        tensors: dict[str, torch.Tensor] | None = None,
        non_tensors: dict[str, Any] | None = None,
        meta_info: dict | None = None,
        num_batch_dims: int = 1,
    ):
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None"
        tensors = tensors or {}
        non_tensors = non_tensors or {}
        meta_info = meta_info or {}

        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            current_batch = tuple(tensor.shape[:num_batch_dims])
            if batch_size is None:
                batch_size = current_batch
                pivot_key = key
            else:
                assert current_batch == batch_size, (
                    f"Not all tensors have the same batch size. {pivot_key} has {batch_size}, "
                    f"{key} has {current_batch}"
                )

        normalized_non_tensors = {}
        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                val = np.array(val, dtype=object)
            normalized_non_tensors[key] = val
            if batch_size is None:
                batch_size = (val.shape[0],)
            else:
                assert val.shape[0] == batch_size[0], (
                    f"non_tensor {key} length {val.shape[0]} != batch size {batch_size[0]}"
                )

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        return cls(batch=tensor_dict, non_tensor_batch=normalized_non_tensors, meta_info=meta_info)

    @classmethod
    def from_remote(
        cls,
        remote_batch: RemoteBatch,
        batch: TensorDict | None = None,
        non_tensors: dict[str, np.ndarray] | None = None,
        meta_info: dict | None = None,
    ):
        return cls(
            batch=batch,
            non_tensor_batch=non_tensors or {},
            meta_info=meta_info or {},
            remote_batch=remote_batch,
        )

    def materialize_remote_batch(self):
        if self.remote_batch is None:
            return self
        fetched = self.remote_batch.materialize()
        if self.batch is not None:
            assert self.batch.batch_size == fetched.batch_size, (
                f"TensorDict batch size mismatch: {self.batch.batch_size} != {fetched.batch_size}"
            )
            for key, val in fetched.items():
                assert key not in self.batch.keys() or self.batch[key].equal(val), (
                    f"{key} exists in both TensorDicts with different values"
                )
                self.batch[key] = val
        else:
            self.batch = fetched
        self.remote_batch = None
        self.check_consistency()
        return self


def split_rollout_data_by_dp_dataproto(
    args: Any,
    data: dict,
    dp_size: int,
    partitions: list,
    dynamic_global_batch_size: int | None = None,
) -> list[DataProto]:
    if len(partitions) != dp_size:
        raise ValueError(f"expected {dp_size} partitions, got {len(partitions)}")
    return _split_rollout_data_by_dp_remote_batch(args, data, partitions, dynamic_global_batch_size)


def _split_rollout_data_by_dp_remote_batch(
    args: Any,
    data: dict,
    partitions: list,
    dynamic_global_batch_size: int | None = None,
) -> list[DataProto]:
    store_init_kwargs = _store_init_kwargs(args)
    store = get_cached_mooncake_store(store_init_kwargs)
    transfer_id = uuid.uuid4().hex
    refs = []
    try:
        for dp_rank, partition in enumerate(partitions):
            indices = [int(idx) for idx in partition]
            shard = _slice_partitioned_data(data, indices)
            shard["partition"] = np.asarray(indices, dtype=np.int64)
            meta_info = {key: data[key] for key in GLOBAL_KEYS if key in data}
            if dynamic_global_batch_size is not None:
                meta_info["dynamic_global_batch_size"] = dynamic_global_batch_size

            remote_tensors, remote_lengths = _extract_remote_tensors(shard)
            meta_info.update(remote_lengths)
            remote_batch = None
            if remote_tensors:
                remote_batch = MooncakeRemoteBatch.from_tensors(
                    remote_tensors,
                    store,
                    prefix=f"slime-rollout/{transfer_id}/dp{dp_rank}",
                    store_init_kwargs=store_init_kwargs,
                    use_hard_pin=getattr(args, "mooncake_dataproto_hard_pin", True),
                )
                _attach_cleanup_info(meta_info, remote_batch, store_init_kwargs)

            try:
                proto = (
                    DataProto.from_remote(remote_batch, non_tensors=_dict_to_non_tensors(shard), meta_info=meta_info)
                    if remote_batch is not None
                    else DataProto.from_dict(non_tensors=shard, meta_info=meta_info)
                )
            except Exception:
                if remote_batch is not None:
                    remote_batch.cleanup()
                raise
            refs.append(proto)
    except Exception:
        cleanup_dataproto_refs(refs)
        raise
    return refs


def maybe_cleanup_dataproto_refs(args: Any, refs: list[DataProto], suppress_errors: bool = False) -> None:
    if getattr(args, "transfer_backend", "ray") != "mooncake_dataproto":
        return
    if not suppress_errors:
        cleanup_dataproto_refs(refs)
        return
    try:
        cleanup_dataproto_refs(refs)
    except Exception:
        return


def cleanup_dataproto_refs(refs: list[DataProto]) -> None:
    keys = set()
    store_init_kwargs = None
    for proto in refs:
        keys.update(proto.meta_info.get("mooncake_cleanup_keys", []))
        if store_init_kwargs is None and "mooncake_cleanup_store_kwargs" in proto.meta_info:
            store_init_kwargs = dict(proto.meta_info["mooncake_cleanup_store_kwargs"])
    if not keys or store_init_kwargs is None:
        return
    store = get_cached_mooncake_store(store_init_kwargs)
    remove_mooncake_keys(store, sorted(keys))


def _attach_cleanup_info(
    meta_info: dict,
    remote_batch: MooncakeRemoteBatch,
    store_init_kwargs: dict[str, Any],
) -> None:
    meta_info["mooncake_cleanup_keys"] = list(remote_batch.keys_to_cleanup)
    meta_info["mooncake_cleanup_store_kwargs"] = dict(store_init_kwargs)


def dataproto_to_rollout_data(proto: DataProto, preserve_remote_tensors: bool = True) -> dict:
    """Materialize a transfer DataProto into slime's legacy rollout dict."""
    if proto.remote_batch is not None:
        proto.materialize_remote_batch()
    rollout_data = {key: val.tolist() for key, val in proto.non_tensor_batch.items()}
    rollout_data.update({key: val for key, val in proto.meta_info.items() if not key.startswith("mooncake_cleanup_")})
    if proto.batch is not None:
        for key, tensor in proto.batch.items():
            lengths = proto.meta_info.get(f"{key}_lengths")
            if preserve_remote_tensors and key in REMOTE_TENSOR_KEYS:
                rollout_data[key] = _tensor_to_row_tensors(tensor, lengths)
            else:
                rollout_data[key] = _tensor_to_list(tensor, lengths)
    return rollout_data


def _slice_partitioned_data(data: dict, indices: list[int]) -> dict:
    shard = {}
    for key in PARTITIONED_KEYS:
        if key in data:
            shard[key] = [data[key][idx] for idx in indices]
    return shard


def _extract_remote_tensors(shard: dict) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    tensors = {}
    lengths = {}
    for key in REMOTE_TENSOR_KEYS:
        if key not in shard:
            continue
        values = shard.pop(key)
        tensor, field_lengths = _list_to_padded_tensor(values, torch.long if key == "tokens" else torch.int)
        tensors[key] = tensor
        lengths[f"{key}_lengths"] = field_lengths
    return tensors, lengths


def _list_to_padded_tensor(values: list, dtype: torch.dtype) -> tuple[torch.Tensor, list[int]]:
    if not values:
        return torch.empty((0, 0), dtype=dtype), []
    tensors = [torch.as_tensor(value, dtype=dtype).reshape(-1) for value in values]
    lengths = [int(tensor.numel()) for tensor in tensors]
    return pad_sequence(tensors, batch_first=True, padding_value=0), lengths


def _tensor_to_row_tensors(tensor: torch.Tensor, lengths: list[int] | None = None) -> list[torch.Tensor]:
    if tensor.ndim == 2:
        if lengths is not None:
            return [tensor[idx, : int(length)] for idx, length in zip(range(tensor.shape[0]), lengths, strict=True)]
        return [tensor[idx] for idx in range(tensor.shape[0])]
    return [tensor[idx] for idx in range(tensor.shape[0])]


def _tensor_to_list(tensor: torch.Tensor, lengths: list[int] | None = None) -> list:
    if tensor.ndim == 2:
        rows = tensor.cpu().tolist()
        if lengths is not None:
            return [row[: int(length)] for row, length in zip(rows, lengths, strict=True)]
        return rows
    return tensor.cpu().numpy().tolist()


def _dict_to_non_tensors(data: dict) -> dict[str, np.ndarray]:
    return {key: val if isinstance(val, np.ndarray) else np.asarray(val, dtype=_infer_numpy_dtype(val)) for key, val in data.items()}


def _infer_numpy_dtype(val: Any) -> Any:
    if isinstance(val, list) and all(isinstance(item, (bool, int, float, np.number)) for item in val):
        return None
    return object


def _store_init_kwargs(args: Any) -> dict[str, Any]:
    kwargs = getattr(args, "mooncake_dataproto_store_init_kwargs", None)
    return normalize_store_init_kwargs(kwargs)
