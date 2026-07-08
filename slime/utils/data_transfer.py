"""Thin Mooncake rollout data transport: put/get/cleanup."""

import logging
import os
from functools import cache
from typing import Any

from slime.utils.misc import Box

try:
    from mooncake.structured_object_store import FieldSchema, MooncakeBundleTransfer, export_ref, import_ref
    from mooncake.store import MooncakeDistributedStore

    _MOONCAKE_AVAILABLE = True
except ImportError:
    _MOONCAKE_AVAILABLE = False

logger = logging.getLogger(__name__)

_ROLLOUT_FIELD_SCHEMA_SPECS = {
    # rollout.py tensorizes these row-aligned fields before transport.
    "tokens": ("ragged_tensor", None, "non_tensor_batch"),
    "loss_masks": ("ragged_tensor", None, "non_tensor_batch"),
    "rollout_log_probs": ("ragged_tensor", None, "non_tensor_batch"),
    "rollout_top_p_token_ids": ("ragged_tensor", None, "non_tensor_batch"),
    "rollout_top_p_token_offsets": ("ragged_tensor", None, "non_tensor_batch"),
    "teacher_log_probs": ("ragged_tensor", None, "non_tensor_batch"),
    "rollout_routed_experts": ("ragged_tensor", None, "non_tensor_batch"),
    # Row-aligned scalar fields.
    "partition": ("ndarray", "int64", "non_tensor_batch"),
    "response_lengths": ("ndarray", "int64", "non_tensor_batch"),
    "rewards": ("ndarray", "float32", "non_tensor_batch"),
    "truncated": ("ndarray", "int64", "non_tensor_batch"),
    "round_number": ("ndarray", "int64", "non_tensor_batch"),
    "sample_indices": ("ndarray", "int64", "non_tensor_batch"),
    "rollout_ids": ("ndarray", "int64", "non_tensor_batch"),
    "rollout_mask_sums": ("tensor", "float32", "batch"),
    # Optional row-aligned text fields.
    "prompt": ("utf8_ragged", None, "non_tensor_batch"),
    # Metadata fields carried with each DP partition.
    "raw_reward": ("auto", None, "meta_info"),
    "total_lengths": ("auto", None, "meta_info"),
    "global_batch_sizes": ("auto", None, "meta_info"),
    "num_microbatches": ("auto", None, "meta_info"),
    "micro_batch_indices": ("auto", None, "meta_info"),
}

_ROLLOUT_FIELD_SCHEMAS = (
    {
        key: FieldSchema(
            codec=codec,
            nullable=False,
            metadata={"section": section, **({"dtype": dtype} if dtype else {})},
        )
        for key, (codec, dtype, section) in _ROLLOUT_FIELD_SCHEMA_SPECS.items()
    }
    if _MOONCAKE_AVAILABLE
    else {}
)


def check_mooncake_available() -> None:
    """Call during argument parsing to fail fast if mooncake is not installed."""
    if not _MOONCAKE_AVAILABLE:
        raise ImportError(
            "rollout-data-transport='mooncake' requires the mooncake package. "
            "Install it with: pip install mooncake"
        )


def put_mooncake_rollout_data(args: Any, data: dict[str, Any], partition: str) -> Box:
    ref = _mooncake_transfer(args, contribute_segment=True).put_legacy_dict(
        data,
        namespace="slime",
        partition=partition,
        stage="rollout",
        field_schemas=_rollout_field_schemas_for_data(data),
    )
    return Box(export_ref(ref))


def _rollout_field_schemas_for_data(data: dict[str, Any]) -> dict:
    return {key: schema for key, schema in _ROLLOUT_FIELD_SCHEMAS.items() if key in data}


def get_mooncake_rollout_data(args: Any, ref: Box) -> dict[str, Any]:
    return _mooncake_transfer(args, contribute_segment=False).get_legacy_dict(import_ref(ref.inner))


def release_mooncake_rollout_data(args: Any, data: dict[str, Any]) -> None:
    """Release pool-backed buffers after training has fully consumed the data."""
    from mooncake.structured_object_store import MooncakeBundleTransfer

    MooncakeBundleTransfer.release_result(data)


def cleanup_mooncake_rollout_data(args: Any, ref: Box) -> None:
    _mooncake_transfer(args, contribute_segment=False).remove_legacy_dict(import_ref(ref.inner))


def cleanup_mooncake_rollout_refs(args: Any, refs: list[Box]) -> None:
    for ref in refs:
        cleanup_mooncake_rollout_data(args, ref)


def _mooncake_transfer(args: Any, contribute_segment: bool):
    config = _mooncake_store_config(args, contribute_segment=contribute_segment)
    return _cached_mooncake_transfer(tuple(sorted(config.items())))


@cache
def _cached_mooncake_transfer(config_items: tuple[tuple[str, Any], ...]):
    store = MooncakeDistributedStore()
    ret = store.setup(dict(config_items))
    if ret:
        raise RuntimeError(f"Mooncake store setup failed: {ret}")
    return MooncakeBundleTransfer(store, key_prefix="slime-rollout")


def _mooncake_store_config(args: Any, contribute_segment: bool) -> dict[str, Any]:
    mc_kwargs = getattr(args, "mooncake_store_init_kwargs", None) or {}
    global_segment_size = _parse_size(
        mc_kwargs.get("global_segment_size") or os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(8 * 1024**3))
    )
    if not contribute_segment:
        global_segment_size = 0
    return {
        "local_hostname": str(mc_kwargs.get("local_hostname") or _local_hostname()),
        "metadata_server": str(
            mc_kwargs.get("metadata_server") or os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
        ),
        "global_segment_size": global_segment_size,
        "local_buffer_size": _parse_size(
            mc_kwargs.get("local_buffer_size") or os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(32 * 1024**3))
        ),
        "protocol": str(mc_kwargs.get("protocol") or os.getenv("MOONCAKE_PROTOCOL", "rdma")),
        "rdma_devices": str(mc_kwargs.get("device_name") or os.getenv("MOONCAKE_DEVICE", "")),
        "master_server_addr": str(mc_kwargs.get("master_server_address") or os.getenv("MOONCAKE_MASTER", "")),
    }


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    units = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "k": 1024, "m": 1024**2, "g": 1024**3}
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def _local_hostname() -> str:
    value = os.getenv("MOONCAKE_LOCAL_HOSTNAME") or os.getenv("LOCAL_HOSTNAME")
    if value:
        return value
    import ray

    return ray.util.get_node_ip_address()
