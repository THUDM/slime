"""Thin Mooncake rollout data transport: put/get/cleanup."""

import logging
import os
from functools import cache
from typing import Any

from slime.utils.misc import Box

try:
    from mooncake.structured_object_store import (
        FieldSchema,
        MooncakeBundleTransfer,
        export_ref,
        import_ref,
    )
    from mooncake.store import MooncakeDistributedStore

    _MOONCAKE_AVAILABLE = True
except ImportError:
    _MOONCAKE_AVAILABLE = False

logger = logging.getLogger(__name__)


def check_mooncake_available() -> None:
    """Call during argument parsing to fail fast if mooncake is not installed."""
    if not _MOONCAKE_AVAILABLE:
        raise ImportError(
            "rollout-data-transport='mooncake' requires the mooncake package. "
            "Install it with: pip install mooncake"
        )


def put_mooncake_rollout_data(args: Any, data: dict[str, Any], partition: str) -> Box:
    ref = _mooncake_transfer(args).put_legacy_dict(
        data,
        namespace="slime",
        partition=partition,
        stage="rollout",
        field_schemas=_rollout_field_schemas(),
    )
    return Box(export_ref(ref))


@cache
def _rollout_field_schemas() -> dict:
    from slime.ray.rollout import _ROLLOUT_DATA_TENSOR_DTYPES

    ragged = FieldSchema(codec="typed_ragged", nullable=False)
    return {k: ragged for k in _ROLLOUT_DATA_TENSOR_DTYPES}


def get_mooncake_rollout_data(args: Any, ref: Box) -> dict[str, Any]:
    transfer = _mooncake_transfer(args)
    result = transfer.get_legacy_dict(import_ref(ref.inner))
    transfer.release_result(result)
    return result


def cleanup_mooncake_rollout_data(args: Any, ref: Box) -> None:
    _mooncake_transfer(args).remove_legacy_dict(import_ref(ref.inner))


def cleanup_mooncake_rollout_refs(args: Any, refs: list[Box] | None) -> None:
    if getattr(args, "rollout_data_transport", "object-store") != "mooncake" or refs is None:
        return
    for ref in refs:
        cleanup_mooncake_rollout_data(args, ref)


@cache
def _mooncake_transfer(args: Any):
    store = MooncakeDistributedStore()
    mc_kwargs = getattr(args, "mooncake_store_init_kwargs", None) or {}
    ret = store.setup(
        {
            "local_hostname": mc_kwargs.get("local_hostname") or _local_hostname(),
            "metadata_server": mc_kwargs.get("metadata_server") or os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE"),
            "global_segment_size": int(
                mc_kwargs.get("global_segment_size") or os.getenv("MC_SEGMENT_SIZE", str(8 * 1024**3))
            ),
            "local_buffer_size": int(
                mc_kwargs.get("local_buffer_size") or os.getenv("MC_BUFFER_SIZE", str(32 * 1024**3))
            ),
            "protocol": mc_kwargs.get("protocol") or os.getenv("MC_PROTOCOL", "rdma"),
            "rdma_devices": mc_kwargs.get("rdma_devices") or os.getenv("MC_DEVICE", ""),
            "master_server_addr": mc_kwargs.get("master_server_addr") or os.getenv("MC_MASTER_SERVER", ""),
        }
    )
    if ret:
        raise RuntimeError(f"Mooncake store setup failed: {ret}")
    return MooncakeBundleTransfer(store, key_prefix="slime-rollout")


def _local_hostname() -> str:
    value = os.getenv("MC_LOCAL_HOSTNAME") or os.getenv("LOCAL_HOSTNAME")
    if value:
        return value
    import ray

    return ray.util.get_node_ip_address()
