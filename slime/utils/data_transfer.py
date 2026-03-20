import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class DataTransferBackend(ABC):
    """Abstract base class for data transfer backends."""

    @abstractmethod
    def put(self, data: Any) -> Any:
        """
        Store data and return a handle/key.
        """
        pass

    @abstractmethod
    def get(self, handle: Any) -> Any:
        """
        Retrieve data using the handle/key.
        """
        pass

    def cleanup(self, handle: Any):  # noqa: B027
        """
        Clean up data associated with the handle (optional).
        """
        pass


class RayDataTransfer(DataTransferBackend):
    """Default data transfer using Ray Object Store."""

    def put(self, data: Any) -> Any:
        import ray

        from slime.utils.misc import Box

        return Box(ray.put(data))

    def get(self, handle: Any) -> Any:
        import ray

        from slime.utils.misc import Box

        if isinstance(handle, Box):
            return ray.get(handle.inner)
        return ray.get(handle)


DEFAULT_MOUNT_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB (larger for benchmark/training)
DEFAULT_LOCAL_BUFFER_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB


def _parse_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError("Invalid segment size: missing number before 'gb'")
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    mount_segment_size: int  # Segment to mount (bytes). 0 = no mount.
    local_buffer_size: int
    protocol: str
    device_name: str | None
    master_server_address: str

    @staticmethod
    def load_from_env(overrides: dict | None = None) -> "MooncakeStoreConfig":
        """Load config from environment variables.

        Required: MOONCAKE_MASTER
        Optional: MOONCAKE_PROTOCOL (default tcp), MOONCAKE_DEVICE,
        MOONCAKE_TE_META_DATA_SERVER (default P2PHANDSHAKE),
        MOONCAKE_MOUNT_SEGMENT_SIZE or MOONCAKE_GLOBAL_SEGMENT_SIZE (default 4 GiB, 0 = no mount),
        MOONCAKE_LOCAL_BUFFER_SIZE (default 2 GiB).
        Set MOONCAKE_PROTOCOL=rdma for best performance (requires InfiniBand/RoCE).
        """
        # MC_STORE_MEMCPY=0 for cross-node RDMA: LOCAL_MEMCPY path can SIGSEGV when
        # Put buffers are not in Client's mounted segment. Use TRANSFER_ENGINE (RDMA).
        os.environ.setdefault("MC_STORE_MEMCPY", "0")
        if not os.getenv("MOONCAKE_MASTER"):
            raise ValueError(
                "Neither the environment variable 'MOONCAKE_CONFIG_PATH' nor 'MOONCAKE_MASTER' is set."
            )
        local_hostname = os.getenv("MOONCAKE_LOCAL_HOSTNAME", "")
        if not local_hostname or local_hostname in ("localhost", "127.0.0.1"):
            try:
                import ray
                if ray.is_initialized():
                    local_hostname = ray.util.get_node_ip_address()
                else:
                    import socket
                    local_hostname = socket.gethostbyname(socket.gethostname())
            except Exception:
                local_hostname = "127.0.0.1"

        overrides = overrides or {}
        mount_sz = overrides.get("mount_segment_size")
        if mount_sz is None:
            env_val = os.getenv("MOONCAKE_MOUNT_SEGMENT_SIZE") or os.getenv(
                "MOONCAKE_GLOBAL_SEGMENT_SIZE"
            )
            mount_sz = (
                _parse_segment_size(env_val)
                if (env_val is not None and env_val.strip())
                else DEFAULT_MOUNT_SEGMENT_SIZE
            )
        else:
            mount_sz = int(mount_sz)

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
            mount_segment_size=mount_sz,
            local_buffer_size=_parse_segment_size(
                os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),  # use "rdma" for RDMA
            device_name=os.getenv("MOONCAKE_DEVICE", ""),
            master_server_address=os.getenv("MOONCAKE_MASTER"),
        )


def get_data_transfer_backend(args):
    """Factory function to get the appropriate backend."""
    backend_name = getattr(args, "transfer_backend", "ray")
    if backend_name in ("mooncake", "mooncake_legacy"):
        from slime.utils.rollout_hybrid_transfer import MooncakeHybridRolloutTransfer

        use_legacy = backend_name == "mooncake_legacy"
        mount_segment_size = getattr(args, "mooncake_mount_segment_size", None)
        return MooncakeHybridRolloutTransfer(
            tensor_min_bytes=1024 * 1024,
            enable_auto_cleanup=True,
            use_legacy_path=use_legacy,
            mount_segment_size=mount_segment_size,
        )
    else:
        return RayDataTransfer()
