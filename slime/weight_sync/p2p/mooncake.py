"""The data plane depends only on Mooncake's pointer-based transfer API."""

import json
import os
import threading


def require_transfer_engine():
    """Load the optional native dependency only when this backend is selected."""
    try:
        from mooncake.engine import TransferEngine
    except ImportError as exc:
        raise ImportError(
            "--update-weight-transport=p2p requires Mooncake Transfer Engine in both "
            "the training and serving Python environments. Install the wheel matching "
            "each runtime's CUDA; see docs/en/advanced/p2p-weight-sync.md."
        ) from exc
    return TransferEngine


class MooncakeTransport:
    """Own an engine independently of SGLang's shared PD/KV engine.

    Registrations and remote-segment caches belong to this instance. Never
    borrow SGLang's engine: resetting a weight transfer must leave KV sessions
    and their long-lived registrations intact.
    """

    def __init__(self, host, *, device_names=None, device=None):
        self.host = host
        self.device_names = os.getenv("SLIME_MOONCAKE_IB_DEVICES", "") if device_names is None else device_names
        self._lock = threading.Lock()
        self._engine = self._create_engine(device=device)

    def _create_engine(self, *, device=None):
        engine = require_transfer_engine()()
        if device is not None and str(device).startswith("cuda:"):
            # Mooncake resolves CUDA_VISIBLE_DEVICES when discovering topology.
            # Restrict this GPU's registrations to its nearest eligible HCAs,
            # instead of registering every buffer on every HCA in the node.
            topology = json.loads(engine.get_local_topology(self.device_names))
            if not any(devices for groups in topology.values() for devices in groups):
                raise RuntimeError(
                    "P2P Mooncake transport found no usable RDMA devices. "
                    "Expose the RDMA device nodes and install the matching verbs provider."
                )
            preferred = topology.get(str(device), [[]])[0]
            if preferred:
                self.device_names = ",".join(preferred)
        self._check(
            engine.initialize(self.host, "P2PHANDSHAKE", "rdma", self.device_names),
            "initialize",
        )
        return engine

    @property
    def engine(self):
        with self._lock:
            if self._engine is None:
                self._engine = self._create_engine()
            return self._engine

    @property
    def session(self):
        return f"{self.host}:{self.engine.get_rpc_port()}"

    def reset(self):
        """Discard cached remote registrations after all local buffers are freed.

        Mooncake 0.3.11's Python API cannot invalidate cached remote rkeys.
        A fresh engine avoids stale keys when GPU addresses are reused.
        The pool calls this only when no transfers hold a buffer lease.
        """
        with self._lock:
            self._engine = None

    @staticmethod
    def _check(result, operation):
        if result != 0:
            raise RuntimeError(f"Mooncake {operation} failed: {result}")

    def register(self, tensor):
        self._check(self.engine.register_memory(tensor.data_ptr(), tensor.numel()), "register")

    def unregister(self, tensor):
        self._check(self.engine.unregister_memory(tensor.data_ptr()), "unregister")

    def unregister_many(self, tensors):
        self._check(self.engine.batch_unregister_memory([tensor.data_ptr() for tensor in tensors]), "unregister")

    def read(self, session, tensor, address, size):
        self._check(
            self.engine.batch_transfer_sync_read(
                session,
                [tensor.data_ptr()],
                [address],
                [size],
            ),
            "read",
        )
