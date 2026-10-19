"""P2P adapter for SGLang's public custom-weight-loader API."""

import base64
import json
import logging
import pickle
import socket
import time

LOADER = "slime.backends.sglang_utils.p2p.load_weights"
_MANIFEST = "__p2p_manifest__"
logger = logging.getLogger(__name__)


def validate_server_args(server_args, *, flush_cache_interval=1):
    if int(server_args.get("pp_size") or server_args.get("pipeline_parallel_size") or 1) != 1:
        raise ValueError("The P2P tensor loader currently requires serving PP=1")
    if int(server_args.get("dp_size") or 1) > 1 and not server_args.get("enable_dp_attention", False):
        raise ValueError("P2P requires serving DP=1 or enable_dp_attention")
    if server_args.get("disaggregation_mode") in ("prefill", "decode") and flush_cache_interval != 1:
        raise ValueError("P2P with PD requires --flush-cache-interval=1 to drain requests and invalidate old KV")


def _clone_bucket(bucket):
    """Give native loaders owned storage with one device copy per packed bucket."""
    import torch

    if not bucket:
        return []
    entries = [
        (name, tensor, tensor.storage_offset() * tensor.element_size(), tensor.numel() * tensor.element_size())
        for name, tensor in bucket
    ]
    size = max(offset + nbytes for _, _, offset, nbytes in entries)
    first = bucket[0][1]
    packed = torch.empty(0, dtype=torch.uint8, device=first.device).set_(first.untyped_storage(), 0, (size,), (1,))
    owned = packed.clone()
    return [
        (name, owned.narrow(0, offset, nbytes).view(tensor.dtype).view(tensor.shape))
        for name, tensor, offset, nbytes in entries
    ]


class _ManifestTensor:
    def __init__(self, data):
        self.data = data

    def __reduce__(self):
        import torch

        # Construct a fresh CPU tensor from inline bytes. Tensor/storage pickle
        # reducers can be patched by Megatron, and multiprocessing reducers may
        # emit shared-memory handles. Neither belongs in a cross-runtime message.
        return torch.ByteTensor, (list(self.data),)


def encode_manifest(manifest):
    tensor = _ManifestTensor(json.dumps(manifest).encode())
    return base64.b64encode(pickle.dumps([(_MANIFEST, tensor)])).decode()


def load_weights(model, named_tensors):
    import torch

    from slime.weight_sync import p2p

    if len(named_tensors) != 1 or named_tensors[0][0] != _MANIFEST:
        raise ValueError("Expected one P2P transfer manifest")
    if getattr(model, "_slime_p2p_failed", False):
        raise RuntimeError("A P2P load failed; restart the engine before loading another version")
    manifest = json.loads(bytes(named_tensors[0][1].cpu().tolist()))
    started = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((manifest["endpoints"][0]["host"], manifest["endpoints"][0]["port"]))
        host = route.getsockname()[0]
    if getattr(model, "_slime_p2p_pool", None) is None:
        model._slime_p2p_pool = p2p.BufferPool(host, device=device)
    seen = set()
    exhausted = False
    try:
        with p2p.Receiver(
            manifest["endpoints"],
            version=manifest["version"],
            consumer=manifest["consumer"],
            host=host,
            device=device,
            timeout=manifest["timeout"],
            pool=model._slime_p2p_pool,
        ) as receiver:

            def tensors():
                nonlocal exhausted
                for bucket in receiver:
                    for name, tensor in _clone_bucket(bucket):
                        if name in seen:
                            raise ValueError(f"Duplicate P2P weight: {name}")
                        seen.add(name)
                        # Native loaders may retain a projection until its pair
                        # arrives in a later bucket. Give them owned storage before
                        # the receiver reuses this bucket's RDMA buffer.
                        yield name, tensor
                exhausted = True

            # A single call preserves fused-projection pairing and performs model
            # post-processing once after all parameters arrive.
            model.load_weights(tensors())
            if not exhausted:
                raise RuntimeError("The model loader did not consume the complete P2P weight stream")
    except Exception:
        model._slime_p2p_failed = True
        raise
    finally:
        # Release GPU storage and stale registrations between updates.
        # trim() preserves any active leases quarantined after a failed read.
        model._slime_p2p_pool.trim()
        torch.cuda.empty_cache()
    logger.info(
        "P2P loaded %d tensors for version %s in %.3fs",
        len(seen),
        manifest["version"],
        time.perf_counter() - started,
    )
