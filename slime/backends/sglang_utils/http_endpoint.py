"""Helpers for rollout backends served by an opaque HTTP endpoint."""

from __future__ import annotations

import dataclasses
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def normalize_rollout_http_endpoint_url(url: str) -> str:
    """Normalize an HTTP endpoint base URL used by rollout generation."""
    url = url.rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.netloc == "":
        raise ValueError(f"Invalid rollout HTTP endpoint URL {url!r}. Use an absolute http:// or https:// URL.")
    return url


def uses_rollout_http_endpoint(args) -> bool:
    return bool(getattr(args, "rollout_http_endpoint_url", None))


def rollout_http_endpoint_url(args, endpoint: str = "/generate") -> str:
    base = normalize_rollout_http_endpoint_url(args.rollout_http_endpoint_url)
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base}{endpoint}"


@dataclasses.dataclass
class HttpEndpointRolloutServer:
    """Rollout server backed by an opaque HTTP endpoint.

    The endpoint is intentionally not assumed to be an SGLang router: it may not
    expose worker-management APIs such as ``/workers`` and it may represent an
    elastic fleet with no stable per-engine handles.
    """

    endpoint_url: str
    model_name: str = "default"
    update_weights: bool = True
    router_ip: str | None = None
    router_port: int | None = None
    server_groups: list = dataclasses.field(default_factory=list)
    engines: list = dataclasses.field(default_factory=list)
    engine_gpu_counts: list[int] = dataclasses.field(default_factory=list)
    engine_gpu_offsets: list[int] = dataclasses.field(default_factory=list)
    num_new_engines: int = 0

    @property
    def all_engines(self):
        return self.engines

    def recover(self):
        logger.warning("Fault tolerance is not supported for opaque HTTP rollout endpoints; skip recover.")

    def offload(self):
        return []

    def onload(self, tags: list[str] | None = None):
        return []

    def onload_weights(self):
        return []

    def onload_kv(self):
        return []


def start_http_endpoint_rollout_servers(args) -> dict[str, HttpEndpointRolloutServer]:
    endpoint_url = normalize_rollout_http_endpoint_url(args.rollout_http_endpoint_url)
    args.rollout_http_endpoint_url = endpoint_url
    args.sglang_model_routers = {}
    if getattr(args, "rollout_num_engines", None) is None:
        args.rollout_num_engines = 1
    logger.info("Using opaque HTTP rollout endpoint: %s", endpoint_url)
    return {
        "default": HttpEndpointRolloutServer(
            endpoint_url=endpoint_url,
            model_name="default",
            update_weights=True,
        )
    }
