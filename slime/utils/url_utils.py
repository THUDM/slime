from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class ExternalEngineAddress:
    base_url: str
    host: str
    port: int
    dist_init_addr: str
    is_url: bool


def normalize_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected an http(s) URL, got {base_url!r}")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def join_url(base_url: str, endpoint: str) -> str:
    base = normalize_base_url(base_url)
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{base}{endpoint}"


def make_http_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def get_default_router_url_from_args(args, endpoint: str = "/generate") -> str:
    if getattr(args, "sglang_router_url", None):
        return join_url(args.sglang_router_url, endpoint)
    return join_url(make_http_base_url(args.sglang_router_ip, args.sglang_router_port), endpoint)


def get_model_url_from_args(args, model_name: str, endpoint: str = "/generate") -> str:
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        router = routers[model_name]
        if isinstance(router, str):
            return join_url(router, endpoint)
        ip, port = router
        return join_url(make_http_base_url(ip, port), endpoint)
    return get_default_router_url_from_args(args, endpoint)


def _format_host_for_addr(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def parse_external_engine_addr(addr: str) -> ExternalEngineAddress:
    """Parse either ``host:port`` or an http(s) external engine base URL."""
    if "://" in addr:
        parsed = urlsplit(addr)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Expected an http(s) external engine URL, got {addr!r}")
        default_port = 443 if parsed.scheme == "https" else 80
        port = parsed.port or default_port
        host = _format_host_for_addr(parsed.hostname)
        return ExternalEngineAddress(
            base_url=normalize_base_url(addr),
            host=host,
            port=port,
            dist_init_addr=f"{host}:{port}",
            is_url=True,
        )

    parsed = urlsplit(f"//{addr}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"Expected external engine address as host:port, got {addr!r}")
    host = _format_host_for_addr(parsed.hostname)
    port = parsed.port
    return ExternalEngineAddress(
        base_url=make_http_base_url(host, port),
        host=host,
        port=port,
        dist_init_addr=f"{host}:{port}",
        is_url=False,
    )
