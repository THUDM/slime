"""Helpers for pre-launched external SGLang engines."""

from __future__ import annotations

import dataclasses
from urllib.parse import urlparse

import requests


@dataclasses.dataclass(frozen=True)
class ExternalEngineInfo:
    url: str
    host: str
    port: int
    worker_type: str
    num_gpus: int
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    disaggregation_bootstrap_port: int | None = None
    server_info: dict = dataclasses.field(default_factory=dict)

    @property
    def is_pd_worker(self) -> bool:
        return self.worker_type in ("prefill", "decode")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def normalize_external_engine_addr(addr: str) -> str:
    """Normalize ``host:port`` or ``http://host:port`` to an HTTP base URL."""
    if "://" not in addr:
        addr = f"http://{addr}"
    addr = addr.rstrip("/")
    parsed = urlparse(addr)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"Invalid external SGLang engine address {addr!r}. "
            "Use host:port or http://host:port (IPv6 must be bracketed)."
        )
    return addr


def external_engine_info_from_dict(data: dict) -> ExternalEngineInfo:
    return ExternalEngineInfo(**data)


def _positive_int(value, default: int) -> int:
    if value is None:
        return default
    value = int(value)
    return value if value > 0 else default


def _get_server_info(url: str, timeout: float = 30.0) -> dict:
    errors = []
    for endpoint in ("/server_info", "/get_server_info"):
        try:
            response = requests.get(f"{url}{endpoint}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError(f"Failed to fetch SGLang server info from {url}: {'; '.join(errors)}")


def _infer_worker_type(server_info: dict) -> str:
    if server_info.get("encoder_only"):
        return "encoder"
    mode = server_info.get("disaggregation_mode")
    if mode in ("prefill", "decode"):
        return mode
    return "regular"


def discover_external_engines(addrs: list[str], timeout: float = 30.0) -> list[ExternalEngineInfo]:
    infos = []
    for addr in addrs:
        url = normalize_external_engine_addr(addr)
        parsed = urlparse(url)
        assert parsed.hostname is not None and parsed.port is not None
        server_info = _get_server_info(url, timeout=timeout)

        pp_size = _positive_int(server_info.get("pp_size") or server_info.get("pipeline_parallel_size"), 1)
        tp_size = _positive_int(server_info.get("tp_size") or server_info.get("tensor_parallel_size"), 1)
        dp_size = _positive_int(server_info.get("dp_size") or server_info.get("data_parallel_size"), 1)
        ep_size = _positive_int(server_info.get("ep_size") or server_info.get("expert_parallel_size"), 1)
        num_gpus = _positive_int(
            server_info.get("num_gpus") or server_info.get("num_gpus_per_engine"),
            tp_size * pp_size,
        )
        bootstrap_port = server_info.get("disaggregation_bootstrap_port")
        bootstrap_port = int(bootstrap_port) if bootstrap_port is not None else None

        infos.append(
            ExternalEngineInfo(
                url=url,
                host=parsed.hostname,
                port=parsed.port,
                worker_type=_infer_worker_type(server_info),
                num_gpus=num_gpus,
                tp_size=tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                ep_size=ep_size,
                disaggregation_bootstrap_port=bootstrap_port,
                server_info=server_info,
            )
        )
    return infos


def apply_external_engine_info_to_args(args, logger=None) -> None:
    """Detect external engines and store the derived topology on ``args``."""
    addrs = getattr(args, "rollout_external_engine_addrs", None)
    if not addrs:
        args.rollout_external = False
        return

    args.rollout_external = True
    infos = discover_external_engines(addrs)
    if not infos:
        raise ValueError("--rollout-external-engine-addrs did not contain any engines.")

    args.rollout_external_engine_infos = [info.to_dict() for info in infos]
    args.rollout_num_engines = len(infos)
    args.rollout_num_gpus = sum(info.num_gpus for info in infos)

    # Keep legacy homogeneous fields meaningful for code paths that still read
    # them.  Per-group rollout startup uses the exact per-engine values below.
    first = infos[0]
    args.rollout_num_gpus_per_engine = first.num_gpus
    args.sglang_pipeline_parallel_size = first.pp_size
    args.sglang_data_parallel_size = first.dp_size
    args.sglang_expert_parallel_size = first.ep_size
    if any(info.dp_size > 1 for info in infos):
        args.sglang_enable_dp_attention = True

    if logger is not None:
        summary = [
            {
                "url": info.url,
                "worker_type": info.worker_type,
                "num_gpus": info.num_gpus,
                "tp_size": info.tp_size,
                "pp_size": info.pp_size,
                "dp_size": info.dp_size,
                "ep_size": info.ep_size,
            }
            for info in infos
        ]
        logger.info(f"Detected external SGLang engines: {summary}")
