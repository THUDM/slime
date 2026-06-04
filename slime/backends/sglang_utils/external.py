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
    disaggregation_bootstrap_port: int | None = None
    server_info: dict = dataclasses.field(default_factory=dict)

    @property
    def is_pd_worker(self) -> bool:
        return self.worker_type in ("prefill", "decode")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ExternalServerGroupInfo:
    worker_type: str
    num_gpus_per_engine: int
    engine_infos: tuple[ExternalEngineInfo, ...]

    @property
    def num_gpus(self) -> int:
        return sum(info.num_gpus for info in self.engine_infos)

    def to_dict(self) -> dict:
        return {
            "worker_type": self.worker_type,
            "num_gpus": self.num_gpus,
            "num_gpus_per_engine": self.num_gpus_per_engine,
            "engine_infos": [info.to_dict() for info in self.engine_infos],
        }


@dataclasses.dataclass(frozen=True)
class ExternalModelInfo:
    name: str
    server_groups: tuple[ExternalServerGroupInfo, ...]
    update_weights: bool = True

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(g.worker_type in ("prefill", "decode") for g in self.server_groups)

    @property
    def engine_infos(self) -> list[ExternalEngineInfo]:
        return [info for group in self.server_groups for info in group.engine_infos]

    @property
    def total_num_gpus(self) -> int:
        return sum(group.num_gpus for group in self.server_groups)

    @property
    def num_engines(self) -> int:
        return sum(len(group.engine_infos) for group in self.server_groups)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "server_groups": [group.to_dict() for group in self.server_groups],
            "update_weights": self.update_weights,
        }


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


def build_external_model_info(infos: list[ExternalEngineInfo], name: str = "default") -> ExternalModelInfo:
    specs_by_topology: dict[tuple[str, int], list[ExternalEngineInfo]] = {}
    for info in infos:
        key = (info.worker_type, info.num_gpus)
        specs_by_topology.setdefault(key, []).append(info)

    server_groups = tuple(
        ExternalServerGroupInfo(
            worker_type=worker_type,
            num_gpus_per_engine=num_gpus_per_engine,
            engine_infos=tuple(group_infos),
        )
        for (worker_type, num_gpus_per_engine), group_infos in specs_by_topology.items()
    )
    return ExternalModelInfo(name=name, server_groups=server_groups)


def get_server_info(url: str, timeout: float = 30.0) -> dict:
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
        server_info = get_server_info(url, timeout=timeout)

        pp_size = int(server_info.get("pp_size") or server_info.get("pipeline_parallel_size") or 1)
        tp_size = int(server_info.get("tp_size") or server_info.get("tensor_parallel_size") or 1)
        num_gpus = int(server_info.get("num_gpus") or server_info.get("num_gpus_per_engine") or tp_size * pp_size)
        bootstrap_port = server_info.get("disaggregation_bootstrap_port")
        bootstrap_port = int(bootstrap_port) if bootstrap_port is not None else None

        infos.append(
            ExternalEngineInfo(
                url=url,
                host=parsed.hostname,
                port=parsed.port,
                worker_type=_infer_worker_type(server_info),
                num_gpus=num_gpus,
                disaggregation_bootstrap_port=bootstrap_port,
                server_info=server_info,
            )
        )
    return infos


def apply_external_engine_info_to_args(args, logger=None) -> None:
    """Detect external engines and store the derived topology on ``args``."""
    addrs = args.rollout_external_engine_addrs
    if not addrs:
        raise ValueError("apply_external_engine_info_to_args requires --rollout-external-engine-addrs.")

    infos = discover_external_engines(addrs)
    if not infos:
        raise ValueError("--rollout-external-engine-addrs did not contain any engines.")

    args.rollout_external_engine_infos = [info.to_dict() for info in infos]
    model_info = build_external_model_info(infos)
    args.rollout_external_model_info = model_info.to_dict()
    args.rollout_num_engines = model_info.num_engines
    args.rollout_num_gpus = model_info.total_num_gpus

    if logger is not None:
        summary = [
            {
                "url": info.url,
                "worker_type": info.worker_type,
                "num_gpus": info.num_gpus,
                "disaggregation_bootstrap_port": info.disaggregation_bootstrap_port,
            }
            for info in infos
        ]
        logger.info(f"Detected external SGLang engines: {summary}")
