"""PD and EPD-specific SGLang deployment sequencing."""

import logging
from typing import Any

import ray

from slime.backends.sglang_utils.engine_group import ServerGroup, ServerGroupPlacement
from slime.backends.sglang_utils.sglang_config import ModelConfig

logger = logging.getLogger(__name__)


def start_pd_server_groups(
    model_config: ModelConfig,
    placement: ServerGroupPlacement,
    router_ip: str,
    router_port: int,
) -> tuple[list[ServerGroup], list[Any]]:
    """Start prefill/decode groups without waiting for final engine initialization."""
    server_groups = []
    init_handles = []
    port_cursors: dict[int, int] = {}
    for group_config in model_config.server_groups:
        group = placement.create(group_config, router_ip, router_port)
        handles, port_cursors = group.start_engines(port_cursors)
        init_handles.extend(handles)
        server_groups.append(group)
    return server_groups, init_handles


def start_epd_server_groups(
    model_config: ModelConfig,
    placement: ServerGroupPlacement,
    router_ip: str,
    router_port: int,
) -> tuple[list[ServerGroup], list[Any]]:
    """Start encoder groups first, then inject their URLs into LLM groups."""
    server_groups = []
    port_cursors: dict[int, int] = {}

    encoder_urls: list[str] = []
    for group_config in model_config.server_groups:
        if group_config.worker_type != "encoder":
            continue
        group = placement.create(group_config, router_ip, router_port)
        handles, port_cursors = group.start_engines(port_cursors)
        if handles:
            ray.get(handles)
        urls = ray.get([engine.get_url.remote() for engine in group.engines])
        encoder_urls.extend(url for url in urls if url is not None)
        server_groups.append(group)

    logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

    init_handles = []
    for group_config in model_config.server_groups:
        if group_config.worker_type == "encoder":
            continue
        overrides_extra = {}
        if encoder_urls and group_config.worker_type in ("prefill", "regular"):
            overrides_extra["language_only"] = True
            overrides_extra["encoder_urls"] = encoder_urls
        group = placement.create(
            group_config,
            router_ip,
            router_port,
            overrides_extra=overrides_extra,
        )
        handles, port_cursors = group.start_engines(port_cursors)
        init_handles.extend(handles)
        server_groups.append(group)

    return server_groups, init_handles
