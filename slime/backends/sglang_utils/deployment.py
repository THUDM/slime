"""Launch and connect SGLang rollout deployments."""

import logging
import multiprocessing
import random
import time
from typing import Any

from slime.backends.sglang_utils.disaggregation import start_epd_server_groups, start_pd_server_groups
from slime.backends.sglang_utils.engine_group import RolloutServer, ServerGroupPlacement
from slime.backends.sglang_utils.external import start_external_rollout_servers
from slime.backends.sglang_utils.sglang_config import resolve_sglang_config
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info

logger = logging.getLogger(__name__)


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sglang_router and return (router_ip, router_port)."""
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True

    # RDMA transfer timeouts are transient and should not mark decode workers dead.
    router_args.disable_circuit_breaker = True

    # RolloutHealthMonitor owns engine health checks.
    if hasattr(router_args, "disable_health_check"):
        router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in placement-group bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    return args.actor_num_nodes * args.actor_num_gpus_per_node


def _compute_megatron_num_gpus(args) -> int:
    """Total number of Megatron GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    return args.actor_num_nodes * args.actor_num_gpus_per_node


def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start configured rollout servers without waiting for final engine initialization."""
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = resolve_sglang_config(args)
    placement = ServerGroupPlacement(
        args=args,
        pg=pg,
        rollout_pg_offset=_compute_rollout_offset(args),
        megatron_num_gpus=_compute_megatron_num_gpus(args),
    )

    servers: dict[str, RolloutServer] = {}
    pending_init_handles: list[Any] = []

    for model_idx, model_config in enumerate(config.models):
        model_config.resolve(args)

        router_ip, router_port = _start_router(
            args,
            has_pd_disaggregation=model_config.has_pd_disaggregation,
            force_new=(model_idx > 0),
        )

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        if model_config.has_encoder_disaggregation:
            server_groups, init_handles = start_epd_server_groups(
                model_config,
                placement,
                router_ip,
                router_port,
            )
        elif model_config.has_pd_disaggregation:
            server_groups, init_handles = start_pd_server_groups(
                model_config,
                placement,
                router_ip,
                router_port,
            )
        else:
            server_groups = []
            init_handles = []
            port_cursors: dict[int, int] = {}
            for group_config in model_config.server_groups:
                group = placement.create(group_config, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                init_handles.extend(handles)
                server_groups.append(group)

        pending_init_handles.extend(init_handles)
        servers[model_config.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_config.name,
            update_weights=model_config.update_weights,
        )

    args.sglang_model_routers = {name: (server.router_ip, server.router_port) for name, server in servers.items()}
    return servers, pending_init_handles
