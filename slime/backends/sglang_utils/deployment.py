"""Launch and connect SGLang rollout deployments."""

import json
import logging
import multiprocessing
import os
import random
import time
from typing import Any

import ray
import requests

from slime.backends.sglang_utils.disaggregation import start_epd_server_groups, start_pd_server_groups
from slime.backends.sglang_utils.engine_group import RolloutServer, ServerGroupPlacement
from slime.backends.sglang_utils.external import start_external_rollout_servers
from slime.backends.sglang_utils.sglang_config import resolve_sglang_config
from slime.utils.cleanup import Cleanup
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
    if getattr(args, "rollout_backend", "sglang") == "sglang-omni":
        return start_omni_rollout_servers(args, pg)
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
    # Models initialize concurrently, so ports allocated to earlier models may
    # not be bound yet. Carry the cursors across every model and group.
    port_cursors: dict[int, int] = {}

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
            server_groups, init_handles, port_cursors = start_epd_server_groups(
                model_config,
                placement,
                router_ip,
                router_port,
                port_cursors,
            )
        elif model_config.has_pd_disaggregation:
            server_groups, init_handles, port_cursors = start_pd_server_groups(
                model_config,
                placement,
                router_ip,
                router_port,
                port_cursors,
            )
        else:
            server_groups = []
            init_handles = []
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


OMNI_ROUTER_PROGRAM = """
import json
import sys
from contextlib import asynccontextmanager
import uvicorn
from sglang_omni_router.python.app import create_app
from sglang_omni_router.python.config import RouterConfig

configuration, disabled = json.loads(sys.argv[1])
app = create_app(RouterConfig(**configuration))
original_lifespan = app.router.lifespan_context

@asynccontextmanager
async def lifespan(application):
    async with original_lifespan(application):
        if disabled:
            for worker in application.state.workers:
                worker.set_disabled(True)
        yield

app.router.lifespan_context = lifespan
uvicorn.run(app, host=configuration['host'], port=configuration['port'], workers=1)
"""


def launch_omni_router(python: str, configuration: dict, disabled: bool, admin_key: str | None) -> None:
    environment = os.environ.copy()
    environment.pop("SGLANG_OMNI_ADMIN_KEY", None)
    if admin_key:
        environment["SGLANG_OMNI_ADMIN_KEY"] = admin_key
    command = [python, "-c", OMNI_ROUTER_PROGRAM, json.dumps([configuration, disabled])]
    os.execvpe(python, command, environment)


def start_omni_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start complete pipelines before seeding their single-process Python router."""
    if args.rollout_external:
        raise ValueError("Omni rollout requires internally owned pipeline replicas")
    from slime.backends.sglang_utils.omni_engine import OMNI_WORKER_CAPABILITIES

    configuration = resolve_sglang_config(args)
    placement = ServerGroupPlacement(
        args=args,
        pg=pg,
        rollout_pg_offset=_compute_rollout_offset(args),
        megatron_num_gpus=_compute_megatron_num_gpus(args),
    )
    servers = {}
    port_cursors = {}
    created_groups, router_processes = [], []
    try:
        for model_index, model in enumerate(configuration.models):
            model.resolve(args)
            groups, handles = [], []
            for group_configuration in model.server_groups:
                if group_configuration.worker_type not in {"regular", "placeholder"}:
                    raise ValueError("Omni uses complete regular pipeline replicas, not PD/EPD server groups")
                replica_gpus = group_configuration.num_gpus_per_engine
                if (
                    not replica_gpus
                    or replica_gpus < 1
                    or replica_gpus > args.num_gpus_per_node
                    or group_configuration.num_gpus % replica_gpus
                ):
                    raise ValueError("Omni replica GPUs must fit one node and divide the group's GPU allocation")
                group = placement.create(group_configuration, None, None)
                created_groups.append(group)
                pending, port_cursors = group.start_engines(port_cursors)
                handles.extend(pending)
                groups.append(group)
            ray.get(handles)
            workers = [
                {"url": group.engine_urls[index], "capabilities": OMNI_WORKER_CAPABILITIES}
                for group in groups
                for index in range(0, len(group.all_engines), group.nodes_per_engine)
            ]
            if not workers:
                raise ValueError("The Omni Python router requires at least one complete pipeline worker")
            router_ip = _wrap_ipv6(get_host_info()[1])
            router_port = args.sglang_router_port if model_index == 0 else None
            router_port = router_port or find_available_port(random.randint(3000, 4000))
            router_configuration = {
                "host": router_ip.strip("[]"),
                "port": router_port,
                "workers": workers,
                "policy": args.omni_router_policy,
                "request_timeout_secs": args.sglang_router_request_timeout_secs,
            }
            process = multiprocessing.Process(
                target=launch_omni_router,
                args=(args.omni_python, router_configuration, model.update_weights, args.omni_admin_key),
            )
            process.start()
            router_processes.append(process)
            deadline = time.monotonic() + getattr(args, "distributed_timeout_minutes", 10) * 60
            with requests.Session() as client:
                client.trust_env = False
                while time.monotonic() < deadline:
                    if not process.is_alive():
                        raise RuntimeError(f"Omni router exited with code {process.exitcode}")
                    try:
                        response = client.get(f"http://{router_ip}:{router_port}/live", timeout=1)
                        response.raise_for_status()
                        break
                    except requests.RequestException:
                        time.sleep(0.1)
                else:
                    raise TimeoutError("Omni router startup timed out")
            for group in groups:
                group.router_ip, group.router_port = router_ip, router_port
            ray.get(
                [engine.attach_router.remote(router_ip, router_port) for group in groups for engine in group.engines]
            )
            servers[model.name] = RolloutServer(
                server_groups=groups,
                router_ip=router_ip,
                router_port=router_port,
                model_name=model.name,
                update_weights=model.update_weights,
            )
            if model_index == 0:
                args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
    except BaseException:
        cleanup = Cleanup(args.rollout_cleanup_timeout)
        shutdowns = []
        engines = [engine for group in created_groups for engine in group.all_engines if engine is not None]
        for engine in engines:
            handle = cleanup.run("request Omni shutdown", engine.shutdown.remote)
            if handle is not None:
                shutdowns.append(handle)
        if shutdowns:
            cleanup.run(
                "wait for Omni shutdown",
                ray.wait,
                shutdowns,
                num_returns=len(shutdowns),
                timeout=cleanup.remaining / 2,
            )
        for engine in engines:
            cleanup.run("kill incomplete Omni actor", ray.kill, engine, no_restart=True)
        for process in router_processes:
            if process.is_alive():
                cleanup.run("terminate incomplete Omni router", process.terminate)
                cleanup.run("join incomplete Omni router", process.join, timeout=cleanup.remaining / 2)
                if process.is_alive():
                    cleanup.run("kill incomplete Omni router", process.kill)
                    cleanup.run("reap incomplete Omni router", process.join, timeout=cleanup.remaining)
        raise
    args.sglang_model_routers = {name: (server.router_ip, server.router_port) for name, server in servers.items()}
    return servers, []
