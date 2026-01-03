import logging
import threading
from urllib.parse import urlparse

import ray
import requests
import sglang_router
from packaging.version import parse
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine


logger = logging.getLogger(__name__)


def get_active_seed_instance(args, exclude_urls: list[str] | None = None):
    """Get an active seed instance from the router for fault tolerance restart.

    When restarting failed engines, this function queries the router to find active workers
    and returns the connection info for one of them to be used as a seed instance for
    remote weight loading.

    Args:
        args: The global arguments containing router IP and port.
        exclude_urls: A list of worker URLs to exclude (e.g., the URLs of killed engines).

    Returns:
        A dict with 'ip' and 'port' keys for the seed instance, or None if no active
        workers are found.
    """
    router_ip = args.sglang_router_ip
    router_port = args.sglang_router_port
    exclude_urls = exclude_urls or []

    if not router_ip or not router_port:
        logger.warning("Router IP or port not set, cannot get active seed instance.")
        return None

    try:
        # Query the router to get active workers
        if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
            response = requests.get(f"http://{router_ip}:{router_port}/list_workers", timeout=5)
            response.raise_for_status()
            data = response.json()
            worker_urls = data.get("urls", [])
        else:
            response = requests.get(f"http://{router_ip}:{router_port}/workers", timeout=5)
            response.raise_for_status()
            data = response.json()
            workers = data.get("workers", [])
            worker_urls = [w["url"] for w in workers]

        if not worker_urls:
            logger.warning("No active workers found in router.")
            return None

        # Filter out excluded URLs (normalize for comparison)
        def normalize_url(url):
            """Normalize URL for comparison (remove trailing slash, etc.)"""
            return url.rstrip("/").lower()

        exclude_urls_normalized = {normalize_url(u) for u in exclude_urls}
        available_urls = [u for u in worker_urls if normalize_url(u) not in exclude_urls_normalized]

        if not available_urls:
            logger.warning(f"No active workers found after excluding {exclude_urls}. All workers: {worker_urls}")
            return None

        # Parse the first available worker's URL to get IP and port
        seed_url = available_urls[0]
        parsed = urlparse(seed_url)

        # Handle IPv6 addresses (may be wrapped in [])
        host = parsed.hostname or parsed.netloc.rsplit(":", 1)[0]
        port = parsed.port or 30000

        logger.info(f"Found active seed instance for fault tolerance: {host}:{port} (excluded: {exclude_urls})")
        return {"ip": host, "port": port}

    except Exception as e:
        logger.warning(f"Failed to get active seed instance from router: {e}")
        return None


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, engine_group, args):
        self._engine_group = engine_group
        self._args = args

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._engine_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self._engine_group.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing actor. Exception: {e}"
            )
            self._kill_and_restart_engine(rollout_engine_id=rollout_engine_id)

    def _kill_and_restart_engine(self, rollout_engine_id: int):
        """Kill a failed engine and immediately restart it with remote weight loading."""
        logger.info(f"Killing and restarting engine group {rollout_engine_id}...")

        args = self._rollout_manager.args
        nodes_per_engine = self._rollout_manager.nodes_per_engine

        # Collect URLs of engines being killed (to exclude from seed instance selection)
        killed_engine_urls = []

        # Kill the failed engine(s)
        for i in range(
            rollout_engine_id * self._engine_group.nodes_per_engine,
            (rollout_engine_id + 1) * self._engine_group.nodes_per_engine,
        ):
            engine = self._engine_group.all_engines[i]
            if engine:
                # Try to get the engine's URL before killing
                try:
                    server_host = ray.get(engine.get_server_host.remote())
                    server_port = ray.get(engine.get_server_port.remote())
                    killed_url = f"http://{server_host}:{server_port}"
                    killed_engine_urls.append(killed_url)
                    logger.info(f"Engine at index {i} has URL: {killed_url}")
                except Exception as e:
                    logger.warning(f"Could not get URL for engine at index {i}: {e}")

                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self._engine_group.all_engines[i] = None

        # Get active seed instance for remote weight loading (excluding killed engines)
        logger.info(f"Looking for seed instance, excluding URLs: {killed_engine_urls}")
        remote_seed_instance = get_active_seed_instance(self._args, exclude_urls=killed_engine_urls)
        if remote_seed_instance is None:
            logger.error(f"Cannot restart engine {rollout_engine_id}: no active seed instance found.")
            return

        logger.info(
            f"Restarting engine {rollout_engine_id} with remote weight loading from "
            f"{remote_seed_instance['ip']}:{remote_seed_instance['port']}"
        )

        # Restart the engine(s)
        try:
            self._restart_engine_group(rollout_engine_id, remote_seed_instance)
            logger.info(f"Successfully restarted engine group {rollout_engine_id}")
        except Exception as e:
            logger.error(f"Failed to restart engine group {rollout_engine_id}: {e}")

    def _restart_engine_group(self, rollout_engine_id: int, remote_seed_instance: dict):
        """Restart a single engine group with remote weight loading."""
        from slime.ray.rollout import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        args = self._engine_group.args
        pg, reordered_bundle_indices, reordered_gpu_ids = self._engine_group.pg
        nodes_per_engine = self._engine_group.nodes_per_engine

        num_gpu_per_engine = min(self._engine_group.num_gpus_per_engine, args.num_gpus_per_node)
        num_engines_per_node = max(
            1, min(args.num_gpus_per_node, args.rollout_num_gpus) // self._engine_group.num_gpus_per_engine
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        # Restart all nodes for this engine
        new_engines = []
        for node_offset in range(nodes_per_engine):
            i = rollout_engine_id * nodes_per_engine + node_offset

            global_rank = self._engine_group.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            gpu_index = self._engine_group.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                args,
                rank=global_rank,
                worker_type=self._engine_group.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self._engine_group.sglang_overrides,
                num_gpus_per_engine=self._engine_group.num_gpus_per_engine,
            )

            new_engines.append((global_rank, rollout_engine))
            self._engine_group.all_engines[i] = rollout_engine

        # Allocate ports for the new engine(s)
        addr_and_ports = self._allocate_ports_for_engine_group(
            rollout_engine_id, new_engines, num_engines_per_node
        )

        # Add remote_seed_instance to addr_and_ports
        for rank, _ in new_engines:
            addr_and_ports[rank]["remote_seed_instance"] = remote_seed_instance

        # Initialize the new engine(s)
        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self._engine_group.router_ip,
                router_port=self._engine_group.router_port,
            )
            for rank, engine in new_engines
        ]
        ray.get(init_handles)

    def _allocate_ports_for_engine_group(
        self, rollout_engine_id: int, new_engines: list, num_engines_per_node: int
    ) -> dict:
        """Allocate ports for a single engine group being restarted."""
        args = self._engine_group.args
        nodes_per_engine = self._engine_group.nodes_per_engine
        num_gpu_per_engine = min(self._engine_group.num_gpus_per_engine, args.num_gpus_per_node)
        num_engines = len(self._engine_group.all_engines)

        addr_and_ports = [{} for _ in range(num_engines)]

        # Get the first engine to allocate ports
        first_rank, first_engine = new_engines[0]

        def get_addr_and_ports(engine):
            start_port = 15000

            def port(consecutive=1):
                nonlocal start_port
                _, p = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = p + consecutive
                return p

            def addr():
                a, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return a

            return addr, port

        get_addr, get_port = get_addr_and_ports(first_engine)

        for rank, _engine in new_engines:
            addr_and_ports[rank]["host"] = get_addr()
            addr_and_ports[rank]["port"] = get_port()
            addr_and_ports[rank]["nccl_port"] = get_port()

            if self._engine_group.worker_type == "prefill":
                addr_and_ports[rank]["disaggregation_bootstrap_port"] = get_port()

        # Handle multi-node engine case
        if self._engine_group.num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = self._engine_group.num_gpus_per_engine // args.num_gpus_per_node
            base_rank = rollout_engine_id * nodes_per_engine
            if base_rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for rank, _ in new_engines:
                    addr_and_ports[rank]["dist_init_addr"] = dist_init_addr
        else:
            for rank, _ in new_engines:
                addr_and_ports[rank]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

        return addr_and_ports
