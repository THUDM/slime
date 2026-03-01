import logging
import threading
from urllib.parse import urlparse

import ray
import requests
import sglang_router
from packaging.version import parse


logger = logging.getLogger(__name__)


def get_active_seed_instance(args):
    """Get an active seed instance from the router for fault tolerance restart.

    When restarting failed engines, this function queries the router to find active workers
    and returns the connection info for one of them to be used as a seed instance for
    remote weight loading.

    Args:
        args: The global arguments containing router IP and port.

    Returns:
        A dict with 'ip' and 'port' keys for the seed instance, or None if no active
        workers are found.
    """
    router_ip = args.sglang_router_ip
    router_port = args.sglang_router_port

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

        # Parse the first available worker's URL to get IP and port
        seed_url = worker_urls[0]
        parsed = urlparse(seed_url)

        # Handle IPv6 addresses (may be wrapped in [])
        host = parsed.hostname or parsed.netloc.rsplit(":", 1)[0]
        port = parsed.port or 30000

        logger.info(f"Found active seed instance for fault tolerance: {host}:{port}")
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
                f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing and restarting actor. Exception: {e}"
            )
            self._kill_and_restart_engine(rollout_engine_id=rollout_engine_id)

    def _kill_and_restart_engine(self, rollout_engine_id: int):
        """Kill a failed engine and immediately restart it with remote weight loading."""
        logger.info(f"Killing and restarting engine group {rollout_engine_id}...")

        args = self._rollout_manager.args
        nodes_per_engine = self._rollout_manager.nodes_per_engine

        # Kill the failed engine(s)
        for i in range(
            rollout_engine_id * self._engine_group.nodes_per_engine,
            (rollout_engine_id + 1) * self._engine_group.nodes_per_engine,
        ):
            engine = self._engine_group.all_engines[i]
            if engine:
                try:
                    ray.get(engine.shutdown.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self._engine_group.all_engines[i] = None


        # Restart the engine(s)
        try:
            init_handles, _ = self._engine_group.start_engines()
            ray.get(init_handles)
            logger.info(f"Successfully restarted engine group {rollout_engine_id}")
        except Exception as e:
            logger.error(f"Failed to restart engine group {rollout_engine_id}: {e}")
