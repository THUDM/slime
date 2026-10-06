import logging
import threading

import ray
import requests

from slime.backends.sglang_utils.server_control import unregister_worker

logger = logging.getLogger(__name__)


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

    def __init__(self, server_group, args):
        self._server_group = server_group

        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Engines may be offloaded before the first rollout.
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True  # Need to wait after each resume
        self._check_lock = threading.Lock()

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._server_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event.clear()
        self._pause_event.set()
        self._need_first_wait = True
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
        self._stop_event.set()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        # Finish an in-flight check before weights or memory ownership change.
        with self._check_lock:
            pass

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()

    def _health_monitor_loop(self) -> None:
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
                try:
                    self._run_health_checks()
                except requests.RequestException:
                    logger.exception("Failed to unregister an unhealthy worker; retaining it for the next check")

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def check_once(self) -> None:
        """Check at the rollout boundary, regardless of interval or warmup grace."""
        self._run_health_checks(force=True)

    def _run_health_checks(self, *, force=False) -> None:
        # The background thread and rollout-completion RPC share this lock, so
        # they cannot retire the same engine while another check uses its handle.
        with self._check_lock:
            if self._stop_event.is_set():
                return
            if not force and self._pause_event.is_set():
                return
            checks = {
                engine.health_generate.remote(timeout=self._check_timeout): rollout_engine_id
                for rollout_engine_id, engine in enumerate(self._server_group.engines)
                if engine is not None
            }
            if checks:
                # Bound queued/wedged actor RPCs as well as the underlying HTTP
                # requests. All engines are probed concurrently.
                ray.wait(list(checks), num_returns=len(checks), timeout=self._check_timeout)
            for handle, rollout_engine_id in checks.items():
                try:
                    ray.get(handle, timeout=0)
                except Exception as error:
                    # Both HTTP failures and unresponsive/dead actor RPCs retire
                    # the engine; HTTP timeout alone cannot detect a wedged actor.
                    logger.error("Health check failed for engine %s: %s", rollout_engine_id, error)
                    self._kill_engine(rollout_engine_id)

    def _kill_engine(self, rollout_engine_id: int):
        logger.info(f"Killing server group {rollout_engine_id}...")
        group = self._server_group
        first = rollout_engine_id * group.nodes_per_engine
        if group.worker_type != "encoder":
            # Remove routing first, independently of shutdown RPC success. If
            # the router rejects removal, leave handles intact and surface the
            # error; proceeding would send later controls to a stale worker.
            unregister_worker(
                f"http://{group.router_ip or group.args.sglang_router_ip}:{group.router_port or group.args.sglang_router_port}",
                group.engine_urls[first],
                timeout=self._check_timeout,
            )
        # A multi-node engine is one serving unit: retire every node together.
        for i in range(first, first + group.nodes_per_engine):
            engine = group.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote(), timeout=self._check_timeout)
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
                finally:
                    # Shutdown RPCs are best-effort; stop the actor even if its
                    # HTTP server has already died or the RPC timed out.
                    ray.kill(engine, no_restart=True)
            # Leave a missing slot for recovery immediately before weight sync.
            group.all_engines[i] = None
