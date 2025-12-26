import logging
import threading

import ray


logger = logging.getLogger(__name__)


class RolloutHealthMonitor:
    def __init__(self, rollout_manager, args):
        # TODO may remove this dependency after refactoring
        self._rollout_manager = rollout_manager

        self._thread = None
        self._stop_event = None
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait

    def start(self) -> bool:
        if not self._rollout_manager.rollout_engines:
            return False

        assert self._thread is None, "Health monitor thread is already running."

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started.")
        return True

    def stop(self) -> None:
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        logger.info(f"Health monitor loop started. Waiting for first wait: {self._check_first_wait}s")
        # TODO: need to be waiting for the large moe to be ready. this is hacky.
        if self._stop_event.wait(self._check_first_wait):
            logger.info("Health monitor stopped during first wait.")
            return
        while not self._stop_event.is_set():
            self._run_health_checks()
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self._rollout_manager.rollout_engines):
            if self._stop_event is not None and self._stop_event.is_set():
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
            self._kill_engine(rollout_engine_id=rollout_engine_id)

    def _kill_engine(self, rollout_engine_id: int):
        logger.info(f"Killing engine group {rollout_engine_id}...")
        for i in range(
            rollout_engine_id * self._rollout_manager.nodes_per_engine,
            (rollout_engine_id + 1) * self._rollout_manager.nodes_per_engine,
        ):
            engine = self._rollout_manager.all_rollout_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self._rollout_manager.all_rollout_engines[i] = None
