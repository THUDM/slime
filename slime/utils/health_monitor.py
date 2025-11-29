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

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        if not self._thread:
            return

        assert self._stop_event is not None
        self._stop_event.set()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)

        self._thread = None
        self._stop_event = None

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        # TODO: need to be waiting for the large moe to be ready. this is hacky.
        if self._stop_event.wait(self._check_first_wait):
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
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            logger.info(
                f"Health check timed out for rollout engine {rollout_engine_id} (ray timeout). Killing actor. (original exception: {e})"
            )
            self._kill_engine(rollout_engine_id=rollout_engine_id)

    def _kill_engine(self, rollout_engine_id: int):
        for i in range(
            rollout_engine_id * self._rollout_manager.nodes_per_engine,
            (rollout_engine_id + 1) * self._rollout_manager.nodes_per_engine,
        ):
            engine = self._rollout_manager.all_rollout_engines[i]
            try:
                ray.get(engine.shutdown.remote())
                ray.kill(engine)
            except Exception as e:
                logger.info(f"Fail to kill engine and skip (e: {e})")
            self._rollout_manager.all_rollout_engines[i] = None
