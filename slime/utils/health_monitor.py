import logging
import threading

import ray


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

    def __init__(self, rollout_manager, args):
        # TODO may remove this dependency after refactoring
        self._rollout_manager = rollout_manager

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
        if not self._rollout_manager.all_rollout_engines:
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
        """Pause health checking. Called when engines are offloaded.

        Before pausing, performs a final health check to ensure all engines are healthy.
        Any unhealthy engines will be killed before pausing.
        """
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor (running final health check first)...")
        # Run a final health check before pausing to catch any unhealthy engines
        if self._is_checking_enabled:
            self._run_health_checks()
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
        """Run health checks for all engines in parallel."""
        engines = self._rollout_manager.rollout_engines
        if not engines:
            return

        # Collect all valid engines with their indices
        engine_tasks = [
            (i, engine, engine.health_generate.remote(timeout=self._check_timeout))
            for i, engine in enumerate(engines)
            if engine is not None
        ]

        if not engine_tasks:
            return

        # Wait for all health checks in parallel
        refs = [task for _, _, task in engine_tasks]
        try:
            results = ray.get(refs, timeout=self._check_timeout + 5)
            # All succeeded
            for (engine_id, _, _), result in zip(engine_tasks, results, strict=True):
                if result is not True:
                    logger.error(f"Health check returned non-True for engine {engine_id}: {result}. Killing actor.")
                    self._kill_engine(rollout_engine_id=engine_id)
                else:
                    logger.debug(f"Health check passed for rollout engine {engine_id}")
        except ray.exceptions.GetTimeoutError:
            # Timeout - need to check which ones failed
            logger.warning("Some health checks timed out, checking individual results...")
            self._check_individual_results(engine_tasks)
        except Exception as e:
            # Some other error - check each one individually
            logger.warning(f"Batch health check failed with error: {e}, checking individually...")
            self._check_individual_results(engine_tasks)

    def _check_individual_results(self, engine_tasks: list) -> None:
        """Check health check results individually after batch failure."""
        for engine_id, _engine, ref in engine_tasks:
            try:
                result = ray.get(ref, timeout=0)  # Non-blocking check
                if result is not True:
                    logger.error(f"Health check returned non-True for engine {engine_id}: {result}. Killing actor.")
                    self._kill_engine(rollout_engine_id=engine_id)
                else:
                    logger.debug(f"Health check passed for rollout engine {engine_id}")
            except ray.exceptions.GetTimeoutError:
                logger.error(f"Health check timed out for rollout engine {engine_id}. Killing actor.")
                self._kill_engine(rollout_engine_id=engine_id)
            except Exception as e:
                logger.error(f"Health check failed for rollout engine {engine_id}: {e}. Killing actor.")
                self._kill_engine(rollout_engine_id=engine_id)

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
