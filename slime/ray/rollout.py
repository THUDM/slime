import itertools
import logging
import time
from typing import Any

import ray

from slime.observability import logging_utils
from slime.observability.logging_utils import configure_logger, init_tracking
from slime.observability.rollout_data_utils import (
    load_debug_rollout_data,
    save_debug_rollout_data,
    validate_rollout_id_annotated,
)
from slime.observability.rollout_metrics import log_eval_rollout_data, log_rollout_data
from slime.rollout.base_types import call_rollout_fn
from slime.rollout.batch_builder import BatchBuilder
from slime.rollout.sample_hooks import set_current_rollout_id
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.misc import load_function
from slime.utils.rollout_transport import accept_raw_rollout, check_rollout_storage, load_rollout_samples
from slime.utils.staleness import fully_async_metrics_enabled

from .utils import Lock, add_default_ray_env_vars

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args
        if args.rollout_data_transport == "straw":
            check_rollout_storage(args)
            from slime.rollout.queue_data_source import create_queue_controller

            create_queue_controller(args)

        rollout_init_handles: list[Any] = []
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            from slime.backends.sglang_utils.deployment import start_rollout_servers

            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.batch_builder = BatchBuilder(args)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if rollout_init_handles:
            ray.get(rollout_init_handles)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self.server
            and self.server.server_groups
            and self.server.server_groups[0].all_engines
            and self.server.server_groups[0].all_engines[0]
        ):
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def pause_rollout_admission(self):
        """Stop the distributed producer before engines drain for a weight update."""
        worker = getattr(self.data_source, "consumers", {}).get("fully_async")
        return worker.pause(drain=False) if worker is not None else True

    def resume_rollout_admission(self, was_paused):
        if not was_paused:
            self.data_source.consumers["fully_async"].resume()

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()
        if close := getattr(self.data_source, "close", None):
            close()
        if controller := getattr(self.args, "_rollout_queue_controller", None):
            ray.get(controller.close.remote())
            ray.kill(controller, no_restart=True)
            self.args._rollout_queue_controller = None
        from slime.utils.rollout_transport import seal_rollout_store

        seal_rollout_store(self.args)
        logging_utils.finish_tracking(self.args)

    @property
    def server(self) -> Any | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> Any | None:
        """Return the server with ``update_weights=True``.

        When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported).
        """
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates.

        Returns engines from the first model that has
        ``update_weights=True``.  Frozen models (reference, reward,
        etc.) are automatically excluded.
        """
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        parallel_configs = srv.engine_parallel_configs if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets, parallel_configs

    def get_num_rollout_per_epoch(self):
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.batch_builder.rollout_id = rollout_id
        set_current_rollout_id(rollout_id)
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        save_debug_rollout_data(
            self.args.save_debug_rollout_data,
            data,
            rollout_id=rollout_id,
            evaluation=False,
            args=self.args,
        )
        log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        cached = self.batch_builder.begin(data)
        if cached is not None:
            return cached
        data = self.batch_builder.convert(data)
        return self.batch_builder.split_by_dp(data)

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        set_current_rollout_id(rollout_id)
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        save_debug_rollout_data(
            self.args.save_debug_rollout_data,
            data,
            rollout_id=rollout_id,
            evaluation=True,
            args=self.args,
        )
        log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        # Keep admission frozen across source and builder snapshots. Source.save
        # preserves this pre-existing pause instead of resuming between files.
        paused = []
        try:
            for consumer in getattr(self.data_source, "consumers", {}).values():
                paused.append((consumer, consumer.pause()))
            self.data_source.save(rollout_id)
            self.batch_builder.save(rollout_id)
        finally:
            for consumer, was_paused in paused:
                if not was_paused:
                    consumer.resume()

    def training_completed(self, rollout_id):
        self.batch_builder.training_completed(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)
        self.batch_builder.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        """Restart dead updatable rollout engines before the next weight update.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            return

        srv.recover()

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = load_debug_rollout_data(
                self.args.load_debug_rollout_data,
                rollout_id=rollout_id,
                subsample_ratio=self.args.load_debug_rollout_data_subsample,
            )
            metrics = None
        else:
            if fully_async_metrics_enabled(self.args):
                # The training loop keeps serving weights fixed while collecting
                # this batch. Query only the updatable model.
                server = self._get_updatable_server()
                engines = [engine for engine in server.engines if engine is not None] if server else []
                versions = ray.get([engine.get_weight_version.remote() for engine in engines])
                valid = bool(versions) and all(
                    str(version).isascii() and str(version).isdigit() for version in versions
                )
                self.args._rollout_weight_version = max(map(int, versions)) if valid else None
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            if self.args.rollout_data_transport == "straw":
                data = accept_raw_rollout(data, self.args, rollout_id)
                self.batch_builder.raw_ref = data
                metrics = data.metrics
                data = load_rollout_samples(data)
            else:
                metrics = data.metrics
                data = load_rollout_samples(data.samples)
            # Enforce the rollout_id contract before flattening: any list[Sample]
            # encountered in the nested output must have rollout_id set on every
            # element. Default rollouts inherit it from the data source; compact /
            # subagent paths that split one rollout into N training samples must
            # set the same rollout_id on every sibling so the loss reducer counts
            # the rollout once instead of N times.
            validate_rollout_id_annotated(data)
            # flatten the data if it is a list of lists
            while data and isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def set_train_parallel_config(self, config: dict):
        self.batch_builder.train_parallel_config = config
