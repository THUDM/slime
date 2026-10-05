import itertools
import logging
import time
from pathlib import Path
from typing import Any

import ray

from slime.data.batch_builder import BatchBuilder
from slime.data.transport import DiskPayloadRef, accept_raw_rollout, check_rollout_storage, load_rollout_samples
from slime.observability import logging_utils
from slime.observability.logging_utils import configure_logger, init_tracking
from slime.observability.rollout_data_utils import (
    load_debug_rollout_data,
    save_debug_rollout_data,
    validate_rollout_id_annotated,
)
from slime.observability.rollout_metrics import log_eval_rollout_data, log_rollout_data
from slime.rollout.base_types import RolloutFnTrainOutput, call_rollout_fn
from slime.rollout.sample_hooks import set_current_rollout_id
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.misc import load_function
from slime.utils.staleness import fully_async_metrics_enabled

from .utils import Lock, add_default_ray_env_vars

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg, *, restore_plan=None, persistent=False):
        configure_logger()

        self.recovery = None
        self.placement_groups = None
        self._recovery_admission_was_paused = None
        self._trainer_reconnect_pending = False
        if persistent:
            from slime.ray.placement_group import create_placement_groups
            from slime.ray.training_recovery import TrainingRecovery, training_recovery_enabled

            assert training_recovery_enabled(args)
            self.recovery = TrainingRecovery(args, restore_plan)
            # This detached actor owns the placement groups and all serving
            # children. Their lifetimes therefore survive the training driver.
            self.placement_groups = create_placement_groups(args, independent_rollout=True)
            pg = self.placement_groups["rollout"]

        self.pg = pg
        self.args = args
        self.controller = None
        self._owns_controller = False
        self.weight_version = None
        if args.rollout_data_transport == "straw":
            check_rollout_storage(args)

        rollout_init_handles: list[Any] = []
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            from slime.backends.sglang_utils.deployment import start_rollout_servers

            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

        data_source_cls = load_function(self.args.data_source_path)
        if args.rollout_data_transport == "straw":
            from slime.data.queue_data_source import QueueDataSource, QueueReader, create_queue_controller

            if data_source_cls is QueueDataSource:
                self.controller = create_queue_controller(args, restore_plan=restore_plan)
                self._owns_controller = True
                self.data_source = data_source_cls(args, controller=self.controller, restore_plan=restore_plan)
            else:
                self.data_source = data_source_cls(args)
                if isinstance(self.data_source, QueueReader):
                    self.controller = self.data_source.controller
                else:
                    self.controller = create_queue_controller(args, restore_plan=restore_plan)
                    self._owns_controller = True
        else:
            self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.batch_builder = BatchBuilder(args, controller=self.controller)
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
        if self.recovery is not None:
            self.recovery.release_trainers()
            self.recovery.release_batches()
        for monitor in self._health_monitors:
            monitor.stop()
        if close := getattr(self.data_source, "close", None):
            close()
        if self._owns_controller:
            controller = self.controller
            ray.get(controller.close.remote())
            ray.kill(controller, no_restart=True)
            self.controller = None
            self._owns_controller = False
        if self.recovery is not None:
            from ray.util.placement_group import remove_placement_group

            engines = [
                engine for server in self.servers.values() for engine in server.all_engines if engine is not None
            ]
            for result in [engine.shutdown.remote() for engine in engines]:
                try:
                    ray.get(result)
                except ray.exceptions.RayActorError:
                    pass  # An unhealthy engine may already have exited.
            for engine in engines:
                ray.kill(engine, no_restart=True)
            ray.kill(self.rollout_engine_lock, no_restart=True)
            groups = {placement[0] for placement in self.placement_groups.values() if placement and placement[0]}
            for group in groups:
                remove_placement_group(group)
        from slime.data.transport import seal_rollout_store

        seal_rollout_store(self.args)
        logging_utils.finish_tracking(self.args)

    def attach_training(self, args, job_id):
        """Claim a stopped driver's session and prepare a new trainer topology."""
        from ray.util.placement_group import remove_placement_group

        from slime.ray.placement_group import _create_placement_group
        from slime.ray.training_recovery import TrainingResume

        recovery = self.recovery
        assert recovery is not None
        recovery.validate_attachment(args)
        reused = recovery.attachments > 0
        self.health_monitoring_pause()
        was_paused = self.pause_rollout_admission()
        if self._recovery_admission_was_paused is None:
            self._recovery_admission_was_paused = was_paused
        recovery.release_trainers()
        num_train_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        actor_pg = self.placement_groups["actor"]
        if args.colocate:
            if num_train_gpus > len(actor_pg[1]):
                raise ValueError(
                    "Restarted colocated trainer exceeds the retained GPU placement; change parallelism within its GPU capacity"
                )
        elif num_train_gpus != len(actor_pg[1]):
            if actor_pg[0] is not None:
                remove_placement_group(actor_pg[0])
            self.placement_groups["actor"] = _create_placement_group(num_train_gpus)
            self.placement_groups["critic"] = self.placement_groups["actor"] if args.use_critic else None

        if reused:
            self._trainer_reconnect_pending = True
            # Old NCCL peers have exited. Remove their serving-side groups before
            # the new Megatron ranks reconnect, without restarting any engines.
            engines = [engine for engine in self.rollout_engines if engine is not None]
            ray.get([engine.reset_weights_update_groups.remote() for engine in engines])
            ray.kill(self.rollout_engine_lock, no_restart=True)
            self.rollout_engine_lock = Lock.options(num_cpus=0, num_gpus=0).remote()
            server = self._get_updatable_server()
            if server:
                for group in server.server_groups:
                    group.num_new_engines = len([engine for engine in group.engines if engine is not None])

        configuration = dict(recovery.resume_configuration)
        if reused:
            server = self._get_updatable_server()
            versions = ray.get([engine.get_weight_version.remote() for engine in server.engines]) if server else []
            configuration["update_weight_start_version"] = max(
                (int(version) for version in versions if str(version).isdigit()), default=0
            )
        for name in ("sglang_router_ip", "sglang_router_port", "sglang_model_routers"):
            if hasattr(self.args, name):
                setattr(args, name, getattr(self.args, name))
        for name, value in configuration.items():
            setattr(args, name, value)
        self.args = args
        self.batch_builder.args = args
        self.data_source.args = args
        recovery.args = args
        recovery.driver_job_id = job_id
        recovery.attachments += 1
        return TrainingResume(self.placement_groups, recovery.restore_plan, configuration, reused)

    def register_training_actors(self, role, actors, configuration):
        return self.recovery.register_trainers(role, actors, configuration)

    def detach_training(self, job_id):
        if self.recovery is None or self.recovery.driver_job_id != job_id:
            raise RuntimeError("Only the owning driver can detach its training session")
        self.health_monitoring_pause()
        self._recovery_admission_was_paused = self.pause_rollout_admission()
        self.recovery.release_trainers()
        self.recovery.driver_job_id = None
        logger.warning("Training stopped; preserving rollout session and replay batches for manual restart")

    def training_ready(self):
        if self._recovery_admission_was_paused is not None:
            self.resume_rollout_admission(self._recovery_admission_was_paused)
            self._recovery_admission_was_paused = None

    def checkpoint_committed(self, rollout_id):
        if self.recovery is not None:
            self.recovery.checkpoint_committed(rollout_id)

    def get_weight_version(self):
        server = self._get_updatable_server()
        engines = [engine for engine in server.engines if engine is not None] if server else []
        versions = ray.get([engine.get_weight_version.remote() for engine in engines])
        if not versions:
            return None
        if len(set(versions)) != 1 or not str(versions[0]).isdigit():
            raise RuntimeError(f"Cannot checkpoint inconsistent serving weight versions: {versions}")
        return int(versions[0])

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
        if self._trainer_reconnect_pending:
            num_new = len(engines)
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
        if self.recovery is not None and rollout_id in self.recovery.batches:
            batch = self.recovery.batches[rollout_id]
            logger.info("Replaying retained rollout %s with the current trainer parallelism", rollout_id)
            if batch.converted is not None:
                self.batch_builder.batch_id = batch.batch_id
                refs = self.batch_builder.split_by_dp(self.recovery.load_converted(rollout_id), publish_batch=False)
                if self.args.rollout_data_transport == "straw":
                    self.recovery.retain_replay_shards(rollout_id, self.batch_builder.replay_refs)
                return refs
            data, metrics = self.recovery.load_raw(rollout_id), None
        else:
            data, metrics = self._get_rollout_data(rollout_id=rollout_id)
            save_debug_rollout_data(
                self.args.save_debug_rollout_data,
                data,
                rollout_id=rollout_id,
                evaluation=False,
                args=self.args,
                reference=self.batch_builder.raw_ref if self.args.rollout_data_transport == "straw" else None,
            )
            if self.recovery is not None:
                self.recovery.remember_raw(
                    rollout_id,
                    (
                        self.batch_builder.raw_ref
                        if self.args.rollout_data_transport == "straw"
                        else self.args.save_debug_rollout_data
                    ),
                )
        log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time, weight_version=self.weight_version
        )
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        cached = self.batch_builder.begin(data)
        if cached is not None:
            return cached
        data = self.batch_builder.convert(data)
        if self.recovery is not None:
            self.recovery.remember_converted(rollout_id, data, self.batch_builder.batch_id)
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
        if self.recovery is not None:
            self.recovery.release_replay_shards(rollout_id)

    def load(self, rollout_id=None):
        from slime.data.checkpoint import SourceRestore

        if self.recovery is not None and self.recovery.loaded:
            return

        source_restore = self.data_source.load(rollout_id)
        # Custom sources keep their existing load() contract; only the built-in
        # queue returns a source/builder restoration handoff.
        self.batch_builder.load(
            rollout_id, source_restore=source_restore if isinstance(source_restore, SourceRestore) else None
        )
        if self.recovery is not None:
            self.recovery.initial_load_completed(rollout_id + 1)

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
        self._trainer_reconnect_pending = False

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
            if (
                self.args.rollout_data_transport == "straw"
                and self.args.load_debug_rollout_data.endswith(".straw.json")
                and self.args.load_debug_rollout_data_subsample is None
            ):
                from slime.data.archive import RolloutArchive

                path = self.args.load_debug_rollout_data.format(rollout_id=rollout_id)
                with RolloutArchive(Path(path).expanduser()) as archive:
                    if (
                        archive.store.backend.root != Path(self.args.rollout_data_dir).resolve()
                        or archive.manifest.manifest.segment.run_id != self.args.rollout_queue_run_id
                    ):
                        raise ValueError("Debug rollout archives must belong to the same Straw storage pool and run")
                    data = archive.load_samples()
                    refs = [archive.contents["raw"]] if "raw" in archive.contents else archive.contents["chunks"]
                    self.batch_builder.raw_ref = accept_raw_rollout(
                        RolloutFnTrainOutput(samples=data, sample_refs=refs),
                        self.args,
                        rollout_id,
                        controller=self.controller,
                    )
                return data, None
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
                self.weight_version = max(map(int, versions)) if valid else None
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            if self.args.rollout_data_transport == "straw":
                samples = getattr(data, "samples", None)
                data = accept_raw_rollout(data, self.args, rollout_id, controller=self.controller)
                self.batch_builder.raw_ref = data
                metrics = data.metrics
                data = (
                    samples
                    if isinstance(samples, list) and all(not isinstance(group, DiskPayloadRef) for group in samples)
                    else load_rollout_samples(data)
                )
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
