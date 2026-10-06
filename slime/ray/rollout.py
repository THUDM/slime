import itertools
import logging
import time
import uuid
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
from slime.utils.http_utils import init_http_client
from slime.utils.misc import load_function
from slime.utils.staleness import fully_async_metrics_enabled

from .utils import Lock, add_default_ray_env_vars

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """Generate and convert batches, borrowing internal engines from serving.

    The serving owner retains engines and queue state across manager death.
    Recovery retains raw/global converted batches across trainer death. This
    manager owns transient readers, conversion and the current trainer's shards.
    """

    def __init__(self, args, pg, *, restore_plan=None, serving=None, deployment=None):
        configure_logger()

        from slime.ray.training_recovery import TrainingRecovery, training_recovery_enabled

        self.serving = serving
        self.recovery = (
            TrainingRecovery(args, restore_plan, retained_serving=deployment.reused)
            if serving is not None and training_recovery_enabled(args)
            else None
        )
        self._recovery_admission_was_paused = None
        self.pg = pg
        self.args = args
        self.controller = deployment.controller if deployment else None
        self._owns_controller = False
        self.weight_version = None
        if args.rollout_data_transport == "straw":
            check_rollout_storage(args)

        rollout_init_handles: list[Any] = []
        if deployment is not None:
            self.servers = deployment.servers
            init_http_client(args)
        elif self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            from slime.backends.sglang_utils.deployment import start_rollout_servers

            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

        data_source_cls = load_function(self.args.data_source_path)
        if args.rollout_data_transport == "straw":
            from slime.data.queue_data_source import QueueDataSource, QueueReader, create_queue_controller

            # Custom sources keep their args-only constructor and may already
            # own a controller. Construct them before creating a default one.
            if data_source_cls is not QueueDataSource:
                self.data_source = data_source_cls(args)
                if isinstance(self.data_source, QueueReader):
                    self.controller = self.data_source.controller
            if self.controller is None:
                if serving is not None:
                    self.controller = ray.get(serving.get_queue_controller.remote())
                else:
                    self.controller = create_queue_controller(args, restore_plan=restore_plan)
                    self._owns_controller = True
            if data_source_cls is QueueDataSource:
                self.data_source = data_source_cls(
                    args,
                    controller=self.controller,
                    restore_plan=restore_plan,
                    reader_generation=uuid.uuid4().hex if serving is not None else "",
                )
                if self.recovery is not None and self.recovery.source_state is not None:
                    # A dead manager's readers may have delivered these samples
                    # already. Exclude retained batches before fencing/replaying
                    # its readers, so each accepted sample is delivered once.
                    excluded = set()
                    for rollout_id in self.recovery.batches:
                        for sample in self.recovery.load_raw(rollout_id):
                            excluded.update(getattr(sample, "_queue_source_positions", []))
                            if receipt := getattr(sample, "_queue_receipt", None):
                                excluded.add(receipt["position"])
                    self.data_source.restore_manager(self.recovery.source_state, excluded=sorted(excluded))
        else:
            self.data_source = data_source_cls(args)
            if self.recovery is not None and self.recovery.source_state is not None:
                self.data_source.load_state_dict(self.recovery.source_state)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.batch_builder = BatchBuilder(args, controller=self.controller)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if rollout_init_handles:
            ray.get(rollout_init_handles)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = (
            deployment.engine_lock
            if deployment is not None
            else Lock.options(
                num_cpus=1,
                num_gpus=0,
                runtime_env={"env_vars": add_default_ray_env_vars()},
            ).remote()
        )
        self.rollout_id = -1
        if self.recovery is not None:
            self.recovery.source_state = self._source_state()
            self.recovery.persist()

    def pause_rollout_admission(self):
        """Stop the distributed producer before engines drain for a weight update."""
        worker = getattr(self.data_source, "consumers", {}).get("fully_async")
        return worker.pause(drain=False) if worker is not None else True

    def resume_rollout_admission(self, was_paused):
        if not was_paused:
            self.data_source.consumers["fully_async"].resume()

    def dispose(self):
        if close := getattr(self.data_source, "close", None):
            close()
        if self.recovery is not None:
            self.recovery.release_batches()
            self.recovery.journal.unlink(missing_ok=True)
        if self.serving is not None:
            ray.get(self.serving.dispose.remote())
        elif self._owns_controller:
            ray.get(self.controller.close.remote())
            ray.kill(self.controller, no_restart=True)
        from slime.data.transport import seal_rollout_store

        seal_rollout_store(self.args)
        logging_utils.finish_tracking(self.args)

    def attach_training(self, args, deployment):
        """Bind a driver to the same serving owner, including after manager death."""
        from slime.ray.training_recovery import TrainingResume

        was_paused = self.pause_rollout_admission()
        if self._recovery_admission_was_paused is None:
            self._recovery_admission_was_paused = was_paused
        self.pg = deployment.placements["rollout"]
        self.servers = deployment.servers
        if self.recovery is not None:
            self.recovery.reconcile_checkpoint()
        configuration = dict(self.recovery.resume_configuration) if self.recovery is not None else {}
        if deployment.reused:
            configuration["update_weight_start_version"] = (
                ray.get(self.serving.get_weight_version.remote(allow_inconsistent=True)) or 0
            )
        for name, value in configuration.items():
            setattr(args, name, value)
        self.args = self.batch_builder.args = self.data_source.args = args
        if self.recovery is not None:
            self.recovery.args = args
        self.rollout_engine_lock = deployment.engine_lock
        return TrainingResume(
            self.recovery.restore_plan if self.recovery is not None else deployment.restore_plan,
            configuration,
            deployment.reused,
        )

    def register_training_actors(self, role, actors, configuration):
        if self.serving is not None:
            ray.get(self.serving.register_trainers.remote(role, actors))
        if self.recovery is not None:
            return self.recovery.resume_role(role, configuration)
        return {"update_weight_start_version": getattr(self.args, "update_weight_start_version", 0)}

    def detach_training(self, job_id):
        self._recovery_admission_was_paused = self.pause_rollout_admission()
        ray.get(self.serving.detach_training.remote(job_id))
        logger.warning("Training stopped; preserving serving and available replay batches for manual restart")

    def _source_state(self):
        """Snapshot the cursor, leaving live queue tasks with their controller."""
        from slime.data.queue_data_source import QueueDataSource

        if isinstance(self.data_source, QueueDataSource):
            return self.data_source.manager_state()
        if state_dict := getattr(self.data_source, "state_dict", None):
            return state_dict()
        return None

    def training_ready(self):
        if self._recovery_admission_was_paused is not None:
            self.resume_rollout_admission(self._recovery_admission_was_paused)
            self._recovery_admission_was_paused = None

    def checkpoint_committed(self, rollout_id):
        if self.recovery is not None:
            self.recovery.checkpoint_committed(rollout_id)

    def get_weight_version(self):
        if self.serving is not None:
            return ray.get(self.serving.get_weight_version.remote())
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
        return next((server for server in self.servers.values() if server.update_weights), None)

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
        if self.serving is not None:
            return ray.get(self.serving.get_updatable_engines_and_lock.remote())
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
        # The legacy flag still selects deliberate CI crash injection. It no
        # longer gates internal health checks, and external servers are never
        # eligible for this internal-engine test.
        if self.serving is not None and self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self.servers = ray.get(self.serving.try_ci_fault_injection.remote())
        result = self._generate(rollout_id, start_time)
        if self.serving is not None:
            # Refresh the local snapshot before offload/pause can target a dead
            # actor. Replacement engines are created only before weight update.
            self.servers = ray.get(self.serving.finish_rollout.remote())
        return result

    def _generate(self, rollout_id, start_time):
        if self.recovery is not None and rollout_id in self.recovery.batches:
            batch = self.recovery.batches[rollout_id]
            logger.info("Replaying retained rollout %s with the current trainer parallelism", rollout_id)
            if batch.converted is not None:
                refs = self.batch_builder.replay_converted(self.recovery.load_converted(rollout_id), batch.batch_id)
                if self.args.rollout_data_transport == "straw" and self.batch_builder.replay_refs:
                    self.recovery.retain_replay_shards(rollout_id, self.batch_builder.replay_refs)
                return refs
            data, metrics = self.recovery.load_raw(rollout_id), None
            if self.args.rollout_data_transport == "straw":
                self.batch_builder.raw_ref = batch.raw
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
                    source_state=self._source_state(),
                )
        log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time, weight_version=self.weight_version
        )
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        cached = self.batch_builder.begin(
            data, replay=self.recovery is not None and rollout_id in self.recovery.batches
        )
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
        if self.serving is not None:
            self.servers = ray.get(self.serving.finish_rollout.remote())
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
            self.recovery.source_state = self._source_state()
            self.recovery.initial_load_completed(rollout_id + 1)

    def offload(self):
        # These controls do not change topology; both internal and external
        # serving use the current handle snapshot. The owner serializes health
        # checks with this pause before we change memory residency.
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
        if self.serving is not None:
            self.servers = ray.get(self.serving.recover_updatable_engines.remote())
            return
        # Keep the existing external-serving recovery policy.
        if self.rollout_id == -1:
            return
        server = self._get_updatable_server()
        if server is not None:
            server.recover()

    def clear_updatable_num_new_engines(self):
        if self.serving is not None:
            return ray.get(self.serving.clear_updatable_num_new_engines.remote())
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        if self.serving is not None:
            return ray.get(self.serving.health_monitoring_pause.remote())

    def health_monitoring_resume(self) -> None:
        if self.serving is not None:
            self.servers = ray.get(self.serving.health_monitoring_resume.remote())
            return

    def check_weights(self, action: str):
        return ray.get(
            [engine.check_weights.remote(action=action) for engine in self.rollout_engines if engine is not None]
        )

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
