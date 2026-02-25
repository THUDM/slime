import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
import yaml
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EngineGroupConfig:
    """Configuration for a single engine group.

    Attributes:
        worker_type: One of "regular", "prefill", "decode", or "placeholder".
                     "placeholder" reserves GPU slots without creating engines.
        num_gpus: Total number of GPUs for this group.
        num_gpus_per_engine: GPUs per engine for this group.  Overrides the
                             model-level or global ``--rollout-num-gpus-per-engine``.
        overrides: Optional dict of SGLang ``ServerArgs`` field overrides.
                   These are applied on top of the base CLI ``--sglang-*``
                   arguments in ``_compute_server_args``.
    """

    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        valid_types = {"regular", "prefill", "decode", "placeholder"}
        assert (
            self.worker_type in valid_types
        ), f"Invalid worker_type '{self.worker_type}', must be one of {valid_types}"
        assert self.num_gpus > 0, f"num_gpus must be > 0, got {self.num_gpus}"


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single model deployment.

    Attributes:
        name: Unique name for this model (e.g. "actor", "reward").
        model_path: HF checkpoint path.  Falls back to ``args.hf_checkpoint``.
        num_gpus_per_engine: Default GPUs per engine for all groups in this
                             model.  Individual groups can override.
        engine_groups: Engine group configurations for this model.
    """

    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    engine_groups: list[EngineGroupConfig] = dataclasses.field(default_factory=list)

    def resolve(self, args) -> None:
        """Resolve per-group defaults from model-level then args-level values."""
        default_gpus_per_engine = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
        default_model_path = self.model_path or args.hf_checkpoint
        for g in self.engine_groups:
            if g.num_gpus_per_engine is None:
                g.num_gpus_per_engine = default_gpus_per_engine
            # Inject model_path into overrides so _compute_server_args picks it up.
            if "model_path" not in g.overrides:
                g.overrides["model_path"] = default_model_path

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(g.worker_type in ("prefill", "decode") for g in self.engine_groups)

    @property
    def total_num_gpus(self) -> int:
        return sum(g.num_gpus for g in self.engine_groups)


@dataclasses.dataclass
class SglangConfig:
    """Configuration for SGLang engine deployment.

    Loaded from ``--sglang-config`` YAML file.

    **Config format**::

        sglang:
          - name: actor
            model_path: /path/to/actor
            num_gpus_per_engine: 2
            engine_groups:
              - worker_type: prefill
                num_gpus: 4
                num_gpus_per_engine: 2
              - worker_type: decode
                num_gpus: 8
                num_gpus_per_engine: 4
          - name: reward
            model_path: /path/to/reward
            engine_groups:
              - worker_type: regular
                num_gpus: 4

    Each model gets its own router.  ``placeholder`` groups reserve GPU
    slots without creating engines.  ``overrides`` are ``ServerArgs``
    field names applied on top of the base ``--sglang-*`` CLI args.
    """

    models: list[ModelConfig]

    @staticmethod
    def from_yaml(path: str) -> "SglangConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        assert "sglang" in data, (
            f"sglang config must have a 'sglang' key, got {list(data.keys())}. "
            f"Wrap your engine_groups inside a model entry under 'sglang'."
        )
        models = []
        for m in data["sglang"]:
            groups = [EngineGroupConfig(**g) for g in m.get("engine_groups", [])]
            models.append(
                ModelConfig(
                    name=m["name"],
                    model_path=m.get("model_path"),
                    num_gpus_per_engine=m.get("num_gpus_per_engine"),
                    engine_groups=groups,
                )
            )
        return SglangConfig(models=models)

    @staticmethod
    def from_prefill_num_servers(args) -> "SglangConfig":
        """Build a config equivalent to the legacy --prefill-num-servers flag."""
        total_gpus = args.rollout_num_gpus
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = total_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {total_gpus}, prefill {prefill_gpus}"
        return SglangConfig(
            models=[
                ModelConfig(
                    name="default",
                    engine_groups=[
                        EngineGroupConfig(worker_type="prefill", num_gpus=prefill_gpus),
                        EngineGroupConfig(worker_type="decode", num_gpus=decode_gpus),
                    ],
                )
            ]
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(m.has_pd_disaggregation for m in self.models)

    @property
    def total_num_gpus(self) -> int:
        return sum(m.total_num_gpus for m in self.models)


@dataclasses.dataclass
class EngineGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple EngineGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", or "decode"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next engine group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            # Compute base_port from the maximum cursor across all nodes that
            # this group's engines may land on (conservative: just use global max).
            base_port = max(port_cursors.values()) if port_cursors else 15000
            addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=rollout_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more engine groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple EngineGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    engine_groups: list[EngineGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.engine_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.engine_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.engine_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.engine_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.engine_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.engine_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.engine_groups}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.engine_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.engine_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        new_engines_all = []
        for g, dead_indices in zip(self.engine_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.args.offload_rollout and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                new_engines_all.extend(new_engines)

        if release_handles:
            ray.get(release_handles)
            ray.get(
                [engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines_all]
            )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.engine_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.engine_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        init_tracking(args, primary=False)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.engine_groups:
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

        if self.server and self.server.engine_groups[0].all_engines and self.server.engine_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.engine_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()

    @property
    def server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_server(self, model_name: str | None = None) -> RolloutServer | None:
        if model_name is None:
            return self.server
        return self.servers.get(model_name)

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_rollout_engines_and_lock(self, model_name: str | None = None):
        srv = self._get_server(model_name)
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def recover_rollout_engines(self, model_name: str | None = None):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        self.health_monitoring_pause()
        srv = self._get_server(model_name)
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_num_new_engines(self, model_name: str | None = None):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_server(model_name)
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
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples and not self.args.debug_rollout_only:
                global_batch_size = self.args.global_batch_size
                if self.args.use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Calculate dynamic global_batch_size to ensure only one training step.

        Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
        This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        # Round down to a multiple of dp_size to ensure only one training step
        dynamic_gbs = (num_samples // dp_size) * dp_size

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        # Calculate how many samples will be discarded
        wasted = num_samples - dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} (num_samples={num_samples}, dp_size={dp_size}, num_steps=1, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different engine groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sgl router or slime router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values.
    When ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        assert not has_pd_disaggregation, "slime router does not support PD disaggregation."
        import copy

        from slime.router.router import run_router

        router_args = copy.copy(args)
        router_args.sglang_router_ip = router_ip
        router_args.sglang_router_port = router_port

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = router_ip
        router_args.port = router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if hasattr(args, "sglang_router_policy") and args.sglang_router_policy:
            router_args.policy = args.sglang_router_policy

        if has_pd_disaggregation:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of engine groups.  Engine groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        engine_groups: list[EngineGroup] = []
        all_init_handles: list = []
        port_cursors: dict[int, int] = {}

        for group_cfg in model_cfg.engine_groups:
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group = EngineGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=group_cfg.overrides,
                router_ip=router_ip,
                router_port=router_port,
            )
            handles, port_cursors = group.start_engines(port_cursors)
            all_init_handles.extend(handles)
            engine_groups.append(group)

            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus

        if all_init_handles:
            ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            engine_groups=engine_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
        )

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                engine_groups=[EngineGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
