import copy
import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import time
from typing import Any

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.external import start_external_rollout_servers
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.observability import logging_utils
from slime.observability.logging_utils import configure_logger, init_tracking
from slime.observability.rollout_data_utils import (
    load_debug_rollout_data,
    save_debug_rollout_data,
    tensorize_rollout_data_for_training,
    validate_rollout_id_annotated,
    validate_rollout_routed_experts_for_replay,
)
from slime.observability.rollout_metrics import log_eval_rollout_data, log_rollout_data
from slime.rollout.base_types import call_rollout_fn
from slime.rollout.sample_hooks import set_current_rollout_id
from slime.utils.data import get_source
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.misc import Box, load_function
from slime.utils.rs_refill import (
    attach_proximal_log_probs,
    fingerprint_rs_train_data,
    merge_replacement_metrics,
    merge_selected_log_prob_caches,
    plan_topology_aligned_rs_refill,
    select_accepted_groups,
    snapshot_sample_masks,
    validate_initial_policy_staleness,
    validate_refill_rollout_ids,
    validate_replacement_policy_version,
    validate_rs_refill_target_batch_alignment,
    validate_rs_train_data_fingerprint,
    validate_sample_masks,
)
from slime.utils.types import Sample

from .rollout_validation import validate_server_group_gpu_indices
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock, add_default_ray_env_vars

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def parallel_config(self) -> dict[str, Any]:
        """Return the SGLang parallel args that affect rank-local expert routing."""
        overrides = {key.replace("-", "_"): value for key, value in self.sglang_overrides.items()}
        pp_size = int(overrides.get("pp_size", getattr(self.args, "sglang_pp_size", 1)))
        tp_size = int(overrides.get("tp_size", self.num_gpus_per_engine // pp_size))
        return {
            "tp_size": tp_size,
            "pp_size": pp_size,
            "ep_size": int(overrides.get("ep_size", getattr(self.args, "sglang_ep_size", 1))),
            "moe_dp_size": int(overrides.get("moe_dp_size", getattr(self.args, "sglang_moe_dp_size", 1))),
        }

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpus_per_engine_on_node = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpus_per_engine_on_node=num_gpus_per_engine_on_node,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpus_per_engine_on_node
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
                    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
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
                    "env_vars": add_default_ray_env_vars(env_vars),
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

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def engine_parallel_configs(self) -> list[dict[str, Any]]:
        """Per-engine SGLang parallel config, parallel to ``engines``."""
        return [g.parallel_config() for g in self.server_groups for _ in g.engines]

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        rollout_init_handles: list[Any] = []
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

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

        if rollout_init_handles:
            ray.get(rollout_init_handles)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote()
        self.rollout_id = -1
        self._pending_rs_batches: dict[int, dict[str, Any]] = {}

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

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()
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
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        if getattr(self.args, "rs_batch_refill", False):
            return self._generate_rs_candidates(rollout_id)

        start_time = time.time()
        self.rollout_id = rollout_id
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
        )
        log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)

    def _call_rollout_for_group_count(
        self,
        rollout_id: int,
        group_count: int,
        *,
        known_rollout_ids: set[Any] | None = None,
    ):
        # Evaluation shares this process-global hook context, so restore the
        # pending batch ID before every reactive generation.
        self.rollout_id = rollout_id
        set_current_rollout_id(rollout_id)
        call_args = copy.copy(self.args)
        call_args.rollout_batch_size = group_count
        # A refill should request only the missing groups. The default rollout
        # loop otherwise inherits the original full-batch sampling granularity.
        call_args.over_sampling_batch_size = group_count
        output = call_rollout_fn(
            self.generate_rollout,
            call_args,
            rollout_id,
            self.data_source,
            evaluation=False,
        )
        groups = output.samples
        if len(groups) != group_count:
            raise RuntimeError(f"RS refill requested {group_count} prompt groups but rollout returned {len(groups)}")
        for group in groups:
            if not isinstance(group, list) or len(group) != self.args.n_samples_per_prompt:
                raise RuntimeError(
                    "RS batch refill requires the rollout function to return one list[Sample] per prompt "
                    f"with n_samples_per_prompt={self.args.n_samples_per_prompt}."
                )
            if any(isinstance(sample, list) for sample in group):
                raise RuntimeError("RS batch refill does not support fan-out/compact nested rollout samples yet.")
            if any(sample.index is None or sample.group_index is None for sample in group):
                raise RuntimeError("RS batch refill requires stable sample.index and sample.group_index values.")
        candidate_rollout_ids = validate_refill_rollout_ids(groups, known_rollout_ids=known_rollout_ids)
        return groups, output.metrics or {}, candidate_rollout_ids

    def _generate_rs_candidates(self, rollout_id: int):
        if rollout_id in self._pending_rs_batches:
            raise RuntimeError(f"RS candidate batch {rollout_id} already exists")
        start_time = time.perf_counter()
        self.rollout_id = rollout_id
        set_current_rollout_id(rollout_id)
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        candidate_count = self.args.rollout_batch_size
        groups, metrics, candidate_rollout_ids = self._call_rollout_for_group_count(rollout_id, candidate_count)
        self._pending_rs_batches[rollout_id] = {
            "accepted": [],
            "unscored": groups,
            "initial_candidate_count": candidate_count,
            "round": 0,
            "seen_sample_indices": set(),
            "seen_group_indices": set(),
            "seen_rollout_ids": candidate_rollout_ids,
            "awaiting_log_prob_indices": None,
            "awaiting_log_prob_bytes": None,
            "proximal_log_probs_by_sample_index": {},
            "retained_logprob_cache_bytes": 0,
            "accepted_mask_fingerprints": {},
            "metrics": dict(metrics),
            "initial_generation_seconds": time.perf_counter() - start_time,
        }
        return rollout_id

    def prepare_rs_candidate_data(self, rollout_id: int):
        pending = self._pending_rs_batches.get(rollout_id)
        if pending is None:
            raise RuntimeError(f"No pending RS candidate batch for rollout_id={rollout_id}")
        if pending["awaiting_log_prob_indices"] is not None or pending["awaiting_log_prob_bytes"] is not None:
            raise RuntimeError(f"RS candidate batch {rollout_id} has selected log probabilities awaiting collection")
        groups = pending["unscored"]
        if not groups:
            raise RuntimeError(f"RS candidate batch {rollout_id} has no unscored groups")

        samples = list(itertools.chain.from_iterable(groups))
        data = self._convert_samples_to_train_data(samples, preflight=True)
        # Preflight is forward-only, so score all candidates in one logical
        # step regardless of the final optimizer global batch size.
        return self._split_train_data_by_dp(data, global_batch_size=len(set(data["rollout_ids"])))

    def apply_rs_candidate_reports(self, rollout_id: int, actor_report_refs, preflight_seconds: float):
        pending = self._pending_rs_batches.get(rollout_id)
        if pending is None:
            raise RuntimeError(f"No pending RS candidate batch for rollout_id={rollout_id}")
        if pending["awaiting_log_prob_indices"] is not None or pending["awaiting_log_prob_bytes"] is not None:
            raise RuntimeError(f"RS candidate batch {rollout_id} already has an uncollected preflight result")

        report_wait_start = time.perf_counter()
        actor_reports = ray.get(actor_report_refs, timeout=self.args.rs_refill_rpc_timeout_seconds)
        preflight_seconds += time.perf_counter() - report_wait_start
        reports = [report for worker_reports in actor_reports if worker_reports for report in worker_reports]
        groups = pending["unscored"]
        remaining_target = self.args.rollout_batch_size - len(pending["accepted"])
        selection = select_accepted_groups(
            groups,
            reports,
            target_size=remaining_target,
            known_sample_indices=pending["seen_sample_indices"],
            known_group_indices=pending["seen_group_indices"],
        )
        if pending["round"] == 0:
            initial_staleness = validate_initial_policy_staleness(groups, reports)
        else:
            validate_replacement_policy_version(groups, reports)

        samples_by_index = {sample.index: sample for group in groups for sample in group}
        cache_bytes_by_sample_index = {}
        actor_cache_bytes = []
        float32_bytes = torch.finfo(torch.float32).bits // 8
        for worker_reports in actor_reports:
            worker_cache_bytes = 0
            for report in worker_reports or []:
                sample_index = report["sample_index"]
                cache_bytes = report.get("candidate_cache_bytes")
                if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
                    raise RuntimeError(
                        "RS candidate cache byte reports must be non-negative integers: "
                        f"sample_index={sample_index}, value={cache_bytes!r}"
                    )
                expected_cache_bytes = samples_by_index[sample_index].response_length * float32_bytes
                if cache_bytes != expected_cache_bytes:
                    raise RuntimeError(
                        "RS candidate cache byte report must exactly match the expected float32 payload: "
                        f"sample_index={sample_index}, reported={cache_bytes}, expected={expected_cache_bytes}"
                    )
                cache_bytes_by_sample_index[sample_index] = cache_bytes
                worker_cache_bytes += cache_bytes
            actor_cache_bytes.append(worker_cache_bytes)

        accepted_sample_indices = [sample.index for group in selection.accepted_groups for sample in group]
        incoming_cache_bytes = sum(cache_bytes_by_sample_index[index] for index in accepted_sample_indices)
        retained_cache_bytes = pending["retained_logprob_cache_bytes"]
        if (
            isinstance(retained_cache_bytes, bool)
            or not isinstance(retained_cache_bytes, int)
            or retained_cache_bytes < 0
        ):
            raise RuntimeError(f"Invalid retained RS proximal-logprob cache size: {retained_cache_bytes!r}")
        required_cache_bytes = retained_cache_bytes + incoming_cache_bytes
        cache_limit = self.args.rs_refill_max_candidate_cache_bytes
        peak_actor_cache_bytes = max(actor_cache_bytes, default=0)
        if peak_actor_cache_bytes > cache_limit:
            raise RuntimeError(
                "RS candidate proximal-logprob cache exceeds the per-actor limit according to actor reports: "
                f"required={peak_actor_cache_bytes}, limit={cache_limit}. "
                "Increase --rs-refill-max-candidate-cache-bytes or reduce the candidate batch/response length."
            )
        if required_cache_bytes > cache_limit:
            raise RuntimeError(
                "RS selected proximal-logprob cache would exceed the RolloutManager retained-payload limit "
                "before transfer: "
                f"retained={retained_cache_bytes}, incoming={incoming_cache_bytes}, "
                f"required={required_cache_bytes}, limit={cache_limit}. "
                "Increase --rs-refill-max-candidate-cache-bytes or reduce the batch/response length."
            )

        pending["seen_sample_indices"].update(sample.index for group in groups for sample in group)
        pending["seen_group_indices"].update(group[0].group_index for group in groups)

        for group in selection.accepted_groups:
            pending["accepted_mask_fingerprints"].update(snapshot_sample_masks(group))
        pending["awaiting_log_prob_indices"] = accepted_sample_indices
        pending["awaiting_log_prob_bytes"] = incoming_cache_bytes
        pending["accepted"].extend(selection.accepted_groups)
        if pending["round"] == 0:
            pending["metrics"]["rollout/rs_refill/initial_policy_staleness"] = initial_staleness

        pending["metrics"].update(
            {
                "rollout/rs_refill/candidate_groups": pending["initial_candidate_count"],
                "rollout/rs_refill/rejected_groups": pending["metrics"].get("rollout/rs_refill/rejected_groups", 0)
                + len(selection.rejected_groups),
                "rollout/rs_refill/surplus_groups": pending["metrics"].get("rollout/rs_refill/surplus_groups", 0)
                + len(selection.surplus_groups),
            }
        )
        pending["metrics"]["rollout/rs_refill/scored_groups"] = pending["metrics"].get(
            "rollout/rs_refill/scored_groups", 0
        ) + len(groups)
        pending["metrics"]["rollout/rs_refill/scored_trainable_tokens"] = pending["metrics"].get(
            "rollout/rs_refill/scored_trainable_tokens", 0
        ) + sum(int(report["valid_tokens"]) for report in reports)
        aggregate_cache_bytes = sum(cache_bytes_by_sample_index.values())
        pending["metrics"]["rollout/rs_refill/aggregate_candidate_logprob_cache_bytes"] = (
            pending["metrics"].get("rollout/rs_refill/aggregate_candidate_logprob_cache_bytes", 0)
            + aggregate_cache_bytes
        )
        pending["metrics"]["rollout/rs_refill/peak_aggregate_candidate_logprob_cache_bytes"] = max(
            pending["metrics"].get("rollout/rs_refill/peak_aggregate_candidate_logprob_cache_bytes", 0),
            aggregate_cache_bytes,
        )
        pending["metrics"]["rollout/rs_refill/peak_actor_candidate_logprob_cache_bytes"] = max(
            pending["metrics"].get("rollout/rs_refill/peak_actor_candidate_logprob_cache_bytes", 0),
            peak_actor_cache_bytes,
        )
        pending["metrics"]["rollout/rs_refill/logprob_cache_limit_bytes"] = cache_limit
        pending["metrics"]["rollout/rs_refill/preflight_seconds"] = (
            pending["metrics"].get("rollout/rs_refill/preflight_seconds", 0.0) + preflight_seconds
        )

        deficit = self.args.rollout_batch_size - len(pending["accepted"])
        pending["unscored"] = []
        return {
            "complete": deficit == 0,
            "exhausted": deficit > 0 and pending["round"] >= self.args.rs_refill_max_rounds,
            "deficit": deficit,
            "round": pending["round"],
            "accepted_groups": len(pending["accepted"]),
            "target_groups": self.args.rollout_batch_size,
            "accepted_sample_indices": accepted_sample_indices,
        }

    def generate_rs_replacement_candidates(self, rollout_id: int):
        """Generate the next bounded replacement set after actor caches are released."""

        pending = self._pending_rs_batches.get(rollout_id)
        if pending is None:
            raise RuntimeError(f"No pending RS candidate batch for rollout_id={rollout_id}")
        if pending["awaiting_log_prob_indices"] is not None or pending["awaiting_log_prob_bytes"] is not None:
            raise RuntimeError(
                f"Cannot generate RS replacements for batch {rollout_id} before collecting selected caches"
            )
        if pending["unscored"]:
            raise RuntimeError(f"RS candidate batch {rollout_id} already has unscored replacement groups")

        deficit = self.args.rollout_batch_size - len(pending["accepted"])
        if deficit <= 0:
            raise RuntimeError(f"RS candidate batch {rollout_id} is already complete")
        if pending["round"] >= self.args.rs_refill_max_rounds:
            raise RuntimeError(
                "RS batch refill exhausted its retry budget before optimizer.step: "
                f"accepted={len(pending['accepted'])}, target={self.args.rollout_batch_size}, "
                f"rounds={pending['round']}, remaining={deficit}."
            )

        refill_start = time.perf_counter()
        replacement_count = plan_topology_aligned_rs_refill(
            self.args,
            self.train_parallel_config,
            deficit,
        )
        refill_round = pending["round"] + 1
        pending["metrics"][f"rollout/rs_refill/round_{refill_round}/candidate_groups"] = replacement_count
        replacement_groups, replacement_metrics, replacement_rollout_ids = self._call_rollout_for_group_count(
            rollout_id,
            replacement_count,
            known_rollout_ids=pending["seen_rollout_ids"],
        )
        pending["unscored"] = replacement_groups
        pending["seen_rollout_ids"].update(replacement_rollout_ids)
        pending["round"] += 1
        merge_replacement_metrics(
            pending["metrics"],
            replacement_metrics,
            round_index=pending["round"],
        )
        pending["metrics"]["rollout/rs_refill/generated_replacement_groups"] = (
            pending["metrics"].get("rollout/rs_refill/generated_replacement_groups", 0) + replacement_count
        )
        pending["metrics"]["rollout/rs_refill/replacement_generation_seconds"] = pending["metrics"].get(
            "rollout/rs_refill/replacement_generation_seconds", 0.0
        ) + (time.perf_counter() - refill_start)
        return {
            "round": pending["round"],
            "candidate_groups": replacement_count,
        }

    def store_rs_accepted_log_probs(self, rollout_id: int, actor_cache_refs):
        """Store only caches selected by the group-atomic manager decision."""

        pending = self._pending_rs_batches.get(rollout_id)
        if pending is None:
            raise RuntimeError(f"No pending RS candidate batch for rollout_id={rollout_id}")
        expected = pending["awaiting_log_prob_indices"]
        expected_bytes = pending["awaiting_log_prob_bytes"]
        if expected is None or expected_bytes is None:
            raise RuntimeError(f"RS candidate batch {rollout_id} has no selected log probabilities to collect")

        transfer_start = time.perf_counter()
        worker_caches = ray.get(actor_cache_refs, timeout=self.args.rs_refill_rpc_timeout_seconds)
        selected_cache = merge_selected_log_prob_caches(worker_caches, expected)
        overlap = set(selected_cache) & set(pending["proximal_log_probs_by_sample_index"])
        if overlap:
            raise RuntimeError(f"RS proximal logprob cache was already stored for sample indices {sorted(overlap)}")

        accepted_by_index = {
            sample.index: sample for group in pending["accepted"] for sample in group if sample.index in selected_cache
        }
        transferred_bytes = 0
        for sample_index, proximal_log_probs in selected_cache.items():
            sample = accepted_by_index[sample_index]
            if (
                not isinstance(proximal_log_probs, torch.Tensor)
                or proximal_log_probs.device.type != "cpu"
                or proximal_log_probs.dtype != torch.float32
                or proximal_log_probs.ndim != 1
                or proximal_log_probs.numel() != sample.response_length
                or not proximal_log_probs.is_contiguous()
            ):
                raise RuntimeError(
                    "RS selected proximal-logprob cache must contain contiguous one-dimensional CPU float32 tensors "
                    "matching each response length: "
                    f"sample_index={sample_index}, type={type(proximal_log_probs).__name__}, "
                    f"device={getattr(proximal_log_probs, 'device', None)}, "
                    f"dtype={getattr(proximal_log_probs, 'dtype', None)}, "
                    f"shape={getattr(proximal_log_probs, 'shape', None)}, "
                    f"response_length={sample.response_length}"
                )
            transferred_bytes += proximal_log_probs.numel() * proximal_log_probs.element_size()
        if transferred_bytes != expected_bytes:
            raise RuntimeError(
                "RS selected proximal-logprob cache byte report does not match the transferred tensors: "
                f"reported={expected_bytes}, actual={transferred_bytes}"
            )

        retained_cache_bytes = pending["retained_logprob_cache_bytes"]
        required_cache_bytes = retained_cache_bytes + transferred_bytes
        cache_limit = self.args.rs_refill_max_candidate_cache_bytes
        if required_cache_bytes > cache_limit:
            raise RuntimeError(
                "RS selected proximal-logprob cache exceeds the RolloutManager retained-payload limit after transfer: "
                f"retained={retained_cache_bytes}, incoming={transferred_bytes}, "
                f"required={required_cache_bytes}, limit={cache_limit}."
            )

        pending["proximal_log_probs_by_sample_index"].update(selected_cache)
        pending["retained_logprob_cache_bytes"] = required_cache_bytes
        pending["metrics"]["rollout/rs_refill/selected_logprob_transfer_bytes"] = (
            pending["metrics"].get("rollout/rs_refill/selected_logprob_transfer_bytes", 0) + transferred_bytes
        )
        pending["metrics"]["rollout/rs_refill/retained_logprob_cache_bytes"] = required_cache_bytes
        pending["metrics"]["rollout/rs_refill/peak_retained_logprob_cache_bytes"] = max(
            pending["metrics"].get("rollout/rs_refill/peak_retained_logprob_cache_bytes", 0),
            required_cache_bytes,
        )
        pending["metrics"]["rollout/rs_refill/selected_logprob_transfer_seconds"] = pending["metrics"].get(
            "rollout/rs_refill/selected_logprob_transfer_seconds", 0.0
        ) + (time.perf_counter() - transfer_start)
        pending["awaiting_log_prob_indices"] = None
        pending["awaiting_log_prob_bytes"] = None

    def abort_rs_batch(self, rollout_id: int) -> bool:
        """Discard transient manager state after a fatal coordination error."""

        return self._pending_rs_batches.pop(rollout_id, None) is not None

    def finalize_rs_batch(self, rollout_id: int, coordinator_seconds: float):
        pending = self._pending_rs_batches.get(rollout_id)
        if pending is None:
            raise RuntimeError(f"No pending RS candidate batch for rollout_id={rollout_id}")
        if pending["awaiting_log_prob_indices"] is not None or pending["awaiting_log_prob_bytes"] is not None:
            raise RuntimeError(f"Cannot finalize RS candidate batch {rollout_id} before collecting selected caches")
        groups = pending["accepted"]
        if len(groups) != self.args.rollout_batch_size:
            raise RuntimeError(
                f"Cannot finalize underfilled RS batch: accepted={len(groups)}, target={self.args.rollout_batch_size}"
            )

        samples = list(itertools.chain.from_iterable(groups))
        pending["metrics"]["rollout/rs_refill/rounds"] = pending["round"]
        pending["metrics"]["rollout/rs_refill/accepted_groups"] = len(groups)
        scored_groups = pending["metrics"]["rollout/rs_refill/scored_groups"]
        rejected_groups = pending["metrics"].get("rollout/rs_refill/rejected_groups", 0)
        pending["metrics"]["rollout/rs_refill/gate_acceptance_rate"] = (
            scored_groups - rejected_groups
        ) / scored_groups
        pending["metrics"]["rollout/rs_refill/selection_utilization"] = len(groups) / scored_groups
        pending["metrics"]["rollout/rs_refill/coordinator_seconds"] = coordinator_seconds
        effective_tokens = sum(
            int(torch.as_tensor(sample.loss_mask).sum().item()) if sample.loss_mask is not None else 0
            for sample in samples
        )
        pending["metrics"]["rollout/rs_refill/effective_trainable_tokens"] = effective_tokens
        pending["metrics"]["rollout/rs_refill/initial_candidate_generation_seconds"] = pending[
            "initial_generation_seconds"
        ]
        refill_path_seconds = pending["initial_generation_seconds"] + coordinator_seconds
        pending["metrics"]["rollout/rs_refill/refill_path_seconds"] = refill_path_seconds
        pending["metrics"]["rollout/rs_refill/effective_tokens_per_refill_path_second"] = (
            effective_tokens / refill_path_seconds if refill_path_seconds > 0 else 0.0
        )
        data = self._convert_samples_to_train_data(samples)
        observability_fingerprint = None
        if (
            getattr(self.args, "custom_rollout_log_function_path", None) is not None
            or self.args.save_debug_rollout_data is not None
        ):
            observability_fingerprint = fingerprint_rs_train_data(
                data,
                group_indices=[sample.group_index for sample in samples],
                weight_versions=[sample.weight_versions for sample in samples],
            )
        save_debug_rollout_data(
            self.args.save_debug_rollout_data,
            samples,
            rollout_id=rollout_id,
            evaluation=False,
        )
        log_rollout_data(
            rollout_id,
            self.args,
            samples,
            pending["metrics"],
            refill_path_seconds,
        )
        if observability_fingerprint is not None:
            data = self._convert_samples_to_train_data(samples)
            validate_rs_train_data_fingerprint(
                data,
                observability_fingerprint,
                group_indices=[sample.group_index for sample in samples],
                weight_versions=[sample.weight_versions for sample in samples],
            )
        validate_sample_masks(samples, pending["accepted_mask_fingerprints"])
        attach_proximal_log_probs(
            data,
            samples,
            pending["proximal_log_probs_by_sample_index"],
        )
        result = self._split_train_data_by_dp(data)
        del self._pending_rs_batches[rollout_id]
        return result

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
        )
        log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

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
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the rollout_id contract before flattening: any list[Sample]
            # encountered in the nested output must have rollout_id set on every
            # element. Default rollouts inherit it from the data source; compact /
            # subagent paths that split one rollout into N training samples must
            # set the same rollout_id on every sibling so the loss reducer counts
            # the rollout once instead of N times.
            validate_rollout_id_annotated(data)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
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

            if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(
        self,
        samples: list[Sample] | list[list[Sample]],
        *,
        preflight: bool = False,
    ):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None and not preflight:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        if preflight:
            raw_rewards = rewards = [0.0] * len(samples)
        else:
            raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        rollout_ids = [sample.rollout_id for sample in samples]
        existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
        tmp_id = 0
        for i in range(len(rollout_ids)):
            if rollout_ids[i] is None:
                while tmp_id in existed_rollout_id_values:
                    tmp_id += 1
                rollout_ids[i] = tmp_id
                existed_rollout_id_values.add(tmp_id)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }
        if preflight:
            train_data["group_indices"] = [sample.group_index for sample in samples]

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

        # Per-rollout aggregate, precomputed at the step level (where we can
        # see every sample of every rollout) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-rollout denominator even
        # when a rollout's samples land in different micro-batches (first-fit
        # packing can split a rollout across mbs):
        #
        #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's rollout. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per rollout.
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if getattr(self.args, "rollout_top_p", 1.0) != 1.0:
            for sample in samples:
                assert sample.rollout_top_p_token_ids is not None
                assert sample.rollout_top_p_token_offsets is not None
                assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1, (
                    f"top-p token offsets length {len(sample.rollout_top_p_token_offsets)} "
                    f"!= response length + 1 {sample.response_length + 1}"
                )
                offset_end = int(sample.rollout_top_p_token_offsets[-1])
                assert offset_end == len(sample.rollout_top_p_token_ids), (
                    f"top-p token offsets[-1] {offset_end} "
                    f"!= token ids length {len(sample.rollout_top_p_token_ids)}"
                )
            train_data["rollout_top_p_token_ids"] = [sample.rollout_top_p_token_ids for sample in samples]
            train_data["rollout_top_p_token_offsets"] = [sample.rollout_top_p_token_offsets for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            routed_experts = [torch.as_tensor(sample.rollout_routed_experts) for sample in samples]
            if getattr(self.args, "use_rollout_routing_replay", False):
                validate_rollout_routed_experts_for_replay(routed_experts, self.args)
            train_data["rollout_routed_experts"] = routed_experts

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        if samples[0].metadata is not None:
            train_data["source_names"] = [get_source(sample) for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        if getattr(self.args, "rs_batch_refill", False):
            validate_rs_refill_target_batch_alignment(self.args, config)
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, global_batch_size: int | None = None):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by rollout id (``samples[i].rollout_id``, falling back
        to ``samples[i].index``); each step holds exactly
        ``args.global_batch_size`` rollouts so the training-step count per
        rollout is fixed at ``rollout_batch_size * n_samples_per_prompt //
        global_batch_size`` regardless of how many training samples each
        rollout produced.
        """
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=global_batch_size or self.args.global_batch_size,
            rollout_indices=data["rollout_ids"],
        )

        # Package per-rank rollout_data
        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "group_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_log_probs",
                "rs_preflight_log_probs",
                "rollout_top_p_token_ids",
                "rollout_top_p_token_offsets",
                "rollout_routed_experts",
                "source_names",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            tensorize_rollout_data_for_training(rollout_data)
            transport = getattr(self.args, "rollout_data_transport", "object-store")
            if transport == "nixl":
                rollout_data_refs.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
            elif transport == "object-store":
                rollout_data_refs.append(Box(ray.put(rollout_data)))
            else:
                raise ValueError(f"Unsupported rollout data transport: {transport!r}")
        return rollout_data_refs


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

    # Track per-node port cursors so that different server groups (called
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
    """Start sglang_router and return (router_ip, router_port).

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

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True

    # Disable circuit breaker to prevent RDMA transfer timeouts from
    # marking decode workers as dead. Timeouts are transient (PCIe
    # contention under high load) and do not indicate a dead server.
    router_args.disable_circuit_breaker = True

    # We will not use the health check from router.
    if hasattr(router_args, "disable_health_check"):
        router_args.disable_health_check = True

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
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start rollout servers without waiting for final engine initialization.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns ``(servers, init_handles)`` where servers maps model name to
    ``RolloutServer`` and init_handles contains pending ``engine.init`` refs.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    pending_init_handles: list[Any] = []
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        port_cursors: dict[int, int] = {}

        has_epd = model_cfg.has_encoder_disaggregation

        def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
            nonlocal engine_offset, gpu_offset
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpus_per_engine_on_node = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpus_per_engine_on_node

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if overrides_extra:
                for k, v in overrides_extra.items():
                    overrides.setdefault(k, v)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus
            return group

        if has_epd:
            # --- Phase 1: start encoder groups, wait, collect URLs ---
            # Encoder URLs are injected into the non-encoder workers' server args,
            # so this phase must stay synchronous even though final LLM init is deferred.
            encoder_urls: list[str] = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type != "encoder":
                    continue
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                if handles:
                    ray.get(handles)
                urls = ray.get([e.get_url.remote() for e in group.engines])
                encoder_urls.extend(u for u in urls if u is not None)
                server_groups.append(group)

            logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

            # --- Phase 2: start non-encoder groups, injecting encoder URLs into
            # language-only LLM workers. Prefill groups use this for full EPD,
            # while regular groups allow encoder/LLM split without PD.
            non_encoder_handles: list = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type == "encoder":
                    continue
                overrides_extra = {}
                if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                    overrides_extra["language_only"] = True
                    overrides_extra["encoder_urls"] = encoder_urls
                group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                handles, port_cursors = group.start_engines(port_cursors)
                non_encoder_handles.extend(handles)
                server_groups.append(group)

            pending_init_handles.extend(non_encoder_handles)
        else:
            # No EPD — start all groups in one pass (original path).
            all_init_handles: list = []
            for group_cfg in model_cfg.server_groups:
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                all_init_handles.extend(handles)
                server_groups.append(group)

            pending_init_handles.extend(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    # Expose per-model router info for custom rollout functions.
    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers, pending_init_handles


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.rollout_num_gpus == 0:
        return SglangConfig(models=[ModelConfig(name="default", server_groups=[])])

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )
