"""SGLang engine groups and their rollout-facing lifecycle."""

import dataclasses
import logging
import os
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt import constants as sglang_constants

from slime.backends.sglang_utils.sglang_config import ServerGroupConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, add_default_ray_env_vars

# Official AMD images ship an older sglang whose constants module has no CUDA graph tag.
GPU_MEMORY_TYPE_KV_CACHE = sglang_constants.GPU_MEMORY_TYPE_KV_CACHE
GPU_MEMORY_TYPE_WEIGHTS = sglang_constants.GPU_MEMORY_TYPE_WEIGHTS
GPU_MEMORY_TYPE_CUDA_GRAPH = getattr(sglang_constants, "GPU_MEMORY_TYPE_CUDA_GRAPH", None)

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
        num_engines = len(self.all_engines)
        required_gpu_slots = self.gpu_offset + num_engines * num_gpus_per_engine_on_node
        if num_engines and not (
            self.gpu_offset >= 0 and num_gpus_per_engine_on_node > 0 and required_gpu_slots <= len(reordered_gpu_ids)
        ):
            raise ValueError(
                "Invalid rollout server group GPU placement: "
                f"worker_type={self.worker_type}, "
                f"gpu_offset={self.gpu_offset}, "
                f"num_gpus_per_engine={self.num_gpus_per_engine}, "
                f"num_gpus_per_engine_on_node={num_gpus_per_engine_on_node}, "
                f"num_engines={num_engines}, "
                f"required_gpu_slots={required_gpu_slots}, "
                f"len(reordered_gpu_ids)={len(reordered_gpu_ids)}, "
                f"rollout_num_gpus={self.args.rollout_num_gpus}, "
                f"rollout_num_gpus_per_engine={self.args.rollout_num_gpus_per_engine}. "
                "Please align --rollout-num-gpus, --rollout-num-gpus-per-engine, "
                "and --sglang-config server_groups."
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
    """A model served behind a shared router, with one or more server groups."""

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
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

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
        """Restore weights for offloaded groups."""
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        tags = [GPU_MEMORY_TYPE_KV_CACHE]
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            tags.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=tags))
        return ray.get(handles) if handles else []


@dataclasses.dataclass
class ServerGroupPlacement:
    """Track rank and GPU offsets while materializing configured server groups."""

    args: Any
    pg: Any
    rollout_pg_offset: int
    megatron_num_gpus: int
    engine_offset: int = 0
    gpu_offset: int = 0

    def create(
        self,
        group_config: ServerGroupConfig,
        router_ip: str,
        router_port: int,
        *,
        overrides_extra: dict | None = None,
    ) -> ServerGroup:
        gpus_per_engine = group_config.num_gpus_per_engine
        assert gpus_per_engine is not None, "ModelConfig.resolve() must set num_gpus_per_engine before deployment"
        num_gpus_per_engine_on_node = min(gpus_per_engine, self.args.num_gpus_per_node)
        num_engines = group_config.num_gpus // num_gpus_per_engine_on_node

        group_abs_start = self.rollout_pg_offset + self.gpu_offset
        needs_offload = self.args.offload_rollout and group_abs_start < self.megatron_num_gpus
        overrides = dict(group_config.overrides)
        if overrides_extra:
            for key, value in overrides_extra.items():
                overrides.setdefault(key, value)
        if self.args.offload_rollout and not needs_offload:
            overrides.setdefault("enable_memory_saver", False)
        logger.info(
            f"Engine group '{group_config.worker_type}' gpu_offset={self.gpu_offset} "
            f"(abs={group_abs_start}): needs_offload={needs_offload}"
        )

        group = ServerGroup(
            args=self.args,
            pg=self.pg,
            all_engines=[None] * num_engines if group_config.worker_type != "placeholder" else [],
            num_gpus_per_engine=gpus_per_engine,
            num_new_engines=0,
            worker_type=group_config.worker_type,
            rank_offset=self.engine_offset,
            gpu_offset=self.gpu_offset,
            sglang_overrides=overrides,
            needs_offload=needs_offload,
            model_path=overrides.get("model_path", self.args.hf_checkpoint),
            router_ip=router_ip,
            router_port=router_port,
        )
        self.engine_offset += num_engines
        self.gpu_offset += group_config.num_gpus
        return group


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    """Allocate rank-local SGLang and distributed-init ports for one group."""
    gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, allocated_port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = allocated_port + consecutive
                node_port_cursor[node_idx] = start_port
                return allocated_port

            def addr():
                address, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return address

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

        if gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for rank, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[rank], f"Engine {rank} {key} is not set."
        logger.info(f"Ports for engine {rank}: {addr_and_ports[rank]}")

    return addr_and_ports, node_port_cursor
