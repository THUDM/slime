"""Measure engine-wave policies at slime's production weight-sync callsite.

This probe invokes ``update_weights_in_engine_group_waves`` directly.  It does
not reimplement the scheduler and it never changes slime's runtime defaults.
Each engine uses the production two-rank ``[trainer, engine]`` process-group
shape; the launched world may contain any number of such engine groups.

One distributed process launch records exactly one policy/evidence-role/run.
Use separate launches for selection and confirmation, then summarize the raw
artifacts with ``--summarize``.  Continuous iterations inside one launch are
never counted as independent runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import time
import types
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


POLICIES = (
    "isolated_a",
    "isolated_b",
    "baseline_overlap",
    "current_legal",
    "candidate_serialized",
    "candidate_windowed",
)
EVIDENCE_ROLES = ("selection", "confirmation")
SCHEMA = "slime.weight_sync_callsite.v1"


@dataclass(frozen=True)
class CallsitePolicy:
    active_engine_indices: tuple[int, ...]
    max_inflight_engine_groups: int


@dataclass
class LaunchObservation:
    engine_index: int
    host_launch_ns: int
    host_wait_return_ns: int | None = None
    start_event: torch.cuda.Event | None = None
    completion_event: torch.cuda.Event | None = None


class _ReadyRemoteMethod:
    """Minimal actor method used only to cross the real callsite boundary."""

    def __init__(self, engine_index: int) -> None:
        self.engine_index = engine_index

    def remote(self, **metadata: Any) -> dict[str, Any]:
        return {"engine_index": self.engine_index, "metadata": metadata}


class _ReadyEngine:
    def __init__(self, engine_index: int) -> None:
        self.update_weights_from_distributed = _ReadyRemoteMethod(engine_index)


class _ObservedWork:
    def __init__(self, work: Any, observation: LaunchObservation, device: torch.device) -> None:
        self._work = work
        self._observation = observation
        self._device = device

    def wait(self) -> object:
        result = self._work.wait()
        if self._device.type == "cuda":
            completion = torch.cuda.Event(enable_timing=True)
            completion.record(torch.cuda.current_stream(self._device))
            self._observation.completion_event = completion
        self._observation.host_wait_return_ns = time.perf_counter_ns()
        return result


def parse_size(value: str) -> int:
    suffixes = {
        "b": 1,
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
        "kb": 1_000,
        "mb": 1_000_000,
        "gb": 1_000_000_000,
    }
    normalized = value.strip().lower().replace("_", "")
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            number = float(normalized[: -len(suffix)])
            result = number * suffixes[suffix]
            if result <= 0 or not result.is_integer():
                raise argparse.ArgumentTypeError(f"invalid positive byte size: {value}")
            return int(result)
    try:
        result = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive byte size: {value}")
    return result


def resolve_policy(policy: str, engine_count: int, window_size: int) -> CallsitePolicy:
    """Resolve benchmark labels to the production callsite's native limit."""
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if engine_count < 2:
        raise ValueError("the A/B callsite probe requires at least two engine groups")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    all_engines = tuple(range(engine_count))
    if policy == "isolated_a":
        return CallsitePolicy((0,), 0)
    if policy == "isolated_b":
        return CallsitePolicy((1,), 0)
    if policy == "baseline_overlap":
        return CallsitePolicy(all_engines, 0)
    if policy == "current_legal":
        return CallsitePolicy(all_engines, engine_count)
    if policy == "candidate_serialized":
        return CallsitePolicy(all_engines, 1)
    return CallsitePolicy(all_engines, min(window_size, engine_count))


def ordered_engine_indices(indices: Sequence[int], order: str) -> tuple[int, ...]:
    if order == "ab":
        return tuple(indices)
    if order == "ba":
        reordered = list(indices)
        if 0 in reordered and 1 in reordered:
            first = reordered.index(0)
            second = reordered.index(1)
            reordered[first], reordered[second] = reordered[second], reordered[first]
        return tuple(reordered)
    raise ValueError(f"unsupported A/B order: {order}")


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def summarize_values(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def artifact_digest(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def verify_artifact(payload: dict[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema: {payload.get('schema')!r}")
    claimed = payload.get("artifact_sha256")
    realized = artifact_digest(payload)
    if claimed != realized:
        raise ValueError(f"artifact digest mismatch: expected {claimed}, computed {realized}")
    if payload.get("evidence_role") not in EVIDENCE_ROLES:
        raise ValueError("invalid evidence_role")
    if payload.get("policy") not in POLICIES:
        raise ValueError("invalid policy")
    expected_ranks = list(range(payload["compatibility"]["launched_world_size"]))
    if payload.get("observed_ranks") != expected_ranks:
        raise ValueError("artifact does not cover every launched rank")
    if not payload.get("payload_validated"):
        raise ValueError("artifact did not validate the received payload")


def _nccl_version() -> str | None:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "nccl"):
        return None
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return ".".join(str(item) for item in version)
    return str(version)


def _device_metadata(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "uuid": getattr(properties, "uuid", None),
        "pci_bus_id": getattr(properties, "pci_bus_id", None),
    }


def _source_commit() -> str | None:
    declared = os.environ.get("SLIME_BENCHMARK_SOURCE_COMMIT")
    if declared:
        return declared
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _numa_nodes() -> str | None:
    path = Path("/sys/devices/system/node/online")
    try:
        return path.read_text().strip()
    except OSError:
        return None


def build_compatibility(
    *,
    backend: str,
    dtype: torch.dtype,
    message_bytes: int,
    world_size: int,
    engine_count: int,
    device: torch.device,
    callsite_import_mode: str,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "source_commit": _source_commit(),
        "python": platform.python_version(),
        "cpu_machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numa_nodes_online": _numa_nodes(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": _nccl_version(),
        "nccl_launch_order_implicit": os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT"),
        "dtype": str(dtype).removeprefix("torch."),
        "message_bytes": message_bytes,
        "tensor_elements": message_bytes // torch.empty((), dtype=dtype).element_size(),
        "launched_world_size": world_size,
        "engine_group_count": engine_count,
        "process_group_membership": [[0, engine_rank] for engine_rank in range(1, world_size)],
        "callsite_import_mode": callsite_import_mode,
        "graph_capture": False,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": _device_metadata(device),
        "container_image_digest": os.environ.get("SLIME_BENCHMARK_CONTAINER_DIGEST"),
    }


def _load_production_callsite_from_source() -> Any:
    """Load the exact source file while stubbing unrelated control-plane imports.

    This mode exists for minimal PyTorch benchmark containers. The exercised
    scheduler, process-group operations, tensors, and Work handles are real;
    only Ray actor references and Megatron conversion helpers that the probe
    never calls are replaced. It is opt-in and recorded in the artifact.
    """
    repo_root = Path(__file__).resolve().parents[1]
    package_paths = {
        "slime.backends.megatron_utils": repo_root / "slime" / "backends" / "megatron_utils",
        "slime.backends.megatron_utils.update_weight": (
            repo_root / "slime" / "backends" / "megatron_utils" / "update_weight"
        ),
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    ray_module = types.ModuleType("ray")
    ray_module.ObjectRef = object
    ray_module.get = lambda refs: refs
    ray_actor_module = types.ModuleType("ray.actor")
    ray_actor_module.ActorHandle = object
    megatron_module = types.ModuleType("megatron")
    megatron_core_module = types.ModuleType("megatron.core")
    megatron_core_module.mpu = types.ModuleType("megatron.core.mpu")
    accelerator_module = types.ModuleType("slime.utils.accelerator")
    distributed_utils_module = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_module.get_gloo_group = lambda: None
    distributed_utils_module.init_process_group = lambda **_kwargs: None
    http_utils_module = types.ModuleType("slime.utils.http_utils")
    http_utils_module._wrap_ipv6 = lambda address: address
    converter_module = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    converter_module.convert_to_hf = lambda *_args, **_kwargs: []
    common_module = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common_module.all_gather_param = lambda _name, param: param
    common_module.named_params_and_buffers = lambda *_args, **_kwargs: []
    sys.modules.update(
        {
            "ray": ray_module,
            "ray.actor": ray_actor_module,
            "megatron": megatron_module,
            "megatron.core": megatron_core_module,
            "megatron.core.mpu": megatron_core_module.mpu,
            "slime.utils.accelerator": accelerator_module,
            "slime.utils.distributed_utils": distributed_utils_module,
            "slime.utils.http_utils": http_utils_module,
            "slime.backends.megatron_utils.megatron_to_hf": converter_module,
            "slime.backends.megatron_utils.update_weight.common": common_module,
        }
    )

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    module_path = package_paths["slime.backends.megatron_utils.update_weight"] / "update_weight_from_distributed.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production callsite from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_production_callsite() -> tuple[Any, str]:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from slime.backends.megatron_utils.update_weight import update_weight_from_distributed

        return update_weight_from_distributed, "installed_runtime"
    except ModuleNotFoundError:
        if os.environ.get("SLIME_CALLSITE_SOURCE_LOAD") != "1":
            raise RuntimeError(
                "the installed slime control-plane dependencies are incomplete; "
                "set SLIME_CALLSITE_SOURCE_LOAD=1 to use the recorded source-loader mode"
            ) from None
        return _load_production_callsite_from_source(), "source_with_control_plane_stubs"


def _make_engine_groups(backend: str, world_size: int) -> tuple[list[Any], Any]:
    groups = []
    for engine_rank in range(1, world_size):
        groups.append(dist.new_group(ranks=[0, engine_rank], backend=backend))
    control_group = (
        dist.group.WORLD if backend == "gloo" else dist.new_group(ranks=list(range(world_size)), backend="gloo")
    )
    return groups, control_group


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_one_iteration(
    *,
    module: Any,
    policy: CallsitePolicy,
    order: str,
    process_groups: Sequence[Any],
    control_group: Any,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    tensor_elements: int,
    iteration: int,
) -> dict[str, Any]:
    active_indices = ordered_engine_indices(policy.active_engine_indices, order)
    value = float((iteration % 17) + 1)
    sender_tensor = torch.full((tensor_elements,), value, device=device, dtype=dtype) if rank == 0 else None
    receiver_tensor = (
        torch.zeros((tensor_elements,), device=device, dtype=dtype)
        if rank > 0 and rank - 1 in active_indices
        else None
    )

    dist.barrier(group=control_group)
    iteration_start_ns = time.perf_counter_ns()
    rank_record: dict[str, Any] = {"rank": rank, "consumer": None}
    controller_observations: list[LaunchObservation] = []

    if rank == 0:
        original_launch = module.launch_weights_from_distributed
        original_ray_get = module.ray.get
        index_by_group_name = {f"callsite-engine-{index}": index for index in active_indices}

        def observed_launch(
            group_name: str,
            group: Any,
            weight_version: int,
            rollout_engines: Sequence[Any],
            converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
            load_format: str | None = None,
        ) -> tuple[list[Any], list[Any]]:
            engine_index = index_by_group_name[group_name]
            observation = LaunchObservation(engine_index=engine_index, host_launch_ns=time.perf_counter_ns())
            if device.type == "cuda":
                observation.start_event = torch.cuda.Event(enable_timing=True)
                observation.start_event.record(torch.cuda.current_stream(device))
            refs, works = original_launch(
                group_name,
                group,
                weight_version,
                rollout_engines,
                converted_named_tensors,
                load_format=load_format,
            )
            controller_observations.append(observation)
            return refs, [_ObservedWork(work, observation, device) for work in works]

        module.launch_weights_from_distributed = observed_launch
        module.ray.get = lambda refs: refs
        try:
            update_groups = [
                module.DistributedWeightUpdateGroup(
                    engine_indices=(index,),
                    group_name=f"callsite-engine-{index}",
                    process_group=process_groups[index],
                    rollout_engines=(_ReadyEngine(index),),
                )
                for index in active_indices
            ]
            module.update_weights_in_engine_group_waves(
                update_groups,
                weight_version=iteration + 1,
                converted_named_tensors=[("probe.weight", sender_tensor)],
                max_inflight_engine_groups=policy.max_inflight_engine_groups,
                load_format="callsite_probe",
            )
        finally:
            module.launch_weights_from_distributed = original_launch
            module.ray.get = original_ray_get
        callsite_return_ns = time.perf_counter_ns()
        _sync_device(device)
        controller_ready_ns = time.perf_counter_ns()
        observations = []
        for item in controller_observations:
            if item.host_wait_return_ns is None:
                raise RuntimeError("production callsite returned without waiting for a collective")
            if item.start_event is not None:
                if item.completion_event is None:
                    raise RuntimeError("CUDA collective did not record a completion dependency")
                duration_ms = item.start_event.elapsed_time(item.completion_event)
                timing_kind = "event_bracket"
            else:
                duration_ms = (item.host_wait_return_ns - item.host_launch_ns) / 1_000_000
                timing_kind = "host_blocking_interval"
            observations.append(
                {
                    "engine_index": item.engine_index,
                    "host_launch_ns": item.host_launch_ns,
                    "host_wait_return_ns": item.host_wait_return_ns,
                    "duration_ms": duration_ms,
                    "timing_kind": timing_kind,
                }
            )
        rank_record.update(
            {
                "controller_observations": observations,
                "callsite_return_ms": (callsite_return_ns - iteration_start_ns) / 1_000_000,
                "controller_device_ready_ms": (controller_ready_ns - iteration_start_ns) / 1_000_000,
            }
        )
    elif rank - 1 in active_indices:
        engine_index = rank - 1
        consumer_wait_start_ns = time.perf_counter_ns()
        work = dist.broadcast(receiver_tensor, src=0, group=process_groups[engine_index], async_op=True)
        work.wait()
        _sync_device(device)
        consumer_ready_ns = time.perf_counter_ns()
        expected = torch.full_like(receiver_tensor, value)
        payload_valid = bool(torch.equal(receiver_tensor, expected))
        rank_record["consumer"] = {
            "engine_index": engine_index,
            "wait_ms": (consumer_ready_ns - consumer_wait_start_ns) / 1_000_000,
            "ready_ms": (consumer_ready_ns - iteration_start_ns) / 1_000_000,
            "payload_valid": payload_valid,
        }

    dist.barrier(group=control_group)
    sync_ready_ns = time.perf_counter_ns()
    rank_record["step_sync_ready_ms"] = (sync_ready_ns - iteration_start_ns) / 1_000_000
    gathered: list[dict[str, Any] | None] | None = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rank_record, gathered, dst=0, group=control_group)

    if rank != 0:
        return {}
    assert gathered is not None
    records = [item for item in gathered if item is not None]
    controller = records[0]
    observations_by_engine = {item["engine_index"]: item for item in controller["controller_observations"]}
    consumers_by_engine = {
        item["consumer"]["engine_index"]: item["consumer"] for item in records if item.get("consumer") is not None
    }
    pair = [observations_by_engine[index] for index in (0, 1) if index in observations_by_engine]
    pair_makespan_ms = None
    realized_offset_us = None
    if len(pair) == 2:
        pair_makespan_ms = (
            max(item["host_wait_return_ns"] for item in pair) - min(item["host_launch_ns"] for item in pair)
        ) / 1_000_000
        realized_offset_us = (
            observations_by_engine[1]["host_launch_ns"] - observations_by_engine[0]["host_launch_ns"]
        ) / 1_000
    return {
        "iteration": iteration,
        "controller_observations": controller["controller_observations"],
        "comm_a_ms": observations_by_engine.get(0, {}).get("duration_ms"),
        "comm_b_ms": observations_by_engine.get(1, {}).get("duration_ms"),
        "rank_local_pair_makespan_ms": pair_makespan_ms,
        "realized_b_minus_a_launch_offset_us": realized_offset_us,
        "consumer_a_wait_ms": consumers_by_engine.get(0, {}).get("wait_ms"),
        "consumer_b_wait_ms": consumers_by_engine.get(1, {}).get("wait_ms"),
        "consumer_ready_ms": max(
            (item["ready_ms"] for item in consumers_by_engine.values()),
            default=None,
        ),
        "callsite_return_ms": controller["callsite_return_ms"],
        "controller_device_ready_ms": controller["controller_device_ready_ms"],
        "step_sync_ready_ms": max(item["step_sync_ready_ms"] for item in records),
        "payload_valid": all(item["payload_valid"] for item in consumers_by_engine.values()),
        "observed_ranks": sorted(item["rank"] for item in records),
    }


def _metric_summary(iterations: Sequence[dict[str, Any]], key: str) -> dict[str, float | None]:
    return summarize_values([float(item[key]) for item in iterations if item.get(key) is not None])


def run_distributed(args: argparse.Namespace) -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requested without CUDA")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend=backend)
    try:
        world_size = dist.get_world_size()
        engine_count = world_size - 1
        resolved = resolve_policy(args.policy, engine_count, args.window_size)
        process_groups, control_group = _make_engine_groups(backend, world_size)
        dtype = getattr(torch, args.dtype)
        element_size = torch.empty((), dtype=dtype).element_size()
        if args.message_bytes % element_size:
            raise ValueError("message_bytes must be divisible by dtype element size")
        tensor_elements = args.message_bytes // element_size

        launch_id = str(uuid.uuid4()) if rank == 0 else None
        launch_id_box = [launch_id]
        dist.broadcast_object_list(launch_id_box, src=0, group=control_group)
        launch_id = launch_id_box[0]
        module, callsite_import_mode = _load_production_callsite()

        for warmup_index in range(args.warmup):
            _run_one_iteration(
                module=module,
                policy=resolved,
                order=args.order,
                process_groups=process_groups,
                control_group=control_group,
                rank=rank,
                device=device,
                dtype=dtype,
                tensor_elements=tensor_elements,
                iteration=-(warmup_index + 1),
            )
        iterations = []
        for iteration in range(args.iterations):
            result = _run_one_iteration(
                module=module,
                policy=resolved,
                order=args.order,
                process_groups=process_groups,
                control_group=control_group,
                rank=rank,
                device=device,
                dtype=dtype,
                tensor_elements=tensor_elements,
                iteration=iteration,
            )
            if rank == 0:
                iterations.append(result)

        if rank == 0:
            compatibility = build_compatibility(
                backend=backend,
                dtype=dtype,
                message_bytes=args.message_bytes,
                world_size=world_size,
                engine_count=engine_count,
                device=device,
                callsite_import_mode=callsite_import_mode,
            )
            payload = {
                "schema": SCHEMA,
                "run_id": args.run_id,
                "process_launch_id": launch_id,
                "evidence_role": args.evidence_role,
                "policy": args.policy,
                "order": args.order,
                "resolved_policy": asdict(resolved),
                "timing_scope": {
                    "communication": "rank-local host interval for Gloo; CUDA event bracket for NCCL",
                    "pair": "controller-rank host launch through Work.wait return",
                    "consumer": "receiver Work.wait plus device synchronization and payload read",
                    "sync_ready": "same-host control barrier after every receiver consumed the payload",
                    "kernel_observed": False,
                },
                "compatibility": compatibility,
                "iterations": iterations,
                "summary": {
                    key: _metric_summary(iterations, key)
                    for key in (
                        "comm_a_ms",
                        "comm_b_ms",
                        "rank_local_pair_makespan_ms",
                        "realized_b_minus_a_launch_offset_us",
                        "consumer_a_wait_ms",
                        "consumer_b_wait_ms",
                        "consumer_ready_ms",
                        "callsite_return_ms",
                        "controller_device_ready_ms",
                        "step_sync_ready_ms",
                    )
                },
                "observed_ranks": iterations[0]["observed_ranks"],
                "payload_validated": all(item["payload_valid"] for item in iterations),
                "claim_limits": [
                    "This is the production scheduler callsite with synthetic tensors and ready metadata refs.",
                    "It is not a Ray/SGLang load benchmark or an end-to-end training throughput claim.",
                    "Event brackets are not labeled as profiler-observed kernel durations.",
                    "One process launch is one independent run regardless of iteration count.",
                ],
            }
            payload["artifact_sha256"] = artifact_digest(payload)
            verify_artifact(payload)
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


def compatibility_cell(payload: dict[str, Any]) -> bytes:
    return _canonical_json(payload["compatibility"])


def summarize_artifacts(paths: Sequence[Path], *, min_runs_per_role: int) -> dict[str, Any]:
    if min_runs_per_role < 1:
        raise ValueError("min_runs_per_role must be positive")
    artifacts = [json.loads(path.read_text()) for path in paths]
    if not artifacts:
        raise ValueError("at least one artifact is required")
    for artifact in artifacts:
        verify_artifact(artifact)
    reference_cell = compatibility_cell(artifacts[0])
    if any(compatibility_cell(artifact) != reference_cell for artifact in artifacts[1:]):
        raise ValueError("artifacts cross an incompatible runtime/message/topology cell")

    run_ids = [artifact["run_id"] for artifact in artifacts]
    launch_ids = [artifact["process_launch_id"] for artifact in artifacts]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id values must be unique")
    if len(launch_ids) != len(set(launch_ids)):
        raise ValueError("process_launch_id values must be unique")

    counts: dict[str, dict[str, int]] = {policy: {role: 0 for role in EVIDENCE_ROLES} for policy in POLICIES}
    for artifact in artifacts:
        counts[artifact["policy"]][artifact["evidence_role"]] += 1
    missing = [
        f"{policy}/{role}={count}"
        for policy, by_role in counts.items()
        for role, count in by_role.items()
        if count < min_runs_per_role
    ]
    if missing:
        raise ValueError(
            f"each policy/evidence role requires {min_runs_per_role} independent process runs; " + ", ".join(missing)
        )

    metric_names = (
        "comm_a_ms",
        "comm_b_ms",
        "rank_local_pair_makespan_ms",
        "consumer_a_wait_ms",
        "consumer_b_wait_ms",
        "consumer_ready_ms",
        "callsite_return_ms",
        "controller_device_ready_ms",
        "step_sync_ready_ms",
    )
    policy_summaries = {}
    for policy in POLICIES:
        policy_summaries[policy] = {}
        for role in EVIDENCE_ROLES:
            selected = [
                artifact
                for artifact in artifacts
                if artifact["policy"] == policy and artifact["evidence_role"] == role
            ]
            policy_summaries[policy][role] = {
                "independent_process_runs": len(selected),
                "orders": sorted({artifact["order"] for artifact in selected}),
                "metrics": {
                    metric: summarize_values(
                        [
                            float(artifact["summary"][metric]["p50"])
                            for artifact in selected
                            if artifact["summary"][metric]["p50"] is not None
                        ]
                    )
                    for metric in metric_names
                },
            }

    result = {
        "schema": "slime.weight_sync_callsite_campaign.v1",
        "compatibility": artifacts[0]["compatibility"],
        "artifact_count": len(artifacts),
        "minimum_independent_runs_per_policy_role": min_runs_per_role,
        "policy_summaries": policy_summaries,
        "selection_and_confirmation_disjoint": True,
        "automatic_policy_selection": False,
        "claim_limits": artifacts[0]["claim_limits"],
        "input_artifact_sha256": sorted(artifact["artifact_sha256"] for artifact in artifacts),
    }
    result["artifact_sha256"] = artifact_digest(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--min-runs-per-role", type=int, default=5)
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--evidence-role", choices=EVIDENCE_ROLES)
    parser.add_argument("--run-id")
    parser.add_argument("--order", choices=("ab", "ba"), default="ab")
    parser.add_argument("--message-bytes", type=parse_size, default=parse_size("1MiB"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.summarize:
        if not args.summary_json:
            raise SystemExit("--summary-json is required with --summarize")
        result = summarize_artifacts(args.summarize, min_runs_per_role=args.min_runs_per_role)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.summary_json} ({result['artifact_count']} independent process artifacts)")
        return 0

    required = {
        "--policy": args.policy,
        "--evidence-role": args.evidence_role,
        "--run-id": args.run_id,
        "--output-json": args.output_json,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit("missing distributed-run arguments: " + ", ".join(missing))
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be non-negative and --iterations must be positive")
    run_distributed(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
