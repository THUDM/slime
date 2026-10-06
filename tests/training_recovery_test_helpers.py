"""Actual CUDA OOM injection and serving/data identity checks for restart e2e."""

import hashlib
import json
import os
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from slime.ray.training_recovery import RECOVERY_NAMESPACE, training_session_name


async def reward(args, sample, **kwargs):
    return float(sample.index % args.n_samples_per_prompt)


def _engine_identity(engine):
    return {"actor_pid": os.getpid(), "server_pid": engine.process.pid, "url": engine.get_url()}


def _serving_identity(serving):
    return {"serving_pid": os.getpid(), "router_pids": [process.pid for process in serving.router_processes]}


def _session_snapshot(manager, rollout_id):
    data = manager.recovery.load_converted(rollout_id)
    serialized = json.dumps(
        {name: data[name] for name in ("tokens", "sample_indices", "rewards")},
        sort_keys=True,
        default=lambda value: value.tolist(),
    )
    # R3 is part of the replay contract: compare the persisted routing bytes,
    # independently of how the restarted trainer partitions samples or layers.
    route_digest = hashlib.sha256()
    route_elements = 0
    for routes in data.get("rollout_routed_experts", []):
        if hasattr(routes, "load"):
            routes = routes.load()
        route_elements += routes.numel()
        route_digest.update(routes.to(dtype=torch.int32).contiguous().numpy().tobytes())
    if getattr(manager.args, "use_rollout_routing_replay", False):
        assert route_elements > 0, "R3 recovery must replay nonempty routing tensors"
    engines = [engine for engine in manager.rollout_engines if engine is not None]
    return {
        "manager_pid": os.getpid(),
        **ray.get(manager.serving.__ray_call__.remote(_serving_identity)),
        "engines": ray.get([engine.__ray_call__.remote(_engine_identity) for engine in engines]),
        "engine_actor_ids": [engine._actor_id.hex() for engine in engines],
        "router": [manager.args.sglang_router_ip, manager.args.sglang_router_port],
        "controller_id": manager.controller._actor_id.hex() if manager.controller else None,
        "rollout_pg_id": manager.pg[0].id.hex(),
        "sample_digest": hashlib.sha256(serialized.encode()).hexdigest(),
        "sample_indices": data["sample_indices"],
        "route_digest": route_digest.hexdigest(),
        "parallel": manager.batch_builder.train_parallel_config,
        "weight_versions": ray.get([engine.get_weight_version.remote() for engine in engines]),
        "start_rollout_id": manager.args.start_rollout_id,
        "checkpoint_step": manager.recovery.checkpoint_step,
    }


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    failure_rollout = int(os.environ.get("SLIME_RECOVERY_TEST_FAILURE_ROLLOUT", "0"))
    directory = Path(os.environ["SLIME_RECOVERY_TEST_DIR"])
    bad_configuration = args.tensor_model_parallel_size == 1
    if rollout_id > failure_rollout:
        if not bad_configuration and args.rollout_data_transport == "object-store" and dist.get_rank() == 0:
            manager = ray.get_actor(training_session_name(args), namespace=RECOVERY_NAMESPACE)
            snapshot = ray.get(manager.__ray_call__.remote(_session_snapshot, rollout_id))
            (directory / f"continued_{rollout_id}.json").write_text(json.dumps(snapshot))
        return
    if dist.get_rank() == 0:
        manager = ray.get_actor(training_session_name(args), namespace=RECOVERY_NAMESPACE)
        snapshot = ray.get(manager.__ray_call__.remote(_session_snapshot, rollout_id))
        snapshot["trainer"] = {
            "pid": os.getpid(),
            "job_id": str(ray.get_runtime_context().get_job_id()),
            "tp_size": args.tensor_model_parallel_size,
        }
        snapshot["scheduler_num_steps"] = opt_param_scheduler.num_steps
        serialized = json.dumps(snapshot, indent=2)
        (directory / f"{'original' if bad_configuration else 'replayed'}_{rollout_id}.json").write_text(serialized)
        if rollout_id == failure_rollout:
            path = directory / ("before.json" if bad_configuration else "after.json")
            path.write_text(serialized)
            if bad_configuration and os.environ.get("SLIME_RECOVERY_TEST_KILL_MANAGER") == "1":
                ray.kill(manager, no_restart=True)
    dist.barrier()
    if bad_configuration and rollout_id == failure_rollout:
        # Exercise the real CUDA allocator's failure and Ray exception handling.
        # The corrected TP=2 configuration proceeds through actual GRPO updates.
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        torch.empty(total_memory * 2, dtype=torch.uint8, device="cuda")
        raise AssertionError("CUDA allocation exceeding twice the device capacity unexpectedly succeeded")
