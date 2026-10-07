"""Crash serving without unregistering it at the actual rollout drain boundary."""

import json
import os
import time
from pathlib import Path

import ray
import requests
import torch.distributed as dist

from slime.ray.training_recovery import RECOVERY_NAMESPACE, training_session_name


async def reward(args, sample, **kwargs):
    return float(sample.index % args.n_samples_per_prompt)


def _stop_server(engine):
    from sglang.srt.utils import kill_process_tree

    identity = {"url": engine.get_url(), "pid": engine.process.pid}
    kill_process_tree(engine.process.pid)
    return identity


def _engines(serving):
    return [engine for server in serving.servers.values() for engine in server.engines]


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout import sglang_rollout

    original_abort = sglang_rollout.abort

    async def crash_then_abort(config, current_rollout):
        assert sglang_rollout.GenerateState(config).active_server_generations
        owner = ray.get_actor(training_session_name(config) + ":serving", namespace=RECOVERY_NAMESPACE)
        engines = ray.get(owner.__ray_call__.remote(_engines))
        start = time.monotonic()
        identity = ray.get(engines[0].__ray_call__.remote(_stop_server), timeout=30)
        if os.environ["SLIME_HEALTH_CRASH_MODE"] == "actor":
            ray.kill(engines[0], no_restart=True)
        router = f"http://{config.sglang_router_ip}:{config.sglang_router_port}"
        response = requests.get(f"{router}/workers", timeout=5)
        response.raise_for_status()
        assert identity["url"] in {worker["url"] for worker in response.json()["workers"]}
        result = await original_abort(config, current_rollout)
        response = requests.get(f"{router}/workers", timeout=5)
        response.raise_for_status()
        urls = {worker["url"] for worker in response.json()["workers"]}
        assert identity["url"] not in urls and len(urls) == 1
        identity["drain_seconds"] = time.monotonic() - start
        assert identity["drain_seconds"] < 60
        Path(os.environ["SLIME_HEALTH_TEST_DIR"], "crash.json").write_text(json.dumps(identity))
        return result

    if rollout_id == 0 and not evaluation:
        sglang_rollout.abort = crash_then_abort
    try:
        return sglang_rollout.generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        sglang_rollout.abort = original_abort


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    if dist.get_rank() != 0 or step_id != 0:
        return
    serving = ray.get_actor(training_session_name(args) + ":serving", namespace=RECOVERY_NAMESPACE)
    engines = ray.get(serving.__ray_call__.remote(_engines))
    live = [engine for engine in engines if engine is not None]
    assert len(live) == (1 if rollout_id == 0 else 2)
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    response = requests.get(f"{router}/workers", timeout=5)
    response.raise_for_status()
    assert len(response.json()["workers"]) == len(live)
    versions = ray.get([engine.get_weight_version.remote() for engine in live])
    assert versions == [str(rollout_id + 1)] * len(live)
    Path(os.environ["SLIME_HEALTH_TEST_DIR"], f"trained_{rollout_id}.json").write_text(
        json.dumps({"live_engines": len(live), "weight_versions": versions})
    )
