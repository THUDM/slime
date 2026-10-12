"""Hooks for a real learner update and full NCCL publication to Omni replicas.

Used by scripts.test_prism_omni in publication mode. The learner uses GPU 0;
the pipeline GPUs supplied to that script must exclude GPU 0. No Ray service
is created by this smoke: the existing engine control methods and the same
weight mapping used by the distributed updater are exercised directly.
"""

from __future__ import annotations

import argparse
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

_engines = []
_learner = None
_group = None
_executor = None
_group_name = "prism-publication-smoke"
_timeout = 600.0


def _wait(futures):
    return [future.result(timeout=_timeout) for future in futures]


def _connect():
    global _group
    from slime.utils.distributed_utils import init_process_group

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    futures = [
        _executor.submit(
            engine.init_weights_update_group,
            master_address="127.0.0.1",
            master_port=port,
            rank_offset=index + 1,
            world_size=len(_engines) + 1,
            group_name=_group_name,
            backend="nccl",
        )
        for index, engine in enumerate(_engines)
    ]
    _group = init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=len(_engines) + 1,
        rank=0,
        group_name=_group_name,
        timeout=timedelta(seconds=_timeout),
    )
    _wait(futures)


def _publish(context, version):
    import torch
    import torch.distributed as dist

    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from slime.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers

    args, model = context["runtime"], context["model"]
    buckets, tensors, byte_count = 0, 0, 0
    buffer = []
    buffer_bytes = 0

    def send():
        nonlocal buckets, tensors, byte_count
        if not buffer:
            return
        futures = [
            _executor.submit(
                engine.update_weights_from_distributed,
                names=[name for name, _ in buffer],
                dtypes=[tensor.dtype for _, tensor in buffer],
                shapes=[tensor.shape for _, tensor in buffer],
                group_name=_group_name,
                weight_version=version,
            )
            for engine in _engines
        ]
        handles = [dist.broadcast(tensor, 0, group=_group, async_op=True) for _, tensor in buffer]
        for handle in handles:
            handle.wait()
        _wait(futures)
        buckets += 1
        tensors += len(buffer)
        byte_count += sum(tensor.numel() * tensor.element_size() for _, tensor in buffer)
        buffer.clear()

    with torch.no_grad():
        for name, parameter in named_params_and_buffers(args, [model]):
            converted = convert_to_hf(args, "moss_tts_prism", name, all_gather_param(name, parameter), None)
            size = sum(tensor.numel() * tensor.element_size() for _, tensor in converted)
            if buffer and buffer_bytes + size > 64 * 1024**2:
                send()
                buffer_bytes = 0
            buffer.extend(converted)
            buffer_bytes += size
        send()
    return {"weight_version": version, "buckets": buckets, "tensors": tensors, "bytes": byte_count}


def before_trajectory(context):
    """Install the native student, activate replicas, then sample its trajectory."""
    _connect()
    report = _publish(context, "1")
    for engine in _engines:
        engine.continue_generation()
    context["args"].expected_weight_version = "1"
    return report


def after_update(context):
    """Install the updated policy and leave admission closed for the controller."""
    for engine in _engines:
        engine.pause_generation()
    return _publish(context, "2")


def _cleanup():
    global _group, _executor
    import torch.distributed as dist

    if _group is not None:
        futures = [_executor.submit(engine.destroy_weights_update_group, _group_name) for engine in _engines]
        group, _group = _group, None
        dist.destroy_process_group(group)
        _wait(futures)
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    if dist.is_initialized():
        from megatron.core import mpu

        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def publication(engines, context):
    """Controller hook: native load → v1 → real rollout → Adam step → v2."""
    global _engines, _learner, _executor, _timeout
    from scripts.test_prism_gpu import run

    if 0 in context["gpu_ids"]:
        raise ValueError("Reserve GPU 0 for the learner; Omni replicas must use other GPUs")
    if not context.get("native_checkpoint"):
        raise ValueError("Publication smoke requires --native-checkpoint")
    options = context.get("publication_args", {})
    if "alignment_atol" not in options:
        raise ValueError("Set an explicit alignment_atol in --publication-args-json")
    _engines = engines
    _timeout = context["startup_timeout"]
    _executor = ThreadPoolExecutor(max_workers=len(engines))
    output = Path(context["artifact_dir"]) / "learner"
    output.mkdir(parents=True, exist_ok=True)
    prompt = output / "prompt.json"
    prompt.write_text(
        json.dumps({"script": context["text"], "task_type": context["task_type"], "show_language": False})
    )
    args = argparse.Namespace(
        hf_checkpoint=context["hf_checkpoint"],
        prism_pretrained_checkpoint=context["native_checkpoint"],
        prism_pretrained_step=options.get("native_step"),
        artifact_dir=str(output),
        numpy_overlay=None,
        omni_url=context["router_url"],
        trajectory_json=None,
        prompt_json=str(prompt),
        omni_model_name="student",
        expected_weight_version="1",
        max_frames=options.get("learner_max_frames", 8),
        request_timeout=context["request_timeout"],
        alignment_atol=options["alignment_atol"],
        alignment_rtol=options.get("alignment_rtol", 0.0),
        learning_rate=options.get("learning_rate", 1e-3),
        stop_loss_weight=1.0,
        before_trajectory_hook="scripts.prism_publication_smoke.before_trajectory",
        after_update_hook="scripts.prism_publication_smoke.after_update",
    )
    try:
        _learner = run(args, ["--attention-backend", "flash"])
        return {**_learner["report"]["after_update"], "learner": _learner["report"]}
    except BaseException:
        _cleanup()
        raise


def verify_generated(context, response, sample):
    """Compare updated Omni actions with the live updated Megatron student."""
    import torch

    from scripts.test_prism_gpu import compare_omni_scores, forward, make_batch
    from slime_plugins.models.prism import extract_policy_log_probs

    if _learner is None:
        raise RuntimeError("Run the publication hook before post-generation verification")
    try:
        _, batch = make_batch([sample])
        with torch.no_grad():
            scores = extract_policy_log_probs(_learner["learner_args"], batch, forward(_learner["model"], batch))
        report = compare_omni_scores(_learner["args"], [sample], scores)
        if not all(values["aligned"] for values in report.values()):
            raise AssertionError(f"Published Omni policy differs from the updated student: {report}")
        return report
    finally:
        _cleanup()
