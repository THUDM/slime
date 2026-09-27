from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from slime.utils.types import Sample

if TYPE_CHECKING:
    from slime.utils.rollout_transport import DiskPayloadRef


@dataclass
class RolloutFnTrainOutput:
    # straw rollouts return a batch manifest; legacy functions may return Samples.
    samples: list[list[Sample]] | DiskPayloadRef
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    from slime.utils.rollout_transport import DiskPayloadRef, RawRolloutRef

    output = fn(*args, **kwargs, evaluation=evaluation)
    if isinstance(output, RawRolloutRef):
        if evaluation:
            raise TypeError("An accepted training collection cannot be used as evaluation output")
        return output

    if evaluation:
        if isinstance(output, RolloutFnEvalOutput):
            return output
        if isinstance(output, dict):
            return RolloutFnEvalOutput(data=output)
    else:
        if isinstance(output, RolloutFnTrainOutput):
            if not isinstance(output.samples, (list, DiskPayloadRef)):
                raise TypeError("Training output must contain Samples or a supported rollout reference")
            return output
        if isinstance(output, (list, DiskPayloadRef)):
            return RolloutFnTrainOutput(samples=output)
    raise TypeError(f"Unsupported rollout output: {type(output).__name__}")


def finalize_rollout_groups(args, rollout_id, groups, metrics=None):
    """Order selected groups, run the batch hook once, and publish the batch."""
    from slime.utils.misc import load_function
    from slime.utils.rollout_transport import RolloutGroupRef, load_rollout_samples, pack_rollout_payload

    groups.sort(
        key=lambda group: (group.index if isinstance(group, RolloutGroupRef) else next(iter_samples(group)).index) or 0
    )
    dropped = set()
    incoming = set()
    if args.rollout_data_transport != "straw":
        incoming = {
            sample._queue_receipt["position"] for sample in iter_samples(groups) if hasattr(sample, "_queue_receipt")
        }
    if args.rollout_sample_filter_path is not None:
        # A custom batch hook can mutate arbitrary Sample fields. Its result
        # must be serialized again; without a hook, retain existing group refs.
        groups = load_rollout_samples(groups)
        incoming = {
            sample._queue_receipt["position"] for sample in iter_samples(groups) if hasattr(sample, "_queue_receipt")
        }
        load_function(args.rollout_sample_filter_path)(args, groups)
        selected = {
            sample._queue_receipt["position"] for sample in iter_samples(groups) if hasattr(sample, "_queue_receipt")
        }
        dropped = incoming - selected
    samples = pack_rollout_payload(groups, args, rollout_id) if args.rollout_data_transport == "straw" else groups
    if dropped and args.rollout_data_transport == "straw":
        import ray

        decision = pack_rollout_payload(
            {"positions": sorted(dropped), "reason": "rollout_sample_filter", "output": samples}, args, rollout_id
        )
        ray.get(args._rollout_queue_controller.record_dispositions.remote(decision.manifest))
    elif incoming and args.rollout_data_transport != "straw":
        import ray

        decision = pack_rollout_payload(
            {"positions": sorted(incoming), "reason": "legacy object-store delivery after batch filter"},
            args,
            rollout_id,
        )
        ray.get(args._rollout_queue_controller.record_dispositions.remote(decision.manifest))
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)


def iter_samples(value):
    """Visit Sample leaves without changing custom generation's nested shape."""
    if isinstance(value, Sample):
        yield value
    else:
        for child in value:
            yield from iter_samples(child)
