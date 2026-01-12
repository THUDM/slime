import logging
import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.sglang_rollout import GenerateState, eval_rollout
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def build_tokens_and_mask_from_messages(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], str, int]:

    if not messages or len(messages) < 2:
        return [], [], "", 0

    prompt_msgs = messages[:2]
    response_msgs = messages[2:]

    prompt_tokens = []
    for msg in prompt_msgs:
        content = msg.get("content", "")
        if content:
            prompt_tokens.extend(tokenizer(content, add_special_tokens=False)["input_ids"])

    response_tokens = []
    loss_mask = []
    response_text_parts = []

    for msg in response_msgs:
        content = msg.get("content", "")
        if not content:
            continue

        tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        token_len = len(tokens)

        response_tokens.extend(tokens)
        response_text_parts.append(content)

        mask_val = 1 if msg.get("role") == "assistant" else 0
        loss_mask.extend([mask_val] * token_len)

    all_tokens = prompt_tokens + response_tokens
    response_text = "".join(response_text_parts)
    response_length = len(response_tokens)

    return all_tokens, loss_mask, response_text, response_length


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Custom generation function for SWE-Agent integration.

    Orchestrates the interaction with the external Gym environment:
    1. Sends prompt/metadata to Gym.
    2. Receives execution trace (messages) and rewards.
    3. Formats data for Slime training format.

    Note: Performs in-place modification of `sample` for memory efficiency.
    """
    # Prepare request for Gym /run endpoint
    request = {
        "responses_create_params": {
            "input": [],
        },
        "sampling_params": sampling_params,
        **sample.metadata,
        "sglang_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1",
    }

    gym_url = os.getenv("SWE_AGENT_GYM_URL", "http://localhost:11000")
    response = await post(f"{gym_url}/run", request)

    exit_status = response.get("info", {}).get("exit_status", "")
    logger.debug(f"exit_status: {exit_status}, reward: {response.get('reward', 0.0)}")

    messages = response.get("messages", [])

    if len(messages) >= 2:
        sample.prompt = messages[:2]

    state = GenerateState(args)
    tokens, loss_mask, response_text, response_length = build_tokens_and_mask_from_messages(
        messages=messages,
        tokenizer=state.tokenizer,
    )

    sample.rollout_log_probs = None  # TODO
    sample.tokens = tokens
    sample.loss_mask = loss_mask
    sample.response = response_text
    sample.response_length = response_length
    sample.metadata["reward"] = response.get("reward", 0.0)
    sample.metadata["eval_report"] = response.get("metadata", {})
    sample.metadata["messages"] = messages

    agent_metrics = response.get("info", {}).get("agent_metrics", {})
    sample.metadata["agent_metrics"] = agent_metrics

    if exit_status == "Submitted":
        sample.status = Sample.Status.COMPLETED
    elif exit_status in ("RolloutTruncated", "LimitsExceeded", "CollapseContinued"):
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function - already computed in generate()"""
    reward = sample.metadata.get("reward", 0.0)
    return reward


def dynamic_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Filter out groups with any aborted samples from training"""
    has_aborted = any(sample.status == Sample.Status.ABORTED for sample in samples)
    if has_aborted:
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging"""
    metrics = {}

    all_metrics = []
    for sample in samples:
        if hasattr(sample, "metadata") and sample.metadata:
            agent_metrics = sample.metadata.get("agent_metrics", {})
            if agent_metrics:
                all_metrics.append(agent_metrics)

    if not all_metrics:
        return {}

    # Count metrics - mean and sum
    for key in ["turns", "tool_calls"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    # Time sum metrics - mean across rollouts
    for key in ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    # Time avg metrics - mean of means
    for key in ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    # Ratio metrics (all based on total_time which includes eval)
    for key in ["model_time_ratio", "env_time_ratio", "eval_time_ratio"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    # Total time stats
    values = [m.get("total_time", 0) for m in all_metrics]
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """
    Custom rollout function that wraps sglang_rollout.generate_rollout_async
    and adds agent metrics aggregation.
    """
    from slime.rollout.sglang_rollout import generate_rollout_async as base_generate_rollout_async

    rollout_output, aborted_samples = await base_generate_rollout_async(args, rollout_id, data_source)

    all_samples = []
    for group in rollout_output.samples:
        if isinstance(group[0], list):
            for sample_list in group:
                all_samples.extend(sample_list)
        else:
            all_samples.extend(group)

    agent_metrics = aggregate_agent_metrics(all_samples)

    metrics = rollout_output.metrics or {}
    metrics.update(agent_metrics)

    logger.info(f"Aggregated agent metrics for rollout {rollout_id}: {agent_metrics}")

    return RolloutFnTrainOutput(samples=rollout_output.samples, metrics=metrics), aborted_samples


def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    output, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return output


def generate_abortable_samples(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
) -> tuple[Any, list[list[Sample]]]:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
