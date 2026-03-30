import asyncio
import inspect
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pybase64
import sglang_router
import torch
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_function, trace_span
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model."""
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


@dataclass
class RolloutGroup:
    index: int
    example: dict
    samples: list[Sample]
    completed: bool = False


class GenerateState(metaclass=SingletonMeta):
    """Global state for the generation process."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        self.semaphore = asyncio.Semaphore(concurrency)
        self.chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_tokens=args.rollout_max_context_len,  # total context budget
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.dp_counts = [0] * (getattr(args, "sglang_dp_size", None) or 1)
        self.dp_rank = 0
        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings: set[asyncio.Task] = set()
        self.aborted = False

    def submit_generate_tasks(self, groups: list[RolloutGroup]) -> None:
        for group in groups:
            self.pendings.add(
                asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(groups)


def _sample_full_text(args: Namespace, sample: Sample) -> str:
    ids = sample.tokens.tolist() if hasattr(sample.tokens, "tolist") else list(sample.tokens)
    if not ids:
        return ""
    return GenerateState(args).tokenizer.decode(ids, skip_special_tokens=args.rollout_skip_special_tokens)


def _examples_to_rollout_groups(examples: list[dict], args) -> list[RolloutGroup]:
    groups = []
    for index, example in enumerate(examples):
        groups.append(
            RolloutGroup(
                index=index,
                example=example,
                samples=[Sample.from_example(example) for _ in range(args.n_samples_per_prompt)],
            )
        )
    return groups


def _prepare_sample_tokens(args: Namespace, sample: Sample, max_context_tokens: int) -> None:
    """Tokenize prompt into sample.tokens. Sets sample._max_tokens on first call.

    All tokenization/processing is synchronous — the async event loop manages
    concurrency through the semaphore in GenerateState.
    """
    state = GenerateState(args)

    if sample.tokens:
        return

    prompt_ids = None

    if sample.has_multimodal and state.processor is None:
        raise RuntimeError("Multimodal examples require a processor, but none could be loaded for this checkpoint.")

    if isinstance(sample.prompt, list):
        messages = sample.prompt
        tools = sample.metadata.get("tools")
        if state.processor and sample.has_multimodal:
            prompt_text = state.processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                **state.chat_template_kwargs,
            )
        else:
            prompt_ids = state.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                **state.chat_template_kwargs,
            )
    else:
        prompt_text = sample.prompt

    if state.processor and sample.has_multimodal:
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = state.processor(text=prompt_text, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0].tolist()
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")
        } or None
    elif prompt_ids is None:
        prompt_ids = state.tokenizer.encode(prompt_text, add_special_tokens=False)

    sample.tokens = list(prompt_ids)
    edge_len = max(len(sample.tokens) - 1, 0)
    sample.loss_mask = [0] * edge_len
    sample.rollout_log_probs = [0.0] * edge_len
    sample._max_tokens = max_context_tokens


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using SGLang router with token-based workflow."""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in (Sample.Status.PENDING, Sample.Status.ABORTED), f"Sample status is {sample.status}"

    _prepare_sample_tokens(args, sample, sampling_params["max_tokens"])

    max_new_tokens = sample._max_tokens - len(sample.tokens)
    if max_new_tokens <= 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    sglang_params = {k: v for k, v in sampling_params.items() if k != "max_tokens"}
    sglang_params["max_new_tokens"] = max_new_tokens

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": sglang_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    image_inputs = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if sample.has_multimodal and image_inputs:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in image_inputs]

    headers = None
    if getattr(args, "router_policy", None) == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": max_new_tokens}) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []

    old_edge_len = sample.num_edges
    sample.tokens.extend(new_tokens)
    new_edge_len = sample.num_edges
    added_edges = new_edge_len - old_edge_len

    sample.loss_mask.extend([1] * added_edges)
    sample.rollout_log_probs.extend(new_log_probs)
    sample.response += output["text"]
    sample.response_length += len(new_tokens)

    if "routed_experts" in output["meta_info"]:
        decoded = pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii"))
        sample.rollout_routed_experts = torch.tensor(list(memoryview(decoded).cast("i")), dtype=torch.int32).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.ensure_edge_alignment()
    sample.update_from_meta_info(args, output["meta_info"])
    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    if sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        return sample

    state = GenerateState(args)

    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        custom_func_path = sample.generate_function_path or args.custom_generate_function_path
        if custom_func_path is not None:
            custom_generate_func = load_function(custom_func_path)
            if "evaluation" in inspect.signature(custom_generate_func).parameters:
                sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
            else:
                sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    if not args.group_rm and sample.status != Sample.Status.ABORTED and sample.reward is None:
        with trace_span(sample, "reward_model"):
            sample.reward = await async_rm(args, sample)
    return sample


@trace_function(
    "generate_and_rm_group",
    target="group",
    attrs_getter=lambda args, group, sampling_params, evaluation=False: {"group_size": len(group.samples)},
)
async def generate_and_rm_group(
    args: Namespace, group: RolloutGroup, sampling_params: dict[str, Any], evaluation: bool = False
) -> RolloutGroup:
    state = GenerateState(args)

    if state.aborted:
        return group

    for sample in group.samples:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group.samples):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = state.group_sampling_seeds[idx]
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group.samples = list(await asyncio.gather(*tasks))

    if not state.aborted and args.group_rm:
        with trace_span(group.samples, "group_reward_model"):
            rewards = await batched_async_rm(args, group.samples)
        for sample, reward in zip(group.samples, rewards, strict=False):
            sample.reward = reward

    group.completed = all(s.status != Sample.Status.ABORTED for s in group.samples)
    return group


async def abort(args: Namespace) -> None:
    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")


async def generate_rollout_async(
    args: Namespace, rollout_id: int, get_examples: Callable[[int], list[dict]]
) -> tuple[RolloutFnTrainOutput, list[dict]]:
    assert args.rollout_global_dataset
    state = GenerateState(args)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None

    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size
    kept_groups: list[RolloutGroup] = []
    all_groups: list[RolloutGroup] = []
    do_print = True
    next_group_index = 0
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")

    while len(kept_groups) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            examples = get_examples(args.over_sampling_batch_size)
            groups = _examples_to_rollout_groups(examples, args)
            for group in groups:
                group.index = next_group_index
                next_group_index += 1
            state.submit_generate_tasks(groups)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() is not None:
                logger.error(f"generate_and_rm_group task failed: {task.exception()}", exc_info=task.exception())
                raise task.exception()
            group: RolloutGroup = task.result()

            if do_print:
                s = group.samples[0]
                logger.info(
                    f"First rollout sample: {[_sample_full_text(args, s)]}, label: {s.label}, reward: {s.reward}",
                )
                do_print = False

            assert len(group.samples) == args.n_samples_per_prompt
            all_groups.append(group)

            if not group.completed:
                state.remaining_batch_size -= 1
                continue

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group.samples)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            if len(kept_groups) < target_data_size:
                kept_groups.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    s = kept_groups[-1].samples[0]
    logger.info(
        f"Finish rollout: {[_sample_full_text(args, s)]}, label: {s.label}, reward: {s.reward}",
    )

    await abort(args)

    aborted_examples = []
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = task.result()
            if not group.completed:
                aborted_examples.append(group.example)

    assert (
        len(kept_groups) == args.rollout_batch_size
    ), f"Got {len(kept_groups)} samples, expected {args.rollout_batch_size}"
    kept_groups.sort(key=lambda g: g.index)
    state.reset()

    samples = [sample for group in kept_groups for sample in group.samples]
    for sample in samples:
        sample.ensure_edge_alignment()
        sample.freeze()

    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, kept_groups)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, [g.samples for g in all_groups], get_examples)

    return RolloutFnTrainOutput(samples=samples, metrics=metric_gatherer.collect()), aborted_examples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    combined = {}
    for result in results_list:
        combined.update(result)
    return RolloutFnEvalOutput(data=combined)


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET
    from slime.utils.data import load_hf_dataset

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint,)
    if cache_key not in EVAL_PROMPT_DATASET:
        EVAL_PROMPT_DATASET[cache_key] = load_hf_dataset(dataset_cfg.path)
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_tokens=dataset_cfg.max_context_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    for raw_row in dataset:
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = Sample.from_example(raw_row)
            sample.metadata = dataset_cfg.inject_metadata(sample.metadata)
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True))
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                f"eval_rollout_single_dataset example data: {[_sample_full_text(args, sample)]} reward={sample.reward}"
            )
            do_print = False
        sample.ensure_edge_alignment()
        sample.freeze()
        data.append(sample)
        pbar.update(1)
    pbar.close()

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [s.reward if not reward_key else s.reward[reward_key] for s in data],
            "truncated": [s.status == Sample.Status.TRUNCATED for s in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))

    output, aborted_examples = run(generate_rollout_async(args, rollout_id, data_source.get_examples))
    data_source.add_examples(aborted_examples)
    return output
