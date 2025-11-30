import asyncio
import base64
import copy
import io
import logging
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import numpy as np
import sglang_router
from packaging.version import parse
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.mask_utils import get_response_lengths
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


def _load_and_encode_image(path: str) -> str:
    """Load an image from path, ensure RGB, encode as JPEG base64 string."""
    with Image.open(path) as image:
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.reset()

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # Process prompt to create text and image payload
    image_data = []
    if isinstance(sample.prompt, str):
        text_prompt = sample.prompt
    else:  # Multimodal prompt (list of dicts)
        text_prompt = ""
        # sglang uses a placeholder to insert image features
        image_token = state.tokenizer.special_tokens_map.get("image_token", "<image>")
        for part in sample.prompt:
            if part["type"] == "text":
                text_prompt += part["text"]
            elif part["type"] == "image":
                text_prompt += image_token
                try:
                    img_b64 = await asyncio.to_thread(_load_and_encode_image, part["path"])
                    image_data.append(img_b64)
                except Exception as e:
                    logger.info(f"Error processing image {part['path']}: {e}")
                    sample.status = Sample.Status.ABORTED
                    return sample

    if len(sample.response) > 0:
        # Adjust max_new_tokens for subsequent generation turns
        prompt_len = len(state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"])
        sampling_params["max_new_tokens"] -= len(sample.tokens) - prompt_len

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if image_data:
        payload["image_data"] = image_data

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        prompt_token_ids = state.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        payload["input_ids"] = prompt_token_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_token_ids

    output = await post(url, payload)

    # Extract new response tokens

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        assert not args.partial_rollout, "Currently parital rollout is not suppurted when using slime router"
        retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
        retrieve_payload = {"text": sample.prompt + output["text"], "return_logp": True}
        retrieve_output = await post(retrieve_url, retrieve_payload)
        sample.tokens = retrieve_output["tokens"]
        sample.response += output["text"]
        sample.loss_mask = retrieve_output["loss_mask"]
        sample.response_length = get_response_lengths([sample.loss_mask])[0]
        sample.loss_mask = sample.loss_mask[-sample.response_length :]
        sample.rollout_log_probs = retrieve_output["rollout_logp"][-sample.response_length :]
        # Notice: currently cannot get the spec info from radix router output.
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            sample.spec_info.add(
                meta_info=output["meta_info"],
                response_length=sample.response_length,
            )

    if "weight_version" in output["meta_info"]:
        sample.weight_versions.append(output["meta_info"]["weight_version"])

    if "routed_experts" in output["meta_info"]:
        assert len(output["meta_info"]["routed_experts"]) == len(sample.tokens)
        sample.rollout_routed_experts = np.array(output["meta_info"]["routed_experts"])

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Union[Sample, list[Sample]],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Union[Sample, list[Sample]]:
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will not do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    for url in urls:
        logger.info(f"Abort request for {url}")
        await post(f"{url}/abort_request", {"abort_all": True})

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = _MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            dynamic_filter_output = _call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


def _call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class _MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: Optional[str]):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    name = dataset_cfg.name
    path = dataset_cfg.path

    def _resolve_dataset_setting(dataset_value, eval_value, rollout_value):
        if dataset_value is not None:
            return dataset_value
        if eval_value is not None:
            return eval_value
        return rollout_value

    prompt_key = _resolve_dataset_setting(
        dataset_cfg.prompt_key,
        args.eval_input_key,
        args.input_key,
    )
    label_key = _resolve_dataset_setting(
        dataset_cfg.label_key,
        args.eval_label_key,
        args.label_key,
    )
    tool_key = _resolve_dataset_setting(
        dataset_cfg.tool_key,
        args.eval_tool_key,
        args.tool_key,
    )
    metadata_key = dataset_cfg.metadata_key or getattr(args, "metadata_key", "metadata")

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.eval_max_prompt_len,
            prompt_key=prompt_key,
            label_key=label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=metadata_key,
            tool_key=tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=_resolve_dataset_setting(dataset_cfg.temperature, args.eval_temperature, args.rollout_temperature),
        top_p=_resolve_dataset_setting(
            dataset_cfg.top_p,
            args.eval_top_p,
            args.rollout_top_p,
        ),
        top_k=_resolve_dataset_setting(
            dataset_cfg.top_k,
            args.eval_top_k,
            args.rollout_top_k,
        ),
        max_new_tokens=_resolve_dataset_setting(
            dataset_cfg.max_response_len,
            args.eval_max_response_len,
            args.rollout_max_response_len,
        ),
        stop=dataset_cfg.stop if dataset_cfg.stop is not None else args.rollout_stop,
        stop_token_ids=(
            dataset_cfg.stop_token_ids if dataset_cfg.stop_token_ids is not None else args.rollout_stop_token_ids
        ),
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    min_new_tokens = dataset_cfg.min_new_tokens
    if min_new_tokens is None:
        min_new_tokens = getattr(args, "eval_min_new_tokens", None)
    if min_new_tokens is not None:
        base_sampling_params["min_new_tokens"] = min_new_tokens

    n_samples_per_prompt = (
        dataset_cfg.n_samples_per_eval_prompt
        if dataset_cfg.n_samples_per_eval_prompt is not None
        else args.n_samples_per_eval_prompt
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(n_samples_per_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


# TODO remove this temp function
def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> Union[RolloutFnTrainOutput, RolloutFnEvalOutput]:
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
