# Adapted form https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
import asyncio
import re

from google_search_server import google_search
from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

SEARCH_R1_CONFIGS = {
    "max_turns": 3,
    "topk": 3,
    "google_api_key": "YOUR_API_KEY",  # Replace with your actual API key
    "snippet_only": True,  # Set to True to only return snippets
    "proxy": None,  # Set to your proxy if needed
    "search_concurrency": 256,
    # rm
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):

        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    result = await google_search(
        SEARCH_R1_CONFIGS["google_api_key"],
        query,
        SEARCH_R1_CONFIGS["topk"],
        snippet_only=SEARCH_R1_CONFIGS["snippet_only"],
        proxy=SEARCH_R1_CONFIGS["proxy"],
    )
    return _passages2string(result)


def postprocess_responses(resp: str) -> str:
    return (
        resp.split("</search>")[0] + "</search>"
        if "</search>" in resp
        else resp.split("</answer>")[0] + "</answer>" if "</answer>" in resp else resp
    )


def postprocess_predictions(prediction: str):
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action = match.group(1)
    else:
        content = ""
        action = None

    return action, content


async def execute_predictions(prediction: str) -> str:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = f"\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, f"Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Handle partial rollout samples: continue generation from existing response
    prompt = sample.prompt
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    for _ in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
        }
        output = await post(url, payload, use_http2=args.use_http2)

        # abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_masks = loss_masks
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
