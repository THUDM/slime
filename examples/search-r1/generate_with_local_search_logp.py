# Adapted from generate_with_search.py to use local search engine
# This is an example showing how to use local_search_server.py instead of google_search_server.py

import asyncio
import re

from local_search_server import local_search
from qa_em_format import compute_score_em

from slime import rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Configuration for Search-R1 with local search engine
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    # Local search engine configuration
    "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
    "proxy": None,  # Typically not needed for local server
    "search_concurrency": 256,
    # rm
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    """
    Perform search using local search engine.

    Note: local_search() uses search_url instead of api_key as the first parameter,
    but otherwise maintains the same interface as google_search().
    """
    result = await local_search(
        SEARCH_R1_CONFIGS["search_url"],  # search_url instead of api_key
        query,
        SEARCH_R1_CONFIGS["topk"],
        proxy=SEARCH_R1_CONFIGS["proxy"],
    )
    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is commented out when using this logp-enabled version.
# def postprocess_responses(resp: str) -> str:
#     return (
#         # <search> text <search>
#         # <search> 1, 0 + 2
#         # [1,2,1,2,0,2]
        
#         resp.split("</search>")[0] + "</search>"
#         if "</search>" in resp
#         else resp.split("</answer>")[0] + "</answer>" if "</answer>" in resp else resp
#     )


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
    loss_mask = []
    rollout_log_probs = []
    for _ in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        # abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        
        cur_response = output["text"]

        # Extract log probs from output - required for TIS metrics
        if "output_token_logprobs" not in output["meta_info"]:
            raise RuntimeError(
                "output_token_logprobs not found in output meta_info. "
                "Make sure 'return_logprob': True is set in the payload."
            )

        # Use token IDs and log probs directly from output_token_logprobs
        # This ensures perfect alignment between tokens and log probs
        # output_token_logprobs format: [[log_prob, token_id, ...], ...]
        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

        response += cur_response
        response_token_ids += cur_response_token_ids
        rollout_log_probs += cur_response_log_probs
        loss_mask += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)
        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        rollout_log_probs += [0.0] * len(obs_tokens_ids)

        assert len(response_token_ids) == len(rollout_log_probs), f'log p length is {len(response_token_ids)} and response length is {len(rollout_log_probs)}'
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None
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
