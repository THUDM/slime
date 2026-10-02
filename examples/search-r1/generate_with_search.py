# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import re

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])
_SEARCH_R1_TURN_COUNT_KEY = "search_r1_completed_turns"


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
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. " f"Must be either 'local' or 'google'.")

    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
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


def _last_prediction_action(prediction: str) -> str | None:
    matches = re.findall(r"<(search|answer)>.*?</\1>", prediction, re.DOTALL)
    return matches[-1] if matches else None


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
        next_obs = "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done


def _count_completed_model_turns(response: str) -> int:
    return len(re.findall(r"</(?:search|answer)>", response))


def _get_completed_model_turns(sample: Sample) -> int:
    turn_count = sample.metadata.get(_SEARCH_R1_TURN_COUNT_KEY)
    if isinstance(turn_count, int):
        return turn_count
    return _count_completed_model_turns(sample.response or "")


def _mark_model_turn_completed(sample: Sample, completed_turns: int) -> None:
    sample.metadata[_SEARCH_R1_TURN_COUNT_KEY] = completed_turns


def _append_trainable_response_tokens(
    args,
    sample: Sample,
    *,
    tokens: list[int],
    log_probs: list[float] | None,
    meta_info: dict,
    text: str,
) -> None:
    if log_probs is not None:
        sample.append_response_tokens(
            args,
            tokens=tokens,
            log_probs=log_probs,
            trainable=True,
            meta_info=meta_info,
            text=text,
        )
        return

    sample.response += text
    sample.tokens += tokens
    sample.response_length += len(tokens)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask += [1] * len(tokens)
    match meta_info["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED


async def generate(args, sample: Sample, sampling_params) -> Sample:
    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = sample.prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not sample.tokens:
        sample.tokens = list(prompt_tokens_ids)
    if sample.loss_mask is None:
        sample.loss_mask = [1] * sample.response_length
    elif args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length
    if SEARCH_R1_CONFIGS["return_logprob"] and sample.rollout_log_probs is None and sample.response_length > 0:
        sample.rollout_log_probs = [0.0] * sample.response_length

    response = sample.response or ""

    # BUGFIX: make the inference engine STOP at the tool/answer boundary.
    # Without a stop, sglang keeps emitting tokens after </search> / </answer>
    # (junk, even fabricated new "Question:"s). The example only trimmed that junk
    # via postprocess_responses when return_logprob=False; with return_logprob=True
    # (TIS) trimming is disabled to keep token/logp aligned, so the junk stayed in
    # the trajectory and got trained on (loss_mask=1) AND broke is_valid_sequence
    # (trailing content after </answer> -> format invalid -> lower reward).
    # Stopping at the tag avoids all of that and keeps token/logp aligned natively.
    # slime already sets no_stop_trim=True, so the closing tag is kept in the output.
    _stop_tags = ["</search>", "</answer>"]
    _existing_stop = sampling_params.get("stop") or []
    if isinstance(_existing_stop, str):
        _existing_stop = [_existing_stop]
    sampling_params = {**sampling_params, "stop": list(dict.fromkeys([*_existing_stop, *_stop_tags]))}

    output = None
    completed_turns = _get_completed_model_turns(sample) if args.partial_rollout else 0
    if args.partial_rollout and _last_prediction_action(response) == "answer":
        sample.status = Sample.Status.COMPLETED
        return sample

    for _turn_idx in range(completed_turns, SEARCH_R1_CONFIGS["max_turns"]):
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)

        cur_response = output["text"]
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        # Extract tokens and log probs based on configuration
        cur_response_log_probs = None
        if SEARCH_R1_CONFIGS["return_logprob"]:
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
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        _append_trainable_response_tokens(
            args,
            sample,
            tokens=cur_response_token_ids,
            log_probs=cur_response_log_probs if SEARCH_R1_CONFIGS["return_logprob"] else None,
            meta_info=output["meta_info"],
            text=cur_response,
        )

        if finish_reason == "length":
            break

        if finish_reason == "stop":
            _mark_model_turn_completed(sample, _turn_idx + 1)

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        sample.append_response_tokens(args, tokens=obs_tokens_ids, trainable=False, text=next_obs)

        # Verify alignment when collecting log probs. Observation tokens receive dummy
        # log probs inside append_response_tokens because loss_mask marks them non-trainable.
        if SEARCH_R1_CONFIGS["return_logprob"]:
            assert sample.rollout_log_probs is not None
            assert len(sample.rollout_log_probs) == sample.response_length, (
                f"Token/logp length mismatch: {sample.response_length} tokens vs "
                f"{len(sample.rollout_log_probs)} logps"
            )

    sample.prompt = prompt_text
    if output is None:
        action = _last_prediction_action(response)
        sample.status = Sample.Status.COMPLETED if action == "answer" else Sample.Status.TRUNCATED
        return sample

    action = _last_prediction_action(response)
    if action != "answer" and output["meta_info"]["finish_reason"]["type"] == "stop":
        sample.status = Sample.Status.TRUNCATED
        return sample

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
