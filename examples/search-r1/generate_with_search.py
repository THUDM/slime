# Adapted form https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
import asyncio
import json
import logging
import os
import re
from typing import Tuple

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

SEARCH_R1_CONFIGS = {
    # LLM API Mode: "generate" (XML format) | "chat" (OpenAI function calling)
    "llm_api_mode": os.getenv("LLM_API_MODE", "generate"),
    # Search Backend: "google" (Google API) | "local" (dense retriever)
    "search_backend": os.getenv("SEARCH_BACKEND", "google"),
    # Google configuration
    "google_api_key": os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY"),
    "google_snippet_only": True,
    "google_proxy": None,
    # Local retriever configuration
    "retrieval_server_url": os.getenv("RETRIEVAL_SERVER_URL", "http://localhost:8000"),
    "retrieval_timeout": 60.0,
    "retrieval_max_connections": 256,
    "retrieval_max_keepalive": 64,
    # Token Tracking: "manual" | "router_radix" | "router_handler"
    "token_tracking_mode": os.getenv("TOKEN_TRACKING_MODE", "manual"),
    "track_log_probs": os.getenv("TRACK_LOG_PROBS", "true").lower() == "true",
    # Loss Mask: "strict" (assistant=1, tool=0) | "simple" (all=1)
    "loss_mask_mode": os.getenv("LOSS_MASK_MODE", "strict"),
    # Other parameters
    "max_turns": 3,
    "topk": 3,
    "search_concurrency": 256,
    "format_score": 0.2,
    "max_context_length": 16384,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])

# Shared HTTP client for local retriever (lazy initialized)
SHARED_HTTP_CLIENT = None


def _get_http_client():
    """Lazy load HTTP client for local retriever"""
    global SHARED_HTTP_CLIENT
    if SHARED_HTTP_CLIENT is None and SEARCH_R1_CONFIGS["search_backend"] == "local":
        import httpx

        SHARED_HTTP_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=SEARCH_R1_CONFIGS["retrieval_max_connections"],
                max_keepalive_connections=SEARCH_R1_CONFIGS["retrieval_max_keepalive"],
            ),
            timeout=SEARCH_R1_CONFIGS["retrieval_timeout"],
        )
    return SHARED_HTTP_CLIENT


def _cleanup_http_client():
    """Cleanup HTTP client on exit"""
    global SHARED_HTTP_CLIENT
    if SHARED_HTTP_CLIENT:
        try:
            asyncio.run(SHARED_HTTP_CLIENT.aclose())
        except Exception:
            pass  # Best effort cleanup


import atexit

atexit.register(_cleanup_http_client)


def _passages2string_google(retrieval_result):
    """Format Google search results (original logic)"""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


def _passages2string_local(results, queries):
    """Format local retriever results (adapted from search-r1-oai)"""
    formatted_results = ""
    for query_idx, query_results in enumerate(results):
        if len(queries) > 1:
            formatted_results += f"Results for query {query_idx + 1}: {queries[query_idx]}\n"
        for idx, doc in enumerate(query_results):
            title = doc.get("title", "No title")
            text = doc.get("text", "")
            if not text:
                # Fallback to contents field if text is empty
                contents = doc.get("contents", "")
                if contents:
                    lines = contents.split("\n")
                    title = lines[0].strip('"') if lines else "No title"
                    text = "\n".join(lines[1:]) if len(lines) > 1 else ""
            formatted_results += f"Doc {idx+1}(Title: {title}) {text}\n"
        if query_idx < len(results) - 1:
            formatted_results += "\n"
    return formatted_results


async def search(queries) -> str:
    """Unified search interface supporting both Google and Local backends"""
    if SEARCH_R1_CONFIGS["search_backend"] == "google":
        # Google API mode (single query)
        from google_search_server import google_search

        query = queries[0] if isinstance(queries, list) else queries
        async with SEMAPHORE:
            result = await google_search(
                SEARCH_R1_CONFIGS["google_api_key"],
                query,
                SEARCH_R1_CONFIGS["topk"],
                snippet_only=SEARCH_R1_CONFIGS["google_snippet_only"],
                proxy=SEARCH_R1_CONFIGS["google_proxy"],
            )
        return _passages2string_google(result)

    else:  # local
        # Local retriever mode (supports batch queries)
        queries = [queries] if isinstance(queries, str) else queries
        url = f"{SEARCH_R1_CONFIGS['retrieval_server_url']}/retrieve"
        payload = {
            "queries": queries,
            "topk": SEARCH_R1_CONFIGS["topk"],
            "return_scores": False,
        }

        client = _get_http_client()
        async with SEMAPHORE:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        return _passages2string_local(data["result"], queries)


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


async def execute_predictions(prediction: str) -> Tuple[str, bool]:
    """Execute parsed prediction (search or answer)"""
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        search_results = await search(search_query)  # Use unified search interface
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
            "Let me try again.\n"
        )
        done = False

    return next_obs, done


def _get_tool_loss_mask(length: int) -> list:
    """Get loss mask for tool/search results based on configuration"""
    mask_value = 0 if SEARCH_R1_CONFIGS["loss_mask_mode"] == "strict" else 1
    return [mask_value] * length


def _validate_loss_mask(response_token_ids, loss_mask, mode):
    """Validate loss_mask length matches response tokens (strict, fail-fast)"""
    if len(loss_mask) != len(response_token_ids):
        raise ValueError(
            f"Loss mask length mismatch: response_tokens={len(response_token_ids)}, "
            f"loss_mask={len(loss_mask)}, mode={mode}. "
            f"This indicates a critical bug in token/loss_mask tracking."
        )


# Token Tracking Functions

async def _extract_tokens_manual(output, state, response_text):
    """Extract tokens from API logprobs or tokenizer"""
    token_ids = []
    log_probs = []

    # Chat mode: OpenAI API format (choices[0].logprobs.content)
    if SEARCH_R1_CONFIGS["llm_api_mode"] == "chat" and "choices" in output and len(output["choices"]) > 0:
        choice = output["choices"][0]
        if "logprobs" in choice and choice["logprobs"] and "content" in choice["logprobs"]:
            for item in choice["logprobs"]["content"]:
                if "token_id" in item:
                    token_ids.append(item["token_id"])
                    if SEARCH_R1_CONFIGS["track_log_probs"]:
                        log_probs.append(item.get("logprob", 0.0))

    # ===== NEW: Generate mode support (SGLang /generate API format) =====
    elif (
        SEARCH_R1_CONFIGS["llm_api_mode"] == "generate"
        and isinstance(output, dict)
        and "meta_info" in output
        and "output_token_logprobs" in output.get("meta_info", {})
    ):
        # SGLang /generate returns output_token_logprobs as [[logprob, token_id], ...]
        for logprob, token_id in output["meta_info"]["output_token_logprobs"]:
            token_ids.append(token_id)
            if SEARCH_R1_CONFIGS["track_log_probs"]:
                log_probs.append(float(logprob))
    # =====================================================================

    # Universal fallback: tokenizer encoding (when API doesn't return logprobs)
    if not token_ids:
        token_ids = state.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        if SEARCH_R1_CONFIGS["track_log_probs"]:
            log_probs = [0.0] * len(token_ids)

    return token_ids, log_probs if SEARCH_R1_CONFIGS["track_log_probs"] else []


async def _extract_tokens_router_radix(args, text):
    """Query Router RadixTree cache via /retrieve_from_text

    Returns:
        tuple: (tokens, logprobs, loss_mask) or (None, None, None) on failure
    """
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
    response = await post(url, {"text": text, "return_logprob": SEARCH_R1_CONFIGS["track_log_probs"]})

    if not response:
        logger.warning("Router retrieval failed, falling back to manual extraction")
        return None, None, None

    tokens = response.get("tokens", [])
    logprobs = response.get("rollout_logp", []) if SEARCH_R1_CONFIGS["track_log_probs"] else []
    loss_mask = response.get("loss_mask", [])

    return tokens, logprobs, loss_mask


async def _extract_tokens_router_handler(args, messages, tools):
    """Query Router ChatHandler via /retrieve_from_messages_template

    Returns:
        tuple: (tokens, logprobs, loss_mask) or (None, None, None) on failure
    """
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_messages_template"
    response = await post(
        url, {"messages": messages, "tools": tools, "return_logprob": SEARCH_R1_CONFIGS["track_log_probs"]}
    )

    if not response:
        logger.warning("Router retrieval failed, falling back to manual extraction")
        return None, None, None

    tokens = response.get("tokens", [])
    logprobs = response.get("rollout_logp", []) if SEARCH_R1_CONFIGS["track_log_probs"] else []
    loss_mask = response.get("loss_mask", [])

    return tokens, logprobs, loss_mask


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Unified generation entry point"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    # ===== Mode compatibility validation =====
    mode = SEARCH_R1_CONFIGS["token_tracking_mode"]
    api_mode = SEARCH_R1_CONFIGS["llm_api_mode"]

    # Strict validation: router_handler + generate mode is incompatible
    if mode == "router_handler" and api_mode == "generate":
        raise ValueError(
            f"Invalid configuration: token_tracking_mode='router_handler' is NOT compatible with llm_api_mode='generate'. "
            f"router_handler requires OpenAI messages format (/v1/chat/completions), but generate mode uses text format (/generate). "
            f"Valid combinations:\n"
            f"  - llm_api_mode='generate' + token_tracking_mode='manual' (extract from API response)\n"
            f"  - llm_api_mode='generate' + token_tracking_mode='router_radix' (query /retrieve_from_text)\n"
            f"  - llm_api_mode='chat' + token_tracking_mode='manual' (extract from API response)\n"
            f"  - llm_api_mode='chat' + token_tracking_mode='router_handler' (query /retrieve_from_messages_template)\n"
        )

    # Warning: router_radix + chat mode loses messages semantics
    if mode == "router_radix" and api_mode == "chat":
        logger.warning(
            "token_tracking_mode='router_radix' with llm_api_mode='chat' will flatten messages to text, "
            "losing OpenAI messages structure. Consider using 'router_handler' for better semantic preservation."
        )
    # ==============================================

    if SEARCH_R1_CONFIGS["llm_api_mode"] == "chat":
        return await _generate_chat(args, sample, sampling_params)
    else:
        return await _generate_text(args, sample, sampling_params)


async def _generate_chat(args, sample: Sample, sampling_params) -> Sample:
    """Chat completions mode (OpenAI function calling format)"""
    state = GenerateState(args)

    messages = sample.prompt  # OpenAI messages format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries",
                        }
                    },
                    "required": ["queries"],
                },
            },
        }
    ]

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"

    # Calculate initial prompt tokens
    initial_text = state.tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    prompt_token_ids = state.tokenizer(initial_text, add_special_tokens=False)["input_ids"]

    # Tracking variables
    all_token_ids = prompt_token_ids.copy()
    response_token_ids = []
    response_log_probs = []
    response_loss_mask = []
    response_text = ""

    # ===== Multi-turn generation loop =====
    for turn in range(SEARCH_R1_CONFIGS["max_turns"]):
        # Check context length
        if len(all_token_ids) >= SEARCH_R1_CONFIGS["max_context_length"]:
            logger.warning(
                f"Context length {len(all_token_ids)} exceeds max {SEARCH_R1_CONFIGS['max_context_length']}, stopping"
            )
            sample.status = Sample.Status.TRUNCATED
            break

        # Call API
        payload = {
            "messages": messages,
            "tools": tools,
            "temperature": sampling_params.get("temperature", 0.8),
            "max_tokens": sampling_params.get("max_new_tokens", 512),
        }
        if SEARCH_R1_CONFIGS["track_log_probs"]:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        output = await post(url, payload)

        if not output or "choices" not in output or len(output["choices"]) == 0:
            logger.warning(f"API response missing choices at turn {turn}")
            sample.status = Sample.Status.ABORTED
            break

        message = output["choices"][0]["message"]
        finish_reason = output["choices"][0].get("finish_reason", "stop")

        # ONLY extract tokens per-turn for manual mode
        # (router_handler mode will query once at the end)
        if SEARCH_R1_CONFIGS["token_tracking_mode"] == "manual":
            # Extract tokens from API response
            assistant_token_ids, assistant_log_probs = await _extract_tokens_manual(
                output, state, message.get("content", "")
            )

            # Update tracking
            response_token_ids.extend(assistant_token_ids)
            if SEARCH_R1_CONFIGS["track_log_probs"]:
                response_log_probs.extend(assistant_log_probs)
            all_token_ids.extend(assistant_token_ids)
            response_loss_mask.extend([1] * len(assistant_token_ids))  # Assistant content

        response_text = message.get("content", "")

        # Handle tool calls
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            messages.append(message)
            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
            else:
                sample.status = Sample.Status.COMPLETED
            break

        messages.append(message)

        # Execute search
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call arguments: {e}")
                continue

            if func_name == "search":
                queries = arguments.get("queries", [])
                if queries:
                    try:
                        search_results = await search(queries)
                    except Exception as e:
                        logger.error(f"Search failed: {e}")
                        search_results = f"Error during retrieval: {str(e)}"

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": func_name,
                        "content": search_results,
                    }

                    # Calculate tool message tokens ONLY for manual mode (incremental method)
                    if SEARCH_R1_CONFIGS["token_tracking_mode"] == "manual":
                        text_before = state.tokenizer.apply_chat_template(
                            messages, tools=tools, add_generation_prompt=True, tokenize=False
                        )
                        messages.append(tool_message)
                        text_after = state.tokenizer.apply_chat_template(
                            messages, tools=tools, add_generation_prompt=True, tokenize=False
                        )
                        delta_text = text_after[len(text_before) :]
                        delta_tokens = state.tokenizer(delta_text, add_special_tokens=False)["input_ids"]

                        response_token_ids.extend(delta_tokens)
                        if SEARCH_R1_CONFIGS["track_log_probs"]:
                            response_log_probs.extend([0.0] * len(delta_tokens))
                        all_token_ids.extend(delta_tokens)
                        response_loss_mask.extend(_get_tool_loss_mask(len(delta_tokens)))
                    else:
                        # For router_handler mode, just append message (no token calculation yet)
                        messages.append(tool_message)

        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

    # ===== Extract tokens ONCE after all turns (router_handler mode) =====
    if SEARCH_R1_CONFIGS["token_tracking_mode"] == "router_handler":
        # Use helper function to query Router
        all_token_ids, all_log_probs, all_loss_mask = await _extract_tokens_router_handler(args, messages, tools)

        if all_token_ids is None:
            # Fallback to tokenizer
            logger.warning("Router query failed, falling back to tokenizer")
            final_text = state.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=False, tokenize=False
            )
            all_token_ids = state.tokenizer(final_text, add_special_tokens=False)["input_ids"]
            response_token_ids = all_token_ids[len(prompt_token_ids) :]
            response_log_probs = (
                [0.0] * len(response_token_ids) if SEARCH_R1_CONFIGS["track_log_probs"] else []
            )
            response_loss_mask = [1] * len(response_token_ids)  # Fallback: simplified loss_mask
        else:
            # Extract response portion
            response_token_ids = all_token_ids[len(prompt_token_ids) :]
            response_log_probs = (
                all_log_probs[len(prompt_token_ids) :] if SEARCH_R1_CONFIGS["track_log_probs"] else []
            )
            response_loss_mask = all_loss_mask[len(prompt_token_ids) :]  # Use Router's precise loss_mask

            logger.debug(
                f"router_handler query successful: {len(all_token_ids)} tokens, "
                f"{len(all_loss_mask)} loss_mask entries"
            )

    # Validate loss_mask length (fail-fast)
    _validate_loss_mask(response_token_ids, response_loss_mask, SEARCH_R1_CONFIGS["token_tracking_mode"])

    # Build Sample
    sample.tokens = all_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = response_loss_mask
    if SEARCH_R1_CONFIGS["track_log_probs"]:
        sample.rollout_log_probs = response_log_probs

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    return sample


async def _generate_text(args, sample: Sample, sampling_params) -> Sample:
    """Text generation mode (original XML format, backward compatible)"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    mode = SEARCH_R1_CONFIGS["token_tracking_mode"]

    prompt = sample.prompt
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response = ""

    # Initialize tracking variables based on mode
    if mode == "manual":
        # Manual mode: per-turn extraction, accumulate tokens/logprobs/loss_mask during loop
        response_token_ids = []
        response_log_probs = []
        response_loss_mask = []
    # Router mode: only accumulate response text, extract tokens at end

    # ===== Multi-turn generation loop =====
    for turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        # Context length check (manual mode only, router mode relies on SGLang)
        if mode == "manual":
            current_length = len(prompt_tokens_ids) + len(response_token_ids)
            if current_length >= SEARCH_R1_CONFIGS["max_context_length"]:
                logger.warning(
                    f"Context length {current_length} exceeds max {SEARCH_R1_CONFIGS['max_context_length']}, stopping"
                )
                sample.status = Sample.Status.TRUNCATED
                break

        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
            "return_logprob": SEARCH_R1_CONFIGS["track_log_probs"],
        }
        output = await post(url, payload)

        if not output:
            logger.error(f"Generate API call failed at turn {turn_idx}")
            sample.status = Sample.Status.ABORTED
            break

        # Abort check
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        # ===== Manual mode: Extract tokens per-turn =====
        if mode == "manual":
            cur_tokens, cur_logprobs = await _extract_tokens_manual(output, state, cur_response)
            response_token_ids.extend(cur_tokens)
            if SEARCH_R1_CONFIGS["track_log_probs"]:
                response_log_probs.extend(cur_logprobs)
            response_loss_mask.extend([1] * len(cur_tokens))  # Assistant content

        response += cur_response

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."

        # ===== Manual mode: Extract tool/observation tokens per-turn =====
        if mode == "manual":
            obs_tokens = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response_token_ids.extend(obs_tokens)
            if SEARCH_R1_CONFIGS["track_log_probs"]:
                response_log_probs.extend([0.0] * len(obs_tokens))
            response_loss_mask.extend(_get_tool_loss_mask(len(obs_tokens)))  # Tool content

        response += next_obs

    # ===== End-of-trajectory processing =====
    if mode == "router_radix":
        # Query Router for complete trajectory (leverages RadixTree cache)
        full_text = prompt + response
        full_token_ids, full_log_probs, full_loss_mask = await _extract_tokens_router_radix(args, full_text)

        if full_token_ids is None:
            # Fallback to tokenizer
            logger.warning("Router query failed, falling back to tokenizer")
            full_token_ids = state.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            full_log_probs = [0.0] * len(full_token_ids) if SEARCH_R1_CONFIGS["track_log_probs"] else []
            full_loss_mask = [1] * len(full_token_ids)  # Simplified: all trainable

        # Extract response portion
        response_token_ids = full_token_ids[len(prompt_tokens_ids) :]
        response_log_probs = full_log_probs[len(prompt_tokens_ids) :] if SEARCH_R1_CONFIGS["track_log_probs"] else []
        response_loss_mask = full_loss_mask[len(prompt_tokens_ids) :]

    # Manual mode: already accumulated during loop, no further processing needed

    # Validate loss_mask length (fail-fast)
    _validate_loss_mask(response_token_ids, response_loss_mask, mode)

    # Build Sample
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = response_loss_mask
    if SEARCH_R1_CONFIGS["track_log_probs"]:
        sample.rollout_log_probs = response_log_probs

    # Set final status based on last finish_reason
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Unified reward function for retrieval-based question answering.

    Reward calculation logic:
    - Chat mode: Use sample.response directly (final assistant message)
    - Generate mode: Extract content after last </information> tag

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Extract evaluation string based on API mode
    if SEARCH_R1_CONFIGS["llm_api_mode"] == "chat":
        # Chat mode: use response directly
        eval_str = sample.response
    else:
        # Generate mode: extract content after last </information> tag
        response = sample.response
        last_info_idx = response.rfind("</information>")

        if last_info_idx != -1:
            # Extract everything after </information>
            eval_str = response[last_info_idx + len("</information>") :].strip()
        else:
            # No </information> tag found, use full response
            eval_str = response

    score = compute_score_em(
        solution_str=eval_str,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
