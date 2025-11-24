# Adapted form https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
import asyncio
import json
import logging
import os
import random

import httpx
from qa_em_format import compute_score_em, log_turn_by_turn
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

SEARCH_R1_CONFIGS = {
    "max_turns": 5,
    "topk": 3,
    "retrieval_server_url": os.getenv("RETRIEVAL_SERVER_URL", "http://localhost:8000"),  # Local dense retriever
    "search_concurrency": 256,
    # HTTP client connection pool settings
    "max_connections": int(os.getenv("SEARCH_MAX_CONNECTIONS", "256")),  # Max concurrent TCP connections
    "max_keepalive_connections": int(os.getenv("SEARCH_MAX_KEEPALIVE", "64")),  # Max idle keep-alive connections
    "request_timeout": float(os.getenv("SEARCH_TIMEOUT", "60.0")),  # Request timeout in seconds
    # rm
    "format_score": 0.2,
    "max_context_length": 32768,  # Maximum context length in tokens
    "log_sample_rate": 64,  # Sample 1/N for detailed logging
    "log_truncate_length": 200,  # Character limit for log truncation
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])

# Shared HTTP client for all search requests to avoid resource exhaustion
SHARED_HTTP_CLIENT = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=SEARCH_R1_CONFIGS["max_connections"],
        max_keepalive_connections=SEARCH_R1_CONFIGS["max_keepalive_connections"],
    ),
    timeout=SEARCH_R1_CONFIGS["request_timeout"],
)


def _passages2string(retrieval_result):
    """Format retrieval results from local dense retriever.

    Args:
        retrieval_result: List of document dicts from local retriever
                         Each doc has: {"title": "...", "text": "...", "contents": "..."}
    """
    format_reference = ""
    for idx, doc in enumerate(retrieval_result):
        # Local retriever returns docs with 'title', 'text', 'contents' fields
        title = doc.get("title", "No title")
        text = doc.get("text", "")
        if not text:
            # Fallback to contents field if text is empty
            contents = doc.get("contents", "")
            if contents:
                # Parse title and text from contents
                lines = contents.split("\n")
                title = lines[0].strip('"') if lines else "No title"
                text = "\n".join(lines[1:]) if len(lines) > 1 else ""

        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


@retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def search(queries: list[str]) -> str:
    """Call local dense retriever service with automatic retry.

    Args:
        queries: List of search queries

    Returns:
        Formatted string of retrieval results
    """
    url = f"{SEARCH_R1_CONFIGS['retrieval_server_url']}/retrieve"
    payload = {
        "queries": queries,
        "topk": SEARCH_R1_CONFIGS["topk"],
        "return_scores": False,
    }

    async with SEMAPHORE:
        # Use shared HTTP client instead of creating new one each time
        response = await SHARED_HTTP_CLIENT.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # data["result"] is a list of lists: [[doc1, doc2], [doc3, doc4], ...]
    # Each inner list corresponds to results for one query
    results = data["result"]

    # Format all results into a single string
    formatted_results = ""
    for query_idx, query_results in enumerate(results):
        if len(queries) > 1:
            formatted_results += f"Results for query {query_idx + 1}: {queries[query_idx]}\n"
        formatted_results += _passages2string(query_results)
        if query_idx < len(results) - 1:
            formatted_results += "\n"

    return formatted_results


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate responses using OpenAI function calling format with local retrieval.

    This implementation uses /v1/chat/completions API with tool calls to:
    1. Get accurate token_ids from logprobs for model-generated content
    2. Use apply_chat_template incremental method for tool messages
    3. Maintain correct loss_mask (assistant=1, tool=0)
    """
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    tokenizer = state.tokenizer

    # Initialize from sample
    messages = sample.prompt  # Should be a list of message dicts
    tools = sample.metadata.get("tools") if hasattr(sample, "metadata") and sample.metadata else None

    # Calculate initial prompt tokens (for final calculation)
    initial_text = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    prompt_token_ids = tokenizer(initial_text, add_special_tokens=False)["input_ids"]

    # Tracking variables
    all_token_ids = prompt_token_ids.copy()
    response_token_ids = []
    response_log_probs = []  # Track log probabilities for rollout
    loss_mask = []
    response_text = ""

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"

    for turn in range(SEARCH_R1_CONFIGS["max_turns"]):
        # Check context length before this turn
        if len(all_token_ids) >= SEARCH_R1_CONFIGS["max_context_length"]:
            logger.warning(f"Context length {len(all_token_ids)} exceeds max {SEARCH_R1_CONFIGS['max_context_length']}, stopping generation")
            sample.status = Sample.Status.TRUNCATED
            break

        # Call chat completions API with tools
        payload = {
            "messages": messages,
            "tools": tools,
            "temperature": sampling_params.get("temperature", 0.8),
            "max_tokens": sampling_params.get("max_new_tokens", 512),
            "logprobs": True,  # Request logprobs for accurate token_ids
            "top_logprobs": 1,
        }

        try:
            output = await post(url, payload)
        except Exception as e:
            logger.error(f"API call failed at turn {turn}: {type(e).__name__}: {str(e)}")
            sample.status = Sample.Status.ABORTED
            sample.response = response_text
            sample.tokens = all_token_ids
            sample.response_length = len(response_token_ids)
            sample.loss_mask = loss_mask
            sample.rollout_log_probs = response_log_probs
            return sample

        # Parse response
        if "choices" not in output or len(output["choices"]) == 0:
            logger.warning(f"API response missing choices at turn {turn}")
            sample.status = Sample.Status.ABORTED
            break

        choice = output["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        # Check for tool calls first (needed for token extraction logic)
        tool_calls = message.get("tool_calls")

        # Extract token_ids and log_probs from logprobs (model-generated content)
        assistant_token_ids = []
        assistant_log_probs = []
        if "logprobs" in choice and choice["logprobs"] and "content" in choice["logprobs"]:
            content_logprobs = choice["logprobs"]["content"]
            for item in content_logprobs:
                # Extract log probability (always available in logprobs)
                log_prob = item.get("logprob", 0.0)

                # Try to get token_id if available, otherwise encode the token string
                if "token_id" in item:
                    assistant_token_ids.append(item["token_id"])
                    assistant_log_probs.append(log_prob)
                elif "bytes" in item and item["bytes"]:
                    # Decode bytes and encode to get token_id
                    token_bytes = item["bytes"]
                    if isinstance(token_bytes, list):
                        token_str = bytes(token_bytes).decode("utf-8", errors="ignore")
                    else:
                        token_str = token_bytes
                    # Tokenize to get ID
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_ids:
                        assistant_token_ids.append(token_ids[0])
                        assistant_log_probs.append(log_prob)
                elif "token" in item:
                    # Fallback: encode token string
                    token_str = item["token"]
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_ids:
                        assistant_token_ids.append(token_ids[0])
                        assistant_log_probs.append(log_prob)
        else:
            # Fallback: manually tokenize content
            # Skip retokenization if tool_calls exist (to avoid prefix instability)
            if not tool_calls:
                assistant_content = message.get("content") or ""
                if assistant_content:
                    assistant_token_ids = tokenizer(assistant_content, add_special_tokens=False)["input_ids"]
                    # No log_probs available in fallback, use 0.0
                    assistant_log_probs = [0.0] * len(assistant_token_ids)

        # Update tracking for assistant message
        if assistant_token_ids:
            response_token_ids.extend(assistant_token_ids)
            response_log_probs.extend(assistant_log_probs)
            all_token_ids.extend(assistant_token_ids)
            loss_mask.extend([1] * len(assistant_token_ids))  # Assistant content: loss_mask=1

        assistant_content = message.get("content") or ""
        response_text = assistant_content

        if not tool_calls:
            # No tool calls, generation complete
            # IMPORTANT: Add final assistant message before breaking (for turn-by-turn logging)
            messages.append(message)
            logger.debug(f"Added final assistant message (no tool calls), finish_reason={finish_reason}")

            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
            else:
                sample.status = Sample.Status.COMPLETED
            break

        # Add assistant message (with tool_calls) to conversation
        messages.append(message)
        logger.debug(f"Added assistant message with {len(tool_calls)} tool_calls")

        # Execute tool calls and collect tool messages
        tool_messages = []
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]

            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call arguments for {func_name}: {str(e)}")
                continue

            if func_name == "search":
                queries = arguments.get("queries", [])
                if not queries:
                    continue

                # Call local retrieval service (with automatic retry)
                try:
                    search_results = await search(queries)
                except Exception as e:
                    # This exception is raised after all retry attempts have been exhausted
                    logger.error(f"Search failed after retries for queries {queries}: {type(e).__name__}: {str(e)}")
                    search_results = f"Error during retrieval: {str(e)}"

                # Create tool message
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": search_results,
                }

                # Add to tool response messages list
                tool_messages.append(tool_message)

        # Process all tool messages as a unit using delta-based approach
        if tool_messages:
            # Validate tool_messages structure before processing
            assert isinstance(tool_messages, list), f"tool_messages must be list, got {type(tool_messages)}"
            for tm in tool_messages:
                assert isinstance(tm, dict), f"Each tool message must be dict, got {type(tm)}"
                assert tm.get("role") == "tool", f"Tool message must have role='tool', got {tm.get('role')}"

            logger.debug(f"Adding {len(tool_messages)} tool messages to conversation")

            # Calculate text before adding THIS tool message
            text_before = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )

            # Add tool messages and calculate delta tokens
            messages.extend(tool_messages)
            logger.debug(f"Tool messages added, total messages: {len(messages)}")

            # Calculate delta for this tool message
            text_after = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
            delta_tool_text = text_after[len(text_before):]
            delta_tool_tokens = tokenizer(delta_tool_text, add_special_tokens=False)["input_ids"]

            # Update tracking for tool message
            response_token_ids.extend(delta_tool_tokens)
            response_log_probs.extend([0.0] * len(delta_tool_tokens))  # Tool messages have no log_probs
            all_token_ids.extend(delta_tool_tokens)
            loss_mask.extend([0] * len(delta_tool_tokens))  # Tool content: loss_mask=0

        # Check for length truncation
        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

    # Construct final sample
    sample.tokens = all_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = response_log_probs

    # Ensure metadata is initialized
    if sample.metadata is None:
        sample.metadata = {}

    # Save complete messages history (OpenAI format) to metadata for debugging
    sample.metadata["messages"] = messages

    # Calculate monitoring metrics in a single pass
    last_assistant_msg = None
    total_tool_calls = 0
    num_turns = 0

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            # Count assistant turns
            num_turns += 1
            # Count tool calls
            if msg.get("tool_calls"):
                total_tool_calls += 1
            # Capture last assistant message (first one in reverse order)
            if last_assistant_msg is None:
                last_assistant_msg = msg

    # Check if final response has <answer> tags
    has_final_answer = "<answer>" in response_text and "</answer>" in response_text
    # Check if final turn incorrectly has tool_calls (should be answer-only)
    final_turn_has_tool_calls = last_assistant_msg and last_assistant_msg.get("tool_calls") is not None

    sample.metadata["has_final_answer"] = has_final_answer
    sample.metadata["final_turn_has_tool_calls"] = final_turn_has_tool_calls
    sample.metadata["total_tool_calls"] = total_tool_calls
    sample.metadata["num_turns"] = num_turns

    if sample.status == Sample.Status.PENDING:
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
        solution_str=sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    # Log turn-by-turn rollout for debugging (sample 1/N randomly)
    if random.randint(1, SEARCH_R1_CONFIGS["log_sample_rate"]) == 1:
        log_turn_by_turn(sample, show_full_content=False)
        logger.info(f"Golden answers: {sample.label['ground_truth']['target']}")
        truncate_len = SEARCH_R1_CONFIGS["log_truncate_length"]
        logger.info(f"Extracted answer: {sample.response[:truncate_len]}..." if len(sample.response) > truncate_len else sample.response)
        logger.info(f"Score: {score}")

    return score
