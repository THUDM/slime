# Adapted form https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
import asyncio
import json
import os

import httpx
from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

SEARCH_R1_CONFIGS = {
    "max_turns": 3,
    "topk": 3,
    "retrieval_server_url": os.getenv("RETRIEVAL_SERVER_URL", "http://localhost:8000"),  # Local dense retriever
    "search_concurrency": 256,
    # rm
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


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


async def search(queries: list[str]) -> str:
    """Call local dense retriever service.

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
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30.0)
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
    loss_mask = []
    response_text = ""

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"

    for turn in range(SEARCH_R1_CONFIGS["max_turns"]):
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
            sample.status = Sample.Status.ABORTED
            sample.response = response_text
            sample.tokens = all_token_ids
            sample.response_length = len(response_token_ids)
            sample.loss_mask = loss_mask
            return sample

        # Parse response
        if "choices" not in output or len(output["choices"]) == 0:
            sample.status = Sample.Status.ABORTED
            break

        choice = output["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        # Extract token_ids from logprobs (model-generated content)
        assistant_token_ids = []
        if "logprobs" in choice and choice["logprobs"] and "content" in choice["logprobs"]:
            content_logprobs = choice["logprobs"]["content"]
            for item in content_logprobs:
                # Try to get token_id if available, otherwise encode the token string
                if "token_id" in item:
                    assistant_token_ids.append(item["token_id"])
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
                elif "token" in item:
                    # Fallback: encode token string
                    token_str = item["token"]
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_ids:
                        assistant_token_ids.append(token_ids[0])
        else:
            # Fallback: manually tokenize content
            # Skip retokenization if tool_calls exist (to avoid prefix instability)
            if not tool_calls:
                assistant_content = message.get("content") or ""
                if assistant_content:
                    assistant_token_ids = tokenizer(assistant_content, add_special_tokens=False)["input_ids"]

        # Update tracking for assistant message
        if assistant_token_ids:
            response_token_ids.extend(assistant_token_ids)
            all_token_ids.extend(assistant_token_ids)
            loss_mask.extend([1] * len(assistant_token_ids))  # Assistant content: loss_mask=1

        assistant_content = message.get("content") or ""
        response_text = assistant_content

        # Check for tool calls
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            # No tool calls, generation complete
            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
            else:
                sample.status = Sample.Status.COMPLETED
            break

        # Add assistant message (with tool_calls) to conversation
        messages.append(message)

        # Execute tool calls and collect tool messages
        tool_messages = []
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]

            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError:
                # Skip invalid tool call
                continue

            if func_name == "search":
                queries = arguments.get("queries", [])
                if not queries:
                    continue

                # Call local retrieval service
                try:
                    search_results = await search(queries)
                except Exception as e:
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
            # Calculate text before adding THIS tool message
            text_before = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )

            # Add tool messages and calculate delta tokens
            messages.append(tool_messages)

            # Calculate delta for this tool message
            text_after = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
            delta_tool_text = text_after[len(text_before):]
            delta_tool_tokens = tokenizer(delta_tool_text, add_special_tokens=False)["input_ids"]

            # Update tracking for tool message
            response_token_ids.extend(delta_tool_tokens)
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

    return score
