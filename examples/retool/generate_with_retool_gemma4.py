# Gemma4-compatible retool generate function.
#
# Uses tokenizer.apply_chat_template() instead of the hardcoded Qwen ChatML
# Jinja template in generate_with_retool.py. Everything else (postprocessing,
# tool execution, scoring rules) is reused.
#
# Why a Gemma4-specific version:
#   generate_with_retool.py wraps the prompt in a Qwen ChatML template
#   (<|im_start|>/<|im_end|>). Gemma4's tokenizer doesn't recognize those
#   tokens as specials, and Gemma4's native turn format is
#   <|turn>role\n...<turn|>. Feeding a ChatML-wrapped prompt to Gemma4
#   produces mangled input.
#
# Design choice:
#   Fix the chat framing via apply_chat_template, but keep the Qwen-style
#   <tool_call>{json}</tool_call> contract in the system prompt. This lets us
#   reuse postprocess_predictions / postprocess_responses / execute_predictions
#   unchanged. Switching to Gemma4's native <|tool_call>call:...<tool_call|>
#   format would require rewriting those parsers and is deferred.
#
# Note: the companion yaml must drop --apply-chat-template so sample.prompt
# stays as the raw message list; this function re-templates once with a
# custom system message.
#
# Usage in training args:
#   --custom-generate-function-path generate_with_retool_gemma4.generate
#   --custom-rm-path generate_with_retool_gemma4.reward_func
import json
from typing import Any

from generate_with_retool import (
    execute_predictions,
    postprocess_predictions,  # noqa: F401 - re-exported for external callers
    postprocess_responses,  # noqa: F401 - re-exported for external callers
)
from tool_sandbox import TOOL_CONFIGS, tool_registry

from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

_dropped_system_warned = {"v": False}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python tools to solve "
    "mathematical problems. When you need to perform calculations, use "
    "the code_interpreter tool to execute code and get results."
)


def _build_tool_instructions(tools: list[dict]) -> str:
    """Append Qwen-style tool instructions to the system message.

    We keep the <tool_call>{json}</tool_call> contract (not Gemma4's native
    <|tool_call>call:...<tool_call|>) so postprocess_predictions' regex and
    the reward function stay unchanged. Gemma4-it is strong enough at
    instruction-following to emit this format on request.
    """
    if not tools:
        return ""
    tool_specs = "\n".join(json.dumps(tool) for tool in tools)
    return (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_specs}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _coerce_to_messages(raw_prompt) -> list[dict]:
    """Normalize sample.prompt into a list of {role, content} dicts."""
    if isinstance(raw_prompt, list):
        return list(raw_prompt)
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    raise TypeError(f"Unsupported sample.prompt type: {type(raw_prompt)}")


def format_conversation_with_tools(
    raw_prompt,
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    tokenizer=None,
) -> str:
    """Render the chat-templated prompt using Gemma4's native template.

    We do NOT pass `tools=` to apply_chat_template — that would trigger
    Gemma4's native <|tool>declaration:...<tool|> tool-spec format, which
    downstream postprocess_predictions can't parse. Instead we inline tool
    info as text inside the system message (Qwen-style contract).
    """
    system_content = system_prompt or DEFAULT_SYSTEM_PROMPT
    system_content += _build_tool_instructions(tools or [])

    user_messages = _coerce_to_messages(raw_prompt)
    # If the dataset already contains a system message, prefer our system
    # prompt (which carries the tool instructions) and drop theirs.
    dataset_system = [m for m in user_messages if m.get("role") == "system"]
    if dataset_system and not _dropped_system_warned["v"]:
        # One-shot log — useful during dataset migrations, silent thereafter.
        print(
            "[retool-gemma4] dataset supplied a system message; overriding "
            "with tool-instruction system prompt. "
            f"(dropped: {dataset_system[0].get('content', '')[:120]!r})",
            flush=True,
        )
        _dropped_system_warned["v"] = True
    user_messages = [m for m in user_messages if m.get("role") != "system"]

    messages = [{"role": "system", "content": system_content}, *user_messages]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls (Gemma4 version)."""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    # Retried samples (previously aborted / partial) arrive here with stale
    # rollout state from the first attempt. Clear it so this generation starts
    # clean; otherwise the concatenation below appends new tokens to old ones
    # and downstream `slice_log_prob_with_cp` sees a length mismatch.
    sample.rollout_log_probs = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(
        raw_prompt=sample.prompt,
        tools=tool_specs,
        tokenizer=state.tokenizer,
    )

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    for turn in range(TOOL_CONFIGS["max_turns"]):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget so a
        # single turn cannot push total_length past max_context_length.
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )

        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "debug/payload_length": len(prompt_tokens_ids) + len(response_token_ids),
                        "debug/available_tools": len(tool_specs),
                        "debug/tools_used": response.count("<interpreter>"),
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
        else:
            # sglang returned text but no output_token_logprobs — we cannot
            # recover per-token logprobs for this turn, which would desync
            # rollout_log_probs from response_token_ids and blow up
            # slice_log_prob_with_cp downstream. Abort so the rollout manager
            # returns the group to the buffer for retry instead of poisoning
            # the trainer.
            sample.status = Sample.Status.ABORTED
            return sample

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(sample.rollout_log_probs), (
                f"Token/logp length mismatch at turn {turn}: "
                f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
            )

        # Tool output is appended verbatim and can push total_length past
        # max_context_length. Trim tail tokens so the final sample fits the
        # training budget exactly.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    # Overwrite raw list prompt with the rendered string. Upstream slime
    # (e.g. fully_async_rollout.py:215) does sample.prompt + sample.response
    # in log statements and assumes a string; with --apply-chat-template off,
    # sample.prompt arrives as a list of message dicts and the concat raises
    # TypeError. We've already rendered the string above, so reuse it.
    sample.prompt = prompt

    sample.payload_text = prompt + response
    sample.payload_has_system = True
    sample.payload_has_tools = "# Tools" in prompt

    sample.tool_call_count = tool_call_count

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool-call reward function for Gemma4.

    Mirrors generate_with_retool.reward_func but scores on sample.response
    alone — with --apply-chat-template disabled, sample.prompt is a list of
    message dicts and cannot be string-concatenated. math_dapo_compute_score
    only looks for an Answer: \\boxed{...} pattern, which lives in the
    response, so dropping the prompt from the solution string is safe.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = sample.response
    ground_truth = sample.label if sample.label is not None else ""
    num_turns = getattr(sample, "tool_call_count", 0)

    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
