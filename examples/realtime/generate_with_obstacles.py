"""Slime rollout for the clear-obstacles grid game.

This replaces the Python-code-interpreter tool environment with the
``ClearObstaclesToolEnv`` from ``real-time/environment``. The model plays a grid
game: it must move ``F`` to the GOAL row while avoiding obstacles, by emitting
``move_up`` / ``move_down`` / ``move_left`` / ``move_right`` tool calls.

Key design points:
  * The seed for each episode arrives on ``sample.label`` (wired via
    ``--label-key seed``). ``generate`` reconstructs the *exact* grid from that
    seed with ``ClearObstaclesToolEnv.reset(seed=...)`` and keeps that single
    env instance live for the whole tool-calling loop, so the game is fully
    interactive — each move acts on the real, evolving grid.
  * The terminal reward is whatever ``ClearObstaclesToolEnv`` reports
    (``1.0`` win / ``0.0`` loss, per ``ClearObstaclesEnvironment.is_complete``).
    It is stashed on ``sample.metadata`` during the rollout and read back out by
    ``reward_func``.

Plumbing (token-by-token logprob handling, context-length clamping, abort/length
handling) mirrors ``generate_with_retool.py``.
"""

import json
import re
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# The obstacles environment lives in real-time/environment; it must be on
# PYTHONPATH (the training script adds ./real-time).
from environment.clear_obstacles import CLEAR_SYSTEM_PROMPT, ClearObstaclesToolEnv

# Max number of moves (tool calls) before we cut the episode off.
MAX_TURNS = 64

# The four move tools exposed to the model.
MOVES = ("move_up", "move_down", "move_left", "move_right")

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    for name, desc in (
        ("move_up", "Move F up one row (row - 1, toward the GOAL)."),
        ("move_down", "Move F down one row (row + 1)."),
        ("move_left", "Move F left one column (col - 1)."),
        ("move_right", "Move F right one column (col + 1)."),
    )
]

INVALID_ACTION_MSG = (
    "\n\nMy previous action was invalid. To move I must emit exactly one tool call, "
    "e.g.\n<tool_call>\n{\"name\": \"move_up\", \"arguments\": {}}\n</tool_call>\n"
    "where the name is one of move_up, move_down, move_left, move_right. Let me try again.\n\n"
)

# Jinja2 template for tool-enabled conversations (Qwen-style). Only the initial
# prompt is rendered through this; subsequent moves and observations are appended
# to the token stream verbatim.
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one function at a time to move your character in the game.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_initial_prompt(system_prompt: str, initial_obs: str) -> str:
    """Render the opening prompt: game rules (system) + starting grid (user)."""
    template = Template(TOOL_TEMPLATE)
    messages = [
        {"role": "system", "content": system_prompt or CLEAR_SYSTEM_PROMPT},
        {"role": "user", "content": initial_obs},
    ]
    return template.render(messages=messages, tools=TOOL_SPECS)


def parse_action(prediction: str) -> str | None:
    """Extract the move name from the last <tool_call> in the prediction.

    Returns one of MOVES, or None if no valid move tool call is found.
    """
    matches = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL))
    if not matches:
        return None
    json_str = matches[-1].group(1).replace("\n", "\\n")
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    name = data.get("name")
    return name if name in MOVES else None


def postprocess_responses(resp: str) -> str:
    """Trim the response to the end of its last complete <tool_call> block."""
    matches = list(re.finditer(r"<tool_call>\s*\{.*?\}\s*</tool_call>", resp, re.DOTALL))
    if matches:
        return resp[: matches[-1].end()]
    return resp


def step_environment(env: ClearObstaclesToolEnv, prediction: str) -> tuple[str, bool]:
    """Apply the model's move to the live env and return (observation, done)."""
    action = parse_action(prediction)
    if action is None:
        return INVALID_ACTION_MSG, False

    obs = getattr(env, action)()
    next_obs = f"\n\n<observation>\n{obs}\n</observation>\n\n"
    return next_obs, env.done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function: play the obstacles game via move tool calls."""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    # Retried samples arrive with stale rollout state from the first attempt;
    # clear it so this generation starts clean (see generate_with_retool.py).
    sample.rollout_log_probs = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Reconstruct the exact grid from the dataset seed and keep this env instance
    # live across every move below.
    seed = int(sample.label) if sample.label is not None else None
    env = ClearObstaclesToolEnv()
    initial_obs = env.reset(seed=seed) if seed is not None else env.reset()

    # Carry the system prompt from the dataset (falls back to the env's default).
    system_prompt = sample.prompt if isinstance(sample.prompt, str) and sample.prompt else CLEAR_SYSTEM_PROMPT
    prompt = format_initial_prompt(system_prompt, initial_obs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    move_count = 0

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    for turn in range(MAX_TURNS):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget.
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
            # No per-token logprobs -> cannot keep rollout_log_probs in sync;
            # abort so the group is retried instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            return sample

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = step_environment(env, cur_response)
        if parse_action(cur_response) is not None:
            move_count += 1
        if done:
            # Append the final (GAME OVER) observation so the model sees the
            # terminal grid, then stop.
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            sample.status = Sample.Status.COMPLETED
            break

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

        # Observation is appended verbatim and can push us past the budget; trim
        # the tail so the final sample fits the training budget exactly.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

    # Terminal reward from the env (1.0 win / 0.0 otherwise). Stash it for
    # reward_func, which runs as a separate call on this same sample.
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["env_reward"] = float(env.reward)
    sample.metadata["env_done"] = bool(env.done)
    sample.metadata["env_won"] = bool(env.env.won) if env.env is not None else False
    sample.metadata["move_count"] = move_count

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # If we exited the loop without setting a terminal status above, classify by
    # the last finish_reason.
    if sample.status == Sample.Status.PENDING:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Return the obstacles env reward stashed during generation."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    metadata = sample.metadata or {}
    return {
        "score": float(metadata.get("env_reward", 0.0)),
        "won": bool(metadata.get("env_won", False)),
        "move_count": int(metadata.get("move_count", 0)),
    }
