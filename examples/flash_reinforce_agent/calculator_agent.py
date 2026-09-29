"""A multi-turn calculator agent on GSM8K, the custom generate function of this example.

The model solves a word problem and may call a calculator by ending its turn with
``<calc>EXPR</calc>``; the result comes back as the next user message (rendered with the model's
chat template) and a new assistant turn continues the solution. Each model turn is generated with
:func:`slime.rollout.sglang_rollout.generate_turn`, so a turn cut by a weight update during
fully-async rollout continues under the new weights instead of the trajectory restarting. The
result messages are appended with ``trainable=False``: they stay out of the loss, the importance
weights and the trust region.

Use with ``--custom-generate-function-path calculator_agent.generate`` and this directory on
``PYTHONPATH``.
"""

import ast
import asyncio
import functools
import operator
import os
import re

from slime.rollout.sglang_rollout import GenerateState, generate_turn
from slime.utils.types import Sample

MAX_TOOL_CALLS = 5
TURN_MAX_NEW_TOKENS = 1024
# Seconds each calculator call takes. A slow tool (a sandbox, a search) spreads a trajectory's turns over
# time, so some turns are in flight when the weights update and continue under the new ones.
TOOL_LATENCY = float(os.environ.get("CALCULATOR_LATENCY", "0"))
CALL_END = "</calc>"
CALL = re.compile(r"<calc>(.*?)</calc>\s*$", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a careful math assistant. For arithmetic you may use a calculator: end your message with "
    "<calc>expression</calc>, for example <calc>(12 + 7) * 3</calc>, and the result comes back in the next "
    "message; then continue the solution. Give the final answer as \\boxed{number}."
)

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str) -> str:
    """Evaluate numbers with + - * / // % ** and parentheses; anything else is reported as an error."""

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](value(node.operand))
        raise ValueError("only numbers and + - * / // % ** are supported")

    try:
        result = value(ast.parse(expression.replace(",", "").strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
        return f"error: {error}"
    if isinstance(result, float):
        return str(int(result)) if result.is_integer() else f"{result:.10g}"
    return str(result)


@functools.cache
def _result_message(tokenizer) -> tuple[str, str]:
    """The chat-template text around a calculator result: closing the assistant turn that called the
    calculator, the result as a user message, and the opening of the next assistant turn."""
    call, result = "\x00CALL\x00", "\x00RESULT\x00"
    turn = [{"role": "user", "content": "question"}, {"role": "assistant", "content": call}]
    closed = tokenizer.apply_chat_template(turn, tokenize=False)
    answered = tokenizer.apply_chat_template(
        [*turn, {"role": "user", "content": result}], tokenize=False, add_generation_prompt=True
    )
    before, after = answered[len(closed) :].split(result)
    return closed[closed.index(call) + len(call) :] + before, after


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    state = GenerateState(args)
    if not sample.tokens:
        sample.tokens = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    before_result, after_result = _result_message(state.tokenizer)
    budget = sampling_params["max_new_tokens"]  # the whole trajectory's response budget
    for calls in range(MAX_TOOL_CALLS + 1):
        remaining = budget - sample.response_length
        turn_params = {**sampling_params, "max_new_tokens": min(TURN_MAX_NEW_TOKENS, remaining), "stop": [CALL_END]}
        output = await generate_turn(args, sample, turn_params)
        call = CALL.search(output["text"])
        if output["meta_info"]["finish_reason"]["type"] != "stop" or call is None:
            # The final answer, a truncated turn or a cancelled rollout; generate_turn set the status.
            return sample
        if TOOL_LATENCY:
            await asyncio.sleep(TOOL_LATENCY)
        message = before_result + f"<result>{calculate(call.group(1))}</result>" + after_result
        tokens = state.tokenizer.encode(message, add_special_tokens=False)
        if calls == MAX_TOOL_CALLS or len(tokens) >= budget - sample.response_length:
            # No model turn would follow the result. End on the model's own turn instead: result tokens
            # get their routed experts (--use-rollout-routing-replay) from the prefill of the next turn.
            sample.status = Sample.Status.TRUNCATED
            return sample
        sample.append_response_tokens(args, tokens=tokens, trainable=False, text=message)
    return sample
