import logging

from camel.interpreters import SubprocessInterpreter
from strands import Agent, tool
from strands_sglang import SGLangClient, SGLangModel, ToolLimiter
from strands_sglang.tool_parsers import HermesToolParser

from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful math-solving assistant with access to the `execute_python_code` tool.

Guidelines:
- For any numerical or symbolic computation, always use the `execute_python_code` tool rather than performing calculations mentally.
- Break problems into clear steps, calling the Python tool whenever computation is required.
- After completing your reasoning, present the final result enclosed in \\boxed{}.
""".strip()

MAX_TOOL_ITERS = 5
MAX_TOOL_CALLS = None  # No limit

_client_cache: dict[str, SGLangClient] = {}


def get_client(args) -> SGLangClient:
    """Get shared client for connection pooling."""
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if base_url not in _client_cache:
        _client_cache[base_url] = SGLangClient.from_slime_args(args, timeout=300.0)
    return _client_cache[base_url]


@tool
def execute_python_code(code: str) -> str:
    """Execute Python code and return the output."""
    interpreter = SubprocessInterpreter(
        require_confirm=False,
        print_stdout=False,
        print_stderr=False,
        execution_timeout=60.0,
    )
    result = interpreter.run(code, "python")
    logger.info(f"Executing Python code: ```python\n{code}\n``` and get execution result: ```python\n{result}\n```")
    return result


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate with TITO: tokens captured during generation, no retokenization."""
    assert not args.partial_rollout, "Partial rollout not supported."

    state = GenerateState(args)
    model = SGLangModel(
        tokenizer=state.tokenizer,
        client=get_client(args),
        tool_parser=HermesToolParser(),  # tool parsing for wrapped JSON tool calls
        sampling_params=sampling_params,
    )

    tool_limiter = ToolLimiter(max_tool_iters=MAX_TOOL_ITERS, max_tool_calls=MAX_TOOL_CALLS)
    agent = Agent(
        model=model,
        tools=[execute_python_code],
        hooks=[tool_limiter],
        callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]

    try:
        await agent.invoke_async(prompt)
        sample.status = Sample.Status.COMPLETED
    except Exception as e:
        # Always use TRUNCATED instead of ABORTED because slime doesn't properly
        # handle ABORTED samples in reward processing. See: https://github.com/THUDM/slime/issues/200
        sample.status = Sample.Status.TRUNCATED
        logger.warning(f"TRUNCATED: {type(e).__name__}: {e}")

    # Extract token trajectory from token_manager
    tm = model.token_manager
    prompt_len = len(tm.segments[0])  # system + user are first segment
    sample.tokens = tm.token_ids
    sample.loss_mask = tm.loss_mask[prompt_len:]
    sample.rollout_log_probs = tm.logprobs[prompt_len:]
    sample.response_length = len(sample.tokens) - prompt_len
    sample.response = model.tokenizer.decode(sample.tokens[prompt_len:], skip_special_tokens=False)
    # Tool iteration and tool call count are different because multiple parallel tool calls count as 1 iteration
    sample.tool_iters = tool_limiter.tool_iter_count
    sample.tool_calls = tool_limiter.tool_call_count

    model.reset()
    agent.cleanup()
    return sample


async def reward_func(args, sample: Sample, **kwargs):
    """Reward function using math_dapo scoring."""
    ground_truth = sample.label or ""
    tool_iters = getattr(sample, "tool_iters", 0)
    tool_calls = getattr(sample, "tool_calls", 0)

    result = math_dapo_compute_score(sample.response, ground_truth, strict_box_verify=False)
    if result["pred"] == "[INVALID]":
        result = math_dapo_compute_score(sample.response, ground_truth, strict_box_verify=True)

    # Encourage tool use on failures
    if result["score"] < 0:
        result["score"] = min(-0.6, result["score"] + (tool_iters - 2) / 2 * 0.1)

    result["pred"] = result["pred"] or ""
    logger.info(
        f"reward={result['score']:.2f} | status={sample.status.name} | tool_iters={tool_iters} | tool_calls={tool_calls} | tokens={len(sample.tokens)} | resp_len={sample.response_length} | "
    )
    return result["score"]
