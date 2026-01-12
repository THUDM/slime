import logging

from camel.interpreters import SubprocessInterpreter
from strands import Agent, tool
from strands_sglang import SGLangClient, SGLangModel
from strands_sglang.tool_limiter import ToolIterationLimiter

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

MAX_TOOL_ITERATIONS = 5

_client_cache: dict[str, SGLangClient] = {}


def get_client(args) -> SGLangClient:
    """Get shared client for connection pooling (like SLIME)."""
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if base_url not in _client_cache:
        _client_cache[base_url] = SGLangClient.from_slime_args(args)
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
        model_id=args.hf_checkpoint.split("/")[-1],
        params={k: sampling_params[k] for k in ["max_new_tokens", "temperature", "top_p"]},
    )

    limiter = ToolIterationLimiter(max_iterations=MAX_TOOL_ITERATIONS)
    agent = Agent(
        model=model,
        tools=[execute_python_code],
        hooks=[limiter],
        callback_handler=None,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]

    try:
        await agent.invoke_async(prompt)
        sample.status = Sample.Status.COMPLETED
    except Exception as e:
        # Always use TRUNCATED instead of ABORTED because Slime doesn't properly
        # handle ABORTED samples in reward processing. See: https://github.com/THUDM/slime/issues/200
        sample.status = Sample.Status.TRUNCATED
        logger.warning(f"TRUNCATED: {type(e).__name__}: {e}")

    # TITO: extract trajectory from token_manager
    tm = model.token_manager
    prompt_len = len(tm.segments[0])  # system + user are first segment
    sample.tokens = tm.token_ids
    sample.loss_mask = tm.loss_mask[prompt_len:]
    sample.rollout_log_probs = tm.logprobs[prompt_len:]
    sample.response_length = len(sample.tokens) - prompt_len
    sample.response = model.tokenizer.decode(sample.tokens[prompt_len:], skip_special_tokens=False)
    # Tool iteration and tool call count are different because multiple parallel tool calls count as 1 iteration
    sample.tool_iterations = limiter.iteration_count
    trajectory = model.format_request_messages(agent.messages, None)
    sample.tool_call_count = [message["role"] == "tool" for message in trajectory].count(True)

    model.reset()
    agent.cleanup()
    return sample


async def reward_func(args, sample: Sample, **kwargs):
    """Reward function using math_dapo scoring."""
    ground_truth = sample.label or ""
    tool_iterations = getattr(sample, "tool_iterations", 0)

    result = math_dapo_compute_score(sample.response, ground_truth, strict_box_verify=False)
    if result["pred"] == "[INVALID]":
        result = math_dapo_compute_score(sample.response, ground_truth, strict_box_verify=True)

    # Encourage tool use on failures
    if result["score"] < 0:
        result["score"] = min(-0.6, result["score"] + (tool_iterations - 2) / 2 * 0.1)

    result["pred"] = result["pred"] or ""
    logger.info(
        f"reward={result['score']:.2f} | status={sample.status.name} | tool_iters={tool_iterations} | tool_calls={getattr(sample, 'tool_call_count', 0)} | tokens={len(sample.tokens)} | resp_len={sample.response_length} | "
    )
    return result["score"]
