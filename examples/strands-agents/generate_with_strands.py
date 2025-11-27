import logging

from camel.toolkits.code_execution import CodeExecutionToolkit
import openai
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
)
import wandb

from slime.rollout.rm_hub.math_dapo_utils import (
    compute_score as math_dapo_compute_score,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a helpful math-solving assistant with access to the `execute_python_code` tool.

Guidelines:
- For any numerical or symbolic computation, always use the `execute_python_code` tool rather than performing calculations mentally.
- Break problems into clear steps, calling the Python tool whenever computation is required.
- After completing your reasoning, present the final result enclosed in \\boxed{}.
""".strip()


def create_strands_agent(args, sampling_params):
    # Create an OpenAI model from the SGLang server
    model_params = {
        "max_tokens": sampling_params["max_new_tokens"],
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
    }
    sglang_server_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"
    logger.info(
        f"[Strands Agents] Creating OpenAIModel from SGLang server at {sglang_server_url}"
        f" with parameters: {model_params}"
    )
    model = OpenAIModel(
        client_args={
            "api_key": "EMPTY",
            "base_url": sglang_server_url,
            "timeout": 300.0,  # tool parser requires some time buffer
        },
        model_id=args.hf_checkpoint.split("/")[-1],
        params=model_params,
    )

    @tool
    def execute_python_code(code: str) -> str:
        r"""Execute a given Python code snippet.

        Args:
            code (str): The input Python code to the Code Execution tool call.

        Returns:
            str: The text output from the Code Execution tool call.
        """
        code_execution_toolkit = CodeExecutionToolkit(
            sandbox="subprocess",
            timeout=60.0,  # 1 minute
        )
        return code_execution_toolkit.execute_code(code=code, code_type="python")

    agent = Agent(
        model=model,
        tools=[execute_python_code],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    return agent


async def run_strands_agent(agent: Agent, prompt: str) -> Sample.Status:
    """Run the strands agent with the given prompt and set the sample status."""
    try:
        logger.info(f"[Strands Agents] running agent with prompt: {prompt}")
        await agent.invoke_async(prompt=prompt)
        sample_status = Sample.Status.COMPLETED
    except Exception as e:
        truncated_conditions = [
            isinstance(e, MaxTokensReachedException),
            isinstance(e, ContextWindowOverflowException),
            isinstance(e, EventLoopException)
            and isinstance(e.original_exception, openai.APIError)
            and "context length" in str(e.original_exception).lower(),
        ]

        if any(truncated_conditions):
            sample_status = Sample.Status.TRUNCATED
        else:
            sample_status = Sample.Status.ABORTED
        logger.error(f"[Strands Agents] inference not completed due to exception: {e}")

    return sample_status


def get_trajectory(agent: Agent) -> list[dict]:
    """Get the chat template-compatible trajectory of the strands agent."""
    openai_model: OpenAIModel = agent.model
    trajectory = openai_model.format_request_messages(messages=agent.messages, system_prompt=agent.system_prompt)
    # Convert content from list[dict] format to string format for chat template
    # The strands library returns content as [{"type": "text", "text": "..."}]
    # but the tokenizer's chat template expects just the string
    for message in trajectory:
        if "content" in message and isinstance(message["content"], list):
            if len(message["content"]) > 0 and "text" in message["content"][0]:
                message["content"] = message["content"][0]["text"]
            else:
                message["content"] = ""

    return trajectory


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate function using strands-agents as agent scaffolding"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    # Create strands agent and run it with the sample prompt
    agent = create_strands_agent(args, sampling_params)
    prompt_text = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]

    # Run the strands agent
    sample.status = await run_strands_agent(agent, prompt_text)

    if sample.status == Sample.Status.ABORTED:
        return sample

    # Get the trajectory from the agent
    trajectory = get_trajectory(agent)
    # Check if the trajectory is None, this is an infra bug, not a “recoverable” case
    if trajectory is None:
        raise RuntimeError(
            "[Strands Agents] trajectory is `None` after `await agent.invoke_async(prompt=prompt)`; "
            "this run should not be used for training."
        )

    # Get the initial prompt (system + user message)
    initial_prompt_messages = [msg for msg in trajectory if msg["role"] in ["system", "user"]]
    assert len(initial_prompt_messages) == 2, "Initial prompt messages must be exactly 2 for single-turn conversations"
    prompt_text = state.tokenizer.apply_chat_template(
        initial_prompt_messages,
        tokenize=False,
        add_generation_prompt=True,  # Add generation prompt for the assistant
    )
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Build (re-tokenize) the response incrementally
    response_token_ids = []
    loss_masks = []
    response_text = ""

    # Start with the initial prompt messages for progressive chat template application
    current_messages = list(initial_prompt_messages)
    prev_token_count = len(prompt_tokens_ids)

    # Iterate through remaining messages (assistant and tool messages)
    for message in trajectory[len(initial_prompt_messages) :]:
        # Add this message to the conversation
        current_messages.append(message)

        # Apply chat template and tokenize up to this point
        current_text = state.tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=False
        )
        current_token_ids = state.tokenizer(current_text, add_special_tokens=False)["input_ids"]

        # Calculate how many new tokens this message added
        new_token_count = len(current_token_ids)
        message_token_length = new_token_count - prev_token_count

        # Extract the new tokens for this message
        message_tokens = current_token_ids[prev_token_count:]
        assert len(message_tokens) == message_token_length, "Message tokens length mismatch"
        response_token_ids.extend(message_tokens)

        # Mask: 1 for assistant messages (we train on these), 0 for tool results
        if message["role"] == "assistant":
            loss_masks.extend([1] * message_token_length)
        else:  # tool messages
            loss_masks.extend([0] * message_token_length)

        prev_token_count = new_token_count

    # Extract the response text (everything after the initial prompt)
    full_conversation_text = state.tokenizer.apply_chat_template(
        trajectory, tokenize=False, add_generation_prompt=False
    )
    response_text = full_conversation_text[len(prompt_text) :]

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = loss_masks

    # Store information for wandb logging
    sample.payload_text = prompt_text + response_text
    sample.payload_has_system = True
    sample.payload_has_tools = len(agent.tool_names) > 0

    # Store tool call count for reward calculation
    sample.tool_call_count = [message["role"] == "tool" for message in trajectory].count(True)

    # Log to wandb if available
    if wandb.run is not None:
        wandb.log(
            {
                "debug/response_length": len(response_text),
                "debug/available_tools": len(agent.tool_names),
                "debug/tool_calls": sample.tool_call_count,
                "debug/num_messages": len(trajectory),
                "debug/truncated": sample.status == Sample.Status.TRUNCATED,
            }
        )

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.payload_text

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # Accept both Answer: ... and \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=False)
    result_boxed = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    if result["pred"] == "[INVALID]":
        result = result_boxed

    # Encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    logger.info(
        f"[Strands Agents] sample summary: "
        f"status={sample.status} | "
        f"tool_call_count={sample.tool_call_count} | "
        f"response_length={sample.response_length} | "
        f"reward={result} | "
        f"ground_truth={ground_truth}"
    )

    return result
