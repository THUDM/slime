import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import Action, RunConfig
from token_delta import (
    assistant_message_text,
    decode_token_ids,
    encode_token_ids,
    sampled_output_tokens,
    tokens_after_assistant,
)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.score_centering import score_centering_request

# Set up logger for this module
logger = logging.getLogger(__name__)


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class ResponseSegment:
    """One contiguous piece of the response.

    Trainable segments are SGLang's sampled token ids. Non-trainable segments
    are template tokens for the environment observation and the next generation
    prompt. Those were not sampled, so they stay out of the loss.
    """

    token_ids: list[int]
    trainable: bool
    log_probs: list[float] | None = None
    meta_info: dict[str, Any] | None = None


@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    segments: list[ResponseSegment] = field(default_factory=list)
    response_length: int = 0
    status: Status = Status.COMPLETED


def call_to_action_sglang(calls: list[Any], text_response: str) -> Action:
    """
    Convert sglang response message to Action, similar to original message_to_action
    but adapted for sglang response format.
    """
    # Default action if no action was found.
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
    if calls:
        if len(calls) > 1:
            logger.debug("Multiple tool calls identified, only taking first.")
        tool_call = calls[0]
        params = json.loads(tool_call["parameters"])
        if not isinstance(params, dict):
            logger.warning(f"{params} does not follow dict structure for action")
        else:
            action = Action(name=tool_call["name"], kwargs=params)
    return action


TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


class TrainableAgentMixin:
    """
    Mixin class that provides trainable agent functionality for tau-bench environments.

    This mixin extends the original tau-bench agent with async LLM interaction
    capabilities for reinforcement learning training using sglang servers.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Reformulate tool call instruction for tau-bench environment.

        The default tool template assumes one or more function calls, but for
        tau-bench, at most one tool call or skip tool calls are the valid options.

        Args:
            text: Original tool instruction text

        Returns:
            Reformulated tool instruction text
        """
        return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make an LLM call tracking.

        Args:
            url: SGLang server URL
            payload: Request payload containing text and sampling parameters

        Returns:
            LLM response from sglang server
        """
        return await post(url, payload)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        """
        Parse tool calls from LLM response string.

        Args:
            response: Raw response text from sglang

        Returns:
            Parsed tool call result in OpenAI format
        """
        return self.openai_adapter.parse_response_to_openai_format(response)

    async def _execute_tool(self, env, action: Action):
        """
        Execute a tool/action in the environment.

        Args:
            env: Tau-bench environment instance
            action: Action to execute

        Returns:
            Environment step result
        """
        return env.step(action)

    def _initialize_environment(self, env, task_index: int | None) -> tuple[str, dict[str, Any]]:
        """
        Initialize the environment and get initial observation.

        Args:
            env: Tau-bench environment instance
            task_index: Task index to reset to

        Returns:
            Tuple of (observation, info)
        """
        if task_index is not None:
            env_reset_res = env.reset(task_index=task_index)
        else:
            env_reset_res = env.reset()
        return env_reset_res.observation, env_reset_res.info.model_dump()

    def _build_initial_messages(self, obs: str) -> list[dict[str, Any]]:
        """
        Build initial conversation messages.

        Args:
            obs: Initial observation from environment

        Returns:
            List of initial messages
        """
        return [{"role": "system", "content": self.wiki}, {"role": "user", "content": obs}]

    def _prepare_prompt_tokens(self, state: GenerateState, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        """
        Prepare prompt text and tokenize it.

        Args:
            state: GenerateState instance with tokenizer
            messages: Conversation messages

        Returns:
            Tuple of (prompt_text, prompt_token_ids)
        """
        prompt_text = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info
        )
        # Reformulate tool call instruction for tau-bench. This string is
        # tokenized once and reused as input_ids; later turns append to it.
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = encode_token_ids(state.tokenizer, prompt_text)
        return prompt_text, prompt_token_ids

    async def asolve(
        self,
        env,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        task_index: int | None = None,
        max_num_steps: int = 30,
    ) -> InteractionResult:
        """
        Execute async agent-environment interaction for training.

        This method extends the original Agent to support async interaction with LLM
        server for reinforcement learning training. Each turn sends the exact token
        prefix accumulated so far and appends SGLang's sampled ids. Environment
        text is the only thing tokenized after the initial prompt.

        Args:
            env: Tau-bench environment instance
            rollout_args: Rollout configuration arguments
            sampling_params: LLM sampling parameters
            task_index: Specific task index to solve (optional)
            max_num_steps: Maximum number of interaction steps

        Returns:
            InteractionResult containing the complete interaction trajectory
        """
        # Initialize environment and state
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:" f"{rollout_args.sglang_router_port}/generate"

        # Get initial environment state
        obs, info = self._initialize_environment(env, task_index)

        # Build initial conversation
        messages = self._build_initial_messages(obs)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages)

        segments: list[ResponseSegment] = []
        total_reward = 0.0
        res = InteractionResult(
            prompt=prompt_text,
            reward=0,
            messages=[],
            info={},
            prompt_token_ids=list(prompt_token_ids),
            segments=segments,
        )

        def response_token_ids() -> list[int]:
            return [token for segment in segments for token in segment.token_ids]

        def commit(
            token_ids: list[int],
            *,
            trainable: bool,
            log_probs: list[float] | None = None,
            meta_info: dict[str, Any] | None = None,
        ) -> None:
            if not token_ids:
                return
            if trainable and (log_probs is None or len(log_probs) != len(token_ids)):
                raise ValueError("trainable assistant tokens require one logprob per sampled token")
            if not trainable and log_probs is not None:
                raise ValueError("environment tokens must not carry rollout log probabilities")
            segments.append(
                ResponseSegment(
                    token_ids=list(token_ids),
                    trainable=trainable,
                    log_probs=list(log_probs) if log_probs is not None else None,
                    meta_info=meta_info,
                )
            )

        def finish(status: Status) -> InteractionResult:
            res.status = status
            return self._build_final_result(res, total_reward, info, messages, state.tokenizer)

        for _ in range(max_num_steps):
            # The next request is the exact training prefix. Do not render the
            # conversation again: that retokenizes the sampled assistant turn.
            turn_sampling = dict(sampling_params)
            custom_params = turn_sampling.get("custom_params")
            if isinstance(custom_params, dict):
                turn_sampling["custom_params"] = dict(custom_params)
            payload = {
                "input_ids": [*prompt_token_ids, *response_token_ids()],
                "sampling_params": turn_sampling,
                "return_logprob": True,
            }
            payload.update(score_centering_request(rollout_args, turn_sampling))

            output = await self._call_llm(url, payload)
            meta = output.get("meta_info") or {}
            finish_type = (meta.get("finish_reason") or {}).get("type")
            if finish_type == "abort":
                return finish(Status.ABORTED)

            try:
                output_ids, output_log_probs = sampled_output_tokens(output)
            except ValueError as exc:
                logger.warning(f"token-exact generation failed: {exc}")
                return finish(Status.ABORTED)

            if finish_type == "length":
                commit(output_ids, trainable=True, log_probs=output_log_probs, meta_info=meta)
                return finish(Status.TRUNCATED)
            if finish_type != "stop":
                logger.warning(f"unexpected finish reason {finish_type}; refusing to retokenize the response")
                return finish(Status.ABORTED)

            response = assistant_message_text(state.tokenizer, output_ids, tools=self.tools_info)
            logger.debug(f"Using OpenAI adapter to parse response: {response[:100]}...")
            try:
                openai_result = self._parse_tool(response)
                logger.debug(f"OpenAI adapter result: success={openai_result['success']}")
                if not openai_result["success"]:
                    logger.warning(f"OpenAI adapter failed: {openai_result['error']}")
                    logger.warning(f"rollout response: {response} can not be parsed into tool calls {openai_result['error']}")
                    return finish(Status.ABORTED)
                parsed = openai_result["parsed_result"]
                logger.debug(f"Successfully parsed - normal_text: '{parsed['normal_text']}', calls: {parsed['calls']}")
            except Exception as exc:
                logger.warning(f"Exception in OpenAI adapter: {exc}")
                logger.warning(f"rollout response: {response} can not be parsed into tool calls {exc}")
                return finish(Status.ABORTED)

            commit(output_ids, trainable=True, log_probs=output_log_probs, meta_info=meta)
            messages.append({"role": "assistant", "content": response})

            agent_content, calls = parsed["normal_text"], parsed["calls"]
            logger.debug(f"Creating action from - content: '{agent_content}', calls: {calls}")
            action = call_to_action_sglang(calls, agent_content)
            logger.debug(f"Created action: {action}")

            try:
                env_response = await self._execute_tool(env, action)
            except Exception as exc:
                logger.warning("Environment step failed, this is usually related to the User simulation call.")
                logger.warning(f"Error: {exc}")
                return finish(Status.ABORTED)

            logger.debug(f"Environment response: reward={env_response.reward}, done={env_response.done}")
            if action.name != RESPOND_ACTION_NAME:
                observation = {"role": "tool", "name": action.name, "content": env_response.observation}
            else:
                observation = {"role": "user", "content": env_response.observation}
            messages.append(observation)
            commit(
                tokens_after_assistant(
                    state.tokenizer,
                    output_ids,
                    observation,
                    tools=self.tools_info,
                    add_generation_prompt=not env_response.done,
                ),
                trainable=False,
            )

            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if env_response.done:
                return finish(Status.COMPLETED)

        return finish(Status.TRUNCATED)

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        tokenizer: Any,
    ) -> InteractionResult:
        """Populate the trajectory from sampled segments and masked observations."""
        response_token_ids: list[int] = []
        loss_mask: list[int] = []
        for segment in res.segments:
            response_token_ids.extend(segment.token_ids)
            loss_mask.extend([1 if segment.trainable else 0] * len(segment.token_ids))

        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_mask
        res.tokens = list(res.prompt_token_ids or []) + response_token_ids
        res.response = decode_token_ids(tokenizer, response_token_ids)
        res.response_length = len(loss_mask)

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"response_loss_mask_len={len(loss_mask)}, "
            f"prompt_token_len={len(res.prompt_token_ids or [])}, "
            f"response_token_len={len(response_token_ids)}, "
            f"response='{res.response[:100]}...'"
        )
        return res


class TrainableToolCallingAgent(ToolCallingAgent, TrainableAgentMixin):
    """
    A trainable version of ToolCallingAgent that uses sglang rollout for training.

    This agent combines the original ToolCallingAgent functionality with the
    TrainableAgentMixin to support async interaction with sglang servers for
    reinforcement learning training.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        rollout_args: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        # Initialize the parent ToolCallingAgent
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )

        # Store rollout and sampling parameters as instance variables
        self.rollout_args = rollout_args or {
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "use_http2": False,
        }
        self.sampling_params = sampling_params or {
            "temperature": self.temperature,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
        }
        # Initialize OpenAI adapter
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type="qwen25")


def agent_factory(
    tools_info: list[dict[str, Any]],
    wiki,
    config: RunConfig,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> Agent:
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    else:
        raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
