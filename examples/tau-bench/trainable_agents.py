from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import logging

import weave
from transformers import AutoTokenizer
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import RunConfig, Action
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from openai_tool_adapter import create_openai_adapter

# Set up logger for this module
logger = logging.getLogger(__name__)

class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"

@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    response: str = ""
    loss_mask: Optional[List[int]] = None
    tokens: Optional[int] = None
    status: Status = Status.COMPLETED


@weave.op()
def call_to_action_sglang(calls: List[Any], text_response: str
) -> Action:
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
            action = Action(
                name=tool_call["name"],
                kwargs=params)
    return action

TOOL_INSTRUCTION = """ At each turn, you are allowed to call one or no function to assist with task execution using <tools></tools> XML tags.
YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. Each tool call leads to a message returned by the system. 
NEVER confirm execution to the user without seeing confirmation from the tool system. 
"""

class TrainableAgentMixin:

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Default tool template assumes one or more function call. However for tau, at most one tool call 
        or skip tool calls are the valid options. 
        """
        return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)
        
    @weave.op()
    async def _call_llm(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an LLM call with Weave tracking.
        """
        return await post(url, payload)

    @weave.op()
    def _parse_tool(self, response: str):
        """
        Parse action from llm response string
        """
        return self.openai_adapter.parse_response_to_openai_format(response)

    @weave.op()
    async def _execute_tool(self, env, action: Action):
        """
        Execute a tool/action with Weave tracking.
        """
        return env.step(action)

    @weave.op()
    async def asolve(
        self,
        env,
        rollout_args: Dict[str, Any],
        sampling_params: Dict[str, Any],
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> InteractionResult:
        """
        Extend original Agent to support async call interaction with LLM server.

        Given a URL and sampling parameters, this method should asynchronously
        send a request to the LLM server and return the sampling result.

        It book keeps the loss mask and records metadata for the request.
        """
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"

        # Reset environment to the specified task
        if task_index is not None:
            env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        
        # Build initial messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs}
        ]
        
        # Calculate initial prompt tokens (loss_mask = 0)
        prompt_text = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info)
        # Hack: the default template forces 1 or more tool uses, but we want 0 or 1.
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        loss_masks = []
        
        # Initialize response tracking
        response_token_ids = []
        total_reward = 0.0
        
        # Initialize result
        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})
        
        def _get_token_delta(tokenizer: AutoTokenizer, messages: List[Dict]) -> Tuple[List[int], List[int]]:
            # tokenization logic taken from here: https://verl.readthedocs.io/en/v0.4.1/sglang_multiturn/multiturn.html 
            # to calculate the right token count in a multi-turn environment, use the delta between the last messages
            
            curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            token_ids = []
            loss_mask = []
            # Case 1; last message is an assistant response. 
            if messages[-1]["role"] == "assistant":
                prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
                token_ids += tokenizer.encode(curr[len(prev):], add_special_tokens=False)
                loss_mask += [1] * len(tokenizer.encode(curr[len(prev):], add_special_tokens=False))  # Mask only the new assistant tokens
            else:
                # Case 2: last message is a tool response or environment observation. 
                prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)
                token_ids += tokenizer.encode(curr[len(prev):], add_special_tokens=False)
                loss_mask += [0] * len(tokenizer.encode(curr[len(prev):], add_special_tokens=False))  # Mask
            return token_ids, loss_mask

        def _build_result(res):
            res.reward = total_reward
            res.info = info
            res.messages = messages
            
            res.loss_mask = loss_masks
            
            res.tokens = prompt_token_ids + response_token_ids
            res.response = "".join([msg.get("content", "") for msg in messages if msg["role"] == "assistant"])
            
            # response_length should equal the response loss mask length
            res.response_length = len(loss_masks)
            
            logger.debug(f"_build_result: response_length={res.response_length}, "
                        f"response_loss_mask_len={len(loss_masks)}, "
                        f"prompt_token_len={len(prompt_token_ids)}, "
                        f"response_token_len={len(response_token_ids)}, "
                        f"response='{res.response[:100]}...'")
            return res

        # Multi-turn interaction loop
        with weave.thread():
            for _ in range(max_num_steps):
                # Prepare payload for sglang
                text_input = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info)
                # Hack: the default template forces 1 or more tool uses, but we want 0 or 1.
                text_input = self._reformulate_tool_call(text_input)
                payload = {
                    "text": text_input,
                    "sampling_params": sampling_params}
                # Send request to sglang server with tool call settings
                output = await self._call_llm(url, payload)
            
                # Check for abort
                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    res.status = Status.ABORTED
                    return _build_result(res)
                response = output["text"]
                # Hack: Not sure why but for qwen response will have end of conversation token at the end. 
                if response.endswith("<|im_end|>"):
                    response=response[:-10]
                
                # Use OpenAI adapter to parse tool calls
                logger.debug(f"Using OpenAI adapter to parse response: {response[:100]}...")
                try:
                    # Get OpenAI format result
                    openai_result = self._parse_tool(response)
                    logger.debug(f"OpenAI adapter result: success={openai_result['success']}")
                    
                    if not openai_result["success"]:
                        logger.warning(f"OpenAI adapter failed: {openai_result['error']}")
                        logger.warning(f"rollout response: {response} can not be parsed into tool calls {openai_result['error']}")
                        res.status = Status.ABORTED
                        return _build_result(res)
                    
                    # Extract parsed results
                    parsed = openai_result["parsed_result"]
                    openai_message = openai_result["openai_message"]
                    logger.debug(f"Successfully parsed - normal_text: '{parsed['normal_text']}', calls: {parsed['calls']}")
                    logger.debug(f"OpenAI message: {openai_message}")
                    
                except Exception as e:
                    logger.warning(f"Exception in OpenAI adapter: {e}")
                    logger.warning(f"rollout response: {response} can not be parsed into tool calls {e}")
                    res.status = Status.ABORTED
                    return _build_result(res)

                # Raw assistant response is the learnable target for the model. 
                messages.append({"role": "assistant", "content": response})
                assistant_token_ids, assistant_loss_mask = _get_token_delta(state.tokenizer, messages)
                response_token_ids.extend(assistant_token_ids)
                loss_masks.extend(assistant_loss_mask)  

                # Step in environment with the tool call.
                agent_content, calls = parsed['normal_text'], parsed["calls"]
                logger.debug(f"Creating action from - content: '{agent_content}', calls: {calls}")
                action = call_to_action_sglang(calls, agent_content)
                logger.debug(f"Created action: {action}")
                logger.debug(f"Stepping environment with action: {action.name}")
                try:
                    env_response = await self._execute_tool(env, action)
                except Exception as e:
                    logger.warning(f"Envrionment step failed, this is usually related to the User simulation call.")
                    logger.warning(f"Error: {e}")
                    res.status = Status.ABORTED
                    return _build_result(res)
                logger.debug(f"Environment response: reward={env_response.reward}, done={env_response.done}")
                # Update message history
                if action.name != RESPOND_ACTION_NAME:
                    messages.append(
                        {
                            "role": "tool",
                            "name": action.name,
                            "content": env_response.observation,
                        }
                    )
                else:
                    # Direct response from user, similar to ToolCallingAgent logic. 
                    messages.append(
                        {"role": "user", "content": env_response.observation},
                    )

                env_token_ids, env_loss_mask = _get_token_delta(state.tokenizer, messages)
                response_token_ids.extend(env_token_ids)
                loss_masks.extend(env_loss_mask)  

                # Update reward and info
                total_reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                
                # Check if done
                if env_response.done:
                    res.status = Status.COMPLETED
                    break
        
        # Handle truncation
        if not env_response.done:
            res.status = Status.TRUNCATED
            
        return _build_result(res)
    
    def get_openai_tools_format(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI format tool definitions
        
        Returns:
            List of OpenAI format tools
        """
        logger.debug(f"Getting OpenAI tools format for {len(self.tools_info)} tools")
        tools = self.openai_adapter.get_openai_tools_format()
        logger.debug(f"OpenAI tools format: {tools}")
        return tools
    
    def get_openai_message_from_response(self, response: str) -> Dict[str, Any]:
        """
        Get OpenAI format message from response
        
        Args:
            response: Raw response text from sglang
            
        Returns:
            OpenAI format message result
        """
        logger.debug(f"Getting OpenAI message from response: {response[:100]}...")
        result = self.openai_adapter.parse_response_to_openai_format(response)
        logger.debug(f"OpenAI message result: {result}")
        return result


class TrainableToolCallingAgent(ToolCallingAgent, TrainableAgentMixin):
    """
    A trainable version of ToolCallingAgent that uses sglang rollout for training.
    
    This agent combines the original ToolCallingAgent functionality with the
    TrainableAgentMixin to support async interaction with sglang servers for
    reinforcement learning training.
    """
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        rollout_args: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
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
        self.openai_adapter = create_openai_adapter(
            tools_info=self.tools_info,
            parser_type="qwen25"
        )
    
    @weave.op()
    async def solve(
        self, env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> "SolveResult":
        """
        Async solve method that uses our new asolve implementation.
        This provides a simple interface for evaluation or standalone usage.
        """
        # Call our async asolve method using instance variables
        interaction_result = await self.asolve(env, self.rollout_args, self.sampling_params, task_index, max_num_steps)
        
        # Convert InteractionResult back to SolveResult format
        from tau_bench.types import SolveResult
        return SolveResult(
            reward=interaction_result.reward,
            info=interaction_result.info,
            messages=interaction_result.messages,
            total_cost=0.0,  # We don't track cost in the new implementation
        )


def agent_factory(
    tools_info: List[Dict[str, Any]], 
    wiki, 
    config: RunConfig,
    rollout_args: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[Dict[str, Any]] = None
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
