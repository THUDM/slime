from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import json
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import RunConfig, Action
from sglang_tool_parser import parse_tools
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

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


def call_to_action_sglang(calls: List[Any], text_response: str
) -> Action:
    """
    Convert sglang response message to Action, similar to original message_to_action
    but adapted for sglang response format.
    """
    # Default action if no action was found. 
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
    if calls:
        if len(calls)>1:
            print("Multiple tool calls identified, only taking first.")
        tool_call = calls[0]
        params = json.loads(tool_call["parameters"])
        if not isinstance(params, dict):
            print(f"{params} does not follow dict structure for action")
        else:
            action = Action(
                name=tool_call["name"],
                kwargs=params)
    return action 

class TrainableAgentMixin:
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
        tool_url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/parse_function_call"

        # Reset environment to the specified task
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        
        # Build initial messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs}
        ]
        
        # Calculate initial prompt tokens (loss_mask = 0)
        prompt_text = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.tools_info)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        loss_masks = [0] * len(prompt_token_ids)  # prompt tokens don't contribute to loss
        
        # Initialize response tracking
        response_token_ids = []
        total_reward = 0.0
        
        # Initialize result
        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})
        
        # Multi-turn interaction loop
        for _ in range(max_num_steps):
            # Prepare payload for sglang
            text_input = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.tools_info)
            payload = {
                "text": text_input,
                "sampling_params": sampling_params}
            # Send request to sglang server
            output = await post(url, payload)
        
            # Check for abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return res
            # Extract tool call 
            response = output["text"]
            try:
                parsed = parse_tools(response, self.tools_info, "qwen25")
            except Exception as e:
                print(f"rollout response: {response} can not be parsed into tool calls {e}")
                res.status = Status.ABORTED
                return res


            agent_content = parsed['normal_text']
            if agent_content:
                # TODO: Verify the loss masks are correct. Aren't we skipping the role tokens?
                cur_token_ids = state.tokenizer(agent_content, add_special_tokens=False)["input_ids"]
                response_token_ids.extend(cur_token_ids)
                loss_masks.extend([1] * len(cur_token_ids))  # agent response contributes to loss
            next_message = {"role": "assistant", "content": agent_content}
            calls = parsed["calls"]
            # Step in environment
            action = call_to_action_sglang(calls, agent_content)
            env_response = env.step(action)

            # Calculate environment observation tokens (loss_mask = 0)
            obs_token_ids = state.tokenizer(env_response.observation, add_special_tokens=False)["input_ids"]
            response_token_ids.extend(obs_token_ids)
            loss_masks.extend([0] * len(obs_token_ids))  # environment observation doesn't contribute to loss

            # Update reward and info
            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Update message history
            if action.name != RESPOND_ACTION_NAME:
                messages.extend([
                    next_message,
                    {
                        "role": "tool",
                        "name": action.name,
                        "content": env_response.observation,
                    }
                ])
            else:
                # Direct response
                messages.extend([
                    next_message,
                    {"role": "user", "content": env_response.observation},
                ])
            
            # Check if done
            if env_response.done:
                res.status = Status.COMPLETED
                break
        
        # Handle truncation
        if not env_response.done:
            res.status = Status.TRUNCATED
            
        # Build final result
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join([msg.get("content", "") for msg in messages if msg["role"] == "assistant"])
        
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
