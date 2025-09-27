from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json

from transformers import AutoTokenizer
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import RunConfig, Action
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from openai_tool_adapter import create_openai_adapter

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
        if len(calls) > 1:
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
        prompt_text = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        loss_masks = [0] * len(prompt_token_ids)  # prompt tokens don't contribute to loss
        
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
            
            # Extract only the response part of loss_mask to align with Search-R1
            # loss_masks contains: [prompt_tokens (mask=0)] + [response_tokens (mask=1)]
            # We only want the response part for training, like Search-R1
            response_loss_mask = loss_masks[len(prompt_token_ids):]
            res.loss_mask = response_loss_mask
            
            res.tokens = prompt_token_ids + response_token_ids
            res.response = "".join([msg.get("content", "") for msg in messages if msg["role"] == "assistant"])
            
            # response_length should equal the response loss mask length
            res.response_length = len(response_loss_mask)
            
            print(f"[DEBUG] _build_result: response_length={res.response_length}, "
                  f"response_loss_mask_len={len(response_loss_mask)}, "
                  f"total_loss_mask_len={len(loss_masks)}, "
                  f"prompt_token_len={len(prompt_token_ids)}, "
                  f"response_token_len={len(response_token_ids)}, "
                  f"response='{res.response[:100]}...'")
            return res

        # Multi-turn interaction loop
        for _ in range(max_num_steps):
            # Prepare payload for sglang
            text_input = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.tools_info)
            payload = {
                "text": text_input,
                "sampling_params": sampling_params}
            # Send request to sglang server with tool call settings
            output = await post(url, payload)
        
            # Check for abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return _build_result(res)
            response = output["text"]
            
            # Use OpenAI adapter to parse tool calls
            print(f"[TrainableAgentMixin] Using OpenAI adapter to parse response: {response}...")
            try:
                # Get OpenAI format result
                openai_result = self.openai_adapter.parse_response_to_openai_format(response)
                print(f"[TrainableAgentMixin] OpenAI adapter result: success={openai_result['success']}")
                
                if not openai_result["success"]:
                    print(f"[TrainableAgentMixin] OpenAI adapter failed: {openai_result['error']}")
                    print(f"rollout response: {response} can not be parsed into tool calls {openai_result['error']}")
                    res.status = Status.ABORTED
                    return _build_result(res)
                
                # Extract parsed results
                parsed = openai_result["parsed_result"]
                openai_message = openai_result["openai_message"]
                print(f"[TrainableAgentMixin] Successfully parsed - normal_text: '{parsed['normal_text']}', calls: {parsed['calls']}")
                print(f"[TrainableAgentMixin] OpenAI message: {openai_message}")
                
            except Exception as e:
                print(f"[TrainableAgentMixin] Exception in OpenAI adapter: {e}")
                print(f"rollout response: {response} can not be parsed into tool calls {e}")
                res.status = Status.ABORTED
                return _build_result(res)

            # Raw assistant response is the learnable target for the model. 
            messages.append({"role": "assistant", "content": response})
            assistant_token_ids, assistant_loss_mask = _get_token_delta(state.tokenizer, messages)
            response_token_ids.extend(assistant_token_ids)
            loss_masks.extend(assistant_loss_mask)  

            # Step in environment with the tool call.
            agent_content, calls = parsed['normal_text'], parsed["calls"]
            print(f"[TrainableAgentMixin] Creating action from - content: '{agent_content}', calls: {calls}")
            action = call_to_action_sglang(calls, agent_content)
            print(f"[TrainableAgentMixin] Created action: {action}")
            print(f"[TrainableAgentMixin] Stepping environment with action: {action.name}")
            env_response = env.step(action)
            print(f"[TrainableAgentMixin] Environment response: reward={env_response.reward}, done={env_response.done}")
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
                # Direct response, similar to ToolCallingAgent logic. 
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
        print(f"[TrainableAgentMixin] Getting OpenAI tools format for {len(self.tools_info)} tools")
        tools = self.openai_adapter.get_openai_tools_format()
        print(f"[TrainableAgentMixin] OpenAI tools format: {tools}")
        return tools
    
    def get_openai_message_from_response(self, response: str) -> Dict[str, Any]:
        """
        Get OpenAI format message from response
        
        Args:
            response: Raw response text from sglang
            
        Returns:
            OpenAI format message result
        """
        print(f"[TrainableAgentMixin] Getting OpenAI message from response: {response[:100]}...")
        result = self.openai_adapter.parse_response_to_openai_format(response)
        print(f"[TrainableAgentMixin] OpenAI message result: {result}")
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
