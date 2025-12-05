import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, Message, ToolCall, ToolMessage, UserMessage
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .sglang_tool_parser import parse_tools

logger = logging.getLogger(__name__)


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: list[int] | None = None
    status: Status = Status.COMPLETED
    response_length: int = 0


def _tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Convert tau2 Tool object to OpenAI schema expected by chat template."""
    return tool.openai_schema


def _tool_call_to_openai(call: ToolCall) -> dict[str, Any]:
    """Convert tau2 ToolCall to OpenAI-compatible tool call payload."""
    return {
        "id": call.id or call.name,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        },
    }


def _tau_message_to_chat(msg: Message) -> dict[str, Any] | None:
    """Convert tau2 message objects to the chat format expected by transformers templates."""
    if isinstance(msg, UserMessage):
        if msg.tool_calls:
            tool_calls = [_tool_call_to_openai(call) for call in msg.tool_calls]
            return {"role": "user", "content": None, "tool_calls": tool_calls}
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AssistantMessage):
        if msg.tool_calls:
            tool_calls = [_tool_call_to_openai(call) for call in msg.tool_calls]
            return {"role": "assistant", "content": None, "tool_calls": tool_calls}
        return {"role": "assistant", "content": msg.content}
    if isinstance(msg, ToolMessage):
        # tool_call_id keeps the chain aligned; name is optional for most templates.
        return {"role": "tool", "content": msg.content or "", "tool_call_id": msg.id}
    logger.debug("Skipping unsupported message type %s", type(msg))
    return None


def res_to_sample(res: InteractionResult, task_index: Any) -> Sample:
    status_mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens or [],
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status_mapping.get(res.status, Sample.Status.ABORTED),
        metadata=res.info,
    )
    sample.response_length = res.response_length
    return sample


class Tau2TrainableAgent:
    """
    Minimal wrapper that lets slime drive a tau2 AgentGymEnv using an sglang-served model.
    """

    def __init__(
        self,
        args,
        sampling_params: dict[str, Any],
        domain: str,
        task_split: str,
        max_steps: int = 100,
        user_llm: str | None = None,
        user_llm_args: dict[str, Any] | None = None,
        solo_mode: bool = False,
        all_messages_as_observation: bool = True,
    ):
        self.args = args
        self.sampling_params = dict(sampling_params or {})
        self._max_response_len = self.sampling_params.get("max_new_tokens") or args.rollout_max_response_len
        self._max_context_len = self.sampling_params.get("max_context_len") or args.rollout_max_context_len
        self._strip_think = getattr(args, "rollout_strip_think", False)
        self.domain = domain
        self.task_split = task_split
        self.max_steps = max_steps
        self.user_llm = user_llm
        self.user_llm_args = user_llm_args or {}
        self.solo_mode = solo_mode
        self.all_messages_as_observation = all_messages_as_observation

        self._task_splits = self._load_task_splits()

    def _load_task_splits(self) -> dict[str, list[str]] | None:
        loader = registry.get_task_splits_loader(self.domain)
        if loader is None:
            return None
        return loader()

    def _resolve_task_id(self, prompt_value: str) -> tuple[str, int]:
        """
        Convert the incoming prompt payload into a concrete task id.
        Accepts raw task ids or integer indices into the configured split.
        """
        raw = str(prompt_value).strip()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw

        if isinstance(payload, dict):
            if "task_id" in payload:
                return str(payload["task_id"]), int(payload.get("index", -1))
            if "index" in payload:
                idx = int(payload["index"])
                return self._task_id_from_index(idx), idx

        try:
            idx = int(payload)
            return self._task_id_from_index(idx), idx
        except Exception:
            return str(payload), -1

    def _task_id_from_index(self, idx: int) -> str:
        if self._task_splits is None:
            raise ValueError("Task splits not available; provide task_id directly or set up splits.")
        if self.task_split not in self._task_splits:
            raise ValueError(f"task_split={self.task_split} not found. Available: {list(self._task_splits)}")
        split_ids = self._task_splits[self.task_split]
        if idx < 0 or idx >= len(split_ids):
            raise IndexError(f"Index {idx} out of range for split '{self.task_split}' with {len(split_ids)} tasks.")
        return str(split_ids[idx])

    def _build_system_message(self, policy: str) -> dict[str, str]:
        system_prompt = SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)
        return {"role": "system", "content": system_prompt}

    def _get_token_delta(self, tokenizer, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
        """
        Compute token delta and loss mask for the newest message.
        """
        curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        token_ids: list[int] = []
        loss_mask: list[int] = []

        if messages[-1]["role"] == "assistant":
            prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [1] * len(new_tokens)
        else:
            prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [0] * len(new_tokens)
        return token_ids, loss_mask

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await post(url, payload)

    def _parse_response(self, response: str, tools_info: list[dict[str, Any]]) -> dict[str, Any]:
        parsed = parse_tools(response, tools_info, parser="qwen25")
        return parsed

    def _build_action_string(self, calls: list[dict[str, Any]], text_response: str) -> str:
        if not calls:
            return text_response
        tool_call = calls[0]
        try:
            params = json.loads(tool_call["parameters"])
        except Exception:
            params = {}
        action = {"id": tool_call.get("id") or tool_call.get("name"), "name": tool_call["name"], "arguments": params}
        return json.dumps(action)

    @staticmethod
    def _strip_think_segments(text: str) -> str:
        if "<think>" not in text:
            return text
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _append_new_messages(
        self,
        tokenizer,
        chat_messages: list[dict[str, Any]],
        env_messages: list[Message],
        seen_count: int,
        response_token_ids: list[int],
        loss_masks: list[int],
    ) -> int:
        for msg in env_messages[seen_count:]:
            chat_msg = _tau_message_to_chat(msg)
            if chat_msg is None:
                continue
            chat_messages.append(chat_msg)
            token_ids, loss_mask = self._get_token_delta(tokenizer, chat_messages)
            response_token_ids.extend(token_ids)
            loss_masks.extend(loss_mask)
        return len(env_messages)

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        chat_messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> InteractionResult:
        res.reward = total_reward
        res.info.update(info)
        res.messages = chat_messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join([m.get("content", "") or "" for m in chat_messages if m["role"] == "assistant"])
        res.response_length = len(loss_masks)
        return res

    async def run_episode(self, task_id: str) -> InteractionResult:
        state = GenerateState(self.args)
        tokenizer = state.tokenizer
        url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"

        env = AgentGymEnv(
            domain=self.domain,
            task_id=task_id,
            max_steps=self.max_steps,
            solo_mode=self.solo_mode,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            all_messages_as_observation=self.all_messages_as_observation,
        )

        _, env_info = env.reset()
        tools_info = [_tool_to_openai_schema(t) for t in env_info["tools"]]
        base_info: dict[str, Any] = {
            "task_id": getattr(env_info.get("task"), "id", task_id),
            "domain": self.domain,
            "task_split": self.task_split,
            "policy": env_info.get("policy"),
        }

        chat_messages: list[dict[str, Any]] = [self._build_system_message(env_info["policy"])]
        initial_env_messages = env._agent.observation if getattr(env, "_agent", None) else []
        seen_env_messages = len(initial_env_messages)
        for m in initial_env_messages:
            converted = _tau_message_to_chat(m)
            if converted:
                chat_messages.append(converted)

        prompt_text = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True, tools=tools_info
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        loss_masks: list[int] = []
        response_token_ids: list[int] = []
        total_reward = 0.0

        res = InteractionResult(
            prompt=prompt_text,
            reward=0.0,
            messages=[],
            info=base_info.copy(),
            status=Status.COMPLETED,
        )

        terminated = False
        for _ in range(self.max_steps):
            text_input = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True, tools=tools_info
            )

            used_tokens = len(prompt_token_ids) + len(response_token_ids)
            if self._max_context_len is not None:
                remaining_context = self._max_context_len - used_tokens
                max_new_tokens = min(self._max_response_len, remaining_context)
            else:
                max_new_tokens = self._max_response_len

            if max_new_tokens <= 0:
                res.status = Status.TRUNCATED
                return self._build_final_result(
                    res, total_reward, res.info, chat_messages, loss_masks, prompt_token_ids, response_token_ids
                )

            self.sampling_params["max_new_tokens"] = max_new_tokens

            payload = {"text": text_input, "sampling_params": self.sampling_params}
            output = await self._call_llm(url, payload)

            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, res.info, chat_messages, loss_masks, prompt_token_ids, response_token_ids
                )

            response = output["text"]
            if self._strip_think:
                response = self._strip_think_segments(response)
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            try:
                parsed = self._parse_response(response, tools_info)
                calls = parsed["calls"]
                normal_text = parsed["normal_text"].strip()
            except Exception as e:
                logger.warning("Failed to parse response: %s", e)
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, res.info, chat_messages, loss_masks, prompt_token_ids, response_token_ids
                )

            if not calls and not (normal_text or response):
                logger.warning("Empty model response; aborting rollout.")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, res.info, chat_messages, loss_masks, prompt_token_ids, response_token_ids
                )

            if calls:
                # Enforce protocol: tool call message should not contain user-facing text.
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.get("id") or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call.get("parameters", "{}"),
                            },
                        }
                        for idx, call in enumerate(calls)
                    ],
                }
            else:
                assistant_message = {"role": "assistant", "content": normal_text or response}

            chat_messages.append(assistant_message)
            token_ids, loss_mask = self._get_token_delta(tokenizer, chat_messages)
            response_token_ids.extend(token_ids)
            loss_masks.extend(loss_mask)

            action_string = self._build_action_string(calls, normal_text or response)
            try:
                _, reward, terminated, _, step_info = env.step(action_string)
            except Exception as e:
                logger.warning("Environment step failed: %s", e)
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, res.info, chat_messages, loss_masks, prompt_token_ids, response_token_ids
                )

            total_reward = reward
            # Update env/task metadata; keep it JSON-serializable where possible.
            reward_info = step_info.get("reward_info")
            if isinstance(reward_info, str):
                try:
                    reward_info = json.loads(reward_info)
                except Exception:
                    pass
            res.info.update({"reward_info": reward_info})

            if getattr(env, "_agent", None):
                seen_env_messages = self._append_new_messages(
                    tokenizer,
                    chat_messages,
                    env._agent.observation,
                    seen_env_messages,
                    response_token_ids,
                    loss_masks,
                )

            if terminated:
                res.status = Status.COMPLETED
                break

        if not terminated:
            res.status = Status.TRUNCATED

        return self._build_final_result(
            res, total_reward, res.info, chat_messages, loss_masks, prompt_token_ids, response_token_ids
        )
