from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from .rebench_eval import (
    evaluate_rebench_result,
    remote_rebench_patch_path,
    render_rebench_eval_script,
    resolve_rebench_base_commit,
    resolve_rebench_workdir,
)
from .rollout_logging import (
    build_eval_log_path,
    build_live_sandbox_log_path,
    build_sample_log_dir,
    prepare_batch_log_dir,
    truncate_text,
    write_sample_artifacts_snapshot,
)
from .runtime_env_paths import expand_runtime_placeholders as _expand_runtime_placeholders
from .sandbox_factory import (
    INSPIRE_SPEC_CPU_HINTS,
    configure_sandbox_defaults,
    install_agent_with_retry,
    normalize_backend,
    resolve_inspire_image,
    resolve_inspire_spec,
    resolve_rock_base_url,
    resolve_sandbox_default_user,
    start_sandbox_with_retry,
)

DEFAULT_SLIME_REPO_ROOT_BOOTSTRAP = str(Path(__file__).resolve().parents[2])
if DEFAULT_SLIME_REPO_ROOT_BOOTSTRAP not in sys.path:
    sys.path.insert(0, DEFAULT_SLIME_REPO_ROOT_BOOTSTRAP)

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, get_model_url
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample


DEFAULT_SESSION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
DEFAULT_ROCK_ROOT = str(Path(__file__).resolve().parents[4] / "ROCK")
DEFAULT_SLIME_REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_AGENT_CONFIG_PATH = str(
    Path(__file__).resolve().with_name("rock_agent_qwen_rebench_template.yaml")
)

_ROLLOUT_STATE: dict[str, Any] = {}


def ensure_rock_root_on_path(rock_root: str) -> None:
    rock_root_path = str(Path(rock_root).resolve())
    if rock_root_path not in sys.path:
        sys.path.insert(0, rock_root_path)


def ensure_workspace_root_on_path(workspace_root: str) -> None:
    workspace_root_path = str(Path(workspace_root).resolve())
    if workspace_root_path not in sys.path:
        sys.path.insert(0, workspace_root_path)


def _env_or_arg(args, env_name: str, arg_name: str, default: Any) -> Any:
    if arg_name not in (None, ""):
        value = getattr(args, arg_name, None)
        if value not in (None, ""):
            return value
    env_value = os.environ.get(env_name, None)
    if env_value in (None, ""):
        return default
    return env_value


def _get_rollout_state(args) -> dict[str, Any]:
    cache_key = f"{args.hf_checkpoint}|{args.loss_mask_type}"
    if cache_key in _ROLLOUT_STATE:
        return _ROLLOUT_STATE[cache_key]

    generate_state = GenerateState(args)
    tokenizer = generate_state.tokenizer
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    state = {
        "generate_state": generate_state,
        "tokenizer": tokenizer,
        "mask_generator": mask_generator,
    }
    _ROLLOUT_STATE[cache_key] = state
    return state


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(str(item.get("content") or ""))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_observation_from_request(request_obj: dict[str, Any]) -> dict[str, Any]:
    messages = request_obj.get("messages") or []
    if not messages:
        return {"role": "", "content": ""}
    last_message = messages[-1]
    return {
        "role": str(last_message.get("role") or ""),
        "content": _stringify_content(last_message.get("content")),
    }


def _sanitize_message(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = {"role": message.get("role", "user"), "content": message.get("content", "")}
    for key in ("tool_call_id", "name", "tool_calls", "function_call"):
        if key in message:
            value = copy.deepcopy(message[key])
            if key == "tool_calls" and isinstance(value, list):
                normalized_tool_calls: list[dict[str, Any]] = []
                for tool_call in value:
                    if not isinstance(tool_call, dict):
                        normalized_tool_calls.append(tool_call)
                        continue
                    normalized_tool_call = copy.deepcopy(tool_call)
                    function_block = normalized_tool_call.get("function")
                    if isinstance(function_block, dict):
                        arguments = function_block.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                parsed_arguments = json.loads(arguments)
                            except Exception:
                                parsed_arguments = arguments
                            if isinstance(parsed_arguments, dict):
                                function_block["arguments"] = parsed_arguments
                    normalized_tool_calls.append(normalized_tool_call)
                value = normalized_tool_calls
            sanitized[key] = value
    return sanitized


def _coerce_param_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    if value in {"true", "false", "null"}:
        try:
            return json.loads(value)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if (value[0] == "[" and value[-1] == "]") or (value[0] == "{" and value[-1] == "}"):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _parse_tool_calls(response: str) -> tuple[bool, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    if "<function" in response:
        function_pattern = r"<function\s*=\s*([^>]+)>(.*?)</function>"
        function_matches = re.findall(function_pattern, response, flags=re.DOTALL)
        for i, (function_name, function_body) in enumerate(function_matches):
            param_pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
            param_matches = re.findall(param_pattern, function_body, flags=re.DOTALL)
            params = {
                key.strip(): _coerce_param_value(value)
                for key, value in param_matches
            }
            actions.append(
                {
                    "type": "function",
                    "id": f"{function_name.strip()}_{int(time.time() * 1000)}_{i}",
                    "function": {
                        "name": function_name.strip(),
                        "arguments": json.dumps(params, ensure_ascii=False),
                    },
                }
            )
        return bool(function_matches), actions

    if "<tool_call>" in response:
        if "</tool_call>" not in response:
            response = response + "</tool_call>"
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
        for i, tool_call_str in enumerate(tool_call_matches):
            try:
                tool_call_json = json.loads(tool_call_str.strip())
            except json.JSONDecodeError:
                continue
            function_name = tool_call_json.get("name", "")
            arguments = tool_call_json.get("arguments", {})
            actions.append(
                {
                    "type": "function",
                    "id": f"{function_name}_{int(time.time() * 1000)}_{i}",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        return bool(tool_call_matches), actions

    return False, actions


def _format_response_payload(response_text: str) -> tuple[str, dict[str, Any]]:
    tool_calls_present, tool_calls = _parse_tool_calls(response_text)
    content = response_text
    if tool_calls_present:
        content = re.split(r"<tool_call>|<function", response_text, maxsplit=1)[0]
    if tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    payload = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }
    if tool_calls:
        payload["choices"][0]["message"]["tool_calls"] = tool_calls

    return json.dumps(payload, ensure_ascii=False), {
        "finish_reason": finish_reason,
        "tool_call_count": len(tool_calls),
    }


def _build_fallback_messages(prompt: str, failure_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": failure_text or "Execution failed.", "step_loss_mask": 1},
    ]


def _finalize_sample(
    args,
    sample: Sample,
    *,
    final_messages: list[dict[str, Any]],
    final_tools: list[dict[str, Any]] | None,
    reward: float,
    status: Sample.Status,
    turn_responses: list[str],
    trajectory: list[dict[str, Any]],
    extra_metadata: dict[str, Any],
) -> Sample:
    state = _get_rollout_state(args)
    mask_generator: MultiTurnLossMaskGenerator = state["mask_generator"]

    if status == Sample.Status.FAILED:
        reward = 0.0
    extra_metadata.setdefault("raw_reward", float(reward if reward is not None else 0.0))

    token_ids, loss_mask = mask_generator.get_loss_mask(final_messages, tools=final_tools)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        fallback_messages = _build_fallback_messages(
            str(sample.prompt),
            "No assistant tokens were produced; converted to failure sample.",
        )
        token_ids, loss_mask = mask_generator.get_loss_mask(fallback_messages, tools=None)
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        final_messages = fallback_messages
        status = Sample.Status.FAILED
        reward = 0.0

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = loss_mask[-response_length:]
    sample.response = "\n\n".join(resp for resp in turn_responses if resp.strip())
    sample.reward = reward
    sample.status = status
    sample.metadata = {
        **(sample.metadata or {}),
        "trajectory": trajectory,
        "training_messages": final_messages,
        **extra_metadata,
    }
    return sample


async def _generate_assistant_response(
    args,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    evaluation: bool,
) -> tuple[str, dict[str, Any]]:
    state = _get_rollout_state(args)
    tokenizer = state["tokenizer"]
    model_url = get_model_url(args, getattr(args, "swe_model_name", "default"))
    sampling_params = dict(state["generate_state"].sampling_params)
    if evaluation:
        sampling_params["temperature"] = 0.0
        sampling_params["top_p"] = 1.0
        sampling_params["top_k"] = 1

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools or None,
        add_generation_prompt=True,
    )
    output = await post(
        model_url,
        {
            "text": prompt_text,
            "sampling_params": sampling_params,
            "return_logprob": False,
        },
    )
    meta_info = output.get("meta_info", {})
    return output.get("text", ""), meta_info


async def _run_eval_script(
    sandbox,
    metadata: dict[str, Any],
    *,
    live_log_path: str | None,
    wait_timeout: int,
) -> tuple[int | None, str]:
    instance_id = str(metadata.get("instance_id") or "task")
    eval_user = resolve_sandbox_default_user(metadata)
    test_patch_remote_path = "/patches/test_patch.diff"
    rendered_eval_script = render_rebench_eval_script(
        metadata,
        test_patch_file=test_patch_remote_path,
    )

    local_test_patch_path: str | None = None
    test_patch = str(metadata.get("test_patch") or "")
    with tempfile.NamedTemporaryFile("w", suffix=".test.diff", delete=False, encoding="utf-8") as f:
        f.write(test_patch)
        local_test_patch_path = f.name
    eval_session = f"rebench-eval-{re.sub(r'[^A-Za-z0-9_.-]+', '-', instance_id)[:40]}"
    try:
        if live_log_path and hasattr(sandbox, "set_live_log_path"):
            sandbox.set_live_log_path(live_log_path)
        from rock.actions import CreateBashSessionRequest

        await sandbox.create_session(
            CreateBashSessionRequest(
                session=eval_session,
                env_enable=True,
                env={},
                remote_user=eval_user,
            )
        )
        await sandbox.arun("mkdir -p /patches", session=eval_session, wait_timeout=wait_timeout)
        await sandbox.upload_by_path(local_test_patch_path, test_patch_remote_path, user=eval_user)
        eval_shell_script = "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false\n" + rendered_eval_script
        observation = await sandbox.arun(
            f"/bin/bash -c {shlex.quote(eval_shell_script)} 2>&1",
            session=eval_session,
            wait_timeout=wait_timeout,
        )
    finally:
        if local_test_patch_path is not None:
            Path(local_test_patch_path).unlink(missing_ok=True)
        try:
            from rock.actions import CloseSessionRequest

            await sandbox.close_session(CloseSessionRequest(session=eval_session))
        except Exception:
            pass

    return observation.exit_code, (observation.output or "")


async def _extract_rebench_candidate_patch(
    sandbox,
    metadata: dict[str, Any],
    *,
    wait_timeout: int,
) -> str:
    workdir = resolve_rebench_workdir(metadata)
    script = "\n".join(
        [
            "set -e",
            f"cd {shlex.quote(workdir)}",
            "git add -N -A",
            "git diff --binary --no-color HEAD",
        ]
    )
    observation = await sandbox.arun(
        f"bash -lc {shlex.quote(script)}",
        wait_timeout=wait_timeout,
    )
    if observation.exit_code != 0:
        raise RuntimeError(f"failed to extract rebench candidate patch: {observation.output or ''}".strip())
    return observation.output or ""

async def _prepare_rebench_agent_workspace(
    sandbox,
    metadata: dict[str, Any],
    *,
    wait_timeout: int,
) -> str:
    workdir = resolve_rebench_workdir(metadata)
    base_commit = resolve_rebench_base_commit(metadata)
    script = "\n".join(
        [
            "set -e",
            f"cd {shlex.quote(workdir)}",
            'echo "=== REBENCH TARGET BASE COMMIT ==="',
            f"printf '%s\\n' {shlex.quote(base_commit)}",
            'echo "=== REBENCH BASELINE HEAD ==="',
            "git rev-parse HEAD",
            'echo "=== REBENCH BASELINE STATUS BEFORE ==="',
            "git status --short",
            'echo "=== REBENCH BASELINE DIFFSTAT BEFORE ==="',
            "git diff --stat HEAD || true",
            'echo "=== REBENCH BASELINE UNTRACKED BEFORE ==="',
            "git ls-files --others --exclude-standard",
            'echo "=== REBENCH BASELINE CLEANUP ==="',
            "printf 'git_reset_hard_head\\n'",
            "git reset --hard HEAD",
            'echo "=== REBENCH BASELINE HEAD AFTER ==="',
            "git rev-parse HEAD",
            'echo "=== REBENCH BASELINE STATUS AFTER ==="',
            "git status --short",
        ]
    )
    observation = await sandbox.arun(
        f"bash -lc {shlex.quote(script)}",
        wait_timeout=wait_timeout,
    )
    if observation.exit_code != 0:
        raise RuntimeError(f"failed to prepare rebench workspace: {observation.output or ''}".strip())
    return observation.output or ""


def _prepare_agent_config_for_sample(
    agent_config_path: str,
    metadata: dict[str, Any],
) -> tuple[str, Path | None]:
    config_path = Path(agent_config_path)
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise RuntimeError(f"Agent config must be a YAML object: {agent_config_path}")

    config_payload = _expand_runtime_placeholders(config_payload)
    config_payload["project_path"] = resolve_rebench_workdir(metadata)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)
        return f.name, Path(f.name)


async def _prepare_agent_run_wrapper_script(
    sandbox,
    *,
    raw_agent_run_cmd: str,
    sample_idx: int,
) -> tuple[str, str]:
    from rock.actions.sandbox.request import WriteFileRequest

    remote_script_path = f"/tmp/rock_agent_run_sample_{sample_idx}.sh"
    script_content = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -x",
            raw_agent_run_cmd,
            "",
        ]
    )
    await sandbox.write_file(WriteFileRequest(path=remote_script_path, content=script_content))
    await sandbox.arun(f"chmod +x {remote_script_path}")
    return remote_script_path, f"bash {shlex.quote(remote_script_path)}"


async def _run_single_sample_once(
    args,
    sample: Sample,
    *,
    evaluation: bool,
    rollout_id: int,
    sample_idx: int,
    sample_semaphore: asyncio.Semaphore,
) -> tuple[Sample, bool]:
    rock_root = _env_or_arg(args, "ROCK_SWE_ROCK_ROOT", "swe_rock_root", DEFAULT_ROCK_ROOT)
    sandbox_backend = normalize_backend(_env_or_arg(args, "ROCK_SWE_SANDBOX_BACKEND", "swe_sandbox_backend", "rock"))
    rock_base_url = resolve_rock_base_url(
        base_url=str(_env_or_arg(args, "ROCK_SWE_BASE_URL", "swe_base_url", "") or "").strip(),
        backend=sandbox_backend,
    )
    agent_config_path = _env_or_arg(
        args, "ROCK_SWE_AGENT_CONFIG_PATH", "swe_agent_config_path", DEFAULT_AGENT_CONFIG_PATH
    )
    startup_timeout = int(_env_or_arg(args, "ROCK_SWE_STARTUP_TIMEOUT", "swe_startup_timeout", 10800))
    wait_timeout = int(_env_or_arg(args, "ROCK_SWE_WAIT_TIMEOUT", "swe_wait_timeout", 10800))
    preview_limit = int(_env_or_arg(args, "ROCK_SWE_OUTPUT_PREVIEW_LIMIT", "swe_output_preview_limit", 12000))
    inspire_spec = resolve_inspire_spec(
        explicit_spec=str(_env_or_arg(args, "ROCK_INSPIRE_SPEC", "inspire_spec", "") or "").strip(),
        cpus=int(_env_or_arg(args, "ROCK_SWE_CPUS", "swe_cpus", 2)),
    )
    memory = str(_env_or_arg(args, "ROCK_SWE_MEMORY", "swe_memory", "4g"))
    cpus = INSPIRE_SPEC_CPU_HINTS[inspire_spec]
    keep_containers = str(_env_or_arg(args, "ROCK_SWE_KEEP_CONTAINERS", "swe_keep_containers", "0")) == "1"
    sandbox_start_retry_times = int(
        _env_or_arg(args, "ROCK_SWE_SANDBOX_START_RETRY_TIMES", "swe_sandbox_start_retry_times", 10)
    )
    sandbox_start_retry_interval = float(
        _env_or_arg(args, "ROCK_SWE_SANDBOX_START_RETRY_INTERVAL", "swe_sandbox_start_retry_interval", 1)
    )
    agent_install_retry_times = int(
        _env_or_arg(args, "ROCK_SWE_AGENT_INSTALL_RETRY_TIMES", "swe_agent_install_retry_times", 10)
    )
    agent_install_retry_interval = float(
        _env_or_arg(args, "ROCK_SWE_AGENT_INSTALL_RETRY_INTERVAL", "swe_agent_install_retry_interval", 1)
    )
    max_turns = int(_env_or_arg(args, "ROCK_SWE_MAX_TURNS", "swe_max_turns", 50))
    agent_finish_timeout = int(
        _env_or_arg(args, "ROCK_SWE_AGENT_FINISH_TIMEOUT", "swe_agent_finish_timeout", 10800)
    )
    turn_limit_agent_grace_timeout = float(
        _env_or_arg(args, "ROCK_SWE_TURN_LIMIT_AGENT_GRACE_TIMEOUT", None, 15)
    )
    log_root = _env_or_arg(args, "ROCK_SWE_LOG_ROOT", "swe_log_root", None)

    ensure_rock_root_on_path(str(rock_root))
    sample_prompt = str(sample.prompt)
    metadata = dict(sample.metadata or {})
    sandbox = None
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is not None:
        sample_log_dir.mkdir(parents=True, exist_ok=True)
    agent_live_sandbox_log_path = build_live_sandbox_log_path(
        log_root,
        sample_idx=sample_idx,
    )

    trajectory: list[dict[str, Any]] = []
    turn_responses: list[str] = []
    final_messages: list[dict[str, Any]] | None = None
    final_tools: list[dict[str, Any]] | None = None
    reward = 0.0
    status = Sample.Status.FAILED
    extra_metadata: dict[str, Any] = {"raw_reward": 0.0}
    if sample_log_dir is not None:
        extra_metadata["sample_log_dir"] = str(sample_log_dir)
    persisted_log_files: list[str] = []
    if agent_live_sandbox_log_path is not None:
        extra_metadata["sandbox_api_log_path"] = str(agent_live_sandbox_log_path)
        extra_metadata["agent_sandbox_api_log_path"] = str(agent_live_sandbox_log_path)
        persisted_log_files.append("sandbox/agent_output.log")
    eval_log_path = build_eval_log_path(
        log_root,
        sample_idx=sample_idx,
    )
    if eval_log_path is not None:
        extra_metadata["eval_log_path"] = str(eval_log_path)
        persisted_log_files.append("sandbox/eval_output.log")
    if persisted_log_files:
        extra_metadata["persisted_log_files"] = persisted_log_files
    agent_task = None
    agent_result_cached = None
    original_create_agent_run_cmd = None
    last_response_payload: str | None = None
    failure_reason = ""
    last_generation_finish_reason = ""
    reached_turn_limit = False
    temp_agent_config_path: Path | None = None

    def _persist_sample_snapshot() -> None:
        write_sample_artifacts_snapshot(
            log_root=log_root,
            rollout_id=rollout_id,
            sample_idx=sample_idx,
            sample_prompt=sample_prompt,
            metadata=metadata,
            extra_metadata=extra_metadata,
            turn_responses=turn_responses,
            trajectory=trajectory,
            final_messages=final_messages,
            last_response_payload=last_response_payload,
        )

    async with sample_semaphore:
        try:
            if sandbox_backend == "inspire":
                template_name = str(metadata.get("inspire_template") or "").strip()
                if template_name:
                    extra_metadata["resolved_inspire_template"] = template_name
                    extra_metadata["inspire_template_source"] = "manifest"
                else:
                    extra_metadata["resolved_inspire_image"] = resolve_inspire_image(metadata)
                    extra_metadata["inspire_template_source"] = "image"

            sandbox = await start_sandbox_with_retry(
                metadata=metadata,
                rock_base_url=rock_base_url,
                startup_timeout=startup_timeout,
                memory=memory,
                cpus=cpus,
                keep_containers=keep_containers,
                retry_times=sandbox_start_retry_times,
                retry_interval_seconds=sandbox_start_retry_interval,
                sandbox_backend=sandbox_backend,
                inspire_spec=inspire_spec,
                live_log_path=(
                    str(agent_live_sandbox_log_path) if agent_live_sandbox_log_path is not None else None
                ),
            )
            extra_metadata["sandbox_id"] = sandbox.sandbox_id
            extra_metadata["sandbox_backend"] = sandbox_backend
            extra_metadata["inspire_spec"] = inspire_spec
            extra_metadata["inspire_cpu_hint"] = cpus
            effective_user, effective_image_env = configure_sandbox_defaults(sandbox, metadata)
            extra_metadata["effective_sandbox_user"] = effective_user
            extra_metadata["effective_sandbox_env_keys"] = sorted(effective_image_env.keys())

            effective_agent_config_path, temp_agent_config_path = _prepare_agent_config_for_sample(
                str(agent_config_path),
                metadata,
            )
            extra_metadata["effective_agent_config_path"] = effective_agent_config_path
            await install_agent_with_retry(
                sandbox,
                config_path=effective_agent_config_path,
                retry_times=agent_install_retry_times,
                retry_interval_seconds=agent_install_retry_interval,
            )
            workspace_prepare_output = await _prepare_rebench_agent_workspace(
                sandbox,
                metadata,
                wait_timeout=wait_timeout,
            )
            extra_metadata["workspace_prepare_preview"] = truncate_text(
                workspace_prepare_output,
                preview_limit,
            )
            raw_agent_run_cmd = await sandbox.agent._create_agent_run_cmd(str(sample.prompt))
            wrapper_script_path, agent_run_cmd = await _prepare_agent_run_wrapper_script(
                sandbox,
                raw_agent_run_cmd=raw_agent_run_cmd,
                sample_idx=sample_idx,
            )
            extra_metadata["raw_agent_run_cmd"] = raw_agent_run_cmd
            extra_metadata["agent_run_wrapper_script"] = wrapper_script_path
            extra_metadata["agent_run_cmd"] = agent_run_cmd
            extra_metadata["output_preview_limit"] = preview_limit
            extra_metadata["log_tail_limit"] = preview_limit
            original_create_agent_run_cmd = sandbox.agent._create_agent_run_cmd

            async def _wrapped_create_agent_run_cmd(_prompt: str) -> str:
                return agent_run_cmd

            sandbox.agent._create_agent_run_cmd = _wrapped_create_agent_run_cmd
            _persist_sample_snapshot()
            agent_task = asyncio.create_task(sandbox.agent.run(sample_prompt))

            implicit_session_end_grace_timeout = 5.0

            index = 0
            while True:
                extra_metadata["anti_call_index"] = index
                if index == 0 and agent_task is not None:
                    anti_call_task = asyncio.create_task(
                        sandbox.agent.model_service.anti_call_llm(
                            index=index,
                            response_payload=last_response_payload,
                        )
                    )
                    done, _ = await asyncio.wait(
                        {anti_call_task, agent_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if agent_task in done and anti_call_task not in done:
                        try:
                            agent_result_cached = await agent_task
                            agent_exit_code = getattr(agent_result_cached, "exit_code", None)
                            agent_output_preview = truncate_text(
                                getattr(agent_result_cached, "output", "") or "",
                                preview_limit,
                            )
                            extra_metadata["agent_exit_code"] = agent_exit_code
                            extra_metadata["agent_output_preview"] = agent_output_preview
                        except Exception as agent_exc:
                            extra_metadata["agent_error"] = str(agent_exc)
                            agent_exit_code = None
                            agent_output_preview = ""
                        anti_call_task.cancel()
                        await asyncio.gather(anti_call_task, return_exceptions=True)
                        raise RuntimeError(
                            "Agent exited before first model request. "
                            f"exit_code={agent_exit_code}; "
                            f"output={agent_output_preview or '<empty>'}"
                        )
                    request_str = await anti_call_task
                    if agent_task in done:
                        try:
                            agent_result_cached = await agent_task
                            extra_metadata["agent_exit_code"] = getattr(agent_result_cached, "exit_code", None)
                            extra_metadata["agent_output_preview"] = truncate_text(
                                getattr(agent_result_cached, "output", "") or "",
                                preview_limit,
                            )
                        except Exception as agent_exc:
                            extra_metadata["agent_error"] = str(agent_exc)
                else:
                    anti_call_task = asyncio.create_task(
                        sandbox.agent.model_service.anti_call_llm(
                            index=index,
                            response_payload=last_response_payload,
                        )
                    )
                    if agent_task is not None:
                        done, _ = await asyncio.wait(
                            {anti_call_task, agent_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if agent_task in done and anti_call_task not in done:
                            try:
                                agent_result_cached = await agent_task
                                extra_metadata["agent_exit_code"] = getattr(agent_result_cached, "exit_code", None)
                                extra_metadata["agent_output_preview"] = truncate_text(
                                    getattr(agent_result_cached, "output", "") or "",
                                    preview_limit,
                                )
                            except Exception as agent_exc:
                                extra_metadata["agent_error"] = str(agent_exc)

                            try:
                                request_str = await asyncio.wait_for(
                                    asyncio.shield(anti_call_task),
                                    timeout=implicit_session_end_grace_timeout,
                                )
                            except asyncio.TimeoutError:
                                anti_call_task.cancel()
                                await asyncio.gather(anti_call_task, return_exceptions=True)
                                request_str = "SESSION_END"
                                extra_metadata["implicit_session_end"] = True
                                extra_metadata["implicit_session_end_reason"] = (
                                    "agent exited before anti_call_llm returned; "
                                    "treating agent exit as SESSION_END"
                                )
                            except Exception:
                                anti_call_task.cancel()
                                await asyncio.gather(anti_call_task, return_exceptions=True)
                                raise
                        else:
                            request_str = await anti_call_task
                            if agent_task in done:
                                try:
                                    agent_result_cached = await agent_task
                                    extra_metadata["agent_exit_code"] = getattr(agent_result_cached, "exit_code", None)
                                    extra_metadata["agent_output_preview"] = truncate_text(
                                        getattr(agent_result_cached, "output", "") or "",
                                        preview_limit,
                                    )
                                except Exception as agent_exc:
                                    extra_metadata["agent_error"] = str(agent_exc)
                    else:
                        request_str = await anti_call_task
                extra_metadata["last_anti_call_request_preview"] = truncate_text(request_str, preview_limit)
                request_str = request_str.strip()
                if request_str == "SESSION_END":
                    break
                if not request_str:
                    raise RuntimeError("anti_call_llm returned empty response")

                try:
                    request_obj = json.loads(request_str)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "anti_call_llm returned non-JSON response: "
                        f"{truncate_text(request_str, preview_limit)}"
                    ) from exc
                messages = [_sanitize_message(msg) for msg in (request_obj.get("messages") or [])]
                tools = request_obj.get("tools") or []
                observation = _extract_observation_from_request(request_obj)

                assistant_text, generation_meta = await _generate_assistant_response(
                    args,
                    messages,
                    tools,
                    evaluation=evaluation,
                )
                response_payload, response_info = _format_response_payload(assistant_text)
                last_response_payload = response_payload
                finish_reason = str(response_info.get("finish_reason") or "")
                last_generation_finish_reason = finish_reason

                final_messages = messages + [{"role": "assistant", "content": assistant_text, "step_loss_mask": 1}]
                final_tools = tools
                turn_responses.append(assistant_text)
                trajectory.append(
                    {
                        "turn_id": index,
                        "observation": observation,
                        "model_request": request_obj,
                        "raw_model_response": assistant_text,
                        "formatted_model_response": json.loads(response_payload),
                        "generation_finish_reason": finish_reason,
                        "raw_generation_finish_reason": (
                            (generation_meta.get("finish_reason", {}) or {}).get("type", "")
                        ),
                        "tool_call_count": response_info["tool_call_count"],
                    }
                )
                _persist_sample_snapshot()

                index += 1
                if index >= max_turns:
                    reached_turn_limit = True
                    failure_reason = f"Reached max turns: {max_turns}"
                    break

            if reached_turn_limit and agent_task is not None and not agent_task.done():
                agent_session_name = getattr(getattr(sandbox, "agent", None), "agent_session", None)
                if agent_session_name:
                    try:
                        from rock.actions import CloseSessionRequest

                        await sandbox.close_session(CloseSessionRequest(session=agent_session_name))
                        extra_metadata["agent_session_closed_on_turn_limit"] = True
                        extra_metadata["agent_session_close_reason"] = failure_reason
                    except Exception as exc:
                        extra_metadata["agent_session_close_error"] = str(exc)

            if agent_task is not None:
                try:
                    wait_timeout = turn_limit_agent_grace_timeout if reached_turn_limit else agent_finish_timeout
                    if agent_result_cached is None:
                        agent_result = await asyncio.wait_for(agent_task, timeout=wait_timeout)
                    else:
                        agent_result = agent_result_cached
                    extra_metadata["agent_exit_code"] = getattr(agent_result, "exit_code", None)
                    extra_metadata["agent_output_preview"] = truncate_text(
                        getattr(agent_result, "output", "") or "",
                        preview_limit,
                    )
                except Exception as exc:
                    extra_metadata["agent_error"] = str(exc)
                    if failure_reason:
                        failure_reason = f"{failure_reason}; agent wait error: {exc}"
                    else:
                        failure_reason = f"agent wait error: {exc}"
                    if not reached_turn_limit:
                        raise

            candidate_patch = await _extract_rebench_candidate_patch(
                sandbox,
                metadata,
                wait_timeout=wait_timeout,
            )
            extra_metadata["candidate_patch_bytes"] = len(candidate_patch.encode("utf-8"))
            extra_metadata["candidate_patch_empty"] = not bool(candidate_patch.strip())
            eval_exit_code, eval_preview = await _run_eval_script(
                sandbox,
                metadata,
                live_log_path=(str(eval_log_path) if eval_log_path is not None else None),
                wait_timeout=wait_timeout,
            )
            rebench_eval = evaluate_rebench_result(
                metadata,
                eval_exit_code=eval_exit_code,
                eval_output=eval_preview,
            )
            reward = float(rebench_eval["reward"])
            extra_metadata.update(
                {
                    "dense_reward": float(rebench_eval["dense_reward"]),
                    "solved": bool(rebench_eval["solved"]),
                    "log_parser": rebench_eval["parser_name"],
                    "passed_actual": rebench_eval["passed_actual"],
                    "failed_actual": rebench_eval["failed_actual"],
                    "from_fail_to_pass": rebench_eval["from_fail_to_pass"],
                    "failed_from_pass_to_pass": rebench_eval["failed_from_pass_to_pass"],
                    "fail_to_pass_expected": rebench_eval["fail_to_pass_expected"],
                    "pass_to_pass_expected": rebench_eval["pass_to_pass_expected"],
                }
            )
            status = Sample.Status.TRUNCATED if reached_turn_limit else Sample.Status.COMPLETED
            if reward <= 0.0 and not turn_responses:
                status = Sample.Status.FAILED
            extra_metadata.update(
                {
                    "eval_exit_code": eval_exit_code,
                    "eval_output_preview": truncate_text(eval_preview, preview_limit),
                    "eval_script_path": f"/tmp/{metadata.get('instance_id', 'task')}.eval.sh",
                    "eval_script_preview": truncate_text(
                        render_rebench_eval_script(
                            metadata,
                            test_patch_file=remote_rebench_patch_path(
                                str(metadata.get("instance_id") or "task"),
                                "test_patch",
                            ),
                        ),
                        preview_limit,
                    ),
                    "raw_reward": reward,
                    "reached_turn_limit": reached_turn_limit,
                    "last_generation_finish_reason": last_generation_finish_reason,
                }
            )
        except Exception as exc:
            failure_reason = str(exc)
            extra_metadata["failure_reason"] = failure_reason
            extra_metadata["failure_type"] = exc.__class__.__name__
            reward = 0.0
            status = Sample.Status.FAILED
        finally:
            had_empty_turn_responses = not bool(turn_responses)
            extra_metadata["turn_responses_empty"] = had_empty_turn_responses
            if agent_task is not None and not agent_task.done():
                agent_task.cancel()
                try:
                    await asyncio.gather(agent_task, return_exceptions=True)
                except Exception as cancel_exc:
                    extra_metadata["agent_cancel_error"] = str(cancel_exc)
            try:
                await sandbox.stop()
            except Exception as stop_exc:
                extra_metadata["sandbox_stop_error"] = str(stop_exc)
            if original_create_agent_run_cmd is not None:
                try:
                    sandbox.agent._create_agent_run_cmd = original_create_agent_run_cmd
                except Exception:
                    pass
            if temp_agent_config_path is not None:
                temp_agent_config_path.unlink(missing_ok=True)
            _persist_sample_snapshot()

    if final_messages is None:
        final_messages = _build_fallback_messages(sample_prompt, failure_reason or "Sandbox rollout failed.")
        final_tools = None
        if not turn_responses:
            turn_responses.append(failure_reason or "Sandbox rollout failed.")

    finalized_sample = _finalize_sample(
        args,
        sample,
        final_messages=final_messages,
        final_tools=final_tools,
        reward=reward,
        status=status,
        turn_responses=turn_responses,
        trajectory=trajectory,
        extra_metadata=extra_metadata,
    )
    return finalized_sample, had_empty_turn_responses


async def _run_single_sample(
    args,
    sample: Sample,
    *,
    evaluation: bool,
    rollout_id: int,
    sample_idx: int,
    sample_semaphore: asyncio.Semaphore,
) -> Sample:
    sample_for_run = copy.deepcopy(sample)
    finalized_sample, _ = await _run_single_sample_once(
        args,
        sample_for_run,
        evaluation=evaluation,
        rollout_id=rollout_id,
        sample_idx=sample_idx,
        sample_semaphore=sample_semaphore,
    )
    return finalized_sample


async def _run_group(
    args,
    group: list[Sample],
    *,
    evaluation: bool,
    rollout_id: int,
    sample_idx_offset: int,
    sample_semaphore: asyncio.Semaphore,
) -> list[Sample]:
    tasks: list[asyncio.Task[Sample]] = []
    for local_idx, sample in enumerate(group):
        tasks.append(
            asyncio.create_task(
                _run_single_sample(
                    args,
                    sample,
                    evaluation=evaluation,
                    rollout_id=rollout_id,
                    sample_idx=sample_idx_offset + local_idx,
                    sample_semaphore=sample_semaphore,
                )
            )
        )
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> RolloutFnTrainOutput:
    if evaluation:
        raise ValueError("SWE debug rollout currently implements training rollout only.")

    fixed_seed = int(os.environ.get("ROCK_SWE_BATCH_SEED", "42"))
    random.seed(fixed_seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(fixed_seed)
    except Exception:
        pass

    target_group_count = max(1, int(getattr(args, "rollout_batch_size", 1)))
    over_sampling_batch_size = max(
        target_group_count,
        int(os.environ.get("ROCK_SWE_OVER_SAMPLING_BATCH_SIZE", str(target_group_count))),
    )
    groups = data_buffer.get_samples(over_sampling_batch_size)
    start_time = time.time()
    processed_groups: list[list[Sample]] = []
    sample_idx_offset = 0
    default_group_concurrency = target_group_count
    default_sample_concurrency = max(
        1,
        target_group_count * int(getattr(args, "n_samples_per_prompt", 1)),
    )
    max_sample_concurrency = int(os.environ.get("ROCK_SWE_MAX_SAMPLE_CONCURRENCY", "0"))
    group_concurrency = max(
        1,
        int(os.environ.get("ROCK_SWE_GROUP_CONCURRENCY", default_group_concurrency)),
    )
    sample_concurrency = max(
        1,
        int(os.environ.get("ROCK_SWE_SAMPLE_CONCURRENCY", default_sample_concurrency)),
    )
    if max_sample_concurrency > 0 and sample_concurrency > max_sample_concurrency:
        sample_concurrency = max_sample_concurrency
    total_groups = len(groups)
    total_samples = sum(len(g) for g in groups)
    print(
        f"[rollout] rollout_id={rollout_id} target_groups={target_group_count} submitted_groups={total_groups} "
        f"total_samples={total_samples} group_concurrency={group_concurrency} "
        f"sample_concurrency={sample_concurrency}",
        file=sys.stderr,
    )
    sample_semaphore = asyncio.Semaphore(sample_concurrency)
    log_root = _env_or_arg(args, "ROCK_SWE_LOG_ROOT", "swe_log_root", None)
    prepare_batch_log_dir(log_root)

    async def _run_group_with_order(
        group_idx: int,
        group: list[Sample],
        group_sample_idx_offset: int,
    ) -> tuple[int, list[Sample]]:
        result = await _run_group( 
            args,
            group,
            evaluation=evaluation,
            rollout_id=rollout_id,
            sample_idx_offset=group_sample_idx_offset,
            sample_semaphore=sample_semaphore,
        )
        return group_idx, result

    pending_tasks: set[asyncio.Task[tuple[int, list[Sample]]]] = set()
    completed_groups: dict[int, list[Sample]] = {}
    completed_group_order: list[int] = []

    for group_idx, group in enumerate(groups):
        if len(completed_groups) >= target_group_count:
            break
        while len(pending_tasks) >= group_concurrency:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                finished_group_idx, finished_group = await task
                if len(completed_groups) < target_group_count:
                    completed_groups[finished_group_idx] = finished_group
                    completed_group_order.append(finished_group_idx)
            if len(completed_groups) >= target_group_count:
                break
        if len(completed_groups) >= target_group_count:
            break
        pending_tasks.add(
            asyncio.create_task(
                _run_group_with_order(
                    group_idx,
                    group,
                    sample_idx_offset,
                )
            )
        )
        sample_idx_offset += len(group)

    try:
        while pending_tasks and len(completed_groups) < target_group_count:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                finished_group_idx, finished_group = await task
                if len(completed_groups) < target_group_count:
                    completed_groups[finished_group_idx] = finished_group
                    completed_group_order.append(finished_group_idx)
    finally:
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    selected_group_indices = completed_group_order[:target_group_count]
    processed_groups = [completed_groups[group_idx] for group_idx in selected_group_indices]

    flat_samples = [sample for group in processed_groups for sample in group]
    success_count = sum(1 for sample in flat_samples if float(sample.reward or 0.0) > 0.0)
    mean_turns = 0.0
    if flat_samples:
        mean_turns = sum(len((sample.metadata or {}).get("trajectory", [])) for sample in flat_samples) / len(flat_samples)

    metrics = {
        "swe/pass_rate": success_count / max(1, len(flat_samples)),
        "swe/success_count": success_count,
        "swe/sample_count": len(flat_samples),
        "swe/mean_turns": mean_turns,
        "swe/target_group_count": target_group_count,
        "swe/submitted_group_count": total_groups,
        "swe/completed_group_count": len(processed_groups),
        "swe/cancelled_group_count": max(0, total_groups - len(processed_groups)),
        "swe/group_concurrency": group_concurrency,
        "swe/sample_concurrency": sample_concurrency,
        "swe/rollout_time_sec": time.time() - start_time,
    }
    return RolloutFnTrainOutput(samples=processed_groups, metrics=metrics)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    return run(generate_rollout_async(args, rollout_id, data_buffer, evaluation=evaluation))
