"""Sandbox acquisition: configuration, build, retryable start, agent install.

The functions here own everything from "I have rebench metadata and a few
knobs" to "I have a started sandbox with the agent installed".  They do not
touch rollout-loop state and take only primitives + the metadata dict + the
sandbox object — the rollout side decodes args/env vars and passes the values
in.  Keeping the dependency direction one-way (rollout → factory) avoids a
circular import with ``rock_swe_rollout``.
"""
from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


INSPIRE_SPEC_CPU_HINTS: dict[str, int] = {
    "G_C1": 1,
    "G_C2": 2,
    "G_C4": 4,
}

DEFAULT_SLIME_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _ensure_workspace_root_on_path(workspace_root: str) -> None:
    workspace_root_path = str(Path(workspace_root).resolve())
    if workspace_root_path not in sys.path:
        sys.path.insert(0, workspace_root_path)


# ---------------------------------------------------------------------------
# Backend / spec / base url resolution
# ---------------------------------------------------------------------------


def normalize_backend(value: str | None) -> str:
    backend = str(value or "rock").strip().lower()
    if backend not in {"rock", "inspire"}:
        raise ValueError(f"Unsupported sandbox backend: {backend}")
    return backend


def _normalize_inspire_spec_name(value: str) -> str:
    spec_name = str(value or "").strip().upper()
    if spec_name not in INSPIRE_SPEC_CPU_HINTS:
        supported = ", ".join(sorted(INSPIRE_SPEC_CPU_HINTS))
        raise ValueError(f"Unsupported ROCK_INSPIRE_SPEC={value!r}; expected one of: {supported}")
    return spec_name


def resolve_inspire_spec(*, explicit_spec: str = "", cpus: int = 2) -> str:
    """Pick an inspire spec name from an explicit override, or fall back to a CPU hint."""
    explicit = str(explicit_spec or "").strip()
    if explicit:
        return _normalize_inspire_spec_name(explicit)
    if cpus <= 1:
        return "G_C1"
    if cpus <= 2:
        return "G_C2"
    return "G_C4"


def _can_curl_base_url(base_url: str, timeout_seconds: int = 10) -> bool:
    result = subprocess.run(
        [
            "curl",
            "-fsS",
            "-L",
            "--max-time",
            str(timeout_seconds),
            base_url.rstrip("/") + "/",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _probe_rock_base_url(base_url: str, retry_times: int = 10, retry_interval_seconds: int = 1) -> str:
    normalized_url = base_url.rstrip("/")
    for attempt in range(1, retry_times + 1):
        if _can_curl_base_url(normalized_url):
            return normalized_url
        if attempt < retry_times:
            print(
                f"[WARN] curl base url failed for {normalized_url}; retry {attempt}/{retry_times} in "
                f"{retry_interval_seconds}s",
                file=sys.stderr,
            )
            time.sleep(retry_interval_seconds)
    raise RuntimeError(
        f"ROCK base url is not reachable after {retry_times} attempts: {normalized_url}"
    )


def resolve_rock_base_url(*, base_url: str, backend: str) -> str:
    """Probe the ROCK base url when backend == 'rock'; return '' otherwise."""
    base_url = str(base_url or "").strip()
    if backend != "rock":
        return ""
    if not base_url:
        raise ValueError("ROCK_SWE_BASE_URL must be set when ROCK_SWE_SANDBOX_BACKEND=rock")
    return _probe_rock_base_url(base_url)


# ---------------------------------------------------------------------------
# Metadata-driven resolvers
# ---------------------------------------------------------------------------


def resolve_inspire_image(metadata: dict[str, Any]) -> str:
    image = str(metadata.get("inspire_image") or metadata.get("local_image_name") or "").strip()
    if image:
        return image
    raise RuntimeError("Unable to resolve inspire image: missing inspire_image/local_image_name")


def resolve_sandbox_default_user(metadata: dict[str, Any]) -> str:
    user = str(metadata.get("docker_image_default_user") or "").strip()
    if not user:
        instance_id = str(metadata.get("instance_id") or "task")
        image_name = str(metadata.get("image_name") or metadata.get("local_image_name") or "").strip()
        raise RuntimeError(
            "rebench metadata missing docker_image_default_user "
            f"(instance_id={instance_id}, image_name={image_name})"
        )
    return user


def _resolve_sandbox_image_env(metadata: dict[str, Any]) -> dict[str, str]:
    raw_env = metadata.get("docker_image_env")
    if not isinstance(raw_env, dict):
        return {}
    return {str(key): str(value) for key, value in raw_env.items()}


# ---------------------------------------------------------------------------
# Sandbox-side default wrapping (create_session / arun)
# ---------------------------------------------------------------------------


def _merge_env_supplementally(base_env: dict[str, str], extra_env: dict[str, str]) -> dict[str, str]:
    merged = {str(key): str(value) for key, value in dict(base_env or {}).items()}
    for key, value in dict(extra_env or {}).items():
        key = str(key)
        value = str(value)
        if key == "PATH":
            current = str(merged.get("PATH") or "").strip()
            if not current:
                merged["PATH"] = value
                continue
            extra_parts = [part for part in value.split(":") if part]
            current_parts = [part for part in current.split(":") if part]
            for part in extra_parts:
                if part not in current_parts:
                    current_parts.append(part)
            merged["PATH"] = ":".join(current_parts)
            continue
        merged.setdefault(key, value)
    return merged


def _wrap_command_with_env_exports(cmd: str, env: dict[str, str]) -> str:
    env = {str(key): str(value) for key, value in dict(env or {}).items() if str(key)}
    if not env:
        return cmd
    exports = " && ".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
    return f"{exports} && {cmd}"


def configure_sandbox_defaults(
    sandbox,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Patch sandbox.create_session / sandbox.arun to inject metadata-derived defaults.

    Idempotent — repeated calls on the same sandbox instance are no-ops.
    Returns ``(default_user, image_env)``.
    """
    default_user = resolve_sandbox_default_user(metadata)
    image_env = _resolve_sandbox_image_env(metadata)
    if getattr(sandbox, "_sandbox_defaults_configured", False):
        return default_user, image_env

    original_create_session = sandbox.create_session
    original_arun = sandbox.arun

    async def _create_session_with_sandbox_defaults(request):
        if getattr(request, "remote_user", None) in (None, ""):
            request = request.model_copy(update={"remote_user": default_user})
        if getattr(request, "env_enable", False):
            merged_env = _merge_env_supplementally(dict(getattr(request, "env", {}) or {}), image_env)
            request = request.model_copy(update={"env": merged_env})
        return await original_create_session(request)

    async def _arun_with_sandbox_defaults(
        cmd: str,
        session: str | None = None,
        wait_timeout: int = 300,
        wait_interval: int = 10,
        mode: str = "normal",
        response_limited_bytes_in_nohup: int | None = None,
        ignore_output: bool = False,
        output_file: str | None = None,
        user: str | None = None,
    ):
        effective_user = user or default_user
        effective_cmd = cmd
        if session is None and image_env:
            effective_cmd = _wrap_command_with_env_exports(cmd, image_env)
        return await original_arun(
            cmd=effective_cmd,
            session=session,
            wait_timeout=wait_timeout,
            wait_interval=wait_interval,
            mode=mode,
            response_limited_bytes_in_nohup=response_limited_bytes_in_nohup,
            ignore_output=ignore_output,
            output_file=output_file,
            user=effective_user,
        )

    sandbox.create_session = _create_session_with_sandbox_defaults
    sandbox.arun = _arun_with_sandbox_defaults
    sandbox._sandbox_defaults_configured = True
    sandbox._sandbox_default_user = default_user
    sandbox._sandbox_image_env = image_env
    return default_user, image_env


# ---------------------------------------------------------------------------
# Build / start (with retry)
# ---------------------------------------------------------------------------


def _build_sandbox(
    *,
    metadata: dict[str, Any],
    rock_base_url: str,
    startup_timeout: int,
    memory: str,
    cpus: int,
    keep_containers: bool,
    sandbox_backend: str,
    inspire_spec: str,
    live_log_path: str | None,
    prefer_image: bool = False,
):
    from rock.sdk.sandbox.config import SandboxConfig

    template_name = "" if prefer_image else str(metadata.get("inspire_template") or "").strip()
    image_runtime_env = metadata.get("docker_image_env")
    if not isinstance(image_runtime_env, dict):
        image_runtime_env = {}
    config = SandboxConfig(
        image=metadata["local_image_name"],
        repo=metadata.get("repo"),
        base_commit=metadata.get("base_commit"),
        dockerfile=metadata.get("dockerfile"),
        base_url=rock_base_url,
        startup_timeout=startup_timeout,
        memory=memory,
        cpus=cpus,
        remove_container=not keep_containers,
    )
    if sandbox_backend == "inspire":
        _ensure_workspace_root_on_path(DEFAULT_SLIME_REPO_ROOT)
        from examples.sandbox_env.rock_inspire_adapter import InspireRockSandbox

        if template_name:
            sandbox = InspireRockSandbox(
                config,
                inspire_spec=inspire_spec,
                template_name=template_name,
                image_runtime_env=image_runtime_env,
            )
            sandbox.set_live_log_path(live_log_path)
            return sandbox
        config.image = resolve_inspire_image(metadata)
        sandbox = InspireRockSandbox(
            config,
            inspire_spec=inspire_spec,
            image_runtime_env=image_runtime_env,
        )
        sandbox.set_live_log_path(live_log_path)
        return sandbox

    from rock.sdk.sandbox.client import Sandbox

    return Sandbox(config)


# Errors that signal a definitively bad request (bad url, missing image, auth
# failure, ...) — there is no point retrying these.  Anything else is treated
# as transient and retried.
_NON_RETRYABLE_START_MARKERS = (
    "file not found",
    "no such file or directory",
    "unsupported config file format",
    "working_dir does not exist",
    "working_dir is not a directory",
    "validation error",
    "invalid url",
    "invalid base url",
    "dockerfile parse error",
    "manifest unknown",
    "pull access denied",
    "image not found",
    "repository does not exist",
    "permission denied",
    "authentication failed",
    "authorization failed",
    "forbidden",
    "unauthorized",
    "400 bad request",
    "404 not found",
)


def _is_retryable_sandbox_start_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return not any(marker in message for marker in _NON_RETRYABLE_START_MARKERS)


async def start_sandbox_with_retry(
    *,
    metadata: dict[str, Any],
    rock_base_url: str,
    startup_timeout: int,
    memory: str,
    cpus: int,
    keep_containers: bool,
    retry_times: int,
    retry_interval_seconds: float,
    sandbox_backend: str,
    inspire_spec: str,
    live_log_path: str | None,
    prefer_image: bool = False,
):
    last_exc: Exception | None = None

    for attempt in range(1, retry_times + 1):
        sandbox = _build_sandbox(
            metadata=metadata,
            rock_base_url=rock_base_url,
            startup_timeout=startup_timeout,
            memory=memory,
            cpus=cpus,
            keep_containers=keep_containers,
            sandbox_backend=sandbox_backend,
            inspire_spec=inspire_spec,
            live_log_path=live_log_path,
            prefer_image=prefer_image,
        )
        try:
            await sandbox.start()
            return sandbox
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_sandbox_start_error(exc)
            if retryable and attempt < retry_times:
                print(
                    f"[WARN] sandbox.start failed for image={metadata.get('local_image_name')} "
                    f"(attempt {attempt}/{retry_times}): {str(exc).strip() or exc.__class__.__name__}; "
                    f"retry in {retry_interval_seconds}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(retry_interval_seconds)
                continue
            if not retryable:
                break

    error_detail = str(last_exc).strip() if last_exc is not None else ""
    if not error_detail:
        error_detail = last_exc.__class__.__name__ if last_exc is not None else "unknown"
    raise RuntimeError(
        "sandbox.start failed "
        f"(image={metadata.get('local_image_name')}, backend={sandbox_backend}, "
        f"base_url={rock_base_url}, inspire_spec={inspire_spec}) "
        f"after {retry_times} attempts: {error_detail}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Agent install (with retry)
# ---------------------------------------------------------------------------


def _is_retryable_agent_install_error(exc: Exception) -> bool:
    error_text = str(exc)
    return (
        "create session failed" in error_text
        or "Upstream server is not reachable" in error_text
        or "Rocklet at" in error_text
    )


async def install_agent_with_retry(
    sandbox,
    *,
    config_path: str,
    retry_times: int,
    retry_interval_seconds: float,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, retry_times + 1):
        try:
            await sandbox.agent.install(config=config_path)
            return
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_agent_install_error(exc)
            if (not retryable) or attempt >= retry_times:
                raise
            print(
                "[WARN] sandbox.agent.install failed "
                f"(sandbox_id={getattr(sandbox, 'sandbox_id', None)}) "
                f"attempt {attempt}/{retry_times}: {exc}; "
                f"retry in {retry_interval_seconds}s",
                file=sys.stderr,
            )
            await asyncio.sleep(retry_interval_seconds)

    if last_exc is not None:
        raise last_exc
