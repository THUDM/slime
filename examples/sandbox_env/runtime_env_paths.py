from __future__ import annotations

from typing import Any


AGENT_RUNTIME_ROOT = "/home/user/.rock/preinstalled/agent-runtime"
MODEL_SERVICE_RUNTIME_ROOT = "/home/user/.rock/preinstalled/model-service-runtime"


def runtime_install_root(kind: str) -> str:
    if kind == "agent":
        return AGENT_RUNTIME_ROOT
    if kind == "model_service":
        return MODEL_SERVICE_RUNTIME_ROOT
    raise ValueError(f"Unsupported runtime kind: {kind}")


def runtime_bin_dir(kind: str) -> str:
    return f"{runtime_install_root(kind)}/runtime-env/bin"


def runtime_path_vars() -> dict[str, str]:
    agent_bin = runtime_bin_dir("agent")
    model_bin = runtime_bin_dir("model_service")
    return {
        "AGENT_RUNTIME_ROOT": AGENT_RUNTIME_ROOT,
        "MODEL_SERVICE_RUNTIME_ROOT": MODEL_SERVICE_RUNTIME_ROOT,
        "AGENT_RUNTIME_BIN_DIR": agent_bin,
        "MODEL_SERVICE_RUNTIME_BIN_DIR": model_bin,
        "AGENT_RUNTIME_NODE": f"{agent_bin}/node",
        "AGENT_RUNTIME_NPM": f"{agent_bin}/npm",
        "AGENT_RUNTIME_QWEN": f"{agent_bin}/qwen",
        "AGENT_RUNTIME_IFLOW": f"{agent_bin}/iflow",
        "MODEL_SERVICE_RUNTIME_PYTHON": f"{model_bin}/python",
        "MODEL_SERVICE_RUNTIME_PIP": f"{model_bin}/pip",
        "MODEL_SERVICE_RUNTIME_ROCK": f"{model_bin}/rock",
    }


def expand_runtime_placeholders(value: Any) -> Any:
    replacements = runtime_path_vars()
    if isinstance(value, str):
        text = value
        for key, replacement in replacements.items():
            text = text.replace(f"${{{key}}}", replacement)
        return text
    if isinstance(value, list):
        return [expand_runtime_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {k: expand_runtime_placeholders(v) for k, v in value.items()}
    return value
