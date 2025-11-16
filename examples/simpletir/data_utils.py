from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


def normalize_prompt(value: Any) -> List[Dict[str, str]]:
    """Convert any supported representation into a chat-style message list."""
    if value is None:
        return [{"role": "user", "content": ""}]

    prompt = value
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    elif isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8", errors="ignore")

    if isinstance(prompt, str):
        stripped = prompt.strip()
        if not stripped:
            return [{"role": "user", "content": ""}]
        try:
            prompt = json.loads(stripped)
        except json.JSONDecodeError:
            return [{"role": "user", "content": stripped}]

    if isinstance(prompt, dict):
        prompt = [prompt]

    if isinstance(prompt, (list, tuple)):
        messages: List[Dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role") or "user")
                content = str(item.get("content") or "")
            else:
                role = "user"
                content = "" if item is None else str(item)
            messages.append({"role": role, "content": content})
        return messages or [{"role": "user", "content": ""}]

    return [{"role": "user", "content": str(prompt)}]


def normalize_reward_model(value: Any) -> Dict[str, Any]:
    """Ensure reward model metadata is a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"ground_truth": value}
        return parsed if isinstance(parsed, dict) else {"ground_truth": value}
    return {}


def ensure_metadata_dict(value: Any) -> Dict[str, Any]:
    """Normalize optional metadata payloads to plain dicts."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
