"""OSWorld environment adapter for slime's multi-turn VLM training pipeline.

It provides:
- Environment construction from task configs
- Screenshot and a11y tree observation formatting
- Action parsing from model responses
- Hybrid reward signal components

Usage:
    Set rollout_interaction_env_path to "examples.osworld.env" in
    your training config YAML.
"""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import logging
import os
import re
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageChops, ImageStat

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# VM pool for parallel rollouts (round-robin assignment)
_vm_pool_counter = itertools.count()
_vm_pool_lock = threading.Lock()


def _get_next_server_url() -> str:
    """Get next server URL in round-robin fashion.

    Supports comma-separated URLs in OSWORLD_SERVER_URL:
      - Single: http://172.17.0.1:8100 (backward compatible)
      - Multiple: http://172.17.0.1:8100,http://172.17.0.1:8101
    """
    urls_str = os.environ.get("OSWORLD_SERVER_URL", "http://localhost:8100")
    urls = [u.strip() for u in urls_str.split(",") if u.strip()]
    if not urls:
        raise ValueError(
            "OSWORLD_SERVER_URL is empty. Set to OSWorld server address, e.g.:\n"
            "  export OSWORLD_SERVER_URL=http://172.17.0.1:8100"
        )
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "OSWORLD_SERVER_URL must include scheme and host, e.g.:\n"
                "  export OSWORLD_SERVER_URL=http://172.17.0.1:8100"
            )
    with _vm_pool_lock:
        idx = next(_vm_pool_counter) % len(urls)
    return urls[idx]


QWEN3_MAX_PIXELS = 1003520  # ~1M pixels (from qwen_vl_utils)
QWEN3_MIN_PIXELS = 256 * 256


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = QWEN3_MIN_PIXELS,
    max_pixels: int = QWEN3_MAX_PIXELS,
) -> tuple[int, int]:
    """Resize dimensions for Qwen VLM compatibility (divisible by factor, within pixel limits)."""
    if height < factor or width < factor:
        raise ValueError(f"Dimensions ({height}, {width}) too small for factor {factor}")

    # Scale down if exceeds max_pixels (critical for OSWorld 1920x1080 screenshots)
    if height * width > max_pixels:
        scale = (max_pixels / (height * width)) ** 0.5
        height = int(height * scale)
        width = int(width * scale)

    # Scale up if below min_pixels
    if height * width < min_pixels:
        scale = (min_pixels / (height * width)) ** 0.5
        height = int(height * scale)
        width = int(width * scale)

    # Make divisible by factor
    height = max(factor, (height // factor) * factor)
    width = max(factor, (width // factor) * factor)

    return height, width


def resize_screenshot_for_vlm(img: Image.Image) -> Image.Image:
    """Resize screenshot to fit within VLM pixel limits."""
    orig_width, orig_height = img.size
    orig_pixels = orig_width * orig_height

    if orig_pixels <= QWEN3_MAX_PIXELS:
        return img  # No resize needed

    new_height, new_width = smart_resize(orig_height, orig_width)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Process-level signal helpers
SCREEN_DIFF_THRESHOLD = float(os.environ.get("OSWORLD_SCREEN_DIFF_THRESHOLD", "0.005"))


def _bytes_to_pil(img: Image.Image | bytes | None) -> Image.Image | None:
    """Convert bytes to PIL Image if needed."""
    if img is None:
        return None
    if isinstance(img, bytes):
        try:
            return Image.open(io.BytesIO(img)).convert("RGB")
        except Exception:
            return None
    return img


def _compute_screen_diff(prev_img: Image.Image | bytes | None, curr_img: Image.Image | bytes | None) -> float:
    """Compute normalized screen diff in [0, 1] using downsampled grayscale."""
    prev_pil = _bytes_to_pil(prev_img)
    curr_pil = _bytes_to_pil(curr_img)
    if prev_pil is None or curr_pil is None:
        return 0.0

    prev_small = prev_pil.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    curr_small = curr_pil.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    diff = ImageChops.difference(prev_small, curr_small)
    stat = ImageStat.Stat(diff)
    mean = stat.mean[0] / 255.0
    return float(min(1.0, max(0.0, mean)))


def _hash_screenshot(img: Image.Image | bytes | None) -> str | None:
    """Hash screenshot for change detection. Handles both PIL Image and raw bytes."""
    if img is None:
        return None
    if isinstance(img, bytes):
        return hashlib.sha1(img).hexdigest()[:12]
    return hashlib.sha1(img.tobytes()).hexdigest()[:12]


# Supported PyAutoGUI actions (includes camelCase and lowercase variants)
VALID_PYAUTOGUI_ACTIONS = {
    "click",
    "doubleClick",
    "doubleclick",
    "rightClick",
    "rightclick",
    "tripleClick",
    "tripleclick",
    "write",
    "typewrite",
    "press",
    "hotkey",
    "scroll",
    "move",
    "moveTo",
    "moveto",
    "drag",
    "dragTo",
    "dragto",
    "screenshot",
}

# Normalize VLM action names to pyautogui equivalents
ACTION_NAME_MAPPING = {
    "type": "write",
    "double_click": "doubleClick",
    "right_click": "rightClick",
    "tripleclick": "tripleClick",
    "triple_click": "tripleClick",
}

TERMINAL_ACTIONS = {"WAIT", "DONE", "FAIL"}


@dataclass
class OSWorldEnvConfig:
    """Configuration for OSWorld environment wrapper."""

    # DesktopEnv settings
    provider_name: str = "docker"  # docker, vmware, aws
    os_type: str = "Ubuntu"
    action_space: str = "pyautogui"  # pyautogui, computer_13, claude_computer_use
    screen_size: tuple[int, int] = (1920, 1080)
    headless: bool = True
    require_a11y_tree: bool = True

    max_steps: int = 8  # Based on trajectory data max
    image_placeholder: str = "<image>"
    include_a11y_in_obs: bool = True
    a11y_max_length: int = int(os.environ.get("OSWORLD_A11Y_MAX_LENGTH", "4096"))

    reuse_env: bool = False
    a11y_mode: str = "full"  # full | off | every_n
    a11y_every_n: int = 2

    track_a11y_grounding: bool = True

    log_dir: str | None = None
    log_prefix: str = "osworld_step"


class OSWorldEnvWrapper:
    """Adapts OSWorld DesktopEnv to slime's pluggable environment interface."""

    def __init__(self, config: OSWorldEnvConfig, task_config: dict):
        self.config = config
        self.task_config = task_config
        self.desktop_env = None
        self.step_count = 0
        self.step_signals: list[dict] = []  # Per-step reward signal components
        self._last_a11y_tree: str = ""
        self._last_screenshot: Image.Image | None = None
        self._last_screen_hash: str | None = None
        self._last_terminal: str = ""
        self._render_count = 0

        # Lazy import - OSWorld may not be installed in all environments
        self._desktop_env_cls = None

    def _get_desktop_env_class(self):
        """Get DesktopEnv class (HTTP server or local)."""
        if self._desktop_env_cls is None:
            osworld_server_url = os.environ.get("OSWORLD_SERVER_URL")
            if osworld_server_url:
                self._desktop_env_cls = HTTPRemoteDesktopEnv
            else:
                try:
                    from desktop_env.desktop_env import DesktopEnv

                    self._desktop_env_cls = DesktopEnv
                except ImportError as e:
                    raise ImportError(
                        "OSWorld desktop_env not installed. Either:\n"
                        "  1. Set OSWORLD_SERVER_URL to use HTTP bridge, or\n"
                        "  2. Install desktop_env: pip install desktop-env"
                    ) from e
        return self._desktop_env_cls

    def reset(self, task_config_override: dict | None = None) -> tuple[dict, dict]:
        """Initialize environment and return first observation."""
        self.step_count = 0
        self.step_signals = []
        self._render_count = 0

        if task_config_override is not None:
            self.task_config = task_config_override

        if self.desktop_env is None or not self.config.reuse_env:
            if self.desktop_env is not None:
                try:
                    self.desktop_env.close()
                except Exception:
                    pass

            DesktopEnvCls = self._get_desktop_env_class()
            self.desktop_env = DesktopEnvCls(
                provider_name=self.config.provider_name,
                os_type=self.config.os_type,
                action_space=self.config.action_space,
                screen_size=self.config.screen_size,
                headless=self.config.headless,
                require_a11y_tree=self.config.require_a11y_tree,
            )
        else:
            try:
                self.desktop_env.is_environment_used = False
            except Exception:
                pass

        obs = self.desktop_env.reset(task_config=self.task_config)
        self._last_a11y_tree = obs.get("accessibility_tree", "")
        self._last_screenshot = obs.get("screenshot")
        self._last_screen_hash = _hash_screenshot(self._last_screenshot)
        self._last_terminal = obs.get("terminal", "")

        formatted_obs = self._format_raw_obs(obs)
        info = {
            "task_id": self.task_config.get("id", "unknown"),
            "domain": self.task_config.get("domain", "unknown"),
            "instruction": obs.get("instruction", self.task_config.get("instruction", "")),
        }

        return formatted_obs, info

    def step(self, action_str: str) -> tuple[dict, bool, dict]:
        """Execute action and return next observation."""
        if self.desktop_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Parse action from model response
        parsed = self._parse_action(action_str)
        action_type = parsed.get("action_type", "unknown")
        action_parsed = parsed.get("parsed", False)
        target_element = parsed.get("target_element")
        action_jittered = False

        # Handle terminal actions (DONE, FAIL, WAIT)
        if action_type in TERMINAL_ACTIONS:
            obs = self.desktop_env._get_observation() if hasattr(self.desktop_env, "_get_observation") else {}
            action_executed = True
            action_error = None

            if action_type == "DONE":
                done = True
                reward = 0.0  # Actual reward computed via evaluate()
            elif action_type == "FAIL":
                done = True
                reward = 0.0
            else:  # WAIT
                done = False
                reward = 0.0
        else:
            # Execute pyautogui action in environment
            try:
                raw_action = parsed.get("raw_action", action_str)
                action_jittered = False
                obs, reward, done, env_info = self.desktop_env.step(raw_action, pause=2)
                action_executed = True
                action_error = None
            except Exception as e:
                logger.warning(f"Action execution failed: {e}")
                obs = self.desktop_env._get_observation() if hasattr(self.desktop_env, "_get_observation") else {}
                reward = 0.0
                done = False
                action_executed = False
                action_error = str(e)
                action_jittered = False

        self.step_count += 1
        current_a11y = obs.get("accessibility_tree", "")

        # Compute a11y grounding signal
        a11y_grounded = False
        if self.config.track_a11y_grounding and target_element and self._last_a11y_tree:
            a11y_grounded = target_element.lower() in self._last_a11y_tree.lower()

        # Compute process-level signals
        current_screen = obs.get("screenshot")
        screen_diff = _compute_screen_diff(self._last_screenshot, current_screen)
        screen_changed = 1.0 if screen_diff >= SCREEN_DIFF_THRESHOLD else 0.0
        screen_hash = _hash_screenshot(current_screen)
        screen_hash_changed = 1.0 if (screen_hash and screen_hash != self._last_screen_hash) else 0.0

        terminal = obs.get("terminal", "")
        terminal_changed = 1.0 if terminal != self._last_terminal else 0.0

        a11y_delta = 0.0
        if self._last_a11y_tree:
            prev_len = len(self._last_a11y_tree)
            curr_len = len(current_a11y)
            denom = max(1, prev_len)
            a11y_delta = min(1.0, abs(curr_len - prev_len) / denom)
        a11y_changed = 1.0 if current_a11y != self._last_a11y_tree else 0.0

        # Record step signals for reward computation
        step_signal = {
            "step": self.step_count,
            "action_type": action_type,
            "action_parsed": action_parsed,
            "action_executed": action_executed,
            "action_error": action_error,
            "target_element": target_element,
            "action_jittered": action_jittered,
            "fallback_used": parsed.get("fallback_used", False),
            "coordinate": parsed.get("coordinate"),  # For repetition penalty
            "a11y_grounded": a11y_grounded,
            "screen_diff": screen_diff,
            "screen_changed": screen_changed,
            "screen_hash": screen_hash,
            "screen_hash_changed": screen_hash_changed,
            "a11y_delta": a11y_delta,
            "a11y_changed": a11y_changed,
            "terminal_changed": terminal_changed,
            "env_reward": reward,
        }
        self.step_signals.append(step_signal)

        if current_screen is not None:
            self._last_screenshot = current_screen
            self._last_screen_hash = screen_hash
        self._last_terminal = terminal
        self._last_a11y_tree = current_a11y

        # Check termination
        if self.step_count >= self.config.max_steps:
            done = True

        # Add action execution feedback to observation
        if action_executed:
            obs["action_result"] = f"Action '{action_type}' executed successfully."
        elif action_error:
            obs["action_result"] = f"Action failed: {action_error}"

        formatted_obs = self._format_raw_obs(obs)
        info = {
            "raw_action": action_str,
            "parsed_action": parsed,
            "step_signal": step_signal,
            "obs_image_path": formatted_obs.get("image_path"),
            "obs_a11y_tree": formatted_obs.get("a11y_tree"),
            "obs_text": formatted_obs.get("obs_str"),
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": action_parsed,
                    "action_is_effective": action_executed,
                    "a11y_grounded": a11y_grounded,
                },
                "traj_metrics": {"success": False},  # Updated in finalize_episode
            },
        }

        return formatted_obs, done, info

    def _format_raw_obs(self, obs: dict) -> dict:
        """Convert OSWorld observation to multimodal format."""
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, bytes):
            try:
                screenshot = Image.open(io.BytesIO(screenshot)).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to decode screenshot bytes: {e}")
                screenshot = None
        if screenshot is None:
            # Create placeholder image if no screenshot available
            screenshot = Image.new("RGB", self.config.screen_size, color=(0, 0, 0))

        screenshot = resize_screenshot_for_vlm(screenshot)

        obs_parts = []

        domain = self.task_config.get("domain", "")
        if domain:
            domain_hint = {
                "chrome": "This is a Chrome/browser task. Focus on the Chrome application.",
                "os": "This is an OS/system settings task. Use Ubuntu Settings app.",
                "libreoffice_calc": "This is a LibreOffice Calc (spreadsheet) task.",
                "libreoffice_writer": "This is a LibreOffice Writer (document) task.",
                "gimp": "This is a GIMP (image editing) task.",
                "vlc": "This is a VLC (media player) task.",
                "vscode": "This is a VS Code (code editor) task.",
            }.get(domain.lower(), f"Task domain: {domain}")
            obs_parts.append(f"[Domain]\n{domain_hint}")

        instruction = obs.get("instruction", "")
        if instruction:
            obs_parts.append(f"[Task]\n{instruction}")

        a11y_tree_text = ""
        if self.config.include_a11y_in_obs and self.config.a11y_mode != "off":
            a11y_tree = obs.get("accessibility_tree", "")
            if a11y_tree:
                if self.config.a11y_mode == "every_n":
                    if self.step_count != 0 and self.step_count % max(1, self.config.a11y_every_n) != 0:
                        a11y_tree = ""
                if a11y_tree:
                    # Truncate if too long
                    if len(a11y_tree) > self.config.a11y_max_length:
                        a11y_tree = a11y_tree[: self.config.a11y_max_length] + "\n... (truncated)"
                    obs_parts.append(f"[Accessibility Tree]\n{a11y_tree}")
                    a11y_tree_text = a11y_tree

        terminal = obs.get("terminal", "")
        if terminal:
            obs_parts.append(f"[Terminal Output]\n{terminal}")

        last_action_result = obs.get("action_result", "")
        if last_action_result:
            obs_parts.append(f"[Last Action Result]\n{last_action_result}")

        obs_text = "\n\n".join(obs_parts) if obs_parts else "(See screenshot)"

        # Save screenshot if logging enabled
        image_path = None
        if self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)
            img_path = os.path.join(self.config.log_dir, f"{self.config.log_prefix}_{self._render_count}.png")
            try:
                screenshot.save(img_path)
                image_path = img_path
            except Exception as e:
                logger.warning(f"Failed to save screenshot: {e}")
            self._render_count += 1

        return {
            "obs_str": obs_text,
            "multi_modal_data": {self.config.image_placeholder: [screenshot]},
            "image_path": image_path,
            "a11y_tree": a11y_tree_text,
        }

    def _parse_action(self, action_str: str) -> dict:
        """Parse action from model response.

        Supports formats:
        - <tool_call>{"name":"computer_use","arguments":{...}}</tool_call> (JSON - PRIMARY)
        - <answer>pyautogui.click(x, y)</answer>
        - <action>pyautogui.type("text")</action>
        - Raw pyautogui commands
        - Terminal actions: WAIT, DONE, FAIL
        """

        def translate_json_to_pyautogui(json_str: str) -> str | None:
            """Translate JSON tool call to pyautogui command.

            Expected format from SFT data:
            {"name":"computer_use","arguments":{"action":"left_click","coordinate":[x,y]}}
            {"name":"computer_use","arguments":{"action":"type","text":"hello"}}
            {"name":"computer_use","arguments":{"action":"key","keys":["ctrl","s"]}}

            Returns pyautogui command string or None if parse fails.
            """
            try:
                data = json.loads(json_str)
                args = data.get("arguments", {})
                action = args.get("action", "").lower()

                # Click actions
                if action in ("left_click", "click"):
                    coord = args.get("coordinate", [0, 0])
                    return f"pyautogui.click({coord[0]}, {coord[1]})"
                elif action == "right_click":
                    coord = args.get("coordinate", [0, 0])
                    return f"pyautogui.rightClick({coord[0]}, {coord[1]})"
                elif action == "double_click":
                    coord = args.get("coordinate", [0, 0])
                    return f"pyautogui.doubleClick({coord[0]}, {coord[1]})"

                # Text input
                elif action == "type":
                    text = args.get("text", "")
                    # Escape quotes in text
                    text = text.replace("\\", "\\\\").replace('"', '\\"')
                    return f'pyautogui.write("{text}")'

                # Key presses
                elif action == "key":
                    keys = args.get("keys", [])
                    if len(keys) == 1:
                        return f'pyautogui.press("{keys[0]}")'
                    elif len(keys) > 1:
                        keys_str = ", ".join(f'"{k}"' for k in keys)
                        return f"pyautogui.hotkey({keys_str})"

                # Scroll
                elif action == "scroll":
                    direction = args.get("direction", "down")
                    amount = args.get("amount", 3)
                    scroll_val = -amount if direction == "down" else amount
                    return f"pyautogui.scroll({scroll_val})"

                # Drag
                elif action == "left_click_drag":
                    end = args.get("end_coordinate")
                    if isinstance(end, (list, tuple)) and len(end) == 2:
                        return f"pyautogui.dragTo({end[0]}, {end[1]})"

                # Terminal actions
                elif action == "wait":
                    return "WAIT"
                elif action == "terminate":
                    status = args.get("status", "success")
                    return "DONE" if status == "success" else "FAIL"

                return None
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                return None

        def translate_action_json(json_str: str) -> str | None:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return None
            if isinstance(data, dict) and "arguments" in data:
                return translate_json_to_pyautogui(json_str)
            if isinstance(data, dict) and "action" in data:
                wrapped = {"name": "computer_use", "arguments": data}
                return translate_json_to_pyautogui(json.dumps(wrapped))
            return None

        def extract_pyautogui_call(text: str) -> str | None:
            match = re.search(r"pyautogui\.\w+\s*\(", text, re.IGNORECASE | re.DOTALL)
            if not match:
                return None
            start = match.start()
            open_paren = text.find("(", match.start())
            if open_paren == -1:
                return text[start:].strip()

            depth = 0
            in_str: str | None = None
            escape = False
            for i in range(open_paren, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == in_str:
                        in_str = None
                else:
                    if ch in ("'", '"'):
                        in_str = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1].strip()
            return text[start:].strip()

        result = {
            "raw_action": action_str,
            "parsed": False,
            "action_type": "unknown",
            "target_element": None,
            "fallback_used": False,
            "coordinate": None,
        }

        # Sanitize: strip common VLM artifacts before parsing
        sanitized = action_str
        # Strip stop tokens
        for token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
            sanitized = sanitized.replace(token, "")
        # Strip markdown code fences (keep content)
        sanitized = re.sub(r"```(?:python)?\s*", "", sanitized)
        sanitized = sanitized.strip()

        # Try to extract from tags - prioritize <tool_call> (primary SFT format)
        tag_patterns = [
            r"<tool_call>(.*?)</tool_call>",  # PRIMARY: matches SFT training data
            r"<answer>(.*?)</answer>",  # Fallback: legacy format
            r"<action>(.*?)</action>",
            r"```python\s*(.*?)\s*```",
        ]

        action_content = sanitized
        is_tool_call = False
        for pattern in tag_patterns:
            match = re.search(pattern, sanitized, re.IGNORECASE | re.DOTALL)
            if match:
                action_content = match.group(1).strip()
                is_tool_call = "tool_call" in pattern
                break

        # JSON-first translation: if we extracted from <tool_call>, try JSON parse
        if is_tool_call or action_content.strip().startswith("{"):
            translated = translate_action_json(action_content)
            if translated:
                action_content = translated

        # Check for terminal actions
        action_upper = action_content.strip().upper()
        if action_upper in TERMINAL_ACTIONS:
            result["parsed"] = True
            result["action_type"] = action_upper
            return result

        # Try to extract a pyautogui call from noisy content
        extracted = extract_pyautogui_call(action_content)
        if extracted:
            action_content = extracted

        # Parse pyautogui command
        pyautogui_match = re.match(r"pyautogui\.(\w+)\s*\((.*)\)", action_content, re.IGNORECASE | re.DOTALL)
        if pyautogui_match:
            action_type = pyautogui_match.group(1).lower()
            args_str = pyautogui_match.group(2)

            # Normalize action names (e.g., type -> write)
            original_action_type = action_type
            if action_type in ACTION_NAME_MAPPING:
                action_type = ACTION_NAME_MAPPING[action_type]

            if action_type in VALID_PYAUTOGUI_ACTIONS:
                result["parsed"] = True
                result["action_type"] = action_type

                # Update raw_action with normalized action name if it was mapped
                if original_action_type != action_type:
                    result["raw_action"] = action_content.replace(
                        f"pyautogui.{original_action_type}(", f"pyautogui.{action_type}("
                    )
                else:
                    result["raw_action"] = action_content

                # Extract coordinates for click actions (for repetition penalty)
                if action_type in ("click", "rightclick", "doubleclick"):
                    coord_match = re.search(r"(\d+)\s*,\s*(\d+)", args_str)
                    if coord_match:
                        result["coordinate"] = [int(coord_match.group(1)), int(coord_match.group(2))]

                # Try to extract target element for grounding
                # Look for element descriptions in the action or surrounding context
                element_match = re.search(r'["\']([^"\']+)["\']', args_str)
                if element_match:
                    result["target_element"] = element_match.group(1)

        if not result["parsed"]:
            # Fallback parsing for non-compliant outputs.
            # Executes a best-effort action but does NOT mark format compliance.
            fallback_action = None
            fallback_action_type = "unknown"

            action_line = sanitized
            if action_line.lower().startswith("action:"):
                action_line = action_line.split(":", 1)[1].strip()

            # Try to recover JSON fragments without <tool_call>
            json_start = action_line.find("{")
            if json_start != -1:
                json_end = action_line.rfind("}")
                if json_end == -1:
                    json_candidate = action_line[json_start:] + "}"
                else:
                    json_candidate = action_line[json_start : json_end + 1]
                translated = translate_action_json(json_candidate)
                if translated:
                    fallback_action = translated

            # Wait actions
            if fallback_action is None and "wait" in action_line.lower():
                fallback_action = "WAIT"
                fallback_action_type = "WAIT"

            # Coordinate-based clicks
            if fallback_action is None:
                coord_match = re.search(r"\[(\d+)\s*,\s*(\d+)\]", action_line)
                if coord_match:
                    x, y = int(coord_match.group(1)), int(coord_match.group(2))
                    result["coordinate"] = [x, y]
                    lower = action_line.lower()
                    if "double" in lower:
                        fallback_action = f"pyautogui.doubleClick({x}, {y})"
                        fallback_action_type = "doubleClick"
                    elif "right" in lower:
                        fallback_action = f"pyautogui.rightClick({x}, {y})"
                        fallback_action_type = "rightClick"
                    elif "click" in lower:
                        fallback_action = f"pyautogui.click({x}, {y})"
                        fallback_action_type = "click"
                elif "click" in action_line.lower():
                    # Best-effort fallback when no coordinates are provided.
                    # Use a conservative center-ish click to avoid out-of-bounds.
                    fallback_action = "pyautogui.click(640, 360)"
                    fallback_action_type = "click"
                    result["coordinate"] = [640, 360]

            # Type actions with quoted text
            if fallback_action is None and "type" in action_line.lower():
                text_match = re.search(r"\"([^\"]+)\"", action_line)
                if text_match:
                    txt = text_match.group(1).replace("\\", "\\\\").replace('"', '\\"')
                    fallback_action = f'pyautogui.write("{txt}")'
                    fallback_action_type = "write"

            # Key or hotkey actions
            if fallback_action is None:
                keys = []
                for key in ["ctrl", "alt", "shift", "enter", "tab", "esc", "escape", "up", "down", "left", "right"]:
                    if re.search(rf"\\b{key}\\b", action_line.lower()):
                        keys.append("esc" if key == "escape" else key)
                if keys:
                    if len(keys) == 1:
                        fallback_action = f'pyautogui.press("{keys[0]}")'
                        fallback_action_type = "press"
                    else:
                        keys_str = ", ".join(f'"{k}"' for k in keys)
                        fallback_action = f"pyautogui.hotkey({keys_str})"
                        fallback_action_type = "hotkey"

            # Scroll actions
            if fallback_action is None and "scroll" in action_line.lower():
                direction = -300 if "down" in action_line.lower() else 300
                fallback_action = f"pyautogui.scroll({direction})"
                fallback_action_type = "scroll"

            if fallback_action is not None:
                result["raw_action"] = fallback_action
                result["action_type"] = fallback_action_type
                result["fallback_used"] = True

        return result

    def compute_reward(self) -> float:
        """Compute final task completion reward using OSWorld's evaluator."""
        if self.desktop_env is None:
            return 0.0

        try:
            return float(self.desktop_env.evaluate())
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0

    def system_prompt(self) -> str:
        """Return system prompt for OSWorld GUI agent."""
        return OSWORLD_SYSTEM_PROMPT

    def close(self):
        """Clean up environment resources."""
        if self.desktop_env is not None:
            try:
                self.desktop_env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
            self.desktop_env = None


class HTTPRemoteDesktopEnv:
    """Remote OSWorld environment via HTTP server.

    Connects to osworld_env_server.py running on HOST (torch ~2.5).
    This allows the container (torch 2.9 + sglang) to use OSWorld
    without importing desktop-env directly.

    Set OSWORLD_SERVER_URL to enable. Supports comma-separated URLs for parallel VMs:
      - Single: http://172.17.0.1:8100
      - Multiple: http://172.17.0.1:8100,http://172.17.0.1:8101,http://172.17.0.1:8102
    """

    def __init__(self, **kwargs):
        self.server_url = _get_next_server_url()
        self.env_config = kwargs
        self.episode_id: str | None = None
        self.step_count = 0
        self.timeout = int(os.environ.get("OSWORLD_HTTP_TIMEOUT", "120"))

    def _request(self, endpoint: str, data: dict | None = None) -> dict:
        """Make HTTP request to OSWorld server."""
        url = f"{self.server_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = Request(url, data=body, headers=headers, method="POST")
        else:
            req = Request(url, headers=headers, method="GET")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            logger.error(f"HTTP request failed: {e}")
            raise RuntimeError(f"OSWorld server request failed: {e}") from e

    def _decode_observation(self, response: dict) -> dict:
        """Decode observation from HTTP response."""
        obs = {
            "accessibility_tree": response.get("accessibility_tree", ""),
            "terminal": response.get("terminal", ""),
            "instruction": response.get("instruction", ""),
        }

        # Decode screenshot from base64
        if "screenshot_base64" in response:
            img_bytes = base64.b64decode(response["screenshot_base64"])
            obs["screenshot"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif "screenshot_path" in response:
            obs["screenshot"] = Image.open(response["screenshot_path"]).convert("RGB")
        else:
            obs["screenshot"] = None

        return obs

    def reset(self, task_config: dict) -> dict:
        """Initialize remote environment."""
        self.step_count = 0

        response = self._request(
            "/reset",
            {
                "task_config": task_config,
                "env_config": self.env_config,
            },
        )

        if "error" in response:
            raise RuntimeError(f"Reset failed: {response['error']}")

        self.episode_id = response.get("episode_id")
        return self._decode_observation(response)

    def step(self, action: str, pause: int = 2) -> tuple[dict, float, bool, dict]:
        """Execute action in remote environment."""
        if not self.episode_id:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        response = self._request(
            "/step",
            {
                "episode_id": self.episode_id,
                "action": action,
                "pause": pause,
            },
        )

        if "error" in response:
            raise RuntimeError(f"Step failed: {response['error']}")

        self.step_count = response.get("step", self.step_count + 1)
        obs = self._decode_observation(response)
        reward = response.get("reward", 0.0)
        done = response.get("done", False)
        info = response.get("info", {})

        return obs, reward, done, info

    def _get_observation(self) -> dict:
        """Get current observation without stepping."""
        if not self.episode_id:
            return {}

        response = self._request("/render", {"episode_id": self.episode_id})
        if "error" in response:
            return {}

        return self._decode_observation(response)

    def evaluate(self) -> float:
        """Compute task completion reward."""
        if not self.episode_id:
            return 0.0

        response = self._request("/evaluate", {"episode_id": self.episode_id})
        if "error" in response:
            logger.warning(f"Evaluate failed: {response['error']}")
            return 0.0

        return response.get("reward", 0.0)

    def close(self):
        """Clean up remote episode."""
        if self.episode_id:
            try:
                self._request("/close", {"episode_id": self.episode_id})
            except Exception:
                pass
            self.episode_id = None


OSWORLD_SYSTEM_PROMPT = """You are a GUI automation agent. Complete the task by interacting with the desktop ONE action at a time.

OUTPUT FORMAT:
Action: <brief description>
<tool_call>
{"name":"computer_use","arguments":{"action":"<action>",<params>}}
</tool_call>

ACTIONS:
- left_click: {"action":"left_click","coordinate":[x,y]}
- right_click: {"action":"right_click","coordinate":[x,y]}
- double_click: {"action":"double_click","coordinate":[x,y]}
- type: {"action":"type","text":"string"}
- key: {"action":"key","keys":["ctrl","s"]}
- scroll: {"action":"scroll","direction":"up|down"}
- wait: {"action":"wait"}
- terminate: {"action":"terminate","status":"success|failure"}

EXAMPLE:
Action: Click the OK button.
<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[960,540]}}
</tool_call>

RULES:
- ONE action per response.
- Use a11y tree coordinates when available.
- Use WAIT after navigation.
"""


DEFAULT_ROLLOUT_CONFIG: dict[str, Any] = {
    "max_turns": 8,  # Based on trajectory data max (all datasets <= 8 turns)
    "max_total_tokens": 16384,  # VLM needs more context for images
    "stop_on_max_tokens": True,
}

DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "provider_name": "docker",
    "os_type": "Ubuntu",
    "action_space": "pyautogui",
    "max_steps": 8,  # Based on trajectory data max
    "headless": True,
    "require_a11y_tree": True,
    "include_a11y_in_obs": True,
}


def build_env(
    sample: Sample | None = None, args: Any | None = None, config_overrides: dict | None = None
) -> OSWorldEnvWrapper:
    """Factory function for rollout integration.

    Args:
        sample: Rollout sample with metadata containing task_config
        args: Training args (unused, for interface compatibility)
        config_overrides: Optional config overrides

    Returns:
        Configured OSWorldEnvWrapper instance
    """
    env_kwargs = deepcopy(DEFAULT_ENV_CONFIG)
    if config_overrides:
        env_kwargs.update(deepcopy(config_overrides))

    sample_metadata = getattr(sample, "metadata", None) or {}
    if isinstance(sample_metadata, str):
        try:
            sample_metadata = json.loads(sample_metadata)
        except json.JSONDecodeError:
            sample_metadata = {}

    env_kwargs.update(deepcopy(sample_metadata.get("env_config", {})))

    # Normalize a11y settings
    a11y_mode = env_kwargs.get("a11y_mode", "full")
    if a11y_mode == "off":
        env_kwargs["require_a11y_tree"] = False
        env_kwargs["include_a11y_in_obs"] = False
    else:
        env_kwargs["require_a11y_tree"] = True
        env_kwargs["include_a11y_in_obs"] = True

    # Handle both cases:
    # 1. sample.metadata IS the task_config (when --metadata-key task_config loads parquet column directly)
    # 2. sample.metadata["task_config"] contains the task_config (nested/legacy structure)
    if "id" in sample_metadata:
        # Metadata IS the task_config (direct load from parquet column)
        task_config = sample_metadata
    else:
        # Task config is nested under "task_config" key
        task_config = sample_metadata.get("task_config", {})

    if isinstance(task_config, str):
        try:
            task_config = json.loads(task_config)
        except json.JSONDecodeError:
            task_config = {}

    if "id" not in task_config and "task_id" in task_config:
        task_config["id"] = task_config["task_id"]

    config = OSWorldEnvConfig(**env_kwargs)
    return OSWorldEnvWrapper(config, task_config)


def format_observation(observation: dict) -> dict:
    """Convert observation to chat message for model input.

    Args:
        observation: Dict with 'obs_str' and optional 'multi_modal_data'

    Returns:
        Chat message dict with role='user' and content list
    """
    content = []

    # Add images first (Qwen3-VL expects images before text)
    multimodal = observation.get("multi_modal_data") or {}
    for _, images in multimodal.items():
        for image in images:
            content.append({"type": "image", "image": image})

    # Add text observation
    content.append({"type": "text", "text": observation.get("obs_str", "")})

    return {"role": "user", "content": content}


def on_reset(
    env: OSWorldEnvWrapper,
    observation: dict,
    sample: Sample | None = None,
    reset_info: dict | None = None,
) -> dict:
    """Capture initial task metadata after reset.

    Args:
        env: The environment instance
        observation: Initial observation
        sample: Current sample
        reset_info: Info dict from reset()

    Returns:
        Metadata updates to merge into sample.metadata
    """
    reset_info = reset_info or {}
    sample_metadata = getattr(sample, "metadata", None) or {}

    osworld_meta = deepcopy(sample_metadata.get("osworld", {}))
    osworld_meta.update(
        {
            "task_id": reset_info.get("task_id"),
            "domain": reset_info.get("domain"),
            "instruction": reset_info.get("instruction"),
            "initial_obs_text": observation.get("obs_str", ""),
        }
    )
    return {"osworld": osworld_meta}


def finalize_episode(
    env: OSWorldEnvWrapper,
    observation: dict,
    sample: Sample | None = None,
    responses: list[str] | None = None,
) -> dict:
    """Collect trajectory metrics for reward computation.

    Args:
        env: The environment instance
        observation: Final observation
        sample: Current sample
        responses: List of model responses

    Returns:
        Metadata updates with trajectory info
    """
    sample_metadata = getattr(sample, "metadata", None) or {}
    osworld_meta = deepcopy(sample_metadata.get("osworld", {}))

    task_reward = env.compute_reward()

    osworld_meta.update(
        {
            "turns": len(responses or []),
            "task_reward": task_reward,
            "step_signals": list(env.step_signals),
            "final_obs_text": observation.get("obs_str", ""),
            "max_steps": env.config.max_steps,
            "a11y_mode": env.config.a11y_mode,
        }
    )

    return {"osworld": osworld_meta}
