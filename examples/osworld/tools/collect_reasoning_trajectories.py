#!/usr/bin/env python3
"""Collect reasoning-rich trajectories using Claude Opus 4.5 with extended thinking.

This script generates bootstrap trajectories for the data flywheel by having
Claude complete OSWorld tasks while showing its reasoning process.

Usage:
    # Start OSWorld server first (on host with KVM)
    cd ~/OSWorld
    sudo -E ~/osworld_venv/bin/python ~/slime/examples/osworld/tools/osworld_env_server.py --port 8100

    # Run collection
    export OSWORLD_SERVER_URL=http://localhost:8100
    export ANTHROPIC_API_KEY=<your-key>
    python examples/osworld/tools/collect_reasoning_trajectories.py \
        --tasks /path/to/osworld_tasks_train.parquet \
        --output /path/to/reasoning_trajectories.jsonl \
        --max-turns 15 \
        --thinking-budget 10000

The script captures both the extended thinking and action output for each turn,
producing trajectories that teach exploration and decision-making.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# System prompt for reasoning-rich trajectory collection
# Per extended thinking best practices: use general instructions, let Claude's creativity guide
REASONING_SYSTEM_PROMPT = """You are an expert GUI automation agent completing desktop tasks.

Think deeply about each step. Consider multiple approaches and explain your reasoning.
If something doesn't work, analyze why and try a different approach.

OUTPUT FORMAT:
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

Think thoroughly before each action. Consider what you see, what options you have, and why you're choosing a particular approach.

IMPORTANT:
- ONE action per response
- Use coordinates from the accessibility tree when available
- If an action doesn't produce the expected result, try a different approach
- Use terminate with status="success" when the task is complete
"""


@dataclass
class TrajectoryMessage:
    """A single message in a trajectory."""

    role: str
    content: str
    thinking: str | None = None
    images: list[str] = field(default_factory=list)  # base64 encoded


@dataclass
class Trajectory:
    """A complete trajectory for a task."""

    task_id: str
    instruction: str
    domain: str
    messages: list[TrajectoryMessage] = field(default_factory=list)
    task_reward: float = 0.0
    metadata: dict = field(default_factory=dict)


class OSWorldHTTPClient:
    """HTTP client for OSWorld environment server."""

    def __init__(self, server_url: str, timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.episode_id: str | None = None

    def _request(self, endpoint: str, data: dict | None = None) -> dict:
        """Make HTTP request to OSWorld server."""
        import urllib.request
        import urllib.error

        url = f"{self.server_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        else:
            req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error(f"HTTP request failed: {e}")
            raise RuntimeError(f"OSWorld server request failed: {e}") from e

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self._request("/health")
            return response.get("status") in ("healthy", "ok")
        except Exception:
            return False

    def reset(self, task_config: dict) -> dict:
        """Reset environment with task config."""
        response = self._request(
            "/reset",
            {
                "task_config": task_config,
                "env_config": {
                    "provider_name": "docker",
                    "os_type": "Ubuntu",
                    "action_space": "pyautogui",
                    "headless": True,
                    "require_a11y_tree": True,
                },
            },
        )

        if "error" in response:
            raise RuntimeError(f"Reset failed: {response['error']}")

        self.episode_id = response.get("episode_id")
        return self._decode_observation(response)

    def step(self, action: str, pause: int = 2) -> tuple[dict, float, bool]:
        """Execute action and return observation, reward, done."""
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

        obs = self._decode_observation(response)
        reward = response.get("reward", 0.0)
        done = response.get("done", False)

        return obs, reward, done

    def evaluate(self) -> float:
        """Evaluate task completion."""
        if not self.episode_id:
            return 0.0

        response = self._request("/evaluate", {"episode_id": self.episode_id})
        if "error" in response:
            logger.warning(f"Evaluate failed: {response['error']}")
            return 0.0

        return response.get("reward", 0.0)

    def close(self):
        """Close current episode."""
        if self.episode_id:
            try:
                self._request("/close", {"episode_id": self.episode_id})
            except Exception:
                pass
            self.episode_id = None

    def _decode_observation(self, response: dict) -> dict:
        """Decode observation from response."""
        obs = {
            "accessibility_tree": response.get("accessibility_tree", ""),
            "terminal": response.get("terminal", ""),
            "instruction": response.get("instruction", ""),
            "screenshot": None,
            "screenshot_base64": None,
        }

        if "screenshot_base64" in response:
            obs["screenshot_base64"] = response["screenshot_base64"]
            img_bytes = base64.b64decode(response["screenshot_base64"])
            obs["screenshot"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return obs


def resize_for_claude(img: Image.Image, max_size: int = 1568) -> Image.Image:
    """Resize image to fit Claude's vision requirements."""
    width, height = img.size
    if width <= max_size and height <= max_size:
        return img

    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_observation_text(obs: dict, include_a11y: bool = True) -> str:
    """Build observation text from environment observation."""
    parts = []

    instruction = obs.get("instruction", "")
    if instruction:
        parts.append(f"[Task]\n{instruction}")

    if include_a11y:
        a11y_tree = obs.get("accessibility_tree", "")
        if a11y_tree:
            # Truncate if too long
            if len(a11y_tree) > 4096:
                a11y_tree = a11y_tree[:4096] + "\n... (truncated)"
            parts.append(f"[Accessibility Tree]\n{a11y_tree}")

    terminal = obs.get("terminal", "")
    if terminal:
        parts.append(f"[Terminal Output]\n{terminal}")

    return "\n\n".join(parts) if parts else "(See screenshot)"


def parse_action_from_response(response_text: str) -> tuple[str | None, str]:
    """Parse action from Claude's response.

    Returns (pyautogui_action, action_type).
    """
    import re

    # Extract from <tool_call> tags
    match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
    if not match:
        # Try to find raw JSON
        json_match = re.search(r'\{"name":\s*"computer_use".*?\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None, "parse_error"
    else:
        json_str = match.group(1).strip()

    try:
        data = json.loads(json_str)
        args = data.get("arguments", {})
        action = args.get("action", "").lower()

        # Translate to pyautogui
        if action in ("left_click", "click"):
            coord = args.get("coordinate", [0, 0])
            return f"pyautogui.click({coord[0]}, {coord[1]})", "click"
        elif action == "right_click":
            coord = args.get("coordinate", [0, 0])
            return f"pyautogui.rightClick({coord[0]}, {coord[1]})", "right_click"
        elif action == "double_click":
            coord = args.get("coordinate", [0, 0])
            return f"pyautogui.doubleClick({coord[0]}, {coord[1]})", "double_click"
        elif action == "type":
            text = args.get("text", "")
            text = text.replace("\\", "\\\\").replace('"', '\\"')
            return f'pyautogui.write("{text}")', "type"
        elif action == "key":
            keys = args.get("keys", [])
            if len(keys) == 1:
                return f'pyautogui.press("{keys[0]}")', "key"
            elif len(keys) > 1:
                keys_str = ", ".join(f'"{k}"' for k in keys)
                return f"pyautogui.hotkey({keys_str})", "hotkey"
        elif action == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            scroll_val = -amount if direction == "down" else amount
            return f"pyautogui.scroll({scroll_val})", "scroll"
        elif action == "wait":
            return "WAIT", "wait"
        elif action == "terminate":
            status = args.get("status", "success")
            return "DONE" if status == "success" else "FAIL", "terminate"

        return None, "unknown_action"

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse action JSON: {e}")
        return None, "parse_error"


def collect_trajectory(
    client: anthropic.Anthropic,
    env: OSWorldHTTPClient,
    task_config: dict,
    max_turns: int = 15,
    thinking_budget: int = 10000,
) -> Trajectory | None:
    """Collect a single trajectory for a task using Claude with extended thinking."""
    task_id = task_config.get("id", "unknown")
    domain = task_config.get("domain", "unknown")

    logger.info(f"Starting trajectory collection for task {task_id} (domain: {domain})")

    trajectory = Trajectory(
        task_id=task_id,
        instruction=task_config.get("instruction", ""),
        domain=domain,
        metadata={
            "max_turns": max_turns,
            "thinking_budget": thinking_budget,
        },
    )

    try:
        # Reset environment
        obs = env.reset(task_config)
        instruction = obs.get("instruction", task_config.get("instruction", ""))
        trajectory.instruction = instruction

        # Build conversation messages for Claude API
        messages: list[dict] = []

        for turn in range(max_turns):
            # Build observation content for Claude
            obs_text = build_observation_text(obs)

            # Prepare image
            screenshot = obs.get("screenshot")
            screenshot_b64 = None
            if screenshot:
                resized = resize_for_claude(screenshot)
                screenshot_b64 = image_to_base64(resized)

            # Build user message with image and text
            user_content = []
            if screenshot_b64:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                })
            user_content.append({"type": "text", "text": obs_text})

            messages.append({"role": "user", "content": user_content})

            # Store user message in trajectory
            trajectory.messages.append(TrajectoryMessage(
                role="user",
                content=obs_text,
                images=[screenshot_b64] if screenshot_b64 else [],
            ))

            # Call Claude with extended thinking
            try:
                response = client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=16000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    },
                    system=REASONING_SYSTEM_PROMPT,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error(f"Claude API error on turn {turn}: {e}")
                break

            # Extract thinking and text from response
            thinking_content = ""
            text_content = ""

            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                elif block.type == "text":
                    text_content = block.text

            logger.info(f"Turn {turn}: Received response with {len(thinking_content)} chars thinking")

            # Store assistant message in trajectory
            trajectory.messages.append(TrajectoryMessage(
                role="assistant",
                content=text_content,
                thinking=thinking_content if thinking_content else None,
            ))

            # Add assistant response to conversation for next turn
            messages.append({"role": "assistant", "content": text_content})

            # Parse action from response
            pyautogui_action, action_type = parse_action_from_response(text_content)

            if pyautogui_action is None:
                logger.warning(f"Turn {turn}: Failed to parse action from response")
                # Add feedback to user
                messages.append({
                    "role": "user",
                    "content": "Could not parse your action. Please provide a valid tool_call."
                })
                trajectory.messages.append(TrajectoryMessage(
                    role="user",
                    content="Could not parse your action. Please provide a valid tool_call.",
                ))
                continue

            # Check for terminal actions
            if pyautogui_action in ("DONE", "FAIL"):
                logger.info(f"Turn {turn}: Agent terminated with {pyautogui_action}")
                break

            if pyautogui_action == "WAIT":
                logger.info(f"Turn {turn}: Agent waiting")
                time.sleep(2)
                # Get updated observation without stepping
                obs, _, _ = env.step("pyautogui.click(1, 1)")  # No-op to get observation
                continue

            # Execute action in environment
            try:
                obs, reward, done = env.step(pyautogui_action)
                logger.info(f"Turn {turn}: Executed {action_type}, done={done}")

                if done:
                    break

            except Exception as e:
                logger.error(f"Turn {turn}: Action execution failed: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Action failed with error: {e}. Please try a different approach."
                })
                trajectory.messages.append(TrajectoryMessage(
                    role="user",
                    content=f"Action failed with error: {e}. Please try a different approach.",
                ))
                continue

        # Evaluate task completion
        task_reward = env.evaluate()
        trajectory.task_reward = task_reward
        trajectory.metadata["turns"] = len([m for m in trajectory.messages if m.role == "assistant"])
        trajectory.metadata["success"] = task_reward >= 0.5

        # Count action types
        action_types = []
        for msg in trajectory.messages:
            if msg.role == "assistant":
                _, action_type = parse_action_from_response(msg.content)
                action_types.append(action_type)
        trajectory.metadata["action_types"] = action_types

        # Count thinking tokens
        total_thinking = sum(
            len(m.thinking or "") for m in trajectory.messages if m.role == "assistant"
        )
        trajectory.metadata["total_thinking_chars"] = total_thinking

        logger.info(
            f"Finished task {task_id}: reward={task_reward:.2f}, "
            f"turns={trajectory.metadata['turns']}, thinking_chars={total_thinking}"
        )

        return trajectory

    except Exception as e:
        logger.error(f"Error collecting trajectory for task {task_id}: {e}")
        return None

    finally:
        env.close()


def trajectory_to_dict(traj: Trajectory) -> dict:
    """Convert trajectory to dict for JSON serialization."""
    return {
        "task_id": traj.task_id,
        "instruction": traj.instruction,
        "domain": traj.domain,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "thinking": m.thinking,
                "images": m.images,
            }
            for m in traj.messages
        ],
        "task_reward": traj.task_reward,
        "metadata": traj.metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect reasoning-rich trajectories using Claude Opus 4.5"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Path to task registry parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum turns per trajectory (default: 15)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=10000,
        help="Token budget for extended thinking (default: 10000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process (for testing)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N tasks (for parallel workers)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have trajectories in output file",
    )
    parser.add_argument(
        "--filter-domain",
        type=str,
        default=None,
        help="Only process tasks from this domain",
    )

    args = parser.parse_args()

    # Check environment
    server_url = os.environ.get("OSWORLD_SERVER_URL")
    if not server_url:
        logger.error("OSWORLD_SERVER_URL environment variable not set")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Load tasks
    logger.info(f"Loading tasks from {args.tasks}")
    tasks_df = pd.read_parquet(args.tasks)
    logger.info(f"Loaded {len(tasks_df)} tasks")

    # Filter by domain if specified
    if args.filter_domain:
        tasks_df = tasks_df[tasks_df["domain"] == args.filter_domain]
        logger.info(f"Filtered to {len(tasks_df)} tasks in domain {args.filter_domain}")

    # Load existing task IDs if skipping
    existing_task_ids = set()
    if args.skip_existing and Path(args.output).exists():
        with open(args.output, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_task_ids.add(data.get("task_id"))
                except json.JSONDecodeError:
                    continue
        logger.info(f"Found {len(existing_task_ids)} existing trajectories")

    # Initialize clients
    env = OSWorldHTTPClient(server_url)
    if not env.health_check():
        logger.error(f"OSWorld server at {server_url} is not healthy")
        sys.exit(1)
    logger.info(f"Connected to OSWorld server at {server_url}")

    client = anthropic.Anthropic(api_key=api_key)

    # Process tasks
    successful = 0
    failed = 0
    skipped = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_configs = []
    for _, row in tasks_df.iterrows():
        task_config = json.loads(row["task_config"]) if isinstance(row["task_config"], str) else row["task_config"]
        if task_config.get("id") in existing_task_ids:
            skipped += 1
            continue
        task_configs.append(task_config)

    # Apply offset for parallel workers
    if args.offset > 0:
        task_configs = task_configs[args.offset:]
        logger.info(f"Skipped first {args.offset} tasks (offset)")

    if args.limit:
        task_configs = task_configs[:args.limit]

    logger.info(f"Processing {len(task_configs)} tasks (skipped {skipped} existing)")

    with open(args.output, "a") as f:
        for i, task_config in enumerate(task_configs):
            task_id = task_config.get("id", "unknown")
            logger.info(f"[{i+1}/{len(task_configs)}] Processing task {task_id}")

            trajectory = collect_trajectory(
                client=client,
                env=env,
                task_config=task_config,
                max_turns=args.max_turns,
                thinking_budget=args.thinking_budget,
            )

            if trajectory is not None:
                # Write immediately to avoid data loss
                f.write(json.dumps(trajectory_to_dict(trajectory)) + "\n")
                f.flush()

                if trajectory.task_reward >= 0.5:
                    successful += 1
                    logger.info(f"Task {task_id}: SUCCESS (reward={trajectory.task_reward:.2f})")
                else:
                    failed += 1
                    logger.info(f"Task {task_id}: FAILED (reward={trajectory.task_reward:.2f})")
            else:
                failed += 1
                logger.warning(f"Task {task_id}: ERROR (no trajectory)")

            # Rate limiting between tasks
            time.sleep(1)

    logger.info(f"\nCollection complete:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
