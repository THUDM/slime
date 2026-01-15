#!/usr/bin/env python3
"""Curate and merge OSWorld SFT datasets with reasoning augmentation.

This script combines:
1. Original SFT dataset (ground-truth successful demonstrations)
2. Reasoning trajectories (rich thinking from Claude Opus 4.5)

Curation principles:
- Preserve high-quality action supervision from original success samples
- Inject reasoning signal without corrupting action supervision
- Normalize formats across both datasets
- Control task imbalance via per-task caps

Turn-aware masking for failed reasoning trajectories:
- Turns 1 to early_cutoff: Keep complete (thinking + action)
- Turns early_cutoff+1 to late_cutoff: Keep thinking, filter repetitive actions
- Turns > late_cutoff: Keep thinking only (no action)

This preserves valuable reasoning signal from all turns while avoiding
training on stuck/repetitive action patterns that occur in late failed turns.

Usage:
    python curate_reasoning_sft.py \
        --original osworld_replay_train.jsonl \
        --reasoning reasoning_trajectories_all.jsonl \
        --output osworld_reasoning_sft_v1.jsonl \
        --max-per-task 5 \
        --reasoning-early-cutoff 5 \
        --reasoning-late-cutoff 15
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# System prompt matching the original SFT format, extended for reasoning
SYSTEM_PROMPT_WITH_REASONING = """You are a GUI automation agent. Complete the task by interacting with the desktop ONE action at a time.

OUTPUT FORMAT:
<thinking>
[Your reasoning about what you observe and what action to take]
</thinking>
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
<thinking>
I can see the desktop with a file manager open. The task asks me to create a new folder.
I should right-click in the empty area to get the context menu, then select "New Folder".
</thinking>
Action: Right-click in empty area to open context menu.
<tool_call>
{"name":"computer_use","arguments":{"action":"right_click","coordinate":[600,400]}}
</tool_call>

RULES:
- ONE action per response.
- Always explain your reasoning in the <thinking> block.
- Use a11y tree coordinates when available.
- Use WAIT after navigation.
- Do not repeat the same action/coordinate if the screen has not changed.
- If something doesn't work, try a different approach.
"""

# Original system prompt (for reference/validation)
ORIGINAL_SYSTEM_PROMPT_START = "You are a GUI automation agent. Complete the task"


@dataclass
class CurationStats:
    """Statistics for dataset curation."""
    original_total: int = 0
    original_kept: int = 0
    original_capped: int = 0
    reasoning_total: int = 0
    reasoning_success: int = 0
    reasoning_failed: int = 0
    reasoning_turns_kept: int = 0
    reasoning_turns_masked: int = 0
    final_total: int = 0
    task_distribution: dict = field(default_factory=dict)
    answer_converted: int = 0


def convert_answer_to_tool_call(content: str) -> str:
    """Convert <answer>pyautogui.xxx(...)</answer> format to <tool_call> format.

    Handles:
        - pyautogui.click(x, y) → left_click
        - pyautogui.rightClick(x, y) → right_click
        - pyautogui.doubleClick(x, y) → double_click
        - pyautogui.write("text") → type
        - pyautogui.typewrite("text") → type
        - pyautogui.press("key") → key
        - pyautogui.hotkey("k1", "k2") → key
        - pyautogui.scroll(n) → scroll
        - DONE → terminate (success)
        - WAIT → wait
    """
    # If already has tool_call, return as-is (remove duplicate <answer> if present)
    if "<tool_call>" in content:
        content = re.sub(r"\s*<answer>.*?</answer>", "", content, flags=re.DOTALL)
        return content

    # Extract answer content
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if not answer_match:
        return content

    answer_content = answer_match.group(1).strip()

    # Parse commands - check in priority order
    tool_call_json = None

    # Terminal actions: DONE, FAIL, WAIT
    if answer_content == "DONE":
        tool_call_json = {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
    elif answer_content == "FAIL":
        tool_call_json = {"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
    elif answer_content == "WAIT":
        tool_call_json = {"name": "computer_use", "arguments": {"action": "wait"}}

    # pyautogui.click(x, y) or pyautogui.click(x,y)
    if not tool_call_json:
        click_match = re.match(r"pyautogui\.click\((\d+),\s*(\d+)\)", answer_content)
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            tool_call_json = {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}

    # pyautogui.rightClick(x, y)
    if not tool_call_json:
        right_click_match = re.match(r"pyautogui\.rightClick\((\d+),\s*(\d+)\)", answer_content)
        if right_click_match:
            x, y = int(right_click_match.group(1)), int(right_click_match.group(2))
            tool_call_json = {"name": "computer_use", "arguments": {"action": "right_click", "coordinate": [x, y]}}

    # pyautogui.doubleClick(x, y)
    if not tool_call_json:
        double_click_match = re.match(r"pyautogui\.doubleClick\((\d+),\s*(\d+)\)", answer_content)
        if double_click_match:
            x, y = int(double_click_match.group(1)), int(double_click_match.group(2))
            tool_call_json = {"name": "computer_use", "arguments": {"action": "double_click", "coordinate": [x, y]}}

    # pyautogui.write("text") or pyautogui.typewrite("text")
    if not tool_call_json:
        type_match = re.match(r'pyautogui\.(?:write|typewrite)\(["\'](.+?)["\']\)', answer_content, re.DOTALL)
        if type_match:
            text = type_match.group(1)
            tool_call_json = {"name": "computer_use", "arguments": {"action": "type", "text": text}}

    # pyautogui.press("key")
    if not tool_call_json:
        press_match = re.match(r'pyautogui\.press\(["\'](.+?)["\']\)', answer_content)
        if press_match:
            key = press_match.group(1)
            tool_call_json = {"name": "computer_use", "arguments": {"action": "key", "keys": [key]}}

    # pyautogui.hotkey("key1", "key2", ...)
    if not tool_call_json:
        hotkey_match = re.match(r'pyautogui\.hotkey\((.+?)\)', answer_content)
        if hotkey_match:
            keys_str = hotkey_match.group(1)
            keys = re.findall(r'["\']([^"\']+)["\']', keys_str)
            if keys:
                tool_call_json = {"name": "computer_use", "arguments": {"action": "key", "keys": keys}}

    # pyautogui.scroll(amount)
    if not tool_call_json:
        scroll_match = re.match(r"pyautogui\.scroll\((-?\d+)\)", answer_content)
        if scroll_match:
            amount = int(scroll_match.group(1))
            direction = "up" if amount > 0 else "down"
            tool_call_json = {"name": "computer_use", "arguments": {"action": "scroll", "direction": direction}}

    if tool_call_json:
        # Replace <answer>...</answer> with <tool_call>...</tool_call>
        tool_call_str = f"<tool_call>\n{json.dumps(tool_call_json)}\n</tool_call>"
        # Use lambda to avoid regex escape issues with JSON content
        content = re.sub(r"<answer>.*?</answer>", lambda m: tool_call_str, content, flags=re.DOTALL)
    else:
        # Unknown format - log warning but keep original
        logger.warning(f"Could not parse answer format: {answer_content[:100]}")

    return content


def extract_action_from_content(content: str) -> str | None:
    """Extract action description from assistant content."""
    # Try to find "Action: <desc>" pattern
    match = re.search(r"Action:\s*(.+?)(?=\n|<tool_call>|$)", content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no Action: prefix, extract text before <tool_call>
    tool_match = re.search(r"^(.+?)<tool_call>", content, re.DOTALL)
    if tool_match:
        text = tool_match.group(1).strip()
        # Clean up and return first sentence/line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            return lines[-1][:200]  # Last line before tool_call, capped

    return None


def normalize_original_sample(sample: dict, add_thinking_stub: bool = True) -> dict:
    """Normalize original SFT sample format.

    Original format:
    - messages: [system, user, assistant, ...]
    - images: [base64, ...]
    - Assistant: "Action: <desc>\n<tool_call>..."

    Target format:
    - Same structure but with <thinking> in assistant messages
    """
    messages = sample.get("messages", [])
    new_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            # Replace with reasoning-aware system prompt
            new_messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT_WITH_REASONING,
            })
        elif role == "assistant":
            # First, convert any <answer> format to <tool_call>
            content = convert_answer_to_tool_call(content)

            if add_thinking_stub:
                # Extract action description
                action_desc = extract_action_from_content(content)

                # Check if already has <tool_call>
                tool_match = re.search(r"(<tool_call>.*?</tool_call>)", content, re.DOTALL)
                tool_call = tool_match.group(1) if tool_match else ""

                # Create minimal thinking stub
                thinking = f"Based on the current screen state, I will proceed with the next step."

                # Reconstruct with thinking
                if action_desc and tool_call:
                    new_content = f"<thinking>\n{thinking}\n</thinking>\nAction: {action_desc}\n{tool_call}"
                else:
                    # Fallback: wrap original content
                    new_content = f"<thinking>\n{thinking}\n</thinking>\n{content}"

                new_messages.append({"role": "assistant", "content": new_content})
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)

    return {
        "messages": new_messages,
        "images": sample.get("images", []),
        "reward": sample.get("reward", 1.0),
        "task_id": sample.get("task_id"),
        "num_steps": sample.get("num_steps"),
        "source_dataset": sample.get("source_dataset", "original"),
        "quality_rank": sample.get("quality_rank", 5),
        "original_task_id": sample.get("original_task_id"),
    }


def extract_action_signature(content: str) -> str | None:
    """Extract action signature for duplicate detection."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            args = data.get("arguments", {})
            action = args.get("action", "")
            coord = args.get("coordinate", [])
            text = args.get("text", "")
            keys = args.get("keys", [])
            return f"{action}:{coord}:{text}:{keys}"
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def normalize_reasoning_sample(
    sample: dict,
    early_cutoff: int = 5,
    late_cutoff: int = 15,
    stats: CurationStats | None = None,
) -> dict | None:
    """Normalize reasoning trajectory to SFT format.

    Reasoning format:
    - messages: [{role, content, thinking, images}, ...]
    - thinking: separate field
    - images: embedded in messages

    Target format:
    - messages: [{role, content}, ...]
    - images: top-level array
    - thinking merged into content as <thinking>...</thinking>

    Turn-aware masking for failed trajectories:
    - Turns 1 to early_cutoff: Keep complete (thinking + action)
    - Turns early_cutoff+1 to late_cutoff: Keep thinking, mask action if repetitive
    - Turns > late_cutoff: Keep thinking only (no action)

    This preserves reasoning signal while avoiding training on stuck/repetitive actions.
    """
    task_id = sample.get("task_id")
    task_reward = sample.get("task_reward", 0.0)
    is_success = task_reward >= 0.5

    messages = sample.get("messages", [])
    if not messages:
        return None

    # Build normalized messages
    new_messages = [{"role": "system", "content": SYSTEM_PROMPT_WITH_REASONING}]
    images = []
    turn_num = 0
    prev_action_sig = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        thinking = msg.get("thinking")
        msg_images = msg.get("images", [])

        if role == "user":
            # Convert images to top-level and add placeholder
            user_parts = []
            for img_b64 in msg_images:
                if img_b64:
                    # Add data URL prefix if not present
                    if not img_b64.startswith("data:"):
                        images.append(f"data:image/png;base64,{img_b64}")
                    else:
                        images.append(img_b64)
                    user_parts.append("<image>")

            # Add text content
            if content:
                user_parts.append(content)

            if user_parts:
                new_messages.append({
                    "role": "user",
                    "content": "\n".join(user_parts),
                })

        elif role == "assistant":
            turn_num += 1

            # Extract action description from content
            action_desc = extract_action_from_content(content)

            # Extract tool_call
            tool_match = re.search(r"(<tool_call>.*?</tool_call>)", content, re.DOTALL)
            tool_call = tool_match.group(1) if tool_match else ""

            # Determine whether to include action based on turn and success
            include_action = True

            if not is_success:
                if turn_num > late_cutoff:
                    # Very late turns: keep thinking only, no action
                    include_action = False
                    if stats:
                        stats.reasoning_turns_masked += 1
                elif turn_num > early_cutoff:
                    # Mid-range turns: filter repetitive actions
                    action_sig = extract_action_signature(content)
                    if action_sig and action_sig == prev_action_sig:
                        # Skip repetitive action, keep thinking
                        include_action = False
                        if stats:
                            stats.reasoning_turns_masked += 1
                    prev_action_sig = action_sig

            if include_action and stats:
                stats.reasoning_turns_kept += 1

            # Build normalized content with thinking
            parts = []
            if thinking:
                parts.append(f"<thinking>\n{thinking.strip()}\n</thinking>")

            if include_action:
                # Full turn: thinking + action + tool_call
                if action_desc:
                    parts.append(f"Action: {action_desc}")

                if tool_call:
                    parts.append(tool_call)
                elif not tool_call and content:
                    # Fallback: include original content if no tool_call found
                    parts.append(content)
            else:
                # Reasoning-only turn: just thinking (for late/repetitive turns)
                # Add a note that action is masked for training purposes
                if thinking:
                    # Only include if we have thinking content
                    pass  # thinking already added above

            if parts:
                new_messages.append({
                    "role": "assistant",
                    "content": "\n".join(parts),
                })

    # Must have at least one assistant message
    assistant_count = sum(1 for m in new_messages if m["role"] == "assistant")
    if assistant_count == 0:
        return None

    return {
        "messages": new_messages,
        "images": images,
        "reward": task_reward,
        "task_id": task_id,
        "num_steps": assistant_count,
        "source_dataset": "reasoning_opus",
        "quality_rank": 5 if is_success else 3,
        "original_task_id": task_id,
        "metadata": {
            "instruction": sample.get("instruction", ""),
            "domain": sample.get("domain", "unknown"),
            "is_success": is_success,
            "original_turns": sample.get("metadata", {}).get("turns", 0),
        },
    }


def curate_dataset(
    original_samples: list[dict],
    reasoning_samples: list[dict],
    max_per_task: int = 5,
    reasoning_early_cutoff: int = 5,
    reasoning_late_cutoff: int = 15,
    add_thinking_stub: bool = True,
) -> tuple[list[dict], CurationStats]:
    """Curate and merge datasets with balancing and format normalization.

    Args:
        original_samples: Original SFT samples (ground-truth success)
        reasoning_samples: Reasoning trajectories (mixed success)
        max_per_task: Maximum samples per task from original dataset
        reasoning_early_cutoff: Keep full turns up to this turn for failed reasoning
        reasoning_late_cutoff: Keep thinking only (no action) after this turn
        add_thinking_stub: Add minimal thinking stub to original samples

    Turn-aware masking for failed reasoning trajectories:
    - Turns 1 to early_cutoff: Keep complete (thinking + action)
    - Turns early_cutoff+1 to late_cutoff: Keep thinking, filter repetitive actions
    - Turns > late_cutoff: Keep thinking only (no action)

    Returns:
        Tuple of (curated_samples, stats)
    """
    stats = CurationStats()
    stats.original_total = len(original_samples)
    stats.reasoning_total = len(reasoning_samples)

    # Group original samples by task_id
    original_by_task = defaultdict(list)
    for s in original_samples:
        task_id = s.get("task_id")
        if task_id:
            original_by_task[task_id].append(s)

    curated = []

    # Process original samples with per-task cap
    for task_id, samples in original_by_task.items():
        # Sort by quality_rank (higher is better) and take top K
        samples_sorted = sorted(samples, key=lambda x: x.get("quality_rank", 0), reverse=True)
        kept = samples_sorted[:max_per_task]
        capped = len(samples_sorted) - len(kept)

        stats.original_kept += len(kept)
        stats.original_capped += capped

        for s in kept:
            normalized = normalize_original_sample(s, add_thinking_stub=add_thinking_stub)
            curated.append(normalized)

    # Process reasoning samples
    for s in reasoning_samples:
        task_reward = s.get("task_reward", 0.0)
        if task_reward >= 0.5:
            stats.reasoning_success += 1
        else:
            stats.reasoning_failed += 1

        normalized = normalize_reasoning_sample(
            s,
            early_cutoff=reasoning_early_cutoff,
            late_cutoff=reasoning_late_cutoff,
            stats=stats,
        )
        if normalized:
            curated.append(normalized)

    stats.final_total = len(curated)

    # Compute task distribution
    task_counts = Counter(s.get("task_id") for s in curated)
    stats.task_distribution = dict(task_counts.most_common(20))

    return curated, stats


def validate_sample(sample: dict, strict_tool_call: bool = False) -> list[str]:
    """Validate a curated sample for format correctness.

    Args:
        sample: The sample to validate
        strict_tool_call: If True, require <tool_call> in ALL assistant messages.
                          If False (default), allow reasoning-only turns.
    """
    issues = []

    messages = sample.get("messages", [])
    if not messages:
        issues.append("No messages")
        return issues

    # Check system prompt
    if messages[0].get("role") != "system":
        issues.append("First message is not system prompt")

    # Check for at least one user and assistant message
    roles = [m.get("role") for m in messages]
    if "user" not in roles:
        issues.append("No user message")
    if "assistant" not in roles:
        issues.append("No assistant message")

    # Check assistant format
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    has_any_tool_call = False

    for msg in assistant_msgs:
        content = msg.get("content", "")
        has_thinking = "<thinking>" in content
        has_tool_call = "<tool_call>" in content

        if has_tool_call:
            has_any_tool_call = True

        if not has_thinking:
            issues.append("Assistant missing <thinking> block")

        if strict_tool_call and not has_tool_call:
            issues.append("Assistant missing <tool_call> (strict mode)")

    # At least one assistant message should have a tool_call
    if not has_any_tool_call:
        issues.append("No assistant message has <tool_call>")

    # Check images
    images = sample.get("images", [])
    user_image_refs = sum(1 for m in messages if m.get("role") == "user" and "<image>" in m.get("content", ""))
    if user_image_refs > 0 and len(images) == 0:
        issues.append(f"User references {user_image_refs} images but images array is empty")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Curate and merge OSWorld SFT datasets"
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original SFT JSONL file",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        required=True,
        help="Path to reasoning trajectories JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output curated JSONL file",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=5,
        help="Maximum samples per task from original dataset (default: 5)",
    )
    parser.add_argument(
        "--reasoning-early-cutoff",
        type=int,
        default=5,
        help="Keep full turns (thinking + action) up to this turn for failed reasoning (default: 5)",
    )
    parser.add_argument(
        "--reasoning-late-cutoff",
        type=int,
        default=15,
        help="Keep thinking only (no action) after this turn for failed reasoning (default: 15)",
    )
    parser.add_argument(
        "--no-thinking-stub",
        action="store_true",
        help="Don't add thinking stub to original samples",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output samples and report issues",
    )

    args = parser.parse_args()

    # Load datasets
    logger.info(f"Loading original dataset from {args.original}")
    with open(args.original) as f:
        original = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(original)} original samples")

    logger.info(f"Loading reasoning dataset from {args.reasoning}")
    with open(args.reasoning) as f:
        reasoning = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(reasoning)} reasoning samples")

    # Curate
    logger.info("Curating dataset...")
    logger.info(f"Turn masking: early_cutoff={args.reasoning_early_cutoff}, late_cutoff={args.reasoning_late_cutoff}")
    curated, stats = curate_dataset(
        original_samples=original,
        reasoning_samples=reasoning,
        max_per_task=args.max_per_task,
        reasoning_early_cutoff=args.reasoning_early_cutoff,
        reasoning_late_cutoff=args.reasoning_late_cutoff,
        add_thinking_stub=not args.no_thinking_stub,
    )

    # Validate if requested
    if args.validate:
        logger.info("Validating samples...")
        issues_count = 0
        for i, sample in enumerate(curated):
            issues = validate_sample(sample)
            if issues:
                issues_count += 1
                if issues_count <= 5:  # Show first 5
                    logger.warning(f"Sample {i} ({sample.get('task_id', 'unknown')[:20]}): {issues}")
        logger.info(f"Validation complete: {issues_count} samples with issues")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for sample in curated:
            f.write(json.dumps(sample) + "\n")

    # Print statistics
    logger.info(f"\n{'='*60}")
    logger.info("CURATION STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"\nOriginal dataset:")
    logger.info(f"  Total samples: {stats.original_total}")
    logger.info(f"  Kept samples: {stats.original_kept}")
    logger.info(f"  Capped (per-task limit): {stats.original_capped}")
    logger.info(f"\nReasoning dataset:")
    logger.info(f"  Total samples: {stats.reasoning_total}")
    logger.info(f"  Successful: {stats.reasoning_success}")
    logger.info(f"  Failed: {stats.reasoning_failed}")
    logger.info(f"  Turns kept: {stats.reasoning_turns_kept}")
    logger.info(f"  Turns masked: {stats.reasoning_turns_masked}")
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Total samples: {stats.final_total}")
    logger.info(f"  Unique tasks: {len(stats.task_distribution)}")
    logger.info(f"\nTop task distribution:")
    for task_id, count in list(stats.task_distribution.items())[:10]:
        logger.info(f"  {task_id[:30]}...: {count} samples")
    logger.info(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
