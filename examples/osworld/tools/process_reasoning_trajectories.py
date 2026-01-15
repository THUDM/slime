#!/usr/bin/env python3
"""Process reasoning trajectories into SFT format with turn-aware masking.

This script converts Claude Opus 4.5 reasoning trajectories into the standard
OSWorld SFT format, applying turn-aware masking for failed trajectories to
avoid reinforcing stuck/looping behavior.

Masking Strategy:
- Successful trajectories: keep all turns
- Failed trajectories:
  - Keep turns 1-5 (early exploration, usually valid)
  - Keep turns 6-9 only if action is non-repetitive
  - Discard turns 10+ (often stuck in loops)

Usage:
    python process_reasoning_trajectories.py \
        --input reasoning_trajectories.jsonl \
        --output reasoning_sft.jsonl \
        --early-cutoff 5 \
        --late-cutoff 10
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# System prompt that includes reasoning format
REASONING_SYSTEM_PROMPT = """You are a GUI automation agent. Complete the task by interacting with the desktop ONE action at a time.

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


@dataclass
class ProcessingStats:
    """Statistics for trajectory processing."""

    total_input: int = 0
    successful_trajs: int = 0
    failed_trajs: int = 0
    total_turns_input: int = 0
    total_turns_output: int = 0
    turns_masked: int = 0
    duplicate_actions_filtered: int = 0


def extract_action_signature(content: str) -> str | None:
    """Extract action signature for duplicate detection."""
    import re

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


def format_assistant_message(thinking: str | None, content: str) -> str:
    """Format assistant message with thinking block."""
    if thinking:
        # Wrap thinking in tags if not already wrapped
        if not thinking.strip().startswith("<thinking>"):
            thinking_block = f"<thinking>\n{thinking.strip()}\n</thinking>\n"
        else:
            thinking_block = thinking.strip() + "\n"
        return thinking_block + content
    return content


def process_trajectory(
    traj: dict,
    early_cutoff: int = 5,
    late_cutoff: int = 10,
    filter_duplicates: bool = True,
    stats: ProcessingStats | None = None,
) -> dict | None:
    """Process a single trajectory into SFT format.

    Args:
        traj: Raw trajectory from collection
        early_cutoff: Keep all turns up to this number for failed trajs
        late_cutoff: Discard all turns >= this number for failed trajs
        filter_duplicates: Filter consecutive duplicate actions in mid-range
        stats: Statistics accumulator

    Returns:
        Processed trajectory in SFT format, or None if empty
    """
    task_id = traj.get("task_id", "unknown")
    instruction = traj.get("instruction", "")
    task_reward = traj.get("task_reward", 0.0)
    is_success = task_reward >= 0.5

    messages = traj.get("messages", [])
    if not messages:
        return None

    # Build SFT messages
    sft_messages = [{"role": "system", "content": REASONING_SYSTEM_PROMPT}]
    images = []
    turn_num = 0
    prev_action_sig = None

    for msg in messages:
        role = msg.get("role")

        if role == "user":
            # Handle user messages
            content = msg.get("content", "")
            msg_images = msg.get("images", [])

            # Add images to top-level list and replace with placeholder
            user_content_parts = []
            for img_b64 in msg_images:
                if img_b64:
                    images.append(f"data:image/png;base64,{img_b64}")
                    user_content_parts.append("<image>")

            # Add text content
            if content:
                user_content_parts.append(content)

            if user_content_parts:
                sft_messages.append({
                    "role": "user",
                    "content": "\n".join(user_content_parts),
                })

        elif role == "assistant":
            turn_num += 1
            thinking = msg.get("thinking")
            content = msg.get("content", "")

            if stats:
                stats.total_turns_input += 1

            # Apply turn-aware masking for failed trajectories
            if not is_success:
                if turn_num > late_cutoff:
                    # Late turns: skip entirely
                    if stats:
                        stats.turns_masked += 1
                    continue

                if turn_num > early_cutoff:
                    # Mid-range: filter duplicates
                    if filter_duplicates:
                        action_sig = extract_action_signature(content)
                        if action_sig and action_sig == prev_action_sig:
                            if stats:
                                stats.duplicate_actions_filtered += 1
                                stats.turns_masked += 1
                            continue
                        prev_action_sig = action_sig

            # Format assistant message with thinking
            formatted_content = format_assistant_message(thinking, content)
            sft_messages.append({
                "role": "assistant",
                "content": formatted_content,
            })

            if stats:
                stats.total_turns_output += 1

    # Must have at least system + user + assistant
    assistant_count = sum(1 for m in sft_messages if m["role"] == "assistant")
    if assistant_count == 0:
        return None

    return {
        "messages": sft_messages,
        "images": images,
        "reward": task_reward,
        "task_id": task_id,
        "num_steps": assistant_count,
        "source_dataset": "reasoning_opus",
        "quality_rank": 5 if is_success else 3,
        "metadata": {
            "instruction": instruction,
            "is_success": is_success,
            "original_turns": traj.get("metadata", {}).get("turns", 0),
            "filtered_turns": assistant_count,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process reasoning trajectories into SFT format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        nargs="+",
        help="Input JSONL file(s) with reasoning trajectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file for SFT training",
    )
    parser.add_argument(
        "--early-cutoff",
        type=int,
        default=5,
        help="Keep all turns up to this number for failed trajectories (default: 5)",
    )
    parser.add_argument(
        "--late-cutoff",
        type=int,
        default=10,
        help="Discard all turns >= this for failed trajectories (default: 10)",
    )
    parser.add_argument(
        "--no-filter-duplicates",
        action="store_true",
        help="Don't filter consecutive duplicate actions in mid-range",
    )
    parser.add_argument(
        "--include-all-failed",
        action="store_true",
        help="Include all turns from failed trajectories (no masking)",
    )

    args = parser.parse_args()

    stats = ProcessingStats()

    # Load all input trajectories
    trajectories = []
    for input_file in args.input:
        logger.info(f"Loading trajectories from {input_file}")
        with open(input_file) as f:
            for line in f:
                try:
                    traj = json.loads(line.strip())
                    trajectories.append(traj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Loaded {len(trajectories)} trajectories")
    stats.total_input = len(trajectories)

    # Process trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    for traj in trajectories:
        task_reward = traj.get("task_reward", 0.0)
        if task_reward >= 0.5:
            stats.successful_trajs += 1
        else:
            stats.failed_trajs += 1

        # Override cutoffs if including all
        early = 999 if args.include_all_failed else args.early_cutoff
        late = 999 if args.include_all_failed else args.late_cutoff

        result = process_trajectory(
            traj,
            early_cutoff=early,
            late_cutoff=late,
            filter_duplicates=not args.no_filter_duplicates,
            stats=stats,
        )

        if result:
            processed.append(result)

    # Write output
    with open(args.output, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    logger.info(f"\n{'='*50}")
    logger.info("Processing Statistics")
    logger.info(f"{'='*50}")
    logger.info(f"Input trajectories: {stats.total_input}")
    logger.info(f"  - Successful: {stats.successful_trajs}")
    logger.info(f"  - Failed: {stats.failed_trajs}")
    logger.info(f"Output samples: {len(processed)}")
    logger.info(f"Total turns input: {stats.total_turns_input}")
    logger.info(f"Total turns output: {stats.total_turns_output}")
    logger.info(f"Turns masked (late-stage): {stats.turns_masked}")
    logger.info(f"Duplicate actions filtered: {stats.duplicate_actions_filtered}")
    logger.info(f"Turn retention rate: {100*stats.total_turns_output/max(1,stats.total_turns_input):.1f}%")
    logger.info(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
