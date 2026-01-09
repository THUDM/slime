"""OSWorld evaluation script for Slime checkpoints.

Evaluates models on OSWorld tasks using official success metrics.
Process rewards logged for diagnostics only.

Usage:
    python examples/osworld/eval.py \
        --checkpoint /path/to/model \
        --tasks /path/to/tasks.parquet \
        --max-turns 12 \
        --seeds 3
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a GUI automation agent. Complete the task by interacting with the desktop ONE action at a time.

ACTIONS:
- left_click: {"action":"left_click","coordinate":[x,y]}
- type: {"action":"type","text":"string"}
- key: {"action":"key","keys":["ctrl","s"]}
- scroll: {"action":"scroll","direction":"up|down"}
- wait: {"action":"wait"}
- terminate: {"action":"terminate","status":"success|failure"}

OUTPUT FORMAT:
Action: <brief description>
<tool_call>
{"name":"computer_use","arguments":{"action":"<action>",<params>}}
</tool_call>

EXAMPLE 1: Clicking a button
Task: Search for flights
Observation: The page shows a search form. There is a "Search" button at coordinates [650, 400].
Reasoning: I need to click the Search button to submit the form.
Action: Click the Search button
<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[650,400]}}
</tool_call>

EXAMPLE 2: Typing in a field
Task: Enter departure city
Observation: There is an input field labeled "From" at coordinates [300, 200]. The cursor is in the field.
Reasoning: I need to type the departure city into this field.
Action: Type the city name
<tool_call>
{"name":"computer_use","arguments":{"action":"type","text":"New York"}}
</tool_call>

EXAMPLE 3: Selecting from dropdown
Task: Select a date
Observation: A calendar dropdown is open. I see "January 15" at coordinates [400, 350].
Reasoning: I need to click the date to select it.
Action: Click January 15
<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[400,350]}}
</tool_call>

RULES:
- ONE action per response.
- Look at the screenshot and accessibility tree to find the right coordinates.
- Think step by step about what action will make progress toward the goal.
- Do not repeat the same action if the screen has not changed.
- If you use the a11y tree, include the element name in the action arguments.
"""


@dataclass
class EvalResult:
    task_id: str
    seed: int
    success: float
    turns: int
    process_signals: list


def load_model(checkpoint_path: str, device: str = "cuda"):
    logger.info(f"Loading model from {checkpoint_path}")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def parse_action(response: str) -> dict:
    """Parse tool_call JSON from model response.

    Supports two formats:
    1. <tool_call>{"name":"computer_use","arguments":{...}}</tool_call>
    2. {"action": "left_click", "coordinate": [...]} (direct JSON)
    """
    import re

    # Try format 1: <tool_call>{"name":"computer_use","arguments":{...}}</tool_call>
    match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if "arguments" in data:
                return data.get("arguments", {"action": "unknown"})
            # Direct action JSON inside tool_call
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Try format 2: Direct JSON like {"action": "...", ...}
    json_match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+(?:_[^"]+)?"[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"action": "unknown", "raw": response}


def action_to_pyautogui(parsed: dict) -> str:
    """Convert parsed action to pyautogui command."""
    action = parsed.get("action", "").lower()
    if action in ("left_click", "click"):
        coord = parsed.get("coordinate", [0, 0])
        return f"pyautogui.click({coord[0]}, {coord[1]})"
    elif action == "right_click":
        coord = parsed.get("coordinate", [0, 0])
        return f"pyautogui.rightClick({coord[0]}, {coord[1]})"
    elif action == "double_click":
        coord = parsed.get("coordinate", [0, 0])
        return f"pyautogui.doubleClick({coord[0]}, {coord[1]})"
    elif action == "type":
        text = parsed.get("text", "").replace('"', '\\"')
        return f'pyautogui.write("{text}")'
    elif action == "key":
        keys = parsed.get("keys", [])
        if len(keys) == 1:
            return f'pyautogui.press("{keys[0]}")'
        keys_str = ", ".join(f'"{k}"' for k in keys)
        return f"pyautogui.hotkey({keys_str})"
    elif action == "scroll":
        direction = parsed.get("direction", "down")
        amount = -300 if direction == "down" else 300
        return f"pyautogui.scroll({amount})"
    elif action == "wait":
        return "WAIT"
    elif action == "terminate":
        status = parsed.get("status", "success")
        return "DONE" if status == "success" else "FAIL"
    return f"# unknown: {parsed}"


def generate_response(model, processor, messages: list, device: str = "cuda") -> str:
    """Generate single response from model."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True)

    image_inputs = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image" and "image" in item:
                    image_inputs.append(item["image"])

    if image_inputs:
        inputs = processor(text=text, images=image_inputs, return_tensors="pt", padding=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    logger.info(f"[MODEL OUTPUT] {response[:500]}")
    return response


def _compute_screen_diff(prev_img: Image.Image | None, curr_img: Image.Image | None) -> float:
    """Compute normalized screen diff in [0, 1]."""
    if prev_img is None or curr_img is None:
        return 0.0
    from PIL import ImageChops, ImageStat

    prev_small = prev_img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    curr_small = curr_img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    diff = ImageChops.difference(prev_small, curr_small)
    stat = ImageStat.Stat(diff)
    return float(min(1.0, max(0.0, stat.mean[0] / 255.0)))


def run_episode(
    model,
    processor,
    env,
    task_config: dict,
    max_turns: int,
    device: str = "cuda",
) -> EvalResult:
    """Run single evaluation episode."""

    env.reset(task_config=task_config)
    time.sleep(2)

    obs = env._get_obs()
    instruction = task_config.get("instruction", "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    process_signals = []
    prev_screenshot = None

    for turn in range(max_turns):
        screenshot = obs.get("screenshot")
        if isinstance(screenshot, bytes):
            import io

            screenshot = Image.open(io.BytesIO(screenshot)).convert("RGB")

        content = []
        if screenshot:
            content.append({"type": "image", "image": screenshot})

        obs_text = f"[Task]\n{instruction}\n\n"
        a11y = obs.get("accessibility_tree", "")
        if a11y:
            obs_text += f"[Accessibility Tree]\n{a11y[:4096]}\n"
        content.append({"type": "text", "text": obs_text})

        messages.append({"role": "user", "content": content})

        response = generate_response(model, processor, messages, device)
        messages.append({"role": "assistant", "content": response})

        parsed = parse_action(response)
        pyautogui_cmd = action_to_pyautogui(parsed)

        screen_diff = _compute_screen_diff(prev_screenshot, screenshot) if prev_screenshot else 0.0
        process_signals.append(
            {
                "turn": turn,
                "action": parsed.get("action", "unknown"),
                "screen_diff": screen_diff,
            }
        )
        prev_screenshot = screenshot

        if pyautogui_cmd in ("DONE", "FAIL"):
            break
        if pyautogui_cmd == "WAIT":
            time.sleep(1)
            obs = env._get_obs()
            continue

        try:
            obs, reward, done, info = env.step(pyautogui_cmd, 2)
            if done:
                break
        except Exception as e:
            logger.warning(f"Step failed: {e}")
            break

    time.sleep(2)
    success = env.evaluate()

    return EvalResult(
        task_id=task_config.get("id", "unknown"),
        seed=0,
        success=float(success),
        turns=len(process_signals),
        process_signals=process_signals,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tasks", required=True, help="Path to tasks parquet file")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--osworld-path", default="/tmp/OSWorld")
    parser.add_argument("--provider", default="docker")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, args.osworld_path)
    from desktop_env.desktop_env import DesktopEnv

    model, processor = load_model(args.checkpoint, args.device)

    tasks_df = pd.read_parquet(args.tasks)
    logger.info(f"Loaded {len(tasks_df)} tasks from {args.tasks}")

    env = DesktopEnv(
        provider_name=args.provider,
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=True,
        require_a11y_tree=True,
    )

    results = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        for _idx, row in tqdm(tasks_df.iterrows(), total=len(tasks_df), desc=f"Seed {seed}"):
            task_config = row.get("task_config", row.to_dict())
            if isinstance(task_config, str):
                task_config = json.loads(task_config)

            result = run_episode(model, processor, env, task_config, args.max_turns, args.device)
            result.seed = seed
            results.append(result)

            logger.info(f"Task {result.task_id} seed={seed}: success={result.success:.2f} turns={result.turns}")

    env.close()

    success_rate = sum(r.success for r in results) / len(results) if results else 0
    logger.info(f"Success rate: {success_rate:.2%} ({sum(r.success for r in results):.0f}/{len(results)})")

    output_data = {
        "checkpoint": args.checkpoint,
        "tasks": args.tasks,
        "max_turns": args.max_turns,
        "seeds": args.seeds,
        "success_rate": success_rate,
        "results": [
            {
                "task_id": r.task_id,
                "seed": r.seed,
                "success": r.success,
                "turns": r.turns,
                "process_signals": r.process_signals,
            }
            for r in results
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
