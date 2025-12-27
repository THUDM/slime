### This script is used to preprocess the sokoban dataset into a format that can be used for training.
### Not used in the training pipeline, just an description of how the processed data set (VeraIsHere/sokoban_processed) is derived from the original dataset (Xiaofeng77/sokoban)
from __future__ import annotations

import base64
import io
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Support running as a script from /root/slime without setting PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from examples.vlm_multi_turn.env_sokoban import SokobanEnv, SokobanEnvConfig

OBS_RE = re.compile(r"\[Current Observation\]:\s*(.*?)\nDecide the next action:", re.DOTALL)
SYMBOL_BLOCK_RE = re.compile(
    r"Symbols:\s*.*?Rules:\s*1\. Push boxes \(can't pull\)\.\s*2\. Avoid walls \(#\)\.\s*",
    re.DOTALL,
)


def _image_to_data_uri(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse_grid(grid_text: str) -> tuple[list[list[int]], set[tuple[int, int]], tuple[int, int]]:
    lines = [ln.strip() for ln in grid_text.strip().splitlines() if ln.strip()]
    grid_tokens = []
    for line in lines:
        tokens = [tok.strip() for tok in line.split("\t") if tok.strip()]
        if tokens:
            grid_tokens.append(tokens)

    if not grid_tokens:
        raise ValueError("Empty grid in observation.")

    h, w = len(grid_tokens), len(grid_tokens[0])
    base_grid = [[1 for _ in range(w)] for _ in range(h)]
    boxes: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None

    symbol_map = {
        "#": ("wall", 0),
        "_": ("floor", 1),
        "O": ("target", 2),
        "X": ("box", 4),
        "√": ("box_on_target", 3),
        "P": ("player", 5),
        "S": ("player_on_target", 6),
    }

    for r, row in enumerate(grid_tokens):
        if len(row) != w:
            raise ValueError(f"Inconsistent row width in grid: expected {w}, got {len(row)}")
        for c, sym in enumerate(row):
            if sym not in symbol_map:
                raise ValueError(f"Unknown symbol '{sym}' in grid.")
            kind, _ = symbol_map[sym]
            if kind == "wall":
                base_grid[r][c] = 0
            elif kind == "target":
                base_grid[r][c] = 2
            elif kind == "box":
                boxes.add((r, c))
            elif kind == "box_on_target":
                base_grid[r][c] = 2
                boxes.add((r, c))
            elif kind == "player":
                player = (r, c)
            elif kind == "player_on_target":
                base_grid[r][c] = 2
                player = (r, c)

    if player is None:
        raise ValueError("Player position not found in grid.")

    return base_grid, boxes, player


def _render_from_grid_components(base_grid: list[list[int]], boxes: set[tuple[int, int]], player: tuple[int, int]):
    h, w = len(base_grid), len(base_grid[0])
    cfg = SokobanEnvConfig(
        grid_size=(h, w),
        num_boxes=len(boxes),
        render_mode="vision",
        wall_fraction=0.0,
        initial_base_grid=base_grid,
        initial_boxes=[list(b) for b in boxes],
        initial_player=list(player),
    )
    env = SokobanEnv(cfg)
    env.base_grid = deepcopy(base_grid)
    env.boxes = set(boxes)
    env.player = player
    return env._render_image()


def _extract_prompt_text(prompt_field: Any) -> str:
    if isinstance(prompt_field, str):
        return prompt_field
    if isinstance(prompt_field, (list, tuple, np.ndarray)) and len(prompt_field) > 0:
        first = prompt_field[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    raise ValueError("Unsupported prompt format; expected string or list/array of dicts with 'content'.")


def _transform_prompt(text: str) -> tuple[str, str]:
    m = OBS_RE.search(text)
    if not m:
        raise ValueError("Could not find observation block in prompt.")
    grid_text = m.group(1)

    visual_legend = (
        "Visual legend:\n"
        "- Walls: red brick blocks with light mortar lines.\n"
        "- Floor: solid black tiles.\n"
        "- Targets: red outlined squares with a red diamond in the center.\n"
        "- Boxes: yellow crates with an orange X.\n"
        "- You: bright green alien/robot icon.\n"
    )
    rules = "Rules:\nPush boxes (can't pull).\nAvoid walls (#).\n"

    text = SYMBOL_BLOCK_RE.sub(visual_legend + rules, text)
    text = OBS_RE.sub("[Current Observation]:\n<image>\nDecide the next action:", text)
    return text, grid_text


def process_row(row: pd.Series, images_dir: str | None, idx: int) -> pd.Series:
    prompt_text = _extract_prompt_text(row["prompt"])
    new_prompt, grid_text = _transform_prompt(prompt_text)
    base_grid, boxes, player = _parse_grid(grid_text)
    image = _render_from_grid_components(base_grid, boxes, player)
    data_uri = _image_to_data_uri(image)

    image_path = None
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)
        base_name = str(row.get("id", idx))
        image_path = os.path.join(images_dir, f"{base_name}.png")
        image.save(image_path)

    env_config = {
        "grid_size": (len(base_grid), len(base_grid[0])),
        "num_boxes": len(boxes),
        "render_mode": "vision",
        "max_steps": SokobanEnvConfig().max_steps,
        "wall_fraction": 0.0,
        "initial_base_grid": base_grid,
        "initial_boxes": [list(b) for b in boxes],
        "initial_player": list(player),
    }

    row = row.copy()
    row["prompt"] = new_prompt
    row["images"] = [data_uri]
    row["extra_info"] = {"env_config": env_config}
    if image_path:
        row["image_path"] = image_path
    return row


def _process_dataset(name: str, input_path: str, output_path: str, limit: int | None, images_dir: str | None):
    df = pd.read_parquet(input_path)
    if limit is not None:
        df = df.head(limit)

    processed_rows = []
    success = 0
    failed = 0
    desc = f"Processing {name}"
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
        try:
            processed_rows.append(process_row(row, images_dir, idx))
            success += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] {name} row {idx} failed: {e}")

    processed = pd.DataFrame(processed_rows)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    processed.to_parquet(output_path, index=False)
    print(
        f"[{name}] Saved processed data to {output_path} | requested={len(df)} success={success} failed={failed} written_rows={len(processed)}"
    )


def main(
    input_path: str = "../sokoban/data/train-00000-of-00001.parquet",
    output_path: str = "../sokoban/data/train.parquet",
    eval_input_path: str | None = None,
    eval_output_path: str | None = None,
    limit: int | None = None,
    images_dir: str | None = None,
):
    _process_dataset("train", input_path, output_path, limit, images_dir)
    if eval_input_path and eval_output_path:
        _process_dataset("eval", eval_input_path, eval_output_path, limit, images_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Sokoban parquet into image-based prompts.")
    parser.add_argument("--input", dest="input_path", default="../sokoban/data/train-00000-of-00001.parquet")
    parser.add_argument("--output", dest="output_path", default="../sokoban/data/train.parquet")
    parser.add_argument(
        "--eval-input",
        dest="eval_input_path",
        default=None,
        help="Optional eval parquet to process with the same pipeline.",
    )
    parser.add_argument(
        "--eval-output",
        dest="eval_output_path",
        default=None,
        help="Output path for the processed eval parquet (requires --eval-input).",
    )
    parser.add_argument(
        "--images-dir",
        dest="images_dir",
        default=None,
        help="Optional directory to also save rendered PNGs (one per row).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows for each split and save as a separate file (e.g., train_10.parquet).",
    )
    args = parser.parse_args()

    # If limit is set, append suffixes to avoid overwriting full datasets.
    train_out_path = args.output_path
    eval_out_path = args.eval_output_path
    if args.limit is not None:
        if train_out_path and train_out_path.endswith(".parquet"):
            stem = train_out_path[:-8]
            train_out_path = f"{stem}_{args.limit}.parquet"
        if eval_out_path and eval_out_path.endswith(".parquet"):
            stem = eval_out_path[:-8]
            eval_out_path = f"{stem}_{args.limit}.parquet"

    main(
        input_path=args.input_path,
        output_path=train_out_path,
        eval_input_path=args.eval_input_path,
        eval_output_path=eval_out_path,
        limit=args.limit,
        images_dir=args.images_dir,
    )
