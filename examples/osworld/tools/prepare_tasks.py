"""Prepare OSWorld Tasks for Training.

This script converts OSWorld evaluation examples into the training format
expected by slime's rollout pipeline. It creates a task index with:
- Task configs for environment initialization
- Initial screenshots as base64 or file references
- Neutral prompt identifiers for logging

Usage:
    python examples/osworld/prepare_tasks.py \\
        --osworld-path /path/to/OSWorld \\
        --output-dir /root/datasets/osworld \\
        --domains chrome,os

Output:
    - train.parquet: Training samples with task configs
    - test.parquet: Evaluation samples
    - task_index.json: Task metadata for debugging
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Domains to include (subset for initial validation)
DEFAULT_DOMAINS = ["chrome", "os"]


def image_to_data_uri(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to data URI string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


def load_osworld_task(task_path: Path) -> dict | None:
    """Load a single OSWorld task from its JSON file.

    Args:
        task_path: Path to task JSON file (e.g., {domain}/{task_id}.json)

    Returns:
        Task config dict or None if invalid
    """
    if not task_path.exists() or not task_path.suffix == ".json":
        return None

    try:
        with open(task_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {task_path}: {e}")
        return None

    config["_task_path"] = str(task_path)
    # No initial screenshots in OSWorld - they're captured at runtime
    config["_screenshot_path"] = None

    return config


def discover_tasks(osworld_path: Path, domains: list[str]) -> list[dict]:
    """Discover all tasks from OSWorld evaluation examples.

    OSWorld structure:
        evaluation_examples/examples/{domain}/{task_id}.json

    Args:
        osworld_path: Path to OSWorld repository
        domains: List of domains to include

    Returns:
        List of task config dicts
    """
    tasks = []
    # OSWorld uses evaluation_examples/examples/ subdirectory
    eval_examples_dir = osworld_path / "evaluation_examples" / "examples"

    if not eval_examples_dir.exists():
        # Try direct path for backwards compatibility
        eval_examples_dir = osworld_path / "evaluation_examples"
        if not eval_examples_dir.exists():
            logger.error(f"Evaluation examples not found at {eval_examples_dir}")
            return tasks

    for domain in domains:
        domain_dir = eval_examples_dir / domain
        if not domain_dir.exists():
            logger.warning(f"Domain directory not found: {domain_dir}")
            continue

        # Each .json file is a task
        for task_path in domain_dir.glob("*.json"):
            task = load_osworld_task(task_path)
            if task:
                task["domain"] = domain
                tasks.append(task)

    logger.info(f"Discovered {len(tasks)} tasks across {len(domains)} domains")
    return tasks


def format_task_for_training(task: dict, include_screenshot: bool = True) -> dict[str, Any]:
    """Convert OSWorld task to training sample format.

    Note: OSWorld doesn't include initial screenshots in task definitions.
    Screenshots are captured at runtime when the environment is initialized.
    The system prompt is injected at runtime from examples/osworld/env.py.

    Args:
        task: Task config from discover_tasks
        include_screenshot: Whether to include initial screenshot (not available for OSWorld)

    Returns:
        Training sample dict with prompt, metadata, images
    """
    task_id = task.get("id", task.get("_task_path", "unknown"))
    instruction = task.get("instruction", "")
    domain = task.get("domain", "unknown")

    prompt = str(task_id)

    # Task config for environment initialization (exclude internal fields)
    task_config = {k: v for k, v in task.items() if not k.startswith("_")}

    return {
        "prompt": prompt,
        "images": [],  # Populated at runtime from environment
        "metadata": {
            "task_config": task_config,
            "task_id": task_id,
            "domain": domain,
            "instruction": instruction,
        },
    }


def split_tasks(tasks: list[dict], test_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split tasks into train and test sets.

    Args:
        tasks: List of task configs
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_tasks, test_tasks)
    """
    import random

    random.seed(seed)
    shuffled = list(tasks)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_to_parquet(samples: list[dict], output_path: Path):
    """Save samples to parquet format.

    Args:
        samples: List of training samples
        output_path: Path to output parquet file
    """
    # Convert samples to flat dict format for parquet
    # Serialize nested structures as JSON strings for compatibility
    records = []
    for sample in samples:
        record = {
            "prompt": json.dumps(sample["prompt"]),  # Serialize to JSON string
            "images": json.dumps(sample.get("images", [])),
            "task_id": sample["metadata"]["task_id"],
            "domain": sample["metadata"]["domain"],
            "instruction": sample["metadata"]["instruction"],
            "task_config": json.dumps(sample["metadata"]["task_config"]),
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(records)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare OSWorld tasks for training")
    parser.add_argument(
        "--osworld-path",
        type=str,
        required=True,
        help="Path to OSWorld repository",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/datasets/osworld",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated list of domains to include",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of tasks for test set",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Skip including screenshots in output",
    )
    args = parser.parse_args()

    osworld_path = Path(args.osworld_path)
    output_dir = Path(args.output_dir)
    domains = [d.strip() for d in args.domains.split(",")]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tasks
    tasks = discover_tasks(osworld_path, domains)
    if not tasks:
        logger.error("No tasks found. Check OSWorld path and domains.")
        return

    # Split into train/test
    train_tasks, test_tasks = split_tasks(tasks, args.test_ratio)
    logger.info(f"Split: {len(train_tasks)} train, {len(test_tasks)} test")

    # Format for training
    train_samples = []
    for task in tqdm(train_tasks, desc="Formatting train tasks"):
        sample = format_task_for_training(task, not args.no_screenshots)
        train_samples.append(sample)

    test_samples = []
    for task in tqdm(test_tasks, desc="Formatting test tasks"):
        sample = format_task_for_training(task, not args.no_screenshots)
        test_samples.append(sample)

    # Save to parquet
    save_to_parquet(train_samples, output_dir / "train.parquet")
    save_to_parquet(test_samples, output_dir / "test.parquet")

    # Save task index for debugging
    task_index = {
        "train": [s["metadata"]["task_id"] for s in train_samples],
        "test": [s["metadata"]["task_id"] for s in test_samples],
        "domains": domains,
        "total_tasks": len(tasks),
    }
    with open(output_dir / "task_index.json", "w") as f:
        json.dump(task_index, f, indent=2)

    logger.info(f"Done. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
