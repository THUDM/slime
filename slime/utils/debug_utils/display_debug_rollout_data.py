import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import torch
import typer

from slime.ray.rollout import compute_perf_metrics_from_samples
from slime.utils.types import Sample

_WHITELIST_KEYS = [
    "group_index",
    "index",
    "prompt",
    "response",
    "response_length",
    "label",
    "reward",
    "status",
    "metadata",
]


def main(
    # Deliberately make this name consistent with main training arguments
    load_debug_rollout_data: Annotated[str, typer.Option()],
    show_metrics: bool = True,
    show_samples: bool = True,
    category: list[str] = None,
):
    if category is None:
        category = ["train", "eval"]
    for rollout_id, path in _get_rollout_dump_paths(load_debug_rollout_data, category):
        print("-" * 80)
        print(f"{rollout_id=} {path=}")
        print("-" * 80)

        pack = torch.load(path)
        sample_dicts = pack["samples"]

        if show_metrics:
            # TODO read these configs from dumps
            args = SimpleNamespace(
                advantage_estimator="grpo",
                reward_key=None,
                log_reward_category=None,
            )
            sample_objects = [Sample.from_dict(s) for s in sample_dicts]
            metrics = compute_perf_metrics_from_samples(args, sample_objects)
            print("metrics", metrics)

        if show_samples:
            for sample in sample_dicts:
                print(json.dumps({k: v for k, v in sample.items() if k in _WHITELIST_KEYS}))


def _get_rollout_dump_paths(load_debug_rollout_data: str, categories: list[str]):
    # may improve later
    for rollout_id in range(1000):
        for category in categories:
            prefix = {
                "train": "",
                "eval": "eval_",
            }[category]
            path = Path(load_debug_rollout_data.format(rollout_id=f"{prefix}{rollout_id}"))
            if path.exists():
                yield rollout_id, path


if __name__ == "__main__":
    """python -m slime.utils.debug_utils.display_debug_rollout_data --load-debug-rollout-data ..."""
    typer.run(main)
