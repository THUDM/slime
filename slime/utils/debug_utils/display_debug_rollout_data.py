import json
from pathlib import Path
from typing import Annotated

import torch
import typer

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
):
    for rollout_id in range(100):
        for path in [
            load_debug_rollout_data.format(rollout_id=f"eval_{rollout_id}"),
            load_debug_rollout_data.format(rollout_id=rollout_id),
        ]:
            path = Path(path)
            if not path.exists():
                continue

            print("-" * 80)
            print(f"{rollout_id=} {path=}")
            print("-" * 80)

            pack = torch.load(path)
            samples = pack["samples"]

            for sample in samples:
                print(json.dumps({k: v for k, v in sample.items() if k in _WHITELIST_KEYS}))


if __name__ == "__main__":
    """python -m slime.utils.debug_utils.display_debug_rollout_data --load-debug-rollout-data ..."""
    typer.run(main)
