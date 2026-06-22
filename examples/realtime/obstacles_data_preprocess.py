"""Build a seed dataset for the clear-obstacles environment.

Each example is just a (system prompt, seed) pair. The seed fully determines the
grid the agent will face (obstacle layout, start position), so "initializing the
environment" is all that is needed to materialize a problem instance.

The seed is the crucial field: it is carried in the ``seed`` column and wired into
slime via ``--label-key seed`` so it arrives on ``sample.label``. At rollout time
``generate_with_obstacles.generate`` reconstructs the *exact* same environment from
that seed (``ClearObstaclesToolEnv.reset(seed=...)``) and keeps that single instance
live across the whole tool-calling loop, so every move acts on the real grid the
seed describes. The constant ``prompt`` column carries the game rules via
``--input-key prompt``.

Run from the directory that contains ``slime/`` with the real-time environment on
the path::

    PYTHONPATH=./real-time python3 slime/examples/realtime/obstacles_data_preprocess.py

Requires the ``environment`` package (``real-time/environment``) to be importable.
"""

import argparse
import json
import os
import random

from environment.clear_obstacles import CLEAR_SYSTEM_PROMPT


def build_rows(n: int, rng: random.Random) -> list[dict]:
    return [
        {"prompt": CLEAR_SYSTEM_PROMPT, "seed": rng.randint(0, 2**31 - 1)}
        for _ in range(n)
    ]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.expanduser("~/obstacles-seeds/train.jsonl"))
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    write_jsonl(args.out, build_rows(args.train_size, rng))


if __name__ == "__main__":
    main()
