#!/usr/bin/env python3
"""Generate eval config YAML for MOPD from pool directory structure."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

EVAL_DATASETS = [
    # (name,  pool-relative path,  n_samples_per_eval_prompt)
    # --- math ---
    ("aime24",          "math/aime24.jsonl",                                      32),
    ("aime25",          "math/aime25.jsonl",                                      32),
    ("amc23",           "math/amc23.jsonl",                                       32),
    ("math500",         "math/math500.jsonl",                                      1),
    ("olympiadmath",    "math/olympiadmath.jsonl",                                 1),
    ("minerva",         "math/minerva.jsonl",                                      1),
    # --- code ---
    ("livecodebench",   "code/livecodebench.jsonl",                                1),
    ("humanevalplus",   "code/humanevalplus.jsonl",                                1),
    ("mbppplus",        "code/mbppplus.jsonl",                                     1),
    # --- tool ---
    ("bfcl_v3",         "tool/eval/bfcl_v3_train-00000-of-00001.jsonl",            1),
    ("bfcl_v3_multi_turn", "tool/eval/bfcl_v3_multi_turn_base_train-00000-of-00001.jsonl", 1),
    # --- stem ---
    ("mmlu_pro",        "stem/eval/mmlu_pro_data_test-00000-of-00001.jsonl",       1),
    ("gpqa",            "stem/eval/gpqa_gpqa_main.jsonl",                          1),
    # --- structured ---
    ("jsonschemabench", "structured/eval/jsonschemabench_data_test-00000-of-00001.jsonl", 1),
    ("ifeval",          "structured/eval/ifeval_ifeval_input_data.jsonl",           1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-response-len", type=int, default=8192)
    args = parser.parse_args()

    pool = Path(args.pool_root)
    datasets = []
    for name, rel, n_samples in EVAL_DATASETS:
        p = pool / rel
        if p.exists():
            datasets.append({"name": name, "path": str(p), "n_samples_per_eval_prompt": n_samples})
        else:
            print(f"  SKIP {name}: {p} not found")

    config = {
        "eval": {
            "defaults": {
                "input_key": "prompt",
                "label_key": "label",
                "metadata_key": "metadata",
                "tool_key": "tools",
                "max_response_len": args.max_response_len,
                "top_k": 1,
            },
            "datasets": datasets,
        }
    }

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote eval config with {len(datasets)} datasets to {dest}")
    for d in datasets:
        print(f"  {d['name']} (n={d['n_samples_per_eval_prompt']})")


if __name__ == "__main__":
    main()
