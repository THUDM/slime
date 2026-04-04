#!/usr/bin/env python3
"""Generate eval config YAML for MOPD from pool directory structure.

Also pre-processes eval JSONL files to ensure system prompts are present.
For samples lacking a system message, a diverse system prompt is injected
deterministically (based on sample content hash) from per-domain candidates.
Processed files are saved to {output_dir}/eval/ and the config points there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from pool_runtime_semantics import materialize_runtime_pool_row


def _preprocess_eval_jsonl(src: Path, dst: Path, domain: str) -> int:
    """Copy eval JSONL from pool to data_cache, materializing runtime prompt semantics."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = materialize_runtime_pool_row(payload)
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Eval dataset definitions
# ---------------------------------------------------------------------------
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

# Domains that get system prompt injection (math/code handled by their own data prep)
_INJECT_DOMAINS = {"stem", "tool", "structured"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-response-len", type=int, default=8192)
    args = parser.parse_args()

    pool = Path(args.pool_root)
    output_dir = Path(args.output).parent
    eval_data_dir = output_dir / "eval"

    datasets = []
    for name, rel, n_samples in EVAL_DATASETS:
        src_path = pool / rel
        if not src_path.exists():
            print(f"  SKIP {name}: {src_path} not found")
            continue

        domain = _infer_domain_from_pool_rel(rel)

        if domain in _INJECT_DOMAINS:
            dst_path = eval_data_dir / f"{name}.jsonl"
            count = _preprocess_eval_jsonl(src_path, dst_path, domain)
            if count == 0:
                print(f"  SKIP {name}: no valid samples after preprocessing")
                continue
            data_path = str(dst_path)
        else:
            # math/code: use pool path directly (system prompts handled by pool template)
            data_path = str(src_path)

        datasets.append({"name": name, "path": data_path, "n_samples_per_eval_prompt": n_samples})

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
