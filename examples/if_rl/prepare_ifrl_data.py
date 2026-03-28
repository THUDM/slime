#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def normalize_prompt(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(part for part in parts if part)
    return str(prompt)


def main():
    parser = argparse.ArgumentParser(description="Normalize IF-RL jsonl into slime prompt dataset format.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_text = normalize_prompt(row.get("prompt", ""))
            out = {
                "prompt": prompt_text,
                "label": "",
                "metadata": {
                    "record_id": row.get("id", 0),
                    "prompt_text": prompt_text,
                    "instruction_id_list": row.get("instruction_id_list") or [],
                    "kwargs": row.get("kwargs") or [],
                    "dataset": row.get("dataset"),
                    "agent_ref": row.get("agent_ref"),
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
