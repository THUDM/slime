import argparse
import json
from pathlib import Path


def main(input_path, task_type="math", output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".processed.jsonl")

    used_ids = set()
    processed = []

    # First pass: load all lines and collect existing instance_ids
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "instance_id" in item:
                used_ids.add(item["instance_id"])
            processed.append(item)

    # Second pass: assign missing instance_ids
    counter = 0
    for item in processed:
        if "instance_id" not in item:
            # Find unused id
            while True:
                candidate_id = f"{task_type}_{counter}"
                counter += 1
                if candidate_id not in used_ids:
                    item["instance_id"] = candidate_id
                    used_ids.add(candidate_id)
                    break

    # Save to new jsonl file
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(processed)} items. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to input JSONL file")
    parser.add_argument(
        "--task_type",
        type=str,
        default="math",
        help="Task type prefix for new instance_id",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to output file")
    args = parser.parse_args()

    main(args.input_path, args.task_type, args.output_path)
