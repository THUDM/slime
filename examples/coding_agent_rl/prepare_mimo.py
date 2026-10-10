"""Convert Xiaomi MiMo's code dataset to slime's coding-agent example schema.

Hidden tests are encoded only in the grading command; they are never uploaded
to the agent sandbox. Grading uses a fresh sandbox and resets test paths to HEAD
before applying the official test patch, matching MiMo's reward contract.
"""

import argparse
import base64
import json
import re
import shlex
from pathlib import Path

import pyarrow.parquet as pq


def evaluation_command(instance):
    touched = []
    for line in instance["test_patch"].splitlines():
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if match:
            for name in match.groups():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Test patch escapes repository")
                if name not in touched:
                    touched.append(name)
    commands = ["set -e"]
    for name in touched:
        q = shlex.quote(name)
        commands.append(
            f"if git cat-file -e HEAD:{q} 2>/dev/null; then git checkout HEAD -- {q}; else git rm -f --cached -- {q} >/dev/null 2>&1 || true; rm -f -- {q}; fi"
        )
    payload = base64.b64encode(instance["test_patch"].encode()).decode()
    commands += [
        f"printf %s {shlex.quote(payload)} | base64 -d > /tmp/mimo-tests.patch",
        "git apply --verbose /tmp/mimo-tests.patch",
        instance["test_command"],
    ]
    return "bash -lc " + shlex.quote("\n".join(commands))


def convert(data: Path, instance_ids: list[str] | None = None) -> list[dict]:
    selected = set(instance_ids) if instance_ids is not None else None
    if selected is not None and len(selected) != len(instance_ids):
        raise ValueError("--ids must not contain duplicate instance IDs")
    mapping = {
        r["dataset_image"]: r["dockerhub_image"]
        for r in map(json.loads, (data / "image-mapping.jsonl").read_text().splitlines())
    }
    rows = []
    found = set()
    for row in pq.read_table(data / "code.parquet", columns=["extra_info"]).to_pylist():
        instance = json.loads(row["extra_info"]["instance_json"])
        instance_id = instance["instance_id"]
        if selected is not None and instance_id not in selected:
            continue
        if instance_id in found:
            raise ValueError(f"Duplicate dataset instance ID: {instance_id}")
        found.add(instance_id)
        rows.append(
            {
                "prompt": [{"role": "user", "content": instance["problem_statement"]}],
                "label": instance["instance_id"],
                "metadata": {
                    "instance_id": instance["instance_id"],
                    "image": mapping[instance["docker_image"]],
                    "workdir": instance["cwd"],
                    "problem_statement": instance["problem_statement"],
                    "eval_cmd": evaluation_command(instance),
                    # Official MiMo images initialize their toolchains from
                    # root's login profile. Only the fresh grader uses root;
                    # the coding CLI continues to run as the agent user.
                    "eval_user": "root",
                    "dataset": "XiaomiMiMo/MiMo-V2.6-RL-oss",
                },
            }
        )
    if selected is not None and (missing := selected - found):
        raise ValueError(f"Missing instance IDs: {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError("No code tasks found")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", type=Path, required=True, help="Directory containing code.parquet and image-mapping.jsonl"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+", help="Select specific instance IDs; omit to convert the full code dataset")
    a = p.parse_args()
    rows = convert(a.data, a.ids)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"output": str(a.output), "tasks": len(rows), "images": len({r["metadata"]["image"] for r in rows})}
    if a.ids:
        summary["instances"] = [r["label"] for r in rows]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
