"""Pre-materialize pool/train data so that downstream consumers
can read the prompt and metadata in the runtime shape expected by training.

New-format pool rows carry supervision in top-level family-specific fields
plus ``supervision_family`` instead of ``metadata.ground_truth``. 
``materialize_runtime_pool_row`` bridges the gap, but
``slime/slime/utils/data.py`` does not call it during training.

For MOPD this is still useful even though the reward comes from the teacher
model: materialization injects the runtime prompt shape and family-specific
metadata before the generic data loader reads the files.

This script reads pool JSONL files, runs materialization, and writes the result
to a cache directory.  The output ``.list`` file points to the materialized
files so the training pipeline can load them transparently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow importing from examples/
_examples_dir = Path(__file__).resolve().parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from pool_runtime_semantics import materialize_runtime_pool_row  # noqa: E402


def materialize_file(src: Path, dst: Path) -> int:
    """Materialize a single pool JSONL file.  Returns number of rows written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open(encoding="utf-8", errors="replace") as fin, \
         dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            materialized = materialize_runtime_pool_row(row)
            if materialized is None:
                # Row filtered (e.g. empty top-level ground_truth for a train family)
                continue
            fout.write(json.dumps(materialized, ensure_ascii=False) + "\n")
            count += 1
    return count


def _file_needs_materialize(src: Path) -> bool:
    with src.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                return bool(row.get("supervision_family"))
    return False


def main():
    pool_root = Path(sys.argv[1]).resolve()
    cache_dir = Path(sys.argv[2]).resolve()
    dest_list = Path(sys.argv[3]).resolve()
    include_domains = [d.strip() for d in sys.argv[4].split(",") if d.strip()]
    exclude_patterns = [p.strip() for p in sys.argv[5].split(",") if p.strip()]

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_list.parent.mkdir(parents=True, exist_ok=True)

    materialized_paths: list[Path] = []

    for domain in include_domains:
        domain_root = pool_root / domain
        if not domain_root.exists():
            continue
        if domain in {"tool", "stem", "structured"}:
            candidates = sorted((domain_root / "train").glob("*.jsonl"))
        else:
            candidates = sorted(domain_root.glob("*.jsonl"))

        for src_path in candidates:
            if src_path.name.startswith("._"):
                continue
            rel = src_path.relative_to(pool_root).as_posix()
            if any(pattern in rel for pattern in exclude_patterns):
                continue

            needs_materialize = False
            try:
                needs_materialize = _file_needs_materialize(src_path)
            except Exception:
                pass

            if needs_materialize:
                dst_path = cache_dir / rel
                count = materialize_file(src_path, dst_path)
                print(f"  materialized {rel}: {count} rows -> {dst_path}")
                materialized_paths.append(dst_path)
            else:
                # Old format: use original file directly
                materialized_paths.append(src_path.resolve())

    with dest_list.open("w", encoding="utf-8") as handle:
        for path in materialized_paths:
            handle.write(f"{path}\n")

    print(f"wrote {len(materialized_paths)} training sources to {dest_list}")
    for path in materialized_paths:
        print(path)


if __name__ == "__main__":
    main()
