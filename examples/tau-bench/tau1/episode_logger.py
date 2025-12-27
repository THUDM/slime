from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any


def _truncate(value: str | None, max_chars: int = 8000) -> str | None:
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n...[truncated {len(value) - max_chars} chars]"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class EpisodeLogger:
    log_dir: str
    run_meta: dict[str, Any]

    def __post_init__(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.log_dir, "episode.jsonl")
        self._summary_path = os.path.join(self.log_dir, "summary.json")
        self._run_meta_path = os.path.join(self.log_dir, "run_meta.json")

        with open(self._run_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.run_meta, f, ensure_ascii=False, indent=2)

    def log_step(self, record: dict[str, Any]) -> None:
        # Control field size: truncate long strings and append hashes.
        for key in [
            "assistant_raw",
            "assistant",
            "user_text",
            "observation",
            "tool_result",
            "env_state",
            "normal_text",
            "tool_parse_error",
            "error",
        ]:
            if key in record and isinstance(record[key], str):
                record[f"{key}_hash"] = _sha256_text(record[key])
                record[key] = _truncate(record[key])

        record["ts"] = time.time()
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finalize(self, summary: dict[str, Any]) -> None:
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
