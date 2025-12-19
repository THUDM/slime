# examples/tau-bench/episode_logger.py
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any


def _truncate(s: str | None, max_chars: int = 8000) -> str | None:
    if s is None:
        return None
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n...[truncated {len(s)-max_chars} chars]"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class EpisodeLogger:
    log_dir: str
    run_meta: dict[str, Any]

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.log_dir, "episode.jsonl")
        self._summary_path = os.path.join(self.log_dir, "summary.json")
        self._run_meta_path = os.path.join(self.log_dir, "run_meta.json")

        with open(self._run_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.run_meta, f, ensure_ascii=False, indent=2)

    def log_step(self, rec: dict[str, Any]):
        # Controlling field size: Truncate long text and append a hash.
        for k in [
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
            if k in rec and isinstance(rec[k], str):
                rec[k + "_hash"] = _sha256_text(rec[k])
                rec[k] = _truncate(rec[k])

        rec["ts"] = time.time()
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def finalize(self, summary: dict[str, Any]):
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
