import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .data_utils import ensure_metadata_dict, normalize_prompt, normalize_reward_model


def _render_messages(tokenizer, messages: list[dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001
        rendered = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            rendered.append(f"{role}: {msg.get('content', '')}")
        rendered.append("Assistant:")
        return "\n".join(rendered)


@dataclass
class SimpleTIRRecord:
    prompt: list[dict[str, Any]]
    ability: str
    reward_style: str
    ground_truth: Any | None
    data_source: Optional[str]
    extra_info: Dict[str, Any]


class SimpleTIRDataset:
    """Loader for SimpleTIR parquet datasets."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"SimpleTIR dataset not found at {self.path}")
        if self.path.suffix == ".parquet":
            self._df = pd.read_parquet(self.path)
        elif self.path.suffix == ".jsonl":
            self._df = pd.read_json(self.path, lines=True)
        else:
            raise ValueError(
                f"Unsupported SimpleTIR dataset format {self.path.suffix!r}. Expected .parquet or .jsonl files."
            )
        if "ability" not in self._df.columns:
            raise ValueError("SimpleTIR dataset must contain an 'ability' column.")
        if "prompt" not in self._df.columns:
            raise ValueError("SimpleTIR dataset must contain a 'prompt' column.")
        if "reward_model" not in self._df.columns:
            raise ValueError("SimpleTIR dataset must contain a 'reward_model' column.")
        if "extra_info" not in self._df.columns:
            self._df["extra_info"] = [{}] * len(self._df)

        self._df["prompt"] = self._df["prompt"].apply(normalize_prompt)
        self._df["reward_model"] = self._df["reward_model"].apply(normalize_reward_model)
        self._df["extra_info"] = self._df["extra_info"].apply(ensure_metadata_dict)

        # Build lookup tables
        self._max_prompt_tokens_filtered: int | None = None

        self._rebuild_indices()

    def __len__(self) -> int:
        return len(self._df)

    def _rebuild_indices(self):
        self._df = self._df.reset_index(drop=True)
        self._df["__row_id__"] = self._df.index
        self._extra_index_to_row_id: dict[int, int] = {}
        for row_id, extra in zip(self._df["__row_id__"], self._df["extra_info"]):
            idx = None
            if isinstance(extra, dict):
                idx = extra.get("index")
            if idx is not None:
                self._extra_index_to_row_id[int(idx)] = int(row_id)

    def ensure_prompt_limit(self, tokenizer, max_tokens: int | None):
        if max_tokens is None or max_tokens <= 0:
            return
        if self._max_prompt_tokens_filtered == max_tokens:
            return

        keep_indices = []
        for row_id, prompt in enumerate(self._df["prompt"]):
            messages = prompt or [{"role": "user", "content": ""}]
            prompt_text = _render_messages(tokenizer, messages)
            length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if length <= max_tokens:
                keep_indices.append(row_id)

        if not keep_indices:
            raise ValueError(f"No SimpleTIR samples remain after enforcing prompt length <= {max_tokens} tokens.")

        self._df = self._df.iloc[keep_indices]
        self._rebuild_indices()
        self._max_prompt_tokens_filtered = max_tokens

    @functools.lru_cache(maxsize=8192)
    def get_record(self, *, row_id: int | None = None, extra_index: int | None = None) -> SimpleTIRRecord:
        if row_id is None:
            if extra_index is None:
                raise ValueError("row_id or extra_index must be provided to fetch SimpleTIR record.")
            if extra_index not in self._extra_index_to_row_id:
                raise KeyError(f"No record found for extra_info.index={extra_index}")
            row_id = self._extra_index_to_row_id[extra_index]

        row = self._df.iloc[row_id]

        reward_model = row["reward_model"] or {}
        style = reward_model.get("style")
        ground_truth = reward_model.get("ground_truth")

        record = SimpleTIRRecord(
            prompt=row["prompt"],
            ability=row["ability"],
            reward_style=style,
            ground_truth=ground_truth,
            data_source=row.get("data_source"),
            extra_info=row.get("extra_info") or {},
        )
        return record


_DATASETS: dict[tuple[str, str], SimpleTIRDataset] = {}


def get_dataset_from_path(path: str | Path, *, split: str) -> SimpleTIRDataset:
    cache_key = (split, str(path))
    if cache_key not in _DATASETS:
        _DATASETS[cache_key] = SimpleTIRDataset(path)
    return _DATASETS[cache_key]


def get_dataset(args, *, split: str) -> SimpleTIRDataset:
    """Retrieve cached dataset instance based on args and split."""
    if split not in ("train", "eval"):
        raise ValueError(f"Unknown split {split!r} for SimpleTIR dataset.")

    if split == "train":
        key = getattr(args, "prompt_data", None)
    else:
        key = getattr(args, "eval_prompt_data", None)
        if isinstance(key, list) and key:
            key = key[1] if len(key) >= 2 else key[0]
    if key is None:
        raise ValueError(
            f"--{'prompt-data' if split == 'train' else 'eval-prompt-data'} must be provided for SimpleTIR integration."
        )
    return get_dataset_from_path(key, split=split)
