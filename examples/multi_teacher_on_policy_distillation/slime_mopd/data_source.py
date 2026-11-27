"""Rollout data source for a domain-labelled prompt mixture.

The example ships one prompt file instead of one file per domain. Every row
carries:

* ``prompt``      -- a chat message list (or a plain string), system prompt included
* ``domain``      -- ``math`` / ``code`` / ``ifeval``, the teacher-routing label
* ``label``       -- reference answer, kept for evaluation, unused by the OPD loss
* ``metadata``    -- optional provenance, kept on the sample

slime's ``Dataset`` keeps only ``prompt``/``label``/``metadata`` columns, so the
domain label would be lost. This data source reads the same file itself, applies
the student chat template, stamps the domain on each sample's ``source`` (the
field slime forwards to the training ranks as ``source_names``), and then hands
the samples to the standard rollout loop. Routing only needs that label.

Use with::

    --prompt-data /data/open-mopd/rl_prompt_mix/train.parquet
    --input-key prompt --apply-chat-template
    --data-source-path slime_mopd.data_source.MopdRolloutDataSource

When no MOPD config is present it falls back to slime's single-file behavior
unchanged, so it stays a drop-in replacement for
``RolloutDataSourceWithBuffer``.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from slime_mopd.config import MopdConfig, MopdConfigError, load_mopd_config
from slime_mopd.routing import DOMAIN_METADATA_KEYS

from slime.data.data_source import RolloutDataSourceWithBuffer
from slime.utils.data import filter_long_prompt, read_file
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class _SampleListDataset:
    """Minimal ``Dataset``-like view over already-built samples."""

    def __init__(self, samples: list, *, seed: int, shuffle: bool):
        self.origin_samples = samples
        self.samples = samples
        self.seed = seed
        self._shuffle_enabled = shuffle
        self.epoch_id = -1

    def shuffle(self, new_epoch_id: int) -> None:
        if not self._shuffle_enabled or self.epoch_id == new_epoch_id:
            return
        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.origin_samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __len__(self) -> int:
        return len(self.samples)


def _iter_parquet_rows(path: str):
    """Stream a parquet file one row group at a time.

    ``slime.utils.data.read_file`` reads parquet with ``iter_batches()``, which
    raises ``ArrowNotImplementedError`` on tables with nested columns like
    ``prompt`` (``list<struct<role, content>>``) on some pyarrow builds.
    Row-group reads convert cleanly and keep the file streaming.
    """

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.num_row_groups):
        yield from parquet.read_row_group(row_group).to_pylist()


def _iter_rows(path: str):
    if str(path).endswith(".parquet"):
        yield from _iter_parquet_rows(path)
    else:
        yield from read_file(path)


def _row_domain(row: dict[str, Any]) -> str | None:
    """Return the ``domain`` label of one row."""

    for key in ("domain", "teacher_id"):
        value = row.get(key)
        if value:
            return str(value)

    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = None
    if isinstance(metadata, dict):
        for key in DOMAIN_METADATA_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
    return None


def _row_messages(row: dict[str, Any], prompt_key: str) -> Any:
    prompt = row.get(prompt_key)
    if prompt is None:
        raise ValueError(f"row has no {prompt_key!r} column; available columns: {sorted(row)}")
    return prompt


def _row_metadata(row: dict[str, Any], domain: str | None) -> dict[str, Any]:
    """Keep the routing label plus the provenance fields worth logging."""

    metadata: dict[str, Any] = {}
    if domain is not None:
        metadata["mopd_domain"] = domain
    data_source = row.get("data_source")
    if data_source:
        metadata["data_source"] = str(data_source)
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        dataset = extra_info.get("dataset")
        if dataset:
            metadata["dataset"] = str(dataset)
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("style"):
        metadata["reward_style"] = str(reward_model["style"])
    return metadata


def _build_samples(
    args: Any,
    path: str,
    config: MopdConfig,
    tokenizer,
    processor,
) -> tuple[list[Sample], dict[str, int]]:
    """Read the prompt file and turn every row into a routable ``Sample``."""

    prompt_key = getattr(args, "input_key", "prompt") or "prompt"
    label_key = getattr(args, "label_key", None)
    template_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    apply_template = bool(getattr(args, "apply_chat_template", False))

    samples: list[Sample] = []
    domain_counts: dict[str, int] = {}

    for row in _iter_rows(path):
        domain = _row_domain(row)
        label = row.get(label_key) if label_key else None

        prompt = _row_messages(row, prompt_key)
        if apply_template:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

        sample = Sample(
            prompt=prompt,
            label=None if label is None else str(label),
            metadata=_row_metadata(row, domain),
            multimodal_inputs=None,
        )
        if domain is not None:
            # ``source`` is what slime forwards to the training ranks.
            sample.source = domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        samples.append(sample)

    max_length = getattr(args, "rollout_max_prompt_len", None)
    if max_length is not None and apply_template:
        before = len(samples)
        samples = filter_long_prompt(samples, tokenizer, processor, max_length)
        if len(samples) != before:
            logger.info(
                "mopd: dropped %d/%d prompts longer than %s tokens",
                before - len(samples),
                before,
                max_length,
            )

    missing = sorted(
        domain
        for domain in domain_counts
        if config.teacher_for_domain(domain) is None and config.default_teacher is None
    )
    if missing:
        raise ValueError(
            f"MOPD config has no teacher for domain(s) {missing}; "
            f"declared teachers: {[teacher.id for teacher in config.teachers]}"
        )

    logger.info(
        "mopd: loaded %d prompts from %s, domains: %s",
        len(samples),
        path,
        dict(sorted(domain_counts.items())),
    )
    return samples, domain_counts


class MopdRolloutDataSource(RolloutDataSourceWithBuffer):
    """DataSource that keeps the row's ``domain`` label on every sample."""

    def __init__(self, args):
        try:
            config = load_mopd_config(args)
        except MopdConfigError:
            config = None

        samples = None
        if config is not None and getattr(args, "prompt_data", None):
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
            samples, _ = _build_samples(args, args.prompt_data, config, tokenizer, processor)

        # ``samples`` already went through the chat template and the length
        # filter; hide the path from the base class so the same file is not
        # parsed and tokenized a second time.
        prompt_data = getattr(args, "prompt_data", None)
        if samples is not None:
            args.prompt_data = None
        try:
            super().__init__(args)
        finally:
            args.prompt_data = prompt_data

        if samples is not None:
            self.dataset = _SampleListDataset(
                samples,
                seed=args.rollout_seed,
                shuffle=args.rollout_shuffle,
            )
            if args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
