"""Discard a stopped queue's future before restarting from a checkpoint."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile
from pathlib import Path


def restore_dataset_cursor(producer, cursor, *, shuffle):
    """Validate the whole cursor before changing the producer."""
    counters = ("sample_offset", "epoch_id", "sample_group_index", "sample_index")
    if any(type(cursor[key]) is not int or cursor[key] < 0 for key in counters):
        raise ValueError("Dataset cursor counters must be nonnegative integers")
    if producer.dataset is not None and cursor["sample_offset"] > len(producer.dataset):
        raise ValueError("Dataset cursor offset exceeds the dataset size")
    state = {key: copy.deepcopy(cursor[key]) for key in (*counters, "metadata")}
    if producer.dataset is not None and shuffle:
        producer.dataset.shuffle(state["epoch_id"])
    for key, value in state.items():
        setattr(producer, key, value)


def fresh_samples(value):
    """Keep prompt inputs and stable IDs; drop all generated/training state."""
    from slime.utils.types import Sample

    if isinstance(value, Sample):
        names = (
            "group_index",
            "index",
            "prompt",
            "multimodal_inputs",
            "apply_chat_template_kwargs",
            "label",
            "metadata",
            "generate_function_path",
            "custom_rm_path",
            "train_metadata",
        )
        return Sample(**{name: copy.deepcopy(getattr(value, name)) for name in names})
    if isinstance(value, (list, tuple)):
        return [fresh_samples(item) for item in value]
    raise TypeError(f"Cannot regenerate checkpoint input of type {type(value).__name__}")


def regeneration_groups(groups, producer, cursor, *, shuffle):
    """Recover original prompt groups once, including fanout/partial results."""
    from slime.data.data_source import RolloutDataSource
    from slime.rollout.base_types import iter_samples
    from slime.utils.types import Sample

    restored, seen = [], set()
    for group in groups:
        samples = list(iter_samples(group))
        originals = [getattr(sample, "_queue_original_input", None) for sample in samples]
        if originals and all(originals):
            for ref in originals:
                if ref.manifest.digest not in seen:
                    seen.add(ref.manifest.digest)
                    restored.append(copy.deepcopy(ref.load()))
        elif all(sample.status == Sample.Status.PENDING and not sample.response_length for sample in samples):
            restored.append(copy.deepcopy(group))
        elif type(producer) is RolloutDataSource and producer.dataset is not None:
            # Existing checkpoints predate original-input references. Re-read
            # their prompts from the saved dataset ordering, including metadata
            # that a generation hook may have changed on completed Samples.
            group_id = samples[0].group_index
            key = ("dataset", group_id)
            if key in seen:
                continue
            seen.add(key)
            distance = cursor["sample_group_index"] - group_id
            size = len(producer.dataset)
            absolute = cursor["epoch_id"] * size + cursor["sample_offset"] - distance
            if distance <= 0 or absolute < 0:
                raise ValueError("Checkpoint sample identity is inconsistent with its dataset cursor")
            producer.epoch_id, producer.sample_offset = divmod(absolute, size)
            producer.sample_group_index = group_id
            producer.sample_index = cursor["sample_index"] - distance * producer.args.n_samples_per_prompt
            producer.metadata = copy.deepcopy(cursor["metadata"])
            if shuffle:
                producer.dataset.shuffle(producer.epoch_id)
            restored.extend(producer.get_samples(1))
        else:
            # Dataset-free/custom producers keep their prompt inputs on Samples.
            restored.append(fresh_samples(group))
    restore_dataset_cursor(producer, cursor, shuffle=shuffle)
    return restored


def reset_journal(queue, backend):
    """Atomically install a valid empty WAL, keeping the queue's identity.

    The caller holds the queue lock and must reopen the coordinator afterwards.
    Reopening creates a new epoch, fencing every old lease and receipt.
    """
    from straw.backend import FilesystemBackend
    from straw.journal import Journal

    path = queue.journal.path
    identity = json.loads((path.parent.parent / "run.json").read_text())
    with tempfile.TemporaryDirectory(prefix=".rewind-", dir=path.parent) as temporary:
        staging = FilesystemBackend(temporary, profile=backend.profile, declaration=backend.declaration)
        journal = Journal(staging, identity)
        journal.close()
        queue.close()
        os.replace(staging.path("control/journal.log"), path)
        backend.sync_directory(path.parent)


def discard_future(args, store, step):
    """Remove later Slime source/batch checkpoints and their storage owners.

    Model payloads stay available for explicit model-only loads. Source and
    joint-commit markers are removed so resumed training can save those steps.
    """
    from straw.reporting import write_report

    roots = {Path(value).resolve() for value in (getattr(args, "load", None), getattr(args, "save", None)) if value}
    candidates = set()
    pattern = re.compile(r"(?:committed|queue_state|builder_state|global_dataset_state_dict)_(\d+)\.(?:json|pt)$")
    for root in roots:
        for path in (root / "rollout").glob("*"):
            match = pattern.fullmatch(path.name)
            if match and int(match[1]) > step:
                candidates.add(path)
    template = getattr(args, "save_debug_rollout_data", None)
    if template and "{rollout_id}" in template and template.endswith(".straw.json"):
        glob = Path(template.replace("{rollout_id}", "*"))
        regex = re.compile(re.escape(Path(template).name).replace(re.escape("{rollout_id}"), r"(eval_)?(\d+)"))
        for path in glob.parent.glob(glob.name):
            match = regex.fullmatch(path.name)
            if match and int(match[2]) > step:
                candidates.add(path)
    owners = {}
    for path in sorted(candidates):
        if path.suffix == ".json":
            try:
                owner = json.loads(path.read_text()).get("storage_owner")
                if owner is not None and not isinstance(owner, str):
                    raise ValueError("Invalid storage owner")
                owners[path] = owner
            except (OSError, ValueError, AttributeError) as error:
                logging.getLogger(__name__).warning("Removing unreadable future checkpoint index %s: %s", path, error)
    for path in sorted(candidates):
        if owner := owners.get(path):
            store.release(owner)
        path.unlink()
    for directory in {path.parent for path in candidates}:
        store.backend.sync_directory(directory)
    for root in roots:
        if root.exists():
            write_report(root / "latest_checkpointed_iteration.txt", step)
