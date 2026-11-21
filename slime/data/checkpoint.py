"""Commit and validate model/source boundaries for in-place Straw recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from straw.protocol import RecordSetRef


@dataclass(frozen=True)
class RestorePlan:
    """Startup decisions, separate from CLI configuration and live queue state."""

    mode: Literal["new", "resume", "rewind", "empty"] = "new"
    root: str | None = None
    expected_session: str | None = None
    session: dict | None = None
    queue_id: str | None = None
    dataset_cursor: str | None = None


@dataclass(frozen=True)
class SourceRestore:
    """Result of restoring a source, consumed by the training batch builder."""

    queue_reset: bool = False
    source_ref: RecordSetRef | None = None


def save_checkpoint(args, rollout_id, actor_model, critic_model, rollout_manager, *, actor_trains):
    """Save training and rollout state, publishing a joint boundary for straw."""
    import ray

    from slime.ray.training_recovery import training_recovery_enabled

    straw_checkpoint = (
        args.rollout_data_transport == "straw"
        and actor_trains
        and not args.debug_train_only
        and not args.debug_rollout_only
    )
    recoverable = training_recovery_enabled(args)
    # A recovery boundary must wait for async model writes to finish before it
    # can declare earlier batches safe to discard.
    force_sync = straw_checkpoint or recoverable or args.release_train or rollout_id == args.num_rollout - 1
    if straw_checkpoint and (Path(args.save) / "rollout" / f"committed_{rollout_id}.json").exists():
        raise FileExistsError("Refusing to overwrite a committed straw checkpoint")
    if actor_trains:
        actor_model.save_model(rollout_id, force_sync=force_sync)
    if args.use_critic:
        critic_model.save_model(rollout_id, force_sync=force_sync)
    ray.get(rollout_manager.save.remote(rollout_id))
    if straw_checkpoint:
        # Failed saves raise above; publish only after all saves return.
        model_args = [actor_model.args]
        if args.use_critic:
            model_args.append(critic_model.args)
        weight_version = ray.get(rollout_manager.get_weight_version.remote())
        commit_checkpoint(args, rollout_id, model_args=model_args, weight_version=weight_version)
    if actor_trains and recoverable:
        # Notify recovery last. Any model, rollout, or joint-commit failure above
        # must leave the previous boundary and its replay batches intact.
        ray.get(rollout_manager.checkpoint_committed.remote(rollout_id))


def commit_checkpoint(args, rollout_id, *, model_args, weight_version=None):
    """Publish the boundary after synchronous model saves and rollout snapshots.

    The driver waits for those calls; a save exception prevents this call.
    Model configurations supply the paths and optimizer/RNG retention policy.
    """
    from straw.reporting import write_report

    root = Path(args.save).resolve()
    path = root / "rollout" / f"committed_{rollout_id}.json"
    if path.exists():
        raise FileExistsError(f"Committed checkpoints are immutable: {path}")
    files = {}
    resumable = True
    for config in model_args:
        model = Path(config.save).resolve() / f"iter_{rollout_id:07d}"
        stateless = getattr(config, "use_stateless_adam", False)
        if stateless and not (model / "opt_param_scheduler.pt").is_file():
            raise ValueError(f"Missing stateless Adam scheduler checkpoint: {model}")
        resumable &= (stateless or not getattr(config, "no_save_optim", False)) and not getattr(
            config, "no_save_rng", False
        )
        if not model.is_relative_to(root):
            raise ValueError("Model checkpoint must be inside --save")
        payloads = [p for p in model.rglob("*") if p.is_file()]
        if not payloads:
            raise ValueError(f"Missing model checkpoint payloads: {model}")
        for item in payloads:
            if item.is_symlink():
                raise ValueError(f"Checkpoint payload must not be a symlink: {item}")
            with item.open("rb") as stream:
                os.fsync(stream.fileno())
                entry = {"size": os.fstat(stream.fileno()).st_size}
                if item.name.endswith(".json") or item.name == ".metadata":
                    entry["sha256"] = hashlib.sha256(stream.read()).hexdigest()
            files[str(item.relative_to(root))] = entry
        for directory in sorted(
            {model, model.parent, *(p.parent for p in payloads)}, key=lambda p: len(p.parts), reverse=True
        ):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for name in ("queue_state", "builder_state"):
        item = root / "rollout" / f"{name}_{rollout_id}.json"
        if not item.exists():
            raise ValueError(f"Missing {name} checkpoint")
        files[str(item.relative_to(root))] = {
            "size": item.stat().st_size,
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        }
    write_report(
        path,
        {
            "version": 1,
            "rollout_id": rollout_id,
            "files": files,
            "resumable": bool(resumable),
            # Manual restarts publish an extra initial sync, so the serving
            # version can advance independently of the training rollout ID.
            "weight_version": rollout_id + 1 if weight_version is None else weight_version,
            "rollout_data_dir": str(Path(args.rollout_data_dir).resolve()),
            "run_id": getattr(args, "rollout_queue_run_id", None) or "rollout",
        },
    )
    write_report(root / "latest_checkpointed_iteration.txt", rollout_id)


def _read_checkpoint(root, step, *, allow_missing_rollout=False):
    """Validate the joint commit, never infer completion from the model tracker."""
    path = root / "rollout" / f"committed_{step}.json"
    value = json.loads(path.read_text())
    if value.get("version") != 1 or value["rollout_id"] != step or not value["resumable"]:
        raise ValueError("Checkpoint is incompatible or omitted optimizer/RNG state")
    queue_state = f"rollout/queue_state_{step}.json"
    dataset_cursor = f"rollout/global_dataset_state_dict_{step}.pt"
    builder_state = f"rollout/builder_state_{step}.json"
    rollout_valid = True
    for name, expected in value["files"].items():
        item = (root / name).resolve()
        if not item.is_relative_to(root):
            raise ValueError(f"Checkpoint component outside save directory: {name}")
        try:
            if item.stat().st_size != expected["size"]:
                raise ValueError(f"Checkpoint component missing or changed: {name}")
            if "sha256" in expected and hashlib.sha256(item.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"Checkpoint component changed: {name}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            if not allow_missing_rollout or name not in {queue_state, dataset_cursor, builder_state}:
                raise
            logging.getLogger(__name__).warning("Cannot restore rollout checkpoint %s: %s", item, error)
            rollout_valid = False
    source_kind = value.get("source_kind")
    if source_kind is None:
        source_kind = "queue" if queue_state in value["files"] else "dataset_cursor"
    source = queue_state if source_kind == "queue" else dataset_cursor
    if (
        source_kind not in {"queue", "dataset_cursor"}
        or source not in value["files"]
        or builder_state not in value["files"]
    ):
        if not allow_missing_rollout:
            raise ValueError("Checkpoint source/builder metadata is missing or invalid")
        logging.getLogger(__name__).warning("Rollout checkpoint metadata is incomplete at %s step %s", root, step)
        rollout_valid = False
    return {**value, "source_kind": source_kind if rollout_valid else None}


def _read_empty_checkpoint(root, step):
    """Require model files even when rollout state cannot be restored."""
    model = root / f"iter_{step:07d}"
    if not model.is_dir() or not any(path.is_file() for path in model.rglob("*")):
        raise FileNotFoundError(f"Requested model checkpoint does not exist: {model}")


def _select_checkpoint(root, step):
    """Honor an explicit step or older tracker; ignore incomplete later saves."""
    tracker = root / "latest_checkpointed_iteration.txt"
    commits = {int(path.stem.split("_")[-1]) for path in (root / "rollout").glob("committed_*.json")}
    if not tracker.exists():
        if step is not None or commits:
            raise FileNotFoundError(f"Missing model checkpoint tracker: {root}")
        return None  # Initial model load, before Slime has saved a checkpoint.
    if step is None:
        step = int(tracker.read_text().strip())
        if commits:
            step = min(step, max(commits))
    if step in commits:
        return root, step, _read_checkpoint(root, step, allow_missing_rollout=True)
    if commits and not (root / f"iter_{step:07d}").exists():
        raise ValueError(f"No committed model/source checkpoint for step {step} in {root}")
    _read_empty_checkpoint(root, step)
    return root, step, None


def _initial_configuration(args):
    """Bind pre-checkpoint WAL recovery to the original model and input stream."""
    return {
        key: getattr(args, key, None)
        for key in (
            "hf_checkpoint",
            "ref_load",
            "prompt_data",
            "input_key",
            "label_key",
            "metadata_key",
            "tool_key",
            "apply_chat_template",
            "n_samples_per_prompt",
            "rollout_seed",
            "rollout_shuffle",
            "rollout_function_path",
            "custom_generate_function_path",
            "custom_rm_path",
            "rollout_queue_run_id",
        )
    }


def resolve_checkpoint(args):
    """Select a model/source boundary without modifying any checkpoint files.

    A checkpoint restart rewinds the existing queue and writes to the requested
    save directory. Before the first checkpoint, recover the original WAL.
    Destructive cleanup is deferred until the controller holds the job lock.
    """
    args = copy.deepcopy(args)
    mode, dataset_cursor = "new", None
    if getattr(args, "rollout_data_transport", None) != "straw":
        return args, RestorePlan()
    if any(getattr(args, key, False) for key in ("debug_train_only", "debug_rollout_only", "load_debug_rollout_data")):
        return args, RestorePlan()
    if not getattr(args, "save", None):
        return args, RestorePlan()
    save_root = Path(args.save).resolve()
    session_path = save_root / "rollout/session.json"
    expected_session = session_path.read_text() if session_path.exists() else None
    previous = json.loads(expected_session) if expected_session else None
    load = getattr(args, "load", None)
    if load is None and previous is not None:
        load = str(save_root)
    root = Path(load).resolve() if load else None
    step = getattr(args, "ckpt_step", None)
    direct_commit = root is not None and root.is_file() and root.name.startswith("committed_")
    if direct_commit:
        selected_step = int(root.stem.split("_")[-1])
        if step is not None and step != selected_step:
            raise ValueError("--ckpt-step differs from the explicit checkpoint marker")
        step, root = selected_step, root.parent.parent
    if root == save_root and previous and previous.get("rewind_pending") and step is None and not direct_commit:
        boundary = previous["restore"]
        root, step = Path(boundary["directory"]), boundary["step"]
    commits = list((root / "rollout").glob("committed_*.json")) if root else []
    if root == save_root and previous and not commits and step is None:
        if boundary := previous.get("restore"):
            root, step = Path(boundary["directory"]), boundary["step"]
        else:
            if previous["initial_configuration"] != _initial_configuration(args):
                raise ValueError("Initial model/input configuration differs from the interrupted run")
            if args.rollout_data_dir is not None and Path(args.rollout_data_dir).resolve() != Path(previous["pool"]):
                raise ValueError("Interrupted run requires its original straw pool")
            args.load, args.save = previous["initial_load"], str(save_root)
            args.rollout_data_dir = previous["pool"]
            session = {**previous, "generation": uuid.uuid4().hex}
            return args, RestorePlan("resume", str(save_root), expected_session, session, session["queue_id"])
    selected = _select_checkpoint(root, step) if root is not None else None

    session = {
        "version": 1,
        "restore": None,
        "rewind_pending": selected is not None,
        "generation": uuid.uuid4().hex,
        "initial_load": getattr(args, "load", None),
        "initial_configuration": _initial_configuration(args),
        "queue_id": previous["queue_id"] if previous else f"run:{uuid.uuid4().hex}",
    }
    if selected is not None:
        root, step, value = selected
        if (
            getattr(args, "finetune", False)
            or getattr(args, "no_load_rng", False)
            or (getattr(args, "no_load_optim", False) and not getattr(args, "use_stateless_adam", False))
        ):
            raise ValueError("Checkpoint restoration requires model, scheduler, RNG and any stateful optimizer state")
        if args.start_rollout_id is not None and args.start_rollout_id != step + 1:
            raise ValueError("--start-rollout-id differs from the checkpoint")
        queue_snapshot = False
        cursor = root / "rollout" / f"global_dataset_state_dict_{step}.pt"
        dataset_cursor = str(cursor) if cursor.exists() else None
        if value is not None:
            if (getattr(args, "rollout_queue_run_id", None) or "rollout") != value["run_id"]:
                raise ValueError("Checkpoint belongs to another straw run")
            if (
                args.rollout_data_dir is not None
                and Path(args.rollout_data_dir).resolve() != Path(value["rollout_data_dir"]).resolve()
            ):
                raise ValueError("Checkpoint restoration requires the original shared straw pool")
            args.rollout_data_dir = value["rollout_data_dir"]
            queue_snapshot = value["source_kind"] == "queue"
        source = root / "rollout" / f"queue_state_{step}.json"
        if source.exists():
            try:
                queue_id = json.loads(source.read_text()).get("queue_id", "rollout")
                if not isinstance(queue_id, str) or not queue_id:
                    raise ValueError("Invalid queue identity")
                session["queue_id"] = queue_id
            except (OSError, ValueError, TypeError, AttributeError) as error:
                logging.getLogger(__name__).warning("Cannot read queue checkpoint %s: %s", source, error)
                queue_snapshot = False
        mode = "rewind" if queue_snapshot else "empty"
        if not queue_snapshot:
            logging.getLogger(__name__).warning(
                "Model checkpoint %s step %s has no usable queue snapshot: starting a new empty queue; %s",
                root,
                step,
                (
                    f"restore dataset cursor from {cursor}"
                    if cursor.exists()
                    else "dataset restarts at offset 0 (not exact data replay)"
                ),
            )
        args.ckpt_step, args.start_rollout_id, args.load = step, step + 1, str(root)
        # Also seed the counter when an older model has no queue snapshot.
        args.update_weight_start_version = value["weight_version"] if value is not None else step + 1
        session["restore"] = {"directory": str(root), "step": step}
    args.save = str(save_root)
    return args, RestorePlan(mode, str(save_root), expected_session, session, session["queue_id"], dataset_cursor)
