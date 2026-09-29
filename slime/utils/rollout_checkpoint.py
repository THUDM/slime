"""Commit and validate model/source boundaries for straw checkpoint forks."""

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path


def save_checkpoint(args, rollout_id, actor_model, critic_model, rollout_manager, *, actor_trains):
    """Save training and rollout state, publishing a joint boundary for straw."""
    import ray

    straw_checkpoint = (
        args.rollout_data_transport == "straw"
        and actor_trains
        and not args.debug_train_only
        and not args.debug_rollout_only
    )
    force_sync = straw_checkpoint or args.release_train or rollout_id == args.num_rollout - 1
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
        commit_checkpoint(args, rollout_id, model_args=model_args)


def commit_checkpoint(args, rollout_id, *, model_args):
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
        resumable &= not getattr(config, "no_save_optim", False) and not getattr(config, "no_save_rng", False)
        model = Path(config.save).resolve() / f"iter_{rollout_id:07d}"
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
            # train.py syncs once before rollout 0, then after each rollout.
            # This save precedes the next sync, so version = rollout_id + 1.
            "weight_version": rollout_id + 1,
            "rollout_data_dir": str(Path(args.rollout_data_dir).resolve()),
            "run_id": getattr(args, "rollout_queue_run_id", None) or "rollout",
        },
    )
    # Mirror the active branch's committed step at the logical root. Publish the
    # tracker first: a crash before the pointer update still selects this valid
    # commit, rather than mistaking the previous step for a user rollback.
    if (plan := getattr(args, "_checkpoint_plan", None)) is not None:
        logical_root = Path(plan["root"])
        current = logical_root / "rollout/current.json"
        value = json.loads(current.read_text())
        if Path(value["directory"]) != root:
            raise RuntimeError("Current branch changed while saving its checkpoint")
        write_report(logical_root / "latest_checkpointed_iteration.txt", rollout_id)
        value["tracker"] = str(rollout_id)
        write_report(current, value)


def _read_checkpoint(root, step):
    """Validate the joint commit, never infer completion from the model tracker."""
    path = root / "rollout" / f"committed_{step}.json"
    value = json.loads(path.read_text())
    if value.get("version") != 1 or value["rollout_id"] != step or not value["resumable"]:
        raise ValueError("Checkpoint is incompatible or omitted optimizer/RNG state")
    for name, expected in value["files"].items():
        item = (root / name).resolve()
        if not item.is_relative_to(root) or item.stat().st_size != expected["size"]:
            raise ValueError(f"Checkpoint component missing or changed: {name}")
        if "sha256" in expected and hashlib.sha256(item.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Checkpoint component changed: {name}")
    return value


def _read_empty_checkpoint(root, step):
    """Allow absent queue state, but never downgrade a broken saved snapshot."""
    for name in ("queue_state", "builder_state"):
        if (root / "rollout" / f"{name}_{step}.json").exists():
            raise ValueError("Queue snapshot exists without a complete joint commit; refusing empty-queue fallback")
    model = root / f"iter_{step:07d}"
    if not model.is_dir() or not any(path.is_file() for path in model.rglob("*")):
        raise FileNotFoundError(f"Requested model checkpoint does not exist: {model}")


def _select_checkpoint(directory, step, *, allow_empty=False):
    """Follow the current branch and its ancestors, excluding abandoned futures."""
    root = directory
    current = root / "rollout/current.json"
    if current.exists():
        root = Path(json.loads(current.read_text())["directory"]).resolve()
    visited = set()
    while root not in visited:
        visited.add(root)
        commits = list((root / "rollout").glob("committed_*.json"))
        if step is None and commits:
            step = max(int(path.stem.split("_")[-1]) for path in commits)
        if step is not None and (root / "rollout" / f"committed_{step}.json").exists():
            return root, step, _read_checkpoint(root, step)
        if step is not None and allow_empty and (root / f"iter_{step:07d}").exists():
            _read_empty_checkpoint(root, step)
            return root, step, None
        branch = root / "rollout/branch.json"
        if not branch.exists():
            break
        parent = json.loads(branch.read_text()).get("parent")
        if parent is None:
            break
        if step is not None and step > parent["step"]:
            break
        step = parent["step"] if step is None else step
        allow_empty = allow_empty or not parent.get("queue_snapshot", True)
        root = Path(parent["directory"]).resolve()
    raise ValueError(f"No committed model/source checkpoint for step {step} in {directory}'s branch history")


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
    """Translate ordinary load/save arguments into an isolated checkpoint branch.

    Each checkpoint restore gets fresh queue authority and output files. A small
    current pointer lets callers keep using the same logical save directory.
    Before the first checkpoint, the original run can instead recover its WAL.
    """
    if (plan := getattr(args, "_checkpoint_plan", None)) is not None:
        if parent := plan["branch"].get("parent"):
            if parent["queue_snapshot"]:
                _read_checkpoint(Path(parent["directory"]), parent["step"])
            else:
                _read_empty_checkpoint(Path(parent["directory"]), parent["step"])
        return
    # Internal decisions, not command-line switches. Every launcher uses the
    # same model checkpoint arguments to select queue restoration.
    args._rollout_queue_resume = False
    args._rollout_queue_fork = False
    if getattr(args, "rollout_data_transport", None) != "straw":
        return
    if any(getattr(args, key, False) for key in ("debug_train_only", "debug_rollout_only", "load_debug_rollout_data")):
        return
    if not getattr(args, "save", None):
        return
    save_root = Path(args.save).resolve()
    current_path = save_root / "rollout/current.json"
    expected_current = current_path.read_text() if current_path.exists() else None
    load = getattr(args, "load", None)
    if load is None and expected_current is not None:
        load = str(save_root)
    root = Path(load).resolve() if load else None
    step = getattr(args, "ckpt_step", None)
    direct_commit = root is not None and root.is_file() and root.name.startswith("committed_")
    if direct_commit:
        selected_step = int(root.stem.split("_")[-1])
        if step is not None and step != selected_step:
            raise ValueError("--ckpt-step differs from the explicit checkpoint marker")
        step, root = selected_step, root.parent.parent
    is_checkpoint = root is not None and (
        direct_commit
        or (root / "rollout/current.json").exists()
        or (root / "rollout/branch.json").exists()
        or any((root / "rollout").glob("committed_*.json"))
    )
    selected = None
    if is_checkpoint:
        current = root / "rollout/current.json"
        pointer = json.loads(current.read_text()) if current.exists() else None
        active = Path(pointer["directory"]).resolve() if pointer else root
        if not direct_commit and step is None:
            tracker = root / "latest_checkpointed_iteration.txt"
            commits = list((active / "rollout").glob("committed_*.json"))
            if tracker.exists():
                text = tracker.read_text()
                tracked_step = int(text.strip())
                latest = max((int(path.stem.split("_")[-1]) for path in commits), default=-1)
                # A changed logical tracker is an explicit user selection; an
                # older physical tracker also selects a rollback. A newer
                # physical tracker may be an interrupted model save and cannot
                # override the joint commit. The recorded value also supports
                # an inherited tracker before the branch's first commit.
                logical_edit = (
                    pointer is not None
                    and "tracker" in pointer
                    and text.strip() != str(pointer["tracker"]).strip()
                    and root != active
                )
                if logical_edit or tracked_step < latest:
                    # An unchanged logical tracker may lag behind a child.
                    if logical_edit or root == active:
                        step = tracked_step
            if step is None and root != active:
                tracker = active / "latest_checkpointed_iteration.txt"
                if tracker.exists() and commits:
                    tracked_step = int(tracker.read_text().strip())
                    if tracked_step < max(int(path.stem.split("_")[-1]) for path in commits):
                        step = tracked_step
        branch_path = active / "rollout/branch.json"
        branch = json.loads(branch_path.read_text()) if branch_path.exists() else None
        if (
            not direct_commit
            and step is None
            and branch is not None
            and branch["parent"] is None
            and not any((active / "rollout").glob("committed_*.json"))
        ):
            if root != save_root:
                raise ValueError("Before the first checkpoint, restart using the original --save directory")
            if branch["initial_configuration"] != _initial_configuration(args):
                raise ValueError("Initial model/input configuration differs from the interrupted run")
            if args.rollout_data_dir is not None and Path(args.rollout_data_dir).resolve() != Path(branch["pool"]):
                raise ValueError("Interrupted run requires its original straw pool")
            args.load, args.save = branch["initial_load"], str(active)
            args.rollout_data_dir = branch["pool"]
            args._rollout_queue_resume = True
            args._rollout_queue_id = branch["queue_id"]
            args._checkpoint_plan = {"root": str(save_root), "expected_current": expected_current, "branch": branch}
            return
        selected = (
            (root, step, _read_checkpoint(root, step))
            if direct_commit
            else _select_checkpoint(root, step, allow_empty=step is not None)
        )
    elif root is not None and (root / "latest_checkpointed_iteration.txt").exists():
        if step is None:
            step = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
        _read_empty_checkpoint(root, step)
        selected = root, step, None
    elif root is not None and step is not None:
        raise FileNotFoundError(f"No model checkpoint tracker for requested step {step}: {root}")

    if selected is not None and not (selected[0] / "latest_checkpointed_iteration.txt").exists():
        raise FileNotFoundError(f"Model checkpoint tracker is missing: {selected[0]}")

    # Never overwrite existing checkpoint files, including a partially written
    # iteration without a commit marker. Repeated rollbacks use distinct paths.
    occupied = (
        expected_current is not None or any(save_root.glob("iter_*")) or (save_root / "rollout/branch.json").exists()
    )
    destination = (
        save_root / "branches" / uuid.uuid4().hex if occupied or (selected and selected[0] == save_root) else save_root
    )
    branch = {
        "version": 1,
        "parent": None,
        "initial_load": getattr(args, "load", None),
        "initial_configuration": _initial_configuration(args),
        "queue_id": f"run:{uuid.uuid4().hex}",
    }
    if selected is not None:
        root, step, value = selected
        if any(getattr(args, key, False) for key in ("finetune", "no_load_optim", "no_load_rng")):
            raise ValueError("Checkpoint restoration requires model, optimizer and RNG state")
        if args.start_rollout_id is not None and args.start_rollout_id != step + 1:
            raise ValueError("--start-rollout-id differs from the checkpoint")
        if value is not None:
            if (getattr(args, "rollout_queue_run_id", None) or "rollout") != value["run_id"]:
                raise ValueError("Checkpoint belongs to another straw run")
            if (
                args.rollout_data_dir is not None
                and Path(args.rollout_data_dir).resolve() != Path(value["rollout_data_dir"]).resolve()
            ):
                raise ValueError("Checkpoint restoration requires the original shared straw pool")
            args.rollout_data_dir = value["rollout_data_dir"]
            args._rollout_queue_fork = True
        else:
            args._rollout_queue_fork = False
            args._rollout_queue_empty_restore = True
            cursor = root / "rollout" / f"global_dataset_state_dict_{step}.pt"
            args._rollout_dataset_cursor = str(cursor) if cursor.exists() else None
            logging.getLogger(__name__).warning(
                "Model checkpoint %s step %s has no queue snapshot: starting a new empty queue; %s",
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
        args.update_weight_start_version = step + 1
        args._rollout_model_step = step
        branch["parent"] = {"directory": str(root), "step": step, "queue_snapshot": value is not None}
    args.save = str(destination)
    args._rollout_queue_id = branch["queue_id"]
    args._checkpoint_plan = {"root": str(save_root), "expected_current": expected_current, "branch": branch}
