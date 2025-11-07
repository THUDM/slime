from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"Warning: failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def load(actor: Any) -> dict[str, Any] | None:
    """Prepare checkpoint payload for a training actor."""
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        print(f"[FSDP] Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            print(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_ckpt = checkpoint_dir / "model.pt"
    if not model_ckpt.exists():
        print(f"[FSDP] Checkpoint {model_ckpt} not found; skipping load.")
        return None

    model_payload = torch.load(model_ckpt, map_location="cpu")
    if isinstance(model_payload, dict) and any(isinstance(v, torch.Tensor) for v in model_payload.values()):
        model_state = model_payload
    else:
        model_state = model_payload.get("model", {})
        if not model_state:
            raise RuntimeError(f"Invalid model checkpoint payload at {model_ckpt}")

    optimizer_state = None
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer_state = torch.load(optimizer_path, map_location="cpu")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "model": model_state,
        "optimizer": optimizer_state,
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    """Finalize checkpoint load by restoring optimizer, RNG, and metadata."""
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("optimizer") is not None and not getattr(actor.args, "no_load_optim", False):
        optimizer_state = checkpoint_payload["optimizer"]
        if actor.args.optimizer == "deepspeed_cpu_adam":
            actor.optimizer.cpu_optimizer.load_state_dict(optimizer_state)
        else:
            actor.optimizer.load_state_dict(optimizer_state)
    checkpoint_payload["optimizer"] = None

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    checkpoint_payload["rng"] = None

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        actor._latest_checkpoint_iteration = int(metadata.get("iteration", iteration))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            actor.args.start_rollout_id = next_rollout
    elif iteration is not None:
        actor._latest_checkpoint_iteration = iteration
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration
    checkpoint_payload["metadata"] = None

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, iteration: int) -> None:
    """Persist model, optimizer, and metadata for the given iteration."""
    torch.cuda.synchronize()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    actor.update_cpu_params_dict(actor.weights["actor"])

    if actor.args.optimizer == "deepspeed_cpu_adam":
        optimizer_state = actor.optimizer.cpu_optimizer.state_dict()
    else:
        optimizer_state = actor.optimizer.state_dict()

    if dist.get_rank() == 0:
        model_payload = {
            "format_version": 1,
            "model": {name: tensor for name, tensor in actor.weights["actor"].items()},
        }
        torch.save(model_payload, checkpoint_dir / "model.pt")
        torch.save(optimizer_state, checkpoint_dir / "optimizer.pt")

        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        print(f"[FSDP] Saved checkpoint to {checkpoint_dir}")
        actor._latest_checkpoint_iteration = step_id

    dist.barrier()
