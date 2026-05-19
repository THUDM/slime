import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Required in the path template so each rollout / rank writes a distinct file.
_REQUIRED_KEYS = ("rollout_id", "rank")


def resolve_debug_train_data_path(template: str, *, rollout_id: int, rank: int) -> Path:
    """Format ``template`` after ensuring ``{rollout_id}`` and ``{rank}`` are present."""
    missing = [k for k in _REQUIRED_KEYS if f"{{{k}}}" not in template]
    if missing:
        suffix = "".join(f"_{{{k}}}" for k in missing)
        path = Path(template)
        template = (
            str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))
            if path.suffix
            else f"{template.rstrip('/')}/debug_train{suffix}.pt"
        )
        logger.warning(
            "save_debug_train_data: path template missing %s, auto-corrected to %s",
            ", ".join(missing),
            template,
        )
    return Path(template.format(rollout_id=rollout_id, rank=rank))


def save_debug_train_data(args, *, rollout_id, rollout_data) -> None:
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = resolve_debug_train_data_path(path_template, rollout_id=rollout_id, rank=rank)
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            path,
        )
