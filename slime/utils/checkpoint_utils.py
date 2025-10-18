from pathlib import Path
from typing import Optional


def get_latest_checkpointed_iteration(args) -> Optional[int]:
    """
    :param args: The Megatron arguments
    """
    if (x := args.ckpt_step) is not None:
        return x

    if args.load is None:
        return None

    path_txt = Path(args.load) / "latest_checkpointed_iteration.txt"
    if not path_txt.exists():
        return None

    return int(path_txt.read_text())
