from pathlib import Path
from typing import Optional


def get_latest_checkpointed_iteration(dir_load: Optional[str]) -> Optional[int]:
    if dir_load is None:
        return None

    path_txt = Path(dir_load) / "latest_checkpointed_iteration.txt"
    if not path_txt.exists():
        return None

    return int(path_txt.read_text())
