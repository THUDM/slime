#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SLIME_ROOT = Path(__file__).resolve().parents[3]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from examples.scripts.maintenance.wandb_consolidate_runner import main


if __name__ == "__main__":
    raise SystemExit(main(default_mode="mopd"))
