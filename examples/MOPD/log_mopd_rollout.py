"""MOPD rollout logging — re-exports from shared log_rollout module."""
from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from log_rollout import log_eval_rollout_data, log_rollout_data  # noqa: E402,F401
