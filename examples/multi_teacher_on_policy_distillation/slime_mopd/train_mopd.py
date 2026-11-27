"""slime train entry point with the MOPD Megatron actor injected.

slime's ``train.py`` calls ``create_training_models`` without an ``actor_cls``
and exposes no flag for it, so this entry loads ``train.py``, patches the global
it resolves, and runs the loop unchanged.

    python -m slime_mopd.train_mopd <slime train args...>

``SLIME_ROOT`` overrides the checkout location; otherwise it is derived from the
imported ``slime`` package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _slime_root() -> Path:
    override = os.environ.get("SLIME_ROOT")
    if override:
        return Path(override)
    import slime

    return Path(slime.__file__).resolve().parents[1]


def _load_slime_train_module(slime_root: Path):
    train_py = slime_root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"slime train entry not found: {train_py}")
    spec = importlib.util.spec_from_file_location("slime_train_entry", train_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["slime_train_entry"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    slime_root = _slime_root()
    if str(slime_root) not in sys.path:
        sys.path.insert(0, str(slime_root))

    from slime_mopd.megatron_actor import MopdMegatronTrainRayActor

    from slime.observability.logging_utils import configure_logger
    from slime.utils.arguments import parse_args

    slime_train = _load_slime_train_module(slime_root)
    original_create_training_models = slime_train.create_training_models

    def create_training_models_with_mopd(args, pgs, rollout_manager):
        return original_create_training_models(args, pgs, rollout_manager, actor_cls=MopdMegatronTrainRayActor)

    slime_train.create_training_models = create_training_models_with_mopd

    args, restore_plan = parse_args(return_restore_plan=True)
    configure_logger()
    slime_train.main(args, restore_plan)


if __name__ == "__main__":
    main()
