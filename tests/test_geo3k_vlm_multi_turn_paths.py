import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_PATH = REPO_ROOT / "examples" / "geo3k_vlm_multi_turn" / "rollout.py"


def _module_constant(path: Path, name: str) -> str:
    module = ast.parse(path.read_text())
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


@pytest.mark.unit
def test_geo3k_vlm_multi_turn_default_env_module_points_to_existing_file():
    module_name = _module_constant(ROLLOUT_PATH, "DEFAULT_ENV_MODULE")
    module_path = REPO_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")

    assert module_path.exists()
    assert module_path.relative_to(REPO_ROOT).as_posix() == "examples/geo3k_vlm_multi_turn/env_geo3k.py"
