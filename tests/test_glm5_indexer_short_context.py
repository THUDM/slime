import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0


def _load_indexer_module():
    module_name = "slime_plugins.models.glm5.ops.indexer_short_context_test"
    for package_name in (
        "slime_plugins",
        "slime_plugins.models",
        "slime_plugins.models.glm5",
        "slime_plugins.models.glm5.ops",
    ):
        package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
        package.__path__ = []

    for dependency in ("tilelang_indexer_bwd", "tilelang_indexer_fwd"):
        dependency_name = f"slime_plugins.models.glm5.ops.{dependency}"
        dependency_module = types.ModuleType(dependency_name)
        setattr(
            dependency_module,
            "indexer_bwd_interface" if dependency.endswith("bwd") else "indexer_fwd_interface",
            None,
        )
        sys.modules[dependency_name] = dependency_module

    source = Path(__file__).parents[1] / "slime_plugins/models/glm5/ops/indexer.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_short_context_topk_is_padded_with_invalid_routes():
    indexer = _load_indexer_module()
    logits = torch.tensor([[0.5, float("-inf"), 0.25]])

    scores, indices = indexer.pytorch_topk_with_invalid_padding(logits, topk=5)

    assert scores.shape == (1, 5)
    assert indices.dtype == torch.int32
    assert sorted(indices[0][indices[0] >= 0].tolist()) == [0, 2]
    assert indices[0].tolist().count(-1) == 3
    assert torch.isneginf(scores[0, 2:]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
