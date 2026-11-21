"""Gradient collectives must be able to allocate outside PyTorch's cache."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0


@pytest.mark.parametrize("free_gib", [0, 2, 3, 8])
def test_gradient_sync_releases_unused_cache_before_collective(free_gib):
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/model.py"
    definition = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_finalize_model_grads"
    )
    memory = {"free": free_gib * 1024**3, "cleared": False}
    model, tokens, groups, result = object(), object(), object(), object()

    def clear_memory():
        memory.update(free=8 * 1024**3, cleared=True)

    def finalize(received_model, received_tokens, *, pg_collection, force_all_reduce):
        assert memory["free"] >= 3 * 1024**3, "NCCL cannot reclaim the PyTorch cache itself"
        assert received_model is model and received_tokens is tokens
        assert pg_collection is groups and force_all_reduce is True
        return result

    namespace = {
        "accelerator": SimpleNamespace(
            current_device=lambda: 0,
            mem_get_info=lambda device: (memory["free"], 80 * 1024**3),
        ),
        "clear_memory": clear_memory,
        "megatron_finalize_model_grads": finalize,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_finalize_model_grads"](model, tokens, pg_collection=groups, force_all_reduce=True) is result
    assert memory["cleared"] == (free_gib < 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
