"""On HIP, the ROCm checkpoint writer patch applies where Megatron ships its own writer and is skipped otherwise."""

import ast
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[1]
STRATEGIES = "megatron.core.dist_checkpointing.strategies"
FILESYSTEM_ASYNC = f"{STRATEGIES}.filesystem_async"


class _ReachedNextStep(Exception):
    """Raised by the first call after the HIP block."""


def _stop(*args, **kwargs):
    """Stand in for the call after the HIP block."""
    raise _ReachedNextStep


def _hip_function(relative_path, name, next_call):
    """Compile one top-level function with torch reporting HIP and the call after the HIP block stubbed out."""
    path = REPO / relative_path
    definition = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": types.SimpleNamespace(version=types.SimpleNamespace(hip="6.4", cuda=None)), next_call: _stop}
    for annotation in ("Namespace", "DDP", "MegatronOptimizer", "OptimizerParamScheduler"):
        namespace[annotation] = type(annotation, (), {})
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class _MegatronWriter:
    """Megatron's FileSystemWriterAsync, as far as the ROCm writer subclasses it."""

    @staticmethod
    def preload_tensors(*args, **kwargs):
        return args, kwargs


@pytest.mark.parametrize(
    "relative_path, name, next_call, call_args",
    [
        (
            "slime/backends/megatron_utils/model.py",
            "initialize_model_and_optimizer",
            "setup_model_and_optimizer",
            (None,),
        ),
        ("tools/convert_hf_to_torch_dist.py", "main", "configure_logger", ()),
    ],
)
@pytest.mark.parametrize("megatron_has_writer", [True, False])
def test_hip_writer_patch_follows_the_megatron_layout(
    monkeypatch, capsys, relative_path, name, next_call, call_args, megatron_has_writer
):
    for package in ("megatron", "megatron.core", "megatron.core.dist_checkpointing", STRATEGIES):
        module = types.ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    monkeypatch.delitem(sys.modules, FILESYSTEM_ASYNC, raising=False)
    monkeypatch.delitem(sys.modules, "slime.utils.rocm_checkpoint_writer", raising=False)
    if megatron_has_writer:
        writer_module = types.ModuleType(FILESYSTEM_ASYNC)
        writer_module.FileSystemWriterAsync = _MegatronWriter
        monkeypatch.setitem(sys.modules, FILESYSTEM_ASYNC, writer_module)
        sys.modules[STRATEGIES].filesystem_async = writer_module
    function = _hip_function(relative_path, name, next_call)

    with pytest.raises(_ReachedNextStep):
        function(*call_args)

    printed = capsys.readouterr().out
    if megatron_has_writer:
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        assert sys.modules[FILESYSTEM_ASYNC].FileSystemWriterAsync is ROCmFileSystemWriterAsync
        assert "Applied FileSystemWriterAsync patch" in printed
    else:
        assert "FileSystemWriterAsync patch skipped" in printed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
