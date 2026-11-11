import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def checkpoint_module(tmp_path, monkeypatch):
    """Load slime's checkpoint adapter against a Megatron loader that records the --no-save-optim it sees."""
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3")
    args = SimpleNamespace(load=str(tmp_path), no_save_optim=True, finetune=False, ckpt_step=None)
    seen = []

    def load_model(*a, **kw):
        """Record the --no-save-optim that Megatron's generate_state_dict reads for the load template."""
        seen.append(args.no_save_optim)
        return 3, 0

    native = SimpleNamespace(save_checkpoint=Mock(), load_checkpoint=Mock(side_effect=load_model))
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", native)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", SimpleNamespace(get_args=lambda: args))
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/checkpoint.py"
    spec = importlib.util.spec_from_file_location("no_save_optim_checkpoint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, args, native, seen


@pytest.mark.parametrize("no_save_optim", [True, False])
def test_megatron_load_keeps_the_optimizer_in_its_template(checkpoint_module, no_save_optim):
    module, args, native, seen = checkpoint_module
    args.no_save_optim = no_save_optim
    assert module.load_checkpoint([], Mock(), Mock(), {}) == (3, 0)
    assert seen == [False]
    assert args.no_save_optim is no_save_optim


def test_no_save_optim_is_restored_when_the_load_fails(checkpoint_module):
    module, args, native, seen = checkpoint_module
    native.load_checkpoint.side_effect = RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        module.load_checkpoint([], Mock(), Mock(), {})
    assert args.no_save_optim is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
