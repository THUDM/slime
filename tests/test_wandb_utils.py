"""CPU tests for W&B run naming in slime.observability.wandb_utils."""

import re
import sys
import types
from types import SimpleNamespace

import pytest

NUM_GPUS = 0


@pytest.fixture
def fake_wandb(monkeypatch):
    """A wandb stand-in without ``wandb.util.generate_id`` (removed in wandb 0.30)."""
    calls = {}
    module = types.ModuleType("wandb")
    module.util = types.ModuleType("wandb.util")
    module.Settings = lambda **kwargs: kwargs
    module.login = lambda **kwargs: calls.setdefault("login", kwargs)
    module.define_metric = lambda *args, **kwargs: None

    def init(**kwargs):
        calls["init"] = kwargs
        module.run = SimpleNamespace(id="run-id")

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.delitem(sys.modules, "slime.observability.wandb_utils", raising=False)
    return calls


def wandb_args(**overrides):
    values = dict(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=True,
        wandb_group="flash",
        wandb_team=None,
        wandb_project="slime",
        wandb_dir=None,
        rank=0,
    )
    return SimpleNamespace(**(values | overrides))


def test_random_suffix_does_not_need_wandb_generate_id(fake_wandb):
    from slime.observability.wandb_utils import init_wandb_primary

    args = wandb_args()
    init_wandb_primary(args)
    group = fake_wandb["init"]["group"]
    assert re.fullmatch(r"flash_[a-z0-9]{8}", group)
    assert fake_wandb["init"]["name"] == f"{group}-RANK_0"
    assert args.wandb_run_id == "run-id"


def test_disabled_suffix_keeps_the_group_name(fake_wandb):
    from slime.observability.wandb_utils import init_wandb_primary

    init_wandb_primary(wandb_args(wandb_random_suffix=False))
    assert fake_wandb["init"]["group"] == fake_wandb["init"]["name"] == "flash"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
