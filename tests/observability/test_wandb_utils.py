from types import SimpleNamespace

import pytest

from slime.observability import wandb_utils

NUM_GPUS = 0


@pytest.mark.unit
def test_init_wandb_primary_generates_random_group_suffix(monkeypatch):
    init_kwargs = {}
    monkeypatch.setattr(wandb_utils, "generate_id", lambda: "fixed123")
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_kwargs.update(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run-123"))
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda: None)

    args = SimpleNamespace(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=True,
        wandb_group="grpo",
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
        rank=0,
    )

    wandb_utils.init_wandb_primary(args)

    assert init_kwargs["group"] == "grpo_fixed123"
    assert init_kwargs["name"] == "grpo_fixed123-RANK_0"
    assert init_kwargs["settings"] == {"mode": "offline"}
    assert args.wandb_run_id == "run-123"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
