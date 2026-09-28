"""The default Ray data path remains usable in environments without straw."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

NUM_GPUS = 0


@pytest.mark.parametrize("transport", ["object-store", "nixl", "straw"])
def test_missing_straw_startup_hint(monkeypatch, caplog, transport):
    module = load_slime_arguments_module(monkeypatch)
    find_spec = module.importlib.util.find_spec
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None if name == "straw" else find_spec(name))
    args = make_slime_validate_args(rollout_data_transport=transport)
    if transport == "straw":
        with pytest.raises(ModuleNotFoundError, match="pip install straw-queue") as raised:
            module.slime_validate_args(args)
        assert raised.value.name == "straw"
    else:
        module.slime_validate_args(args)
        assert args.rollout_data_transport == transport
        assert args.data_source_path == "slime.rollout.data_source.RolloutDataSourceWithBuffer"
        assert args.rollout_data_dir is None
        assert f"continuing with {transport}" in caplog.text
        assert "pip install straw-queue" in caplog.text


def _run_default_without_straw():
    import importlib.abc
    from types import SimpleNamespace

    class NoStraw(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "straw" or fullname.startswith("straw."):
                raise ModuleNotFoundError("straw is not installed", name="straw")

    # Keep the regression effective even when invoked from an installed environment.
    sys.meta_path.insert(0, NoStraw())

    import ray
    import torch

    from slime.rollout import fully_async_rollout
    from slime.rollout.base_types import finalize_rollout_groups
    from slime.rollout.batch_builder import BatchBuilder
    from slime.utils.data import process_rollout_data
    from slime.utils.rollout_transport import pack_rollout_group, pack_rollout_payload, rollout_store
    from slime.utils.types import Sample

    args = SimpleNamespace(
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        global_batch_size=2,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        balance_data=False,
        balance_by_flops=False,
    )
    groups = [[Sample(index=i, tokens=[i + 1, i + 2], response_length=1)] for i in range(2)]
    assert pack_rollout_payload(groups, args, 0) is groups
    assert pack_rollout_group(groups[0], args, 0) is groups[0]
    assert finalize_rollout_groups(args, 0, groups).samples is groups
    builder = BatchBuilder(args)
    assert builder.begin(groups) is None
    builder.save(0)
    builder.load(0)

    async def generate(*values):
        return groups

    fully_async_rollout._generate_rollout_async = generate
    assert fully_async_rollout.generate_rollout_fully_async(args, 0, object()) is groups

    ray.init(address="local", num_cpus=1, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        builder.train_parallel_config = {
            "dp_size": 1,
            "cp_size": 1,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        refs = builder.split_by_dp(
            {
                "tokens": [group[0].tokens for group in groups],
                "rollout_ids": [0, 1],
                "response_lengths": [1, 1],
                "loss_masks": [[1], [1]],
                "rewards": [1.0, 2.0],
                "raw_reward": [1.0, 2.0],
            }
        )
        assert isinstance(refs[0].inner, ray.ObjectRef)
        batch = process_rollout_data(refs, 0, 1)
        assert [tokens.tolist() for tokens in batch["tokens"]] == [[1, 2], [2, 3]]
        assert batch["local_raw_reward"] == [1.0, 2.0]

        # Exercise a CPU optimizer step on the received tensors, without a model download.
        weight = torch.nn.Parameter(torch.zeros(2, 1))
        optimizer = torch.optim.SGD([weight], lr=0.01)
        predictions = torch.stack(batch["tokens"]).float() @ weight
        loss = (predictions.flatten() - torch.tensor(batch["local_raw_reward"])).square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss) and weight.detach().abs().sum() > 0
    finally:
        ray.shutdown()

    with pytest.raises(ModuleNotFoundError, match="pip install straw-queue"):
        rollout_store(args)
    assert not any(name == "straw" or name.startswith("straw.") for name in sys.modules)


def test_default_rollout_and_training_work_without_straw(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from test_optional_straw import _run_default_without_straw; _run_default_without_straw()",
        ],
        check=True,
        timeout=120,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])]),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert not list(tmp_path.iterdir())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
