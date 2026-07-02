import types

import pytest

from slime.ray.rollout_validation import validate_server_group_gpu_indices


NUM_GPUS = 0


@pytest.mark.unit
def test_validate_server_group_gpu_indices_accepts_valid_config():
    validate_server_group_gpu_indices(
        worker_type="regular",
        gpu_offset=2,
        num_gpus_per_engine=1,
        num_gpu_per_engine=1,
        num_engines=2,
        num_available_gpus=4,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
    )


@pytest.mark.unit
def test_validate_server_group_gpu_indices_allows_empty_group():
    validate_server_group_gpu_indices(
        worker_type="placeholder",
        gpu_offset=4,
        num_gpus_per_engine=1,
        num_gpu_per_engine=1,
        num_engines=0,
        num_available_gpus=4,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
    )


@pytest.mark.unit
def test_validate_server_group_gpu_indices_reports_config_context():
    with pytest.raises(ValueError) as exc_info:
        validate_server_group_gpu_indices(
            worker_type="regular",
            gpu_offset=3,
            num_gpus_per_engine=2,
            num_gpu_per_engine=2,
            num_engines=1,
            num_available_gpus=4,
            rollout_num_gpus=4,
            rollout_num_gpus_per_engine=2,
        )

    message = str(exc_info.value)
    assert "worker_type=regular" in message
    assert "gpu_offset=3" in message
    assert "num_gpus_per_engine=2" in message
    assert "num_engines=1" in message
    assert "required_gpu_slots=5" in message
    assert "len(reordered_gpu_ids)=4" in message
    assert "rollout_num_gpus=4" in message
    assert "rollout_num_gpus_per_engine=2" in message


def _make_convert_manager(loss_aggregation):
    from slime.ray.rollout import RolloutManager

    meta = getattr(RolloutManager, "__ray_metadata__", None)
    cls = meta.modified_class if meta is not None else RolloutManager
    manager = cls.__new__(cls)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        loss_aggregation=loss_aggregation,
        reward_key=None,
        advantage_estimator="reinforce",
        rewards_normalization=False,
        grpo_std_normalization=False,
    )
    return manager


def _make_grouped_samples(group_indices):
    from slime.utils.types import Sample

    samples = []
    for i, gid in enumerate(group_indices):
        samples.append(
            Sample(
                index=i,
                group_index=gid,
                rollout_id=i,
                tokens=[0, 1, 2, 3],
                response_length=2,
                reward=0.0,
                loss_mask=[1, 1],
            )
        )
    return samples


@pytest.mark.unit
def test_prompt_mean_fails_loud_on_none_group_index():
    pytest.importorskip("sglang")  # RolloutManager import pulls sglang
    manager = _make_convert_manager("prompt_mean")
    samples = _make_grouped_samples([0, 0, None, 1])

    with pytest.raises(ValueError, match="group_index"):
        manager._convert_samples_to_train_data(samples)


@pytest.mark.unit
def test_prompt_mean_builds_per_group_mask_sums():
    pytest.importorskip("sglang")
    manager = _make_convert_manager("prompt_mean")
    samples = _make_grouped_samples([0, 0, 1])

    train_data = manager._convert_samples_to_train_data(samples)

    assert train_data["prompt_mask_sums"] == [4, 4, 2]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["sample_mean", "constant", "token_mean"])
def test_non_prompt_mean_modes_ignore_none_group_index(mode):
    pytest.importorskip("sglang")
    manager = _make_convert_manager(mode)
    samples = _make_grouped_samples([0, None, 1])

    train_data = manager._convert_samples_to_train_data(samples)

    assert "prompt_mask_sums" not in train_data


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
