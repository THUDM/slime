import pytest

from slime.ray.rollout_validation import validate_server_group_gpu_indices


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
