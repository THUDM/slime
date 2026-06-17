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


# ---------------------------------------------------------------------------
# _convert_samples_to_train_data: prompt_mask_sums build (--loss-aggregation)
# ---------------------------------------------------------------------------
#
# prompt_mask_sums (the per-prompt-group denominator for prompt_mean) is built
# only under --loss-aggregation=prompt_mean, and that build fails loud if any
# sample is missing its prompt group (group_index is None) — a None would
# silently collapse unrelated prompts into one denominator, degrading
# prompt_mean to sample_mean for that sample. The other three modes never read
# group_index, so they must neither build the key nor consult group_index.


def _make_convert_manager(loss_aggregation):
    """A bare RolloutManager (no Ray/sglang init) wired just enough to call
    ``_convert_samples_to_train_data``: no custom hooks, and reward
    post-processing reduced to identity (advantage_estimator outside the
    group-norm set), so the only behavior under test is the prompt_mask_sums
    build + its group_index guard."""
    from slime.ray.rollout import RolloutManager

    # RolloutManager is @ray.remote-decorated (an ActorClass); unwrap to the plain
    # class so a bare instance can be built to exercise the method directly.
    meta = getattr(RolloutManager, "__ray_metadata__", None)
    cls = meta.modified_class if meta is not None else RolloutManager
    manager = cls.__new__(cls)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.args = types.SimpleNamespace(
        loss_aggregation=loss_aggregation,
        reward_key=None,
        advantage_estimator="reinforce",  # outside the group-norm reshape path
        rewards_normalization=False,
        grpo_std_normalization=False,
    )
    return manager


def _make_grouped_samples(group_indices):
    """One Sample per entry; each carries a length-2 loss_mask so the
    per-group mask totals are non-trivial."""
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
    """prompt_mean with a None group_index is a real break (the prompt-grouping
    invariant is violated), so the convert step must raise — not silently
    renumber the sample into its own singleton group."""
    pytest.importorskip("sglang")  # RolloutManager import pulls sglang
    manager = _make_convert_manager("prompt_mean")
    samples = _make_grouped_samples([0, 0, None, 1])

    with pytest.raises(ValueError, match="group_index"):
        manager._convert_samples_to_train_data(samples)


@pytest.mark.unit
def test_prompt_mean_builds_per_group_mask_sums():
    """Sanity: with every group_index set, prompt_mask_sums is the per-group
    mask total broadcast per sample (group 0 has two length-2 samples → 4)."""
    pytest.importorskip("sglang")
    manager = _make_convert_manager("prompt_mean")
    samples = _make_grouped_samples([0, 0, 1])  # group 0: 2 samples, group 1: 1

    train_data = manager._convert_samples_to_train_data(samples)

    # group 0 = 2+2 = 4 (both samples), group 1 = 2.
    assert train_data["prompt_mask_sums"] == [4, 4, 2]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["sample_mean", "constant", "token_mean"])
def test_non_prompt_mean_modes_ignore_none_group_index(mode):
    """The other three modes never read group_index and never build
    prompt_mask_sums, so a None group_index must NOT raise and the key must be
    absent (keeping the default batch byte-identical — no extra key)."""
    pytest.importorskip("sglang")
    manager = _make_convert_manager(mode)
    samples = _make_grouped_samples([0, None, 1])

    train_data = manager._convert_samples_to_train_data(samples)

    assert "prompt_mask_sums" not in train_data


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
