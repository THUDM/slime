"""GRPO statistics count each agent rollout once, regardless of its forks."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.data.batch_builder import BatchBuilder  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def builder(**overrides):
    args = dict(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    return BatchBuilder(SimpleNamespace(**(args | overrides)))


def segment(group, rollout, reward, mask=(1,)):
    return Sample(
        group_index=group,
        index=rollout,
        rollout_id=rollout,
        reward=reward,
        tokens=[100, *range(len(mask))],
        response_length=len(mask),
        loss_mask=list(mask),
        rollout_log_probs=[-0.5] * len(mask),
    )


@pytest.mark.parametrize("std", [False, True])
def test_unequal_forks_and_shuffled_prompt_groups(std):
    samples = [
        segment(20, 4, 8),
        segment(10, 1, 1),
        segment(20, 3, 10),
        segment(10, 1, 1),
        segment(10, 2, 0),
        segment(20, 4, 8),
        segment(10, 1, 1),
    ]
    raw, normalized = builder(grpo_std_normalization=std)._post_process_rewards(samples)
    assert raw == [8, 1, 10, 1, 0, 8, 1]
    expected = [-1, 0.5, 1, 0.5, -0.5, -1, 0.5]
    if std:
        low = 0.5 / (math.sqrt(0.5) + 1e-6)
        high = 1 / (math.sqrt(2) + 1e-6)
        expected = [-high, low, high, low, -low, -high, low]
    assert normalized == pytest.approx(expected)


def test_segment_count_matching_batch_size_does_not_override_group_ids():
    samples = [segment(9, 1, 1), segment(9, 1, 1), segment(9, 1, 1), segment(9, 2, 0)]
    assert builder()._post_process_rewards(samples)[1] == pytest.approx([0.5, 0.5, 0.5, -0.5])


def test_single_rollout_group_has_finite_zero_advantage():
    samples = [segment(3, 8, 1), segment(3, 8, 1), segment(4, 9, 0)]
    assert builder(grpo_std_normalization=True)._post_process_rewards(samples)[1] == [0, 0, 0]


def test_rollout_reward_or_prompt_group_mismatch_is_rejected():
    for samples in (
        [segment(1, 3, 0), segment(1, 3, 1)],
        [segment(1, 3, 0), segment(2, 3, 0)],
    ):
        with pytest.raises(ValueError, match="rollout"):
            builder()._post_process_rewards(samples)


def test_incomplete_prompt_group_metadata_is_rejected():
    with pytest.raises(ValueError, match="group_index"):
        builder()._post_process_rewards([segment(1, 0, 1), segment(None, 1, 0)])


def test_legacy_flat_batches_without_identifiers_keep_positional_groups():
    samples = [Sample(reward=r) for r in [1, 0, 10, 8]]
    assert builder()._post_process_rewards(samples)[1] == pytest.approx([0.5, -0.5, 1, -1])


def test_disabled_normalization_preserves_outcome_rewards():
    samples = [segment(5, 2, 1), segment(5, 2, 1), segment(5, 3, 0)]
    assert builder(rewards_normalization=False)._post_process_rewards(samples) == ([1, 1, 0], [1, 1, 0])


def test_conversion_aggregates_loss_denominators_over_original_rollout():
    samples = [segment(5, 2, 1, (1, 0, 1)), segment(5, 3, 0), segment(5, 2, 1, (1, 1, 1))]
    # An omitted rollout_id falls back to index, as the Sample contract specifies.
    for sample in samples:
        sample.rollout_id = None
    data = builder().convert(samples)
    assert data["rollout_ids"] == [2, 3, 2]
    assert data["rollout_mask_sums"] == [5, 1, 5]
    assert data["rewards"] == pytest.approx([0.5, -0.5, 0.5])
    assert data["tokens"] == [sample.tokens for sample in samples]
    assert data["rollout_log_probs"] == [sample.rollout_log_probs for sample in samples]


def training_builder(global_batch_size, dp_size, **options):
    batch = builder(
        global_batch_size=global_batch_size,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        balance_data=False,
        balance_by_flops=False,
        **options,
    )
    batch.train_parallel_config = dict(dp_size=dp_size, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
    return batch


@pytest.mark.parametrize("transport", ["object-store", "straw", "legacy-straw"])
@pytest.mark.parametrize("precomputed", [True, False])
def test_loss_counts_follow_rollouts_across_ranks_and_steps(transport, precomputed, monkeypatch, tmp_path):
    from slime.data import batch_builder
    from slime.utils.data import process_rollout_data

    legacy = transport == "legacy-straw"
    if legacy:
        transport = "straw"
    if transport == "straw":
        tensor = pytest.importorskip("straw.tensor")
    monkeypatch.setattr(batch_builder.ray, "put", lambda value: value)
    monkeypatch.setattr(batch_builder.ray, "get", lambda value: value)
    batch = training_builder(2, 2, rollout_data_transport=transport, rollout_data_dir=str(tmp_path))
    # Rollout 7 has two live siblings on different ranks and one removed sibling.
    # Both rollouts in the second step are fully masked. The last rollout is trimmed.
    samples = [segment(0, rid, 1) for rid in (7, 9, 7, 7, 3, 4, 5)]
    for i in (0, 1, 4, 5):
        samples[i].remove_sample = True
    data = batch.convert(samples)
    if not precomputed:
        del data["rollout_mask_sums"]
    if legacy:
        # A pre-activity-column conversion, with either tensorized totals or only tensorized masks.
        from slime.data.transport import pack_rollout_payload
        from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training

        data["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
        tensorize_rollout_data_for_training(data)
        data = pack_rollout_payload(data, batch.args, 0)
    with monkeypatch.context() as guard:
        if transport == "straw":
            guard.setattr(tensor.TensorRef, "load", lambda *a, **kw: pytest.fail("scheduler read an archived tensor"))
        refs = batch.split_by_dp(data)
    for rank in range(2):
        shard = process_rollout_data(refs, rank, 2)
        assert shard["loss_rollout_counts"] == [1, 0]
        assert shard["global_batch_sizes"] == [2, 2]
        assert shard["num_microbatches"] == [2, 1]
        assert sum(int(mask.sum()) for mask in shard["loss_masks"]) == 1
        assert shard["rollout_mask_sums"].tolist() == [2 if rid == 7 else 0 for rid in shard["rollout_ids"]]


@pytest.mark.parametrize("removed", [0, 2, 6])
def test_removed_rollouts_preserve_value_loss_and_gradient(removed, monkeypatch):
    import _cp_dist_helpers  # noqa: F401
    import torch
    from megatron.core import mpu

    from slime.backends.megatron_utils.loss import loss_function
    from slime.data import batch_builder
    from slime.observability.train_metric_utils import reduce_train_step_metrics

    monkeypatch.setattr(batch_builder.ray, "put", lambda value: value)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **_: 1, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    samples = [segment(0, i, 1) for i in range(2 + removed)]
    for sample in samples[2:]:
        sample.remove_sample = True
    batch = training_builder(len(samples), 1, rollout_data_transport="object-store")
    shard = batch.split_by_dp(batch.convert(samples))[0].inner
    divisor = shard["loss_rollout_counts"][0]
    args = SimpleNamespace(
        loss_type="value_loss",
        value_clip=1,
        allgather_cp=False,
        calculate_per_token_loss=False,
        recompute_loss_function=False,
    )
    prediction = torch.tensor(1.0, requires_grad=True)
    reports = []
    num_mbs = shard["num_microbatches"][0]
    for indices in shard["micro_batch_indices"]:
        mb = {k: [shard[k][i] for i in indices] for k in ("loss_masks", "response_lengths", "rollout_mask_sums")}
        mb.update(
            total_lengths=[2],
            unconcat_tokens=[torch.tensor([100, 0])],
            values=[torch.ones(1)],
            returns=[torch.zeros(1)],
        )
        loss, _, report = loss_function(args, mb, num_mbs, divisor, prediction.expand(1, 2, 1))
        (loss / num_mbs).backward()
        reports.append(report)
    metrics = reduce_train_step_metrics(
        reports, calculate_per_token_loss=False, loss_divisor=divisor, cp_size=1, dp_with_cp_group=None
    )
    assert metrics["value_loss"] == pytest.approx(1.0)
    assert prediction.grad.item() == pytest.approx(2.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
