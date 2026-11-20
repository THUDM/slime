import base64
from argparse import Namespace

import numpy as np
import pytest
import torch

from slime.observability.rollout_metrics import _compute_agent_compaction_metrics, _compute_top_p_kept_vocab_metrics
from slime.rollout.filter_hub.base_types import DynamicFilterOutput, MetricGatherer, call_dynamic_filter
from slime.utils.misc import decode_int32_meta_array
from slime.utils.types import Sample

NUM_GPUS = 0


def test_raw_reward_before_filters_counts_rejected_trajectories_once_before_mutation():
    args = Namespace(reward_key="score", rollout_sample_filter_path=None)
    shared_verdict = DynamicFilterOutput(keep=True)

    def mutate(args, group, *, keep):
        for trajectory in group:
            for sample in trajectory if isinstance(trajectory, list) else [trajectory]:
                sample.reward = {"score": 99.0}
        shared_verdict.keep = keep
        return shared_verdict

    failed = Sample(reward={"score": 0.0}, status=Sample.Status.COMPLETED)
    success = Sample(reward={"score": 1.0}, status=Sample.Status.TRUNCATED)
    rejected = call_dynamic_filter(mutate, args, [[failed] * 3, [success]], keep=False)
    accepted = call_dynamic_filter(mutate, args, [Sample(reward={"score": 1.0})], keep=True)
    gatherer = MetricGatherer()
    gatherer.on_group_completed(rejected)
    gatherer.on_dynamic_filter_drop("test")
    gatherer.on_group_completed(accepted)
    assert rejected.keep is False
    assert shared_verdict.raw_reward_stats is None
    assert gatherer.collect() == {
        "rollout/dynamic_filter/drop_test": 1,
        "rollout/raw_reward_before_filters": pytest.approx(2 / 3),
    }


@pytest.mark.parametrize("missing", ["recovered", "unscored"])
def test_raw_reward_before_filters_omits_incomplete_candidate_statistics(missing):
    args = Namespace(reward_key=None, rollout_sample_filter_path=None)
    gatherer = MetricGatherer()
    gatherer.on_group_completed(call_dynamic_filter(lambda *a: True, args, [Sample(reward=1.0)]))
    output = (
        DynamicFilterOutput(keep=True)
        if missing == "recovered"
        else call_dynamic_filter(lambda *a: True, args, [Sample(reward=None)])
    )
    gatherer.on_group_completed(output)
    assert "rollout/raw_reward_before_filters" not in gatherer.collect()


@pytest.mark.parametrize("batch_filter", [None, "example.batch_filter"])
def test_raw_reward_before_filters_only_needs_one_value_when_filtering_is_enabled(batch_filter):
    args = Namespace(reward_key=None, rollout_sample_filter_path=batch_filter)
    gatherer = MetricGatherer()
    gatherer.on_group_completed(call_dynamic_filter(None, args, [Sample(reward=0.0)]))
    expected = {"rollout/raw_reward_before_filters": 0.0} if batch_filter else {}
    assert gatherer.collect() == expected


def test_compaction_count_is_one_mean_per_trajectory():
    samples = [
        Sample(
            index=0,
            group_index=0,
            metadata={"agent_metrics": {"compact_count": 1, "compact_seconds": 3, "compact_limit_reached": 1}},
        ),
        Sample(index=0, group_index=0, metadata={"agent_metrics": {"compact_count": 1}}),
        Sample(index=1, group_index=0, metadata={"agent_metrics": {"compact_count": 0}}),
        Sample(index=2, group_index=0),
    ]
    metrics = _compute_agent_compaction_metrics(samples)
    assert metrics == {"compact_count": 0.5}


def test_compaction_count_is_omitted_without_observations():
    assert _compute_agent_compaction_metrics([Sample(index=0, group_index=0)]) == {}


def _make_args():
    return Namespace(sglang_speculative_algorithm=False, num_layers=2, moe_router_topk=2)


@pytest.mark.parametrize(
    "counts",
    [
        {"spec_accept_token_num": 2, "spec_draft_token_num": 6},
        {"spec_num_correct_drafts": 2, "spec_num_proposed_drafts": 6},
        {
            "spec_num_correct_drafts": 2,
            "spec_num_proposed_drafts": 6,
            "spec_accept_token_num": 4,
            "spec_draft_token_num": 8,
        },
    ],
)
def test_spec_info_accepts_sglang_counter_formats_without_double_counting(counts):
    info = Sample.SpecInfo()
    info.add({**counts, "spec_verify_ct": 2, "completion_tokens": 4})
    info.add({**counts, "spec_verify_ct": 2, "completion_tokens": 4})
    assert info.spec_accept_token_num == 4
    assert info.spec_draft_token_num == 12
    assert info.spec_accept_rate == pytest.approx(1 / 3)
    assert info.spec_accept_length == 2
    assert Sample.SpecInfo.from_dict(info.to_dict()) == info


@pytest.mark.unit
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(samples) == {}


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


@pytest.mark.unit
def test_decode_int32_meta_array_decodes_base64_to_tensor():
    decoded = decode_int32_meta_array({"routed_experts": _b64_int32([1, 2, 3])}, "routed_experts")

    assert torch.is_tensor(decoded)
    assert decoded.dtype == torch.int32
    torch.testing.assert_close(decoded, torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.tokens == [0, 1, 10, 20]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
    torch.testing.assert_close(
        sample.rollout_routed_experts,
        torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_append_response_tokens_ignores_split_pd_routed_experts():
    sample = Sample(tokens=[101, 102, 103, 104])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "pd_prefill_routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "pd_decode_routed_experts": _b64_int32([8, 9, 10, 11]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts is None


@pytest.mark.unit
def test_append_response_tokens_rejects_mismatched_routed_experts_shape():
    sample = Sample(tokens=[101, 102, 103])

    with pytest.raises(ValueError, match="routed_experts element count"):
        sample.append_response_tokens(
            _make_args(),
            tokens=[],
            trainable=True,
            meta_info={
                "routed_experts": _b64_int32([0, 1, 2, 3]),
                "finish_reason": {"type": "stop"},
            },
        )


@pytest.mark.unit
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.tokens == [0, 1, 200, 201, 202]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_requires_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="trainable response tokens require rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], trainable=True)


@pytest.mark.unit
def test_append_response_tokens_rejects_non_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="non-trainable response tokens should not pass rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], log_probs=[-0.1], trainable=False)


@pytest.mark.unit
def test_append_response_tokens_validates_pd_merged_routed_experts_and_continuation():
    from slime.utils.routed_experts import validate_routed_experts_tensor

    args = _make_args()
    args.num_experts = 16

    # Turn 1: prompt (3 tokens) + response (2 tokens) -> 4 routed-expert rows:
    # prefill returns [0, 3) (3 rows) and decode returns [3, 4) (1 row), merged by router.
    prefill_rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    decode_rows = [13, 14, 15, 1]
    sample = Sample(tokens=[101, 102, 103])
    sample.append_response_tokens(
        args,
        tokens=[201, 202],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32(prefill_rows + decode_rows),
            "finish_reason": {"type": "length"},
        },
    )
    validate_routed_experts_tensor(
        sample.rollout_routed_experts, args, sample_index=0, expected_rows=len(sample.tokens) - 1
    )

    # Turn 2 continuation with routed_experts_start_len=4 (prompt_len=5, 2 new tokens -> 2 rows):
    # prefill returns [4, 5) (1 row) and decode returns [5, 6) (1 row).
    cont_prefill = [2, 3, 4, 5]
    cont_decode = [6, 7, 8, 9]
    sample.append_response_tokens(
        args,
        tokens=[203, 204],
        log_probs=[-0.3, -0.4],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32(cont_prefill + cont_decode),
            "routed_experts_start_len": 4,
            "finish_reason": {"type": "stop"},
        },
    )
    merged = sample.materialize_rollout_routed_experts()
    assert merged.shape == (6, 2, 2)
    validate_routed_experts_tensor(merged, args, sample_index=0, expected_rows=len(sample.tokens) - 1)


@pytest.mark.unit
@pytest.mark.parametrize("patch_version", ["v0.5.15.post1", "v0.5.21", "latest"])
def test_sglang_patches_preserve_pd_routed_experts_collection_and_decode_clamp(patch_version):
    from pathlib import Path

    patch_path = Path(__file__).resolve().parents[1] / "docker" / "patch" / patch_version / "sglang.patch"
    patch_text = patch_path.read_text()

    assert "+                if req.return_routed_experts:" in patch_text
    assert "+                    self.batch_result_processor._maybe_collect_routed_experts(req)" in patch_text
    assert "+            if self.disaggregation_mode == DisaggregationMode.DECODE:" in patch_text
    assert "+                req.routed_experts_start_len = len(req.origin_input_ids)" in patch_text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
