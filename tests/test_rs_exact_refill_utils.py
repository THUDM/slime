from types import SimpleNamespace

import pytest
import torch

from slime.utils.rs_refill import (
    apply_rs_refill_tis,
    attach_proximal_log_probs,
    clone_rs_masks,
    compute_sequence_rs_masks,
    get_rs_refill_candidate_group_multiple,
    merge_replacement_metrics,
    merge_selected_log_prob_caches,
    plan_topology_aligned_rs_refill,
    run_rs_batch_refill,
    select_accepted_groups,
    snapshot_sample_masks,
    validate_final_rs_masks,
    validate_initial_policy_staleness,
    validate_refill_rollout_ids,
    validate_replacement_policy_version,
    validate_rs_refill_target_batch_alignment,
    validate_sample_masks,
)

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "n_samples_per_prompt": 2,
        "rollout_batch_size": 8,
        "tis_level": "token",
        "tis_mode": "clip",
        "tis_lower_bound": 0.5,
        "tis_upper_bound": 2.0,
        "tis_batch_normalize": False,
        "rs_level": "geometric",
        "rs_lower_bound": 0.6,
        "rs_upper_bound": 1.5,
        "rs_veto_threshold": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _topology(**overrides):
    values = {
        "dp_size": 2,
        "vpp_size": 2,
        "microbatch_group_size_per_vp_stage": 4,
    }
    values.update(overrides)
    return values


def _group(group_index: int, *, size: int = 2, weight_version: str = "7"):
    return [
        SimpleNamespace(
            index=group_index * 10 + sample_index,
            group_index=group_index,
            rollout_id=None,
            response_length=2,
            loss_mask=[1, 1],
            weight_versions=[weight_version],
        )
        for sample_index in range(size)
    ]


def _report(sample, *, passed: bool = True, policy_version: str = "7"):
    return {
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "valid_tokens": 2,
        "gate_passed": passed,
        "policy_version": policy_version,
    }


@pytest.mark.parametrize(
    ("required", "expected"),
    [(1, 4), (4, 4), (5, 8), (8, 8), (9, 12)],
)
def test_exact_planner_rounds_only_to_the_topology_quantum(required, expected):
    args = _args()
    topology = _topology()

    assert get_rs_refill_candidate_group_multiple(args, topology) == 4
    assert plan_topology_aligned_rs_refill(args, topology, required) == expected


@pytest.mark.parametrize("required", [0, -1, 1.5, True])
def test_exact_planner_rejects_invalid_deficits(required):
    with pytest.raises(ValueError, match="positive integer"):
        plan_topology_aligned_rs_refill(_args(), _topology(), required)


def test_final_effective_batch_alignment_fails_closed():
    validate_rs_refill_target_batch_alignment(_args(rollout_batch_size=8), _topology())

    with pytest.raises(ValueError, match=r"effective sample count.*10, alignment = 8"):
        validate_rs_refill_target_batch_alignment(_args(rollout_batch_size=5), _topology())


@pytest.mark.parametrize(
    ("topology", "samples_per_group", "expected"),
    [
        ({"dp_size": 8, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}, 3, 8),
        ({"dp_size": 2, "vpp_size": 2, "microbatch_group_size_per_vp_stage": 4}, 6, 4),
        ({"dp_size": 4, "vpp_size": 2, "microbatch_group_size_per_vp_stage": 2}, 2, 4),
    ],
)
def test_candidate_quantum_covers_dp_vpp_alignment(topology, samples_per_group, expected):
    assert get_rs_refill_candidate_group_multiple(_args(n_samples_per_prompt=samples_per_group), topology) == expected


@pytest.mark.parametrize("bad_value", [True, 1.5, "2"])
def test_candidate_quantum_rejects_non_integer_topology_values(bad_value):
    with pytest.raises(ValueError, match="dp_size must be an integer"):
        get_rs_refill_candidate_group_multiple(_args(), _topology(dp_size=bad_value))


def test_sequence_and_geometric_admission_use_full_log_ratio():
    train = [torch.log(torch.tensor([1.4, 1.4]))]
    rollout = [torch.zeros(2)]
    masks = [torch.ones(2)]

    sequence = compute_sequence_rs_masks(
        _args(rs_level="sequence"),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=masks,
    )
    geometric = compute_sequence_rs_masks(
        _args(rs_level="geometric"),
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=masks,
    )

    torch.testing.assert_close(sequence[0], torch.zeros(2))
    torch.testing.assert_close(geometric[0], torch.ones(2))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_admission_rejects_nonfinite_ratios_even_on_masked_tokens(bad_value):
    modified = compute_sequence_rs_masks(
        _args(rs_lower_bound=0.0, rs_upper_bound=10.0),
        train_log_probs=[torch.tensor([0.0, bad_value])],
        rollout_log_probs=[torch.zeros(2)],
        loss_masks=[torch.tensor([1.0, 0.0])],
    )

    torch.testing.assert_close(modified[0], torch.zeros(2))


def test_veto_rejects_the_whole_sequence():
    modified = compute_sequence_rs_masks(
        _args(rs_lower_bound=0.0, rs_upper_bound=10.0, rs_veto_threshold=0.1),
        train_log_probs=[torch.log(torch.tensor([1.0, 0.01]))],
        rollout_log_probs=[torch.zeros(2)],
        loss_masks=[torch.ones(2)],
    )

    torch.testing.assert_close(modified[0], torch.zeros(2))


def test_tis_applies_clipped_weights_and_reuses_the_admission_rule():
    ratios = torch.tensor([2.0, 0.25])
    train = [torch.log(ratios)]
    rollout = [torch.zeros(2)]
    masks = [torch.ones(2)]
    pg_loss = torch.ones(2)

    weighted_loss, modified_masks, metrics = apply_rs_refill_tis(
        _args(),
        pg_loss=pg_loss,
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=masks,
    )

    torch.testing.assert_close(weighted_loss, torch.tensor([2.0, 0.5]))
    torch.testing.assert_close(modified_masks[0], torch.ones(2))
    torch.testing.assert_close(
        modified_masks[0],
        compute_sequence_rs_masks(
            _args(),
            train_log_probs=train,
            rollout_log_probs=rollout,
            loss_masks=masks,
        )[0],
    )
    assert metrics["mis_tis_clip_fraction_low"].shape == pg_loss.shape
    assert metrics["mis_is_ratio_mean_final"].shape == pg_loss.shape


def test_tis_rejects_unsupported_batch_normalization_and_shape_mismatch():
    values = [torch.zeros(2)]
    masks = [torch.ones(2)]
    with pytest.raises(ValueError, match="batch normalization"):
        apply_rs_refill_tis(
            _args(tis_batch_normalize=True),
            pg_loss=torch.ones(2),
            train_log_probs=values,
            rollout_log_probs=values,
            loss_masks=masks,
        )

    with pytest.raises(ValueError, match="policy-gradient loss shape"):
        apply_rs_refill_tis(
            _args(),
            pg_loss=torch.ones(1),
            train_log_probs=values,
            rollout_log_probs=values,
            loss_masks=masks,
        )


def test_group_atomic_selection_preserves_order_and_separates_surplus():
    groups = [_group(index) for index in range(4)]
    reports = [_report(sample, passed=sample.group_index != 1) for group in groups for sample in group]

    selection = select_accepted_groups(groups, reports, target_size=2)

    assert selection.accepted_groups == [groups[0], groups[2]]
    assert selection.rejected_groups == [groups[1]]
    assert selection.surplus_groups == [groups[3]]
    assert selection.deficit == 0


def test_one_rejected_completion_rejects_the_whole_group():
    groups = [_group(0), _group(1)]
    reports = [_report(sample) for group in groups for sample in group]
    reports[1]["gate_passed"] = False

    selection = select_accepted_groups(groups, reports, target_size=2)

    assert selection.accepted_groups == [groups[1]]
    assert selection.rejected_groups == [groups[0]]
    assert selection.deficit == 1


def test_selection_rejects_duplicate_ids_and_unknown_reports():
    group = _group(0)
    reports = [_report(sample) for sample in group]

    with pytest.raises(ValueError, match="duplicate sample index"):
        select_accepted_groups([group], reports, target_size=1, known_sample_indices={group[0].index})

    reports.append({**reports[0], "sample_index": 999})
    with pytest.raises(ValueError, match="unknown sample indices"):
        select_accepted_groups([group], reports, target_size=1)


@pytest.mark.parametrize(("rollout_version", "actor_version", "expected"), [("7", "7", 0), ("7", "8", 1)])
def test_initial_policy_staleness_is_bounded_to_one_step(rollout_version, actor_version, expected):
    groups = [_group(0, weight_version=rollout_version)]
    reports = [_report(sample, policy_version=actor_version) for sample in groups[0]]

    assert validate_initial_policy_staleness(groups, reports) == expected


@pytest.mark.parametrize(
    ("rollout_versions", "actor_version", "message"),
    [
        (["6", "6"], "8", "hard policy-staleness bound"),
        (["8", "8"], "7", "hard policy-staleness bound"),
        (["6", "7"], "7", "span multiple"),
        (["release-a", "release-a"], "release-b", "integer actor and rollout"),
    ],
)
def test_initial_policy_staleness_fails_closed(rollout_versions, actor_version, message):
    groups = [_group(0)]
    for sample, version in zip(groups[0], rollout_versions, strict=True):
        sample.weight_versions = [version]
    reports = [_report(sample, policy_version=actor_version) for sample in groups[0]]

    with pytest.raises(ValueError, match=message):
        validate_initial_policy_staleness(groups, reports)


def test_replacements_must_match_the_scoring_policy_version():
    groups = [_group(0, weight_version="8")]
    reports = [_report(sample, policy_version="8") for sample in groups[0]]
    assert validate_replacement_policy_version(groups, reports) == "8"

    groups[0][0].weight_versions = ["7"]
    with pytest.raises(ValueError, match="current rollout policy"):
        validate_replacement_policy_version(groups, reports)


def test_effective_rollout_ids_are_unique_across_rounds():
    initial = _group(0)
    initial[0].rollout_id = 100
    known = validate_refill_rollout_ids([initial])
    assert known == {100}

    replacement = _group(1)
    replacement[0].rollout_id = 100
    with pytest.raises(ValueError, match="unique effective rollout_id"):
        validate_refill_rollout_ids([replacement], known_rollout_ids=known)

    replacement[0].rollout_id = []
    with pytest.raises(ValueError, match="hashable"):
        validate_refill_rollout_ids([replacement])


def test_selected_cache_merge_requires_an_exact_bijection():
    assert merge_selected_log_prob_caches([{1: "a"}, None, {2: "b"}], [1, 2]) == {1: "a", 2: "b"}

    with pytest.raises(ValueError, match="duplicate proximal"):
        merge_selected_log_prob_caches([{1: "a"}, {1: "b"}], [1])
    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        merge_selected_log_prob_caches([{1: "a"}], [1, 2])
    with pytest.raises(ValueError, match=r"extra=\[2\]"):
        merge_selected_log_prob_caches([{1: "a"}, {2: "b"}], [1])
    with pytest.raises(ValueError, match="cache sample index must be an integer"):
        merge_selected_log_prob_caches([{1.5: "a"}], [1])


def test_cache_attachment_preserves_final_sample_order():
    samples = [_group(1)[1], _group(0)[0]]
    data = {}

    attach_proximal_log_probs(data, samples, {0: "zero", 11: "eleven"})

    assert data["rs_preflight_log_probs"] == ["eleven", "zero"]

    with pytest.raises(ValueError, match="already attached"):
        attach_proximal_log_probs(data, samples, {0: "zero", 11: "eleven"})


def test_sample_and_training_mask_snapshots_detect_mutation():
    samples = _group(0)
    snapshot = snapshot_sample_masks(samples)
    validate_sample_masks(samples, snapshot)

    samples[0].loss_mask[1] = 0
    with pytest.raises(RuntimeError, match=r"changed=\[0\]"):
        validate_sample_masks(samples, snapshot)

    original = [torch.tensor([1, 1]), torch.tensor([1, 0])]
    cloned = clone_rs_masks(original)
    original[0][0] = 0
    torch.testing.assert_close(cloned[0], torch.tensor([1, 1]))
    with pytest.raises(RuntimeError, match=r"microbatch_positions=\[0\]"):
        validate_final_rs_masks(cloned, original)

    samples[0].loss_mask = [1, 0.5]
    with pytest.raises(ValueError, match="non-binary"):
        snapshot_sample_masks(samples)


def test_replacement_metrics_sum_only_known_counters():
    metrics = {"rollout/dynamic_filter/drop_zero_std": 2, "custom/rate": 0.25}

    merge_replacement_metrics(
        metrics,
        {"rollout/dynamic_filter/drop_zero_std": 3, "custom/rate": 0.75},
        round_index=2,
    )

    assert metrics == {
        "rollout/dynamic_filter/drop_zero_std": 5,
        "custom/rate": 0.25,
        "rollout/rs_refill/replacement_round_2/custom/rate": 0.75,
    }


class _RemoteMethod:
    def __init__(self, function):
        self.remote = function


class _FakeManager:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.events = []
        self.prepare_rs_candidate_data = _RemoteMethod(self._prepare)
        self.apply_rs_candidate_reports = _RemoteMethod(self._apply)
        self.store_rs_accepted_log_probs = _RemoteMethod(self._store)
        self.generate_rs_replacement_candidates = _RemoteMethod(self._generate)
        self.finalize_rs_batch = _RemoteMethod(self._finalize)
        self.abort_rs_batch = _RemoteMethod(self._abort)

    def _prepare(self, rollout_id):
        self.events.append(("prepare", rollout_id))
        return f"candidate-{len(self.events)}"

    def _apply(self, rollout_id, reports, seconds):
        self.events.append(("apply", rollout_id, reports, seconds))
        status = next(self.statuses)
        if isinstance(status, Exception):
            raise status
        return status

    def _store(self, rollout_id, caches):
        self.events.append(("store", rollout_id, caches))
        return True

    def _generate(self, rollout_id):
        self.events.append(("generate", rollout_id))
        return True

    def _finalize(self, rollout_id, seconds):
        self.events.append(("finalize", rollout_id, seconds))
        return "train-data"

    def _abort(self, rollout_id):
        self.events.append(("abort", rollout_id))
        return True


class _FakeActor:
    def __init__(self):
        self.events = []

    def async_score_rs_candidates(self, rollout_id, candidates):
        self.events.append(("score", rollout_id, candidates))
        return f"reports-{candidates}"

    def async_take_rs_candidate_log_probs(self, rollout_id, selected):
        self.events.append(("take", rollout_id, selected))
        return f"caches-{selected}"

    def async_discard_rs_candidate_log_probs(self, rollout_id):
        self.events.append(("discard", rollout_id))
        return "actor-cleanup"


def _status(*, complete=False, exhausted=False, round_index=0, accepted=None):
    return {
        "complete": complete,
        "exhausted": exhausted,
        "deficit": 0 if complete else 1,
        "round": round_index,
        "accepted_groups": 2 if complete else 1,
        "target_groups": 2,
        "accepted_sample_indices": list(accepted or []),
    }


def test_coordinator_runs_exact_replacement_round_and_finalizes():
    manager = _FakeManager(
        [
            _status(accepted=[10]),
            _status(complete=True, round_index=1, accepted=[20]),
        ]
    )
    actor = _FakeActor()
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).__next__

    result = run_rs_batch_refill(actor, manager, 7, resolve=lambda value: value, clock=clock)

    assert result == "train-data"
    assert [event[0] for event in manager.events] == [
        "prepare",
        "apply",
        "store",
        "generate",
        "prepare",
        "apply",
        "store",
        "finalize",
    ]
    assert [event[0] for event in actor.events] == ["score", "take", "score", "take"]
    assert manager.events[-1] == ("finalize", 7, 5.0)


def test_coordinator_cleans_actor_and_manager_state_on_report_failure():
    manager = _FakeManager([RuntimeError("score failed")])
    actor = _FakeActor()
    resolved = []

    def resolve(value):
        resolved.append(value)
        return value

    with pytest.raises(RuntimeError, match="score failed"):
        run_rs_batch_refill(actor, manager, 3, resolve=resolve, clock=iter([0.0, 1.0, 2.0]).__next__)

    assert actor.events[-1] == ("discard", 3)
    assert manager.events[-1] == ("abort", 3)
    assert "actor-cleanup" in resolved
    assert True in resolved


def test_coordinator_releases_selected_cache_before_exhaustion_cleanup():
    manager = _FakeManager([_status(exhausted=True, round_index=2, accepted=[10])])
    actor = _FakeActor()

    with pytest.raises(RuntimeError, match="exhausted its retry budget"):
        run_rs_batch_refill(actor, manager, 5, resolve=lambda value: value, clock=iter([0.0, 1.0, 2.0]).__next__)

    assert [event[0] for event in actor.events] == ["score", "take", "discard"]
    assert [event[0] for event in manager.events] == ["prepare", "apply", "store", "abort"]


@pytest.mark.parametrize(
    ("stage", "statuses"),
    [
        ("store", [_status(complete=True, accepted=[10])]),
        ("generate", [_status(accepted=[10])]),
        ("finalize", [_status(complete=True, accepted=[10])]),
    ],
)
def test_coordinator_cleans_both_sides_when_a_late_stage_fails(stage, statuses):
    manager = _FakeManager(statuses)
    actor = _FakeActor()
    method_name = {
        "store": "store_rs_accepted_log_probs",
        "generate": "generate_rs_replacement_candidates",
        "finalize": "finalize_rs_batch",
    }[stage]

    def fail(*args):
        manager.events.append((stage, *args))
        raise RuntimeError(f"{stage} failed")

    setattr(manager, method_name, _RemoteMethod(fail))

    with pytest.raises(RuntimeError, match=rf"{stage} failed"):
        run_rs_batch_refill(actor, manager, 9, resolve=lambda value: value, clock=iter(range(10)).__next__)

    assert actor.events[-1] == ("discard", 9)
    assert manager.events[-1] == ("abort", 9)


@pytest.mark.parametrize("cleanup_failure", ["submit", "resolve"])
def test_coordinator_preserves_the_primary_error_when_cleanup_fails(cleanup_failure):
    manager = _FakeManager([RuntimeError("primary failure")])
    actor = _FakeActor()

    if cleanup_failure == "submit":

        def actor_submit_failure(_rollout_id):
            raise RuntimeError("actor cleanup submit failed")

        def manager_submit_failure(_rollout_id):
            raise RuntimeError("manager cleanup submit failed")

        actor.async_discard_rs_candidate_log_probs = actor_submit_failure
        manager.abort_rs_batch = _RemoteMethod(manager_submit_failure)
        resolve = lambda value: value
    else:

        def resolve(value):
            if value in {"actor-cleanup", True}:
                raise RuntimeError("cleanup resolve failed")
            return value

    with pytest.raises(RuntimeError, match="primary failure"):
        run_rs_batch_refill(actor, manager, 11, resolve=resolve, clock=iter(range(10)).__next__)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
