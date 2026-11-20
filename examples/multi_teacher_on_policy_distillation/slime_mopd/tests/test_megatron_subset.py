import pytest

from slime_mopd.megatron_subset import build_teacher_subset, scatter_subset_into


def _rollout_data(micro_batch_indices, sample_count, **extra):
    data = {
        "micro_batch_indices": micro_batch_indices,
        "tokens": [f"tok{i}" for i in range(sample_count)],
        "total_lengths": [10 + i for i in range(sample_count)],
        "response_lengths": [4 + i for i in range(sample_count)],
        "loss_masks": [f"mask{i}" for i in range(sample_count)],
    }
    data.update(extra)
    return data


def test_subset_compacts_and_tiles_exactly_once():
    data = _rollout_data([[0, 1], [2]], sample_count=3)
    subset, selected = build_teacher_subset(data, ["a", "b", "a"], "a")

    assert selected == [0, 2]
    assert subset["micro_batch_indices"] == [[0], [1]]
    assert subset["tokens"] == ["tok0", "tok2"]
    assert subset["total_lengths"] == [10, 12]


def test_microbatch_mixing_teachers_is_split():
    data = _rollout_data([[0, 1, 2]], sample_count=3)
    subset_a, selected_a = build_teacher_subset(data, ["a", "b", "a"], "a")
    subset_b, selected_b = build_teacher_subset(data, ["a", "b", "a"], "b")

    assert (selected_a, subset_a["micro_batch_indices"]) == ([0, 2], [[0, 1]])
    assert (selected_b, subset_b["micro_batch_indices"]) == ([1], [[0]])


def test_teacher_without_samples_returns_none():
    data = _rollout_data([[0], [1]], sample_count=2)
    subset, selected = build_teacher_subset(data, ["a", "a"], "b")
    assert subset is None
    assert selected == []


def test_untiled_subset_raises():
    data = _rollout_data([[0]], sample_count=2)
    with pytest.raises(ValueError, match="not tiled exactly once"):
        build_teacher_subset(data, ["a", "a"], "a")


def test_scorer_field_length_mismatch_raises():
    data = _rollout_data([[0, 1]], sample_count=2)
    data["tokens"] = ["only-one"]
    with pytest.raises(ValueError, match="scorer field 'tokens'"):
        build_teacher_subset(data, ["a", "a"], "a")


def test_missing_optional_fields_are_omitted():
    data = {"micro_batch_indices": [[0]], "tokens": ["t0"]}
    subset, selected = build_teacher_subset(data, ["a"], "a")
    assert selected == [0]
    assert "loss_masks" not in subset


def test_scatter_subset_into_original_positions():
    full = [None, None, None, None, None]
    scatter_subset_into(full, [3, 0, 4], ["c", "a", "e"])
    assert full == ["a", None, None, "c", "e"]


def test_scatter_size_mismatch_raises():
    with pytest.raises(ValueError, match="scatter size mismatch"):
        scatter_subset_into([None], [0], [])
