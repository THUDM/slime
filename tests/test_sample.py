"""CPU unit tests for ``slime.utils.types.Sample``.

Pins two contracts that the rollout / training boundary depends on:

  1. ``to_dict`` / ``from_dict`` round-trip — Sample crosses Ray actor
     boundaries as a dict (especially in async / fully-async / partial-
     rollout paths). A silent field drop or enum corruption here means a
     sample loses its status / spec_info / prefix_cache_info / group_id
     on the way to the trainer with no crash signal.

  2. ``update_from_meta_info`` finish_reason → Status enum mapping
     (length→TRUNCATED, abort→ABORTED, stop→COMPLETED). The match
     statement at types.py:176-182 is the only place sglang's
     finish_reason gets translated; a typo'd enum or removed case here
     would silently mis-tag every sample.
"""

from __future__ import annotations

import argparse

import pytest

from slime.utils.types import Sample


NUM_GPUS = 0


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def _make_sample(**overrides) -> Sample:
    """Build a Sample with one non-default value per field-category so the
    round-trip test exercises every code path in to_dict/from_dict, not
    just the trivial defaults case."""
    base = dict(
        group_index=0,
        index=42,
        group_id=7,
        prompt="hello",
        tokens=[1, 2, 3],
        multimodal_inputs={"images": ["fake_url"]},
        response="world",
        response_length=5,
        label="42",
        reward=0.75,
        loss_mask=[1, 1, 0, 1, 1],
        weight_versions=["v1"],
        rollout_log_probs=[-0.1, -0.2],
        rollout_routed_experts=[[0, 1], [2, 3]],
        remove_sample=False,
        teacher_log_probs=[-0.3, -0.4],
        status=Sample.Status.COMPLETED,
        metadata={"rm_type": "math"},
        generate_function_path="some.module.fn",
        train_metadata={"loss_type": "policy_loss"},
        session_id="uuid-1234",
        non_generation_time=1.5,
    )
    base.update(overrides)
    return Sample(**base)


@pytest.mark.unit
def test_to_dict_serializes_status_as_string_value():
    """The ``status`` field is an enum on the dataclass; ``to_dict`` must
    flatten it to its string value so it survives JSON / pickle across
    Ray boundaries."""
    sample = _make_sample()
    d = sample.to_dict()
    assert d["status"] == "completed"  # not Sample.Status.COMPLETED
    assert isinstance(d["status"], str)


@pytest.mark.unit
def test_to_dict_flattens_spec_info_and_prefix_cache_info():
    """``spec_info`` and ``prefix_cache_info`` are nested dataclasses;
    to_dict converts each via its own to_dict (types.py:133-134)."""
    sample = _make_sample()
    sample.spec_info.spec_accept_token_num = 10
    sample.spec_info.spec_draft_token_num = 20
    sample.prefix_cache_info.cached_tokens = 5
    sample.prefix_cache_info.total_prompt_tokens = 50

    d = sample.to_dict()
    assert d["spec_info"] == {
        "spec_accept_token_num": 10,
        "spec_draft_token_num": 20,
        "spec_verify_ct": 0,
        "completion_token_num": 0,
    }
    assert d["prefix_cache_info"] == {"cached_tokens": 5, "total_prompt_tokens": 50}


@pytest.mark.unit
def test_round_trip_preserves_every_field():
    """Serialize → deserialize → compare. If any field gets silently
    dropped on either side, the new sample won't equal the old. Uses
    ``__dict__`` equality (not ``__eq__`` on the dataclass, which Sample
    doesn't define) so nested SpecInfo / PrefixCacheInfo also get
    compared structurally."""
    original = _make_sample()
    original.spec_info.spec_accept_token_num = 3
    original.prefix_cache_info.cached_tokens = 7

    restored = Sample.from_dict(original.to_dict())

    # Status came back as the enum, not the string value.
    assert restored.status is Sample.Status.COMPLETED
    # Nested infos round-tripped as the correct type.
    assert isinstance(restored.spec_info, Sample.SpecInfo)
    assert isinstance(restored.prefix_cache_info, Sample.PrefixCacheInfo)
    assert restored.spec_info.spec_accept_token_num == 3
    assert restored.prefix_cache_info.cached_tokens == 7

    # All non-nested fields preserved.
    for field in (
        "group_index",
        "index",
        "group_id",
        "prompt",
        "tokens",
        "multimodal_inputs",
        "response",
        "response_length",
        "label",
        "reward",
        "loss_mask",
        "weight_versions",
        "rollout_log_probs",
        "rollout_routed_experts",
        "remove_sample",
        "teacher_log_probs",
        "metadata",
        "generate_function_path",
        "train_metadata",
        "session_id",
        "non_generation_time",
    ):
        assert getattr(restored, field) == getattr(original, field), f"field {field} drifted"


@pytest.mark.unit
def test_group_id_accepts_legacy_rollout_id_assignment_only():
    """Older custom rollout code may assign ``rollout_id``; constructor use
    is no longer supported."""
    with pytest.raises(TypeError, match="rollout_id"):
        Sample(index=42, rollout_id=9)

    sample = Sample(index=42)
    with pytest.warns(DeprecationWarning, match="Sample.rollout_id is deprecated"):
        sample.rollout_id = 10
    assert sample.group_id == 10
    assert sample.to_dict()["group_id"] == 10
    with pytest.raises(AttributeError, match="write-only"):
        _ = sample.rollout_id


@pytest.mark.unit
def test_from_dict_accepts_legacy_rollout_id_without_group_id():
    """Older debug dumps may only carry ``rollout_id``."""
    d = _make_sample().to_dict()
    d["rollout_id"] = 13
    del d["group_id"]

    with pytest.warns(DeprecationWarning, match="Sample.rollout_id is deprecated"):
        restored = Sample.from_dict(d)

    assert restored.group_id == 13
    with pytest.raises(AttributeError, match="write-only"):
        _ = restored.rollout_id


@pytest.mark.unit
def test_from_dict_prefers_group_id_over_legacy_rollout_id():
    """If both names exist, the canonical group_id wins."""
    d = _make_sample(group_id=11).to_dict()
    d["rollout_id"] = 13

    restored = Sample.from_dict(d)

    assert restored.group_id == 11


@pytest.mark.unit
def test_group_id_does_not_serialize_legacy_rollout_id_alias():
    """New debug dumps should only carry the canonical ``group_id``."""
    sample = Sample(index=42, group_id=11)

    with pytest.raises(AttributeError, match="write-only"):
        _ = sample.rollout_id
    assert sample.to_dict()["group_id"] == 11
    assert "rollout_id" not in sample.to_dict()


@pytest.mark.unit
def test_from_dict_preserves_unknown_fields_as_attributes():
    """``from_dict`` keeps unknown keys as attributes (types.py:148-150),
    not as dataclass fields. This is what lets newer rollout code stash
    extra metadata that older trainers will simply ignore — a back-compat
    contract worth pinning."""
    d = _make_sample().to_dict()
    d["future_extension"] = "carried through"

    restored = Sample.from_dict(d)
    assert restored.future_extension == "carried through"  # type: ignore[attr-defined]


@pytest.mark.unit
def test_round_trip_through_default_constructed_sample():
    """A bare Sample (only defaults) must also round-trip — this is the
    common case for a freshly-spawned rollout. Catches regressions where
    ``from_dict`` requires a key that ``to_dict`` doesn't always emit."""
    original = Sample()
    restored = Sample.from_dict(original.to_dict())
    assert restored.status is Sample.Status.PENDING
    assert restored.tokens == []
    assert restored.metadata == {}


# ---------------------------------------------------------------------------
# update_from_meta_info — finish_reason → Status mapping
# ---------------------------------------------------------------------------


def _make_args(speculative: bool = False) -> argparse.Namespace:
    """``update_from_meta_info`` only consults ``args.sglang_speculative_algorithm``
    — minimal stub is enough."""
    return argparse.Namespace(sglang_speculative_algorithm=speculative)


@pytest.mark.unit
@pytest.mark.parametrize(
    "finish_reason,expected_status",
    [
        ("length", Sample.Status.TRUNCATED),
        ("abort", Sample.Status.ABORTED),
        ("stop", Sample.Status.COMPLETED),
    ],
)
def test_status_mapping_for_each_finish_reason(finish_reason, expected_status):
    """The match statement at types.py:176-182 is the one place sglang's
    finish_reason ever gets translated. Each branch must hit the right
    enum; a typo in the enum name would crash later in unrelated places."""
    sample = Sample()
    sample.update_from_meta_info(
        _make_args(),
        meta_info={"finish_reason": {"type": finish_reason}},
    )
    assert sample.status is expected_status


@pytest.mark.unit
def test_unknown_finish_reason_leaves_status_unchanged():
    """No ``case`` matches → status stays at whatever it was. Pins the
    "no default clause means no-op" behavior so a future refactor adding
    a default doesn't silently break this contract."""
    sample = Sample()
    sample.status = Sample.Status.PENDING
    sample.update_from_meta_info(
        _make_args(),
        meta_info={"finish_reason": {"type": "something_new"}},
    )
    assert sample.status is Sample.Status.PENDING


@pytest.mark.unit
def test_weight_version_is_appended_when_present():
    """``weight_version`` in meta_info is appended to the sample's list
    (types.py:173-174) — partial-rollout uses this to track which model
    version produced each chunk."""
    sample = Sample()
    sample.weight_versions = ["v1"]
    sample.update_from_meta_info(
        _make_args(),
        meta_info={
            "finish_reason": {"type": "stop"},
            "weight_version": "v2",
        },
    )
    assert sample.weight_versions == ["v1", "v2"]


@pytest.mark.unit
def test_prefix_cache_info_is_accumulated_across_calls():
    """Every call to update_from_meta_info adds to prefix_cache_info
    (types.py:171). Multi-turn rollouts call this once per turn — the
    counts must accumulate, not overwrite."""
    sample = Sample()
    for prompt_tokens, cached_tokens in [(100, 0), (200, 50)]:
        sample.update_from_meta_info(
            _make_args(),
            meta_info={
                "finish_reason": {"type": "stop"},
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
            },
        )
    assert sample.prefix_cache_info.cached_tokens == 50  # 0 + 50
    assert sample.prefix_cache_info.total_prompt_tokens == 300  # 100 + 200


@pytest.mark.unit
def test_spec_info_only_updated_when_speculative_enabled():
    """``spec_info.add`` is gated on ``args.sglang_speculative_algorithm``
    (types.py:166-168). Without the flag, spec stats stay at zero even
    if sglang sends them."""
    meta_info = {
        "finish_reason": {"type": "stop"},
        "spec_accept_token_num": 7,
        "spec_draft_token_num": 10,
    }

    no_spec = Sample()
    no_spec.update_from_meta_info(_make_args(speculative=False), meta_info=meta_info)
    assert no_spec.spec_info.spec_accept_token_num == 0

    with_spec = Sample()
    with_spec.update_from_meta_info(_make_args(speculative=True), meta_info=meta_info)
    assert with_spec.spec_info.spec_accept_token_num == 7
    assert with_spec.spec_info.spec_draft_token_num == 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
