import logging

import pytest

from slime_mopd.config import MopdConfig
from slime_mopd.routing import (
    RoutingError,
    TrainSampleView,
    pop_default_fallback_count,
    reset_default_fallback_stats,
    resolve_teacher,
    resolve_teacher_id,
    sample_domain,
    set_sample_teacher,
)


class _Sample:
    def __init__(self, source=None, metadata=None):
        self.source = source
        self.metadata = metadata or {}


def _config(**overrides):
    raw = {
        "teachers": [
            {"id": "math", "rm_url": "http://127.0.0.1:30001"},
            {"id": "code", "rm_url": "http://127.0.0.1:30002"},
            {"id": "if", "rm_url": "http://127.0.0.1:30003"},
        ],
        "active_teachers": ["math", "code", "if"],
    }
    raw.update(overrides)
    return MopdConfig.from_raw(raw)


@pytest.fixture(autouse=True)
def _reset_fallback():
    reset_default_fallback_stats()
    yield
    reset_default_fallback_stats()


def test_domain_label_routes_by_identity():
    config = _config()

    for domain in ("math", "code", "if"):
        sample = _Sample(source=domain, metadata={"mopd_domain": domain})
        assert resolve_teacher_id(config, None, sample) == domain


def test_domain_metadata_wins_over_source():
    config = _config()
    sample = _Sample(source="code", metadata={"mopd_domain": "math"})

    assert resolve_teacher_id(config, None, sample) == "math"


def test_domains_block_renames_teacher():
    config = MopdConfig.from_raw(
        {
            "teachers": [{"id": "t_math", "rm_url": "http://127.0.0.1:1"}],
            "domains": {"math": "t_math"},
        }
    )
    sample = _Sample(source="math", metadata={"mopd_domain": "math"})

    assert resolve_teacher_id(config, None, sample) == "t_math"


def test_explicit_annotation_beats_domain_label():
    config = _config()
    sample = _Sample(source="math", metadata={"mopd_domain": "math", "teacher": "code"})

    assert resolve_teacher_id(config, None, sample) == "code"


def test_mopd_teacher_id_annotation_is_honoured():
    config = _config()
    sample = _Sample(metadata={"mopd_teacher_id": "if"})

    assert resolve_teacher_id(config, None, sample) == "if"


def test_train_sample_view_routes_from_source_name():
    config = _config()
    view = TrainSampleView(prompt="p", source_name="if", index=3)

    assert sample_domain(view) == "if"
    assert resolve_teacher_id(config, None, view) == "if"


def test_unroutable_sample_raises():
    config = _config()
    sample = _Sample(metadata={"mopd_domain": "vision"})

    with pytest.raises(RoutingError, match="cannot route sample"):
        resolve_teacher_id(config, None, sample)


def test_default_teacher_fallback_is_counted_once(caplog):
    config = _config(default_teacher="math")
    samples = [_Sample(metadata={"mopd_domain": "vision"}) for _ in range(5)]

    with caplog.at_level(logging.WARNING):
        assert {resolve_teacher_id(config, None, s) for s in samples} == {"math"}

    assert pop_default_fallback_count() == 5
    warnings = [record for record in caplog.records if "default_teacher" in record.message]
    assert len(warnings) == 1
    assert pop_default_fallback_count() == 0


def test_inactive_teacher_is_rejected():
    config = _config(active_teachers=["math"], default_teacher="code")
    sample = _Sample(metadata={"mopd_domain": "vision"})

    with pytest.raises(RoutingError, match="inactive"):
        resolve_teacher(config, None, sample)


def test_resolve_teacher_returns_config_object():
    config = _config()
    teacher = resolve_teacher(config, None, _Sample(metadata={"mopd_domain": "code"}))

    assert teacher.id == "code"
    assert teacher.rm_url == "http://127.0.0.1:30002"


def test_routed_teacher_must_be_registered():
    config = _config()
    sample = _Sample(metadata={"teacher": "ghost"})

    with pytest.raises(RoutingError, match="not defined in the registry"):
        resolve_teacher(config, None, sample)


def test_set_sample_teacher_caches_on_attribute():
    sample = _Sample()
    set_sample_teacher(sample, "code")

    assert sample.mopd_teacher_id == "code"
    assert "mopd_teacher_id" not in sample.metadata


def test_set_sample_teacher_falls_back_to_metadata():
    class _Slotted:
        __slots__ = ("metadata",)

        def __init__(self):
            self.metadata = {}

    sample = _Slotted()
    set_sample_teacher(sample, "if")

    assert sample.metadata["mopd_teacher_id"] == "if"
