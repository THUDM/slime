from types import SimpleNamespace

import pytest
from slime_mopd.config import MopdConfig, MopdConfigError, clear_config_cache, load_mopd_config


def _raw(**overrides):
    raw = {
        "teachers": [
            {"id": "math", "backend": "megatron", "load": "/models/rl-math"},
            {"id": "code", "backend": "megatron", "load": "/models/rl-code"},
            {"id": "if", "backend": "megatron", "load": "/models/rl-if"},
        ]
    }
    raw.update(overrides)
    return raw


def test_teacher_ids_are_the_domains():
    config = MopdConfig.from_raw(_raw())

    assert [teacher.id for teacher in config.teachers] == ["math", "code", "if"]
    assert config.teacher_for_domain("math") == "math"
    assert config.teacher_for_domain("code") == "code"
    assert config.teacher_for_domain("if") == "if"


def test_unknown_domain_has_no_teacher():
    config = MopdConfig.from_raw(_raw())

    assert config.teacher_for_domain("vision") is None


def test_domains_block_remaps_labels():
    config = MopdConfig.from_raw(_raw(teachers=[{"id": "t_math", "backend": "megatron"}], domains={"math": "t_math"}))

    assert config.teacher_for_domain("math") == "t_math"
    assert config.teacher_for_domain("code") is None


def test_domain_referencing_unknown_teacher_rejected():
    with pytest.raises(MopdConfigError, match="unknown teacher"):
        MopdConfig.from_raw(_raw(domains={"math": "nope"}))


def test_duplicate_teacher_ids_rejected():
    with pytest.raises(MopdConfigError, match="duplicate teacher ids"):
        MopdConfig.from_raw(
            _raw(
                teachers=[
                    {"id": "math", "backend": "megatron"},
                    {"id": "math", "backend": "megatron"},
                ]
            )
        )


def test_unknown_backend_rejected():
    with pytest.raises(MopdConfigError, match="unsupported backend"):
        MopdConfig.from_raw(_raw(teachers=[{"id": "math", "backend": "vllm"}]))


def test_sglang_rm_url_must_be_http():
    with pytest.raises(MopdConfigError, match="http"):
        MopdConfig.from_raw(_raw(teachers=[{"id": "math", "backend": "sglang", "rm_url": "localhost:1234"}]))


def test_default_teacher_must_exist():
    with pytest.raises(MopdConfigError, match="default_teacher"):
        MopdConfig.from_raw(_raw(default_teacher="ghost"))


def test_router_block_still_accepts_default_teacher():
    config = MopdConfig.from_raw(_raw(router={"default_teacher": "math"}))

    assert config.default_teacher == "math"


def test_relative_teacher_load_resolves_against_config_dir():
    config = MopdConfig.from_raw(
        {"teachers": [{"id": "math", "backend": "megatron", "load": "teachers/math"}]},
        base_dir="/srv/mopd",
    )

    assert config.teacher("math").load == "/srv/mopd/teachers/math"


def test_absolute_teacher_load_is_untouched():
    config = MopdConfig.from_raw(_raw(), base_dir="/srv/mopd")

    assert config.teacher("math").load == "/models/rl-math"


def test_active_teachers_filter():
    config = MopdConfig.from_raw(_raw(active_teachers=["math", "if"]))

    assert [teacher.id for teacher in config.active_teachers()] == ["math", "if"]
    assert not config.is_active("code")


def test_active_teachers_unknown_id_rejected():
    with pytest.raises(MopdConfigError, match="unknown ids"):
        MopdConfig.from_raw(_raw(active_teachers=["ghost"]))


def test_logprob_dtype_validated():
    with pytest.raises(MopdConfigError, match="logprob_dtype"):
        MopdConfig.from_raw(_raw(logprob_dtype="int8"))


def test_missing_teachers_rejected():
    with pytest.raises(MopdConfigError, match="non-empty 'teachers' list"):
        MopdConfig.from_raw({})


def test_mopd_wrapper_is_unwrapped():
    config = MopdConfig.from_raw({"mopd": _raw()})

    assert config.teacher_for_domain("code") == "code"


def test_inline_config_resolves_load_against_custom_config_path(tmp_path):
    clear_config_cache()
    config_path = tmp_path / "mopd.yaml"
    config_path.write_text("mopd:\n  teachers: []\n")
    args = SimpleNamespace(
        mopd=_raw(teachers=[{"id": "math", "backend": "megatron", "load": "models/math"}]),
        custom_config_path=str(config_path),
    )

    config = load_mopd_config(args)

    assert config.base_dir == str(tmp_path)
    assert config.teacher("math").load == str(tmp_path / "models" / "math")


def test_missing_config_raises(monkeypatch):
    clear_config_cache()
    monkeypatch.delenv("SLIME_MOPD_CONFIG", raising=False)

    with pytest.raises(MopdConfigError, match="MOPD config not found"):
        load_mopd_config(SimpleNamespace(mopd=None, mopd_config=None))
