from types import SimpleNamespace

import slime_mopd.train_data as train_data
from slime_mopd.config import MopdConfig, clear_config_cache


def _args():
    return SimpleNamespace(
        mopd={
            "teachers": [
                {"id": "t_a", "backend": "megatron", "load": "/x"},
                {"id": "t_b", "backend": "megatron", "load": "/y"},
            ],
        },
        custom_reward_post_process_path=None,
    )


def _sample(prompt, teacher=None):
    metadata = {"teacher": teacher} if teacher else {}
    return SimpleNamespace(prompt=prompt, metadata=metadata)


def test_convert_forwards_teacher_ids_via_source_names(monkeypatch):
    clear_config_cache()
    monkeypatch.setattr(train_data, "_default_convert", lambda args, samples: {"tokens": []})
    samples = [_sample("q1", teacher="t_a"), _sample("q2", teacher="t_b")]
    out = train_data.mopd_convert_samples_to_train_data(_args(), samples)
    assert out["source_names"] == ["t_a", "t_b"]
    # A non-numeric field would break slime's log_rollout_data (sum over lists).
    assert "prompt" not in out


def test_convert_caches_resolved_id_on_sample(monkeypatch):
    clear_config_cache()
    monkeypatch.setattr(train_data, "_default_convert", lambda args, samples: {})
    sample = _sample("q1", teacher="t_a")
    train_data.mopd_convert_samples_to_train_data(_args(), [sample])
    assert sample.mopd_teacher_id == "t_a"


def test_resolve_train_teacher_ids_prefers_cached_id():
    clear_config_cache()
    config = MopdConfig.from_raw(_args().mopd)
    sample = _sample("q", teacher="t_a")
    sample.mopd_teacher_id = "t_b"
    assert train_data.resolve_train_teacher_ids(config, _args(), [sample]) == ["t_b"]


def test_resolve_train_teacher_ids_routes_by_metadata_teacher():
    clear_config_cache()
    config = MopdConfig.from_raw(_args().mopd)
    sample = _sample("q", teacher="t_b")
    assert train_data.resolve_train_teacher_ids(config, _args(), [sample]) == ["t_b"]
