"""CPU checks for the SGLang control protocol used by PipelineRL."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


@pytest.fixture
def engine_module(monkeypatch):
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    utils = types.ModuleType("sglang.srt.utils")
    utils.kill_process_tree = lambda *_: None
    monkeypatch.setitem(sys.modules, server_args.__name__, server_args)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = Path(__file__).resolve().parents[1] / "slime/backends/sglang_utils/sglang_engine.py"
    spec = importlib.util.spec_from_file_location("pipeline_rl_engine_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flush_cache", [False, True])
def test_pause_mode_and_disk_reload_preserve_cache_only_when_enabled(engine_module, monkeypatch, flush_cache):
    engine = engine_module.SGLangEngine(types.SimpleNamespace(), rank=0)
    engine.node_rank = 0
    engine.server_host = "engine"
    engine.server_port = 30000
    requests = []

    def post(url, *, json):
        requests.append((url, json))
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"success": True})

    monkeypatch.setattr(engine_module.requests, "post", post)
    engine.pause_generation(mode="abort" if flush_cache else "in_place")
    engine.update_weights_from_disk("/weights", weight_version="2", flush_cache=flush_cache)
    engine.continue_generation()
    assert requests == [
        ("http://engine:30000/pause_generation", {"mode": "abort" if flush_cache else "in_place"}),
        (
            "http://engine:30000/update_weights_from_disk",
            {"model_path": "/weights", "flush_cache": flush_cache, "weight_version": "2"},
        ),
        ("http://engine:30000/continue_generation", {}),
    ]


def test_nonzero_engine_rank_does_not_send_control_requests(engine_module, monkeypatch):
    engine = engine_module.SGLangEngine(types.SimpleNamespace(), rank=1)
    engine.node_rank = 1
    monkeypatch.setattr(
        engine_module.requests, "post", lambda *a, **kw: pytest.fail("Only node rank zero controls serving")
    )
    engine.pause_generation()
    engine.update_weights_from_disk("/weights", weight_version="2")
    engine.continue_generation()


@pytest.mark.parametrize(
    "interval,expected",
    [(-1, [1]), (-100, [1]), (0, [1]), (1, [1, 2, 3, 4, 5, 6, 7]), (2, [1, 3, 5, 7]), (3, [1, 4, 7])],
)
def test_flush_schedule_counts_training_updates_and_handles_restored_versions(interval, expected):
    from slime.utils.weight_sync import should_flush_cache

    assert [version for version in range(1, 8) if should_flush_cache(interval, version)] == expected
    # Restoring a nonzero serving version keeps the same phase.
    assert [version for version in range(5, 8) if should_flush_cache(interval, version)] == [
        v for v in expected if v >= 5
    ]


@pytest.mark.parametrize("interval", [-1, 0, 1, 2, 3])
def test_first_publication_after_resume_flushes_health_check_cache(interval):
    from slime.utils.weight_sync import should_flush_cache

    assert should_flush_cache(interval, weight_version=6, initial_weight_version=5)
    assert should_flush_cache(interval, weight_version=7, initial_weight_version=5) == should_flush_cache(interval, 7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
