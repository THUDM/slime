"""Validation tests for SGLang multi-model deployment config."""

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_config(tmp_path: Path, models: list[dict]) -> str:
    path = tmp_path / "sglang.yaml"
    path.write_text(yaml.safe_dump({"sglang": models}))
    return str(path)


def _resolve_args(**overrides):
    values = {
        "hf_checkpoint": "/models/actor",
        "num_gpus_per_node": 8,
        "rollout_num_gpus_per_engine": 2,
        "sglang_pp_size": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def _load_deployment(monkeypatch):
    class FakeServerGroup:
        def start_engines(self, port_cursors=None):
            return [], port_cursors or {}

    class FakePlacement:
        def __init__(self, **_kwargs):
            pass

        def create(self, *_args, **_kwargs):
            return FakeServerGroup()

    class FakeRolloutServer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    disaggregation = types.ModuleType("slime.backends.sglang_utils.disaggregation")
    disaggregation.start_epd_server_groups = lambda *_args, **_kwargs: ([], [])
    disaggregation.start_pd_server_groups = lambda *_args, **_kwargs: ([], [])
    engine_group = types.ModuleType("slime.backends.sglang_utils.engine_group")
    engine_group.RolloutServer = FakeRolloutServer
    engine_group.ServerGroupPlacement = FakePlacement
    external = types.ModuleType("slime.backends.sglang_utils.external")
    external.start_external_rollout_servers = lambda *_args, **_kwargs: ({}, [])
    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils._wrap_ipv6 = lambda host: host
    http_utils.find_available_port = lambda _port: 3000
    http_utils.get_host_info = lambda: ("host", "127.0.0.1")

    monkeypatch.setitem(sys.modules, disaggregation.__name__, disaggregation)
    monkeypatch.setitem(sys.modules, engine_group.__name__, engine_group)
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, http_utils.__name__, http_utils)

    path = REPO_ROOT / "slime/backends/sglang_utils/deployment.py"
    spec = importlib.util.spec_from_file_location("_sglang_deployment_under_test", path)
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    return deployment


def test_model_entries_reject_unknown_fields(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [],
                "unexpected_option": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="Model 'actor' has unknown fields: unexpected_option"):
        SglangConfig.from_yaml(path)


def test_model_entries_reject_both_server_group_aliases(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [],
                "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="Model 'actor' cannot define both 'server_groups' and legacy 'engine_groups'",
    ):
        SglangConfig.from_yaml(path)


def test_resolve_rejects_duplicate_model_names(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {"name": "actor", "update_weights": True, "server_groups": []},
            {"name": "actor", "update_weights": False, "server_groups": []},
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(ValueError, match="Duplicate model name 'actor'"):
        config.resolve(_resolve_args())


def test_resolve_rejects_multiple_updatable_models(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {"name": "actor", "update_weights": True, "server_groups": []},
            {
                "name": "ref",
                "model_path": "/models/ref",
                "update_weights": True,
                "server_groups": [],
            },
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(
        ValueError,
        match="At most one model may set update_weights=True; got: actor, ref",
    ):
        config.resolve(_resolve_args())


def test_resolve_rejects_non_divisible_server_group(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [{"worker_type": "regular", "num_gpus": 3}],
            }
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(
        ValueError,
        match=r"Model 'actor' server group 0 \(regular\) has num_gpus=3, "
        r"which is not divisible by num_gpus_per_engine=2",
    ):
        config.resolve(_resolve_args())


def test_resolve_rejects_non_positive_gpus_per_engine_override(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": 2,
                        "num_gpus_per_engine": 0,
                    }
                ],
            }
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(
        ValueError,
        match=r"Model 'actor' server group 0 \(regular\) has invalid num_gpus_per_engine=0; must be > 0",
    ):
        config.resolve(_resolve_args())


def test_resolve_does_not_replace_invalid_model_override_with_global_default(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "num_gpus_per_engine": 0,
                "server_groups": [{"worker_type": "regular", "num_gpus": 2}],
            }
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(ValueError, match="invalid num_gpus_per_engine=0"):
        config.resolve(_resolve_args(rollout_num_gpus_per_engine=2))


@pytest.mark.parametrize(
    ("model_override", "group_override", "global_default", "expected"),
    [
        (None, None, 2, 2),
        (3, None, 2, 3),
        (4, 3, 2, 3),
    ],
)
def test_resolve_uses_most_specific_gpus_per_engine(
    tmp_path,
    model_override,
    group_override,
    global_default,
    expected,
):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    group = {"worker_type": "regular", "num_gpus": 6}
    if group_override is not None:
        group["num_gpus_per_engine"] = group_override
    model = {
        "name": "actor",
        "num_gpus_per_engine": model_override,
        "server_groups": [group],
    }
    config = SglangConfig.from_yaml(_write_config(tmp_path, [model]))

    config.resolve(_resolve_args(rollout_num_gpus_per_engine=global_default))

    assert config.models[0].server_groups[0].num_gpus_per_engine == expected


@pytest.mark.parametrize(
    ("gpus_per_engine", "group_overrides", "default_pp_size", "error"),
    [
        (3, {}, 2, "num_gpus_per_engine=3, which is not divisible by pp_size=2"),
        (3, {"pp-size": 2}, 1, "num_gpus_per_engine=3, which is not divisible by pp_size=2"),
        (4, {"pp_size": 0}, 1, "invalid pp_size=0; must be > 0"),
        (4, {"pp_size": 2, "tp-size": 3}, 1, "tp_size=3 and pp_size=2 require 6 GPUs"),
        (4, {"pp-size": 2, "tp_size": 0}, 1, "invalid tp_size=0; must be > 0"),
    ],
)
def test_resolve_rejects_invalid_parallel_topology(
    tmp_path,
    gpus_per_engine,
    group_overrides,
    default_pp_size,
    error,
):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": gpus_per_engine * 2,
                        "num_gpus_per_engine": gpus_per_engine,
                        "overrides": group_overrides,
                    }
                ],
            }
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(ValueError, match=error):
        config.resolve(_resolve_args(sglang_pp_size=default_pp_size))


@pytest.mark.parametrize(
    ("gpus_per_engine", "gpus_per_node", "error"),
    [
        (2, 0, "invalid num_gpus_per_node=0; must be > 0"),
        (10, 8, "num_gpus_per_engine=10, which is not divisible by num_gpus_per_node=8"),
    ],
)
def test_resolve_rejects_invalid_multi_node_topology(tmp_path, gpus_per_engine, gpus_per_node, error):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": gpus_per_engine * 2,
                        "num_gpus_per_engine": gpus_per_engine,
                    }
                ],
            }
        ],
    )
    config = SglangConfig.from_yaml(path)

    with pytest.raises(ValueError, match=error):
        config.resolve(_resolve_args(num_gpus_per_node=gpus_per_node))


def test_resolve_allows_consistent_multi_node_parallel_topology(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "server_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": 32,
                        "num_gpus_per_engine": 16,
                        "overrides": {"pp-size": 2, "tp_size": 8},
                    }
                ],
            }
        ],
    )
    config = SglangConfig.from_yaml(path)

    config.resolve(_resolve_args(num_gpus_per_node=8))

    assert config.models[0].server_groups[0].num_gpus_per_engine == 16


def test_resolve_allows_empty_models_and_non_divisible_placeholders(tmp_path):
    from slime.backends.sglang_utils.sglang_config import SglangConfig

    path = _write_config(
        tmp_path,
        [
            {"name": "actor", "update_weights": True, "server_groups": []},
            {
                "name": "reserved",
                "update_weights": False,
                "server_groups": [{"worker_type": "placeholder", "num_gpus": 3}],
            },
        ],
    )
    config = SglangConfig.from_yaml(path)

    config.resolve(_resolve_args(rollout_num_gpus_per_engine=2, num_gpus_per_node=0))

    assert config.models[0].server_groups == []
    assert config.models[1].server_groups[0].num_gpus_per_engine == 2


@pytest.mark.parametrize(
    ("ref_group", "error"),
    [
        (
            {"worker_type": "regular", "num_gpus": 3},
            "num_gpus=3, which is not divisible by num_gpus_per_engine=2",
        ),
        (
            {
                "worker_type": "regular",
                "num_gpus": 6,
                "num_gpus_per_engine": 3,
                "overrides": {"pp-size": 2},
            },
            "num_gpus_per_engine=3, which is not divisible by pp_size=2",
        ),
        (
            {"worker_type": "regular", "num_gpus": 20, "num_gpus_per_engine": 10},
            "num_gpus_per_engine=10, which is not divisible by num_gpus_per_node=8",
        ),
    ],
)
def test_start_rollout_servers_validates_all_models_before_starting_router(
    tmp_path,
    monkeypatch,
    ref_group,
    error,
):
    deployment = _load_deployment(monkeypatch)
    path = _write_config(
        tmp_path,
        [
            {
                "name": "actor",
                "update_weights": True,
                "server_groups": [{"worker_type": "regular", "num_gpus": 2}],
            },
            {
                "name": "ref",
                "update_weights": False,
                "server_groups": [ref_group],
            },
        ],
    )
    router_calls = []

    def record_router_start(*_args, **_kwargs):
        router_calls.append(True)
        return "127.0.0.1", 3000

    monkeypatch.setattr(deployment, "_start_router", record_router_start)
    args = Namespace(
        rollout_external=False,
        sglang_config=path,
        rollout_num_gpus=2 + ref_group["num_gpus"],
        rollout_num_gpus_per_engine=2,
        hf_checkpoint="/models/actor",
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        num_gpus_per_node=8,
        sglang_pp_size=1,
    )

    try:
        with pytest.raises(ValueError, match=error):
            deployment.start_rollout_servers(args, pg=None)
    finally:
        assert router_calls == []
