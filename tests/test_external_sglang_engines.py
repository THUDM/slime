import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.sglang_utils.external import apply_external_engine_info_to_args, discover_external_engines
from slime.utils.http_utils import get_rollout_num_engines

NUM_GPUS = 0


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_discover_external_engines_reads_server_info(monkeypatch):
    def fake_get(url, timeout):
        assert timeout == 30.0
        assert url == "http://host1:10090/server_info"
        return _Response(
            {
                "tp_size": 4,
                "pp_size": 2,
                "dp_size": 1,
                "ep_size": 4,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)

    infos = discover_external_engines(["host1:10090"])

    assert len(infos) == 1
    info = infos[0]
    assert info.url == "http://host1:10090"
    assert info.host == "host1"
    assert info.port == 10090
    assert info.worker_type == "regular"
    assert info.num_gpus == 8
    assert info.server_info["tp_size"] == 4
    assert info.server_info["pp_size"] == 2
    assert info.server_info["dp_size"] == 1
    assert info.server_info["ep_size"] == 4


def test_apply_external_engine_info_handles_pd(monkeypatch):
    payloads = {
        "http://prefill:10090/server_info": {
            "tp_size": 2,
            "pp_size": 1,
            "dp_size": 1,
            "ep_size": 1,
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 12090,
        },
        "http://decode:10091/server_info": {
            "tp_size": 4,
            "pp_size": 1,
            "dp_size": 2,
            "ep_size": 2,
            "disaggregation_mode": "decode",
        },
    }

    def fake_get(url, timeout):
        return _Response(payloads[url])

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["prefill:10090", "decode:10091"],
        rollout_num_gpus=None,
        rollout_num_gpus_per_engine=1,
        sglang_pipeline_parallel_size=1,
        sglang_data_parallel_size=1,
        sglang_expert_parallel_size=1,
        sglang_enable_dp_attention=False,
        router_pd_disaggregation=False,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is False
    assert args.rollout_num_gpus == 6
    assert args.rollout_num_engines == 2
    assert get_rollout_num_engines(args) == 2
    assert [info["worker_type"] for info in args.rollout_external_engine_infos] == ["prefill", "decode"]
    assert [info["num_gpus"] for info in args.rollout_external_engine_infos] == [2, 4]
    assert [info["server_info"]["dp_size"] for info in args.rollout_external_engine_infos] == [1, 2]
    assert args.rollout_external_engine_infos[0]["disaggregation_bootstrap_port"] == 12090


def test_apply_external_engine_info_preserves_router_pd_flag(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://regular:10090/server_info"
        return _Response(
            {
                "tp_size": 2,
                "pp_size": 1,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["regular:10090"],
        router_pd_disaggregation=True,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is True
    assert args.rollout_num_gpus == 2
    assert args.rollout_num_engines == 1


def test_apply_external_engine_info_requires_addrs():
    args = Namespace(rollout_external_engine_addrs=None)

    with pytest.raises(ValueError, match="rollout-external-engine-addrs"):
        apply_external_engine_info_to_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
