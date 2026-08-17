from __future__ import annotations

import asyncio
import os
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent / "plugin_contracts"))

from _shared import install_paths, install_stubs

install_paths()
install_stubs(with_sglang_router=True, with_transformers=True)

from slime.rollout import sglang_rollout
from slime.utils.types import Sample

NUM_GPUS = 0


def _samples(*session_ids: str | None) -> list[Sample]:
    return [Sample(index=index, session_id=session_id) for index, session_id in enumerate(session_ids)]


@pytest.mark.unit
def test_sample_scope_assigns_distinct_ids_to_missing_samples():
    samples = _samples(None, None, None)

    sglang_rollout._assign_missing_session_ids(samples, "sample")

    assert len({sample.session_id for sample in samples}) == 3
    assert all(sample.session_id for sample in samples)


@pytest.mark.unit
def test_sample_scope_preserves_explicit_session_id():
    samples = _samples("caller-id", None)

    sglang_rollout._assign_missing_session_ids(samples, "sample")

    assert samples[0].session_id == "caller-id"
    assert samples[1].session_id and samples[1].session_id != "caller-id"


@pytest.mark.unit
def test_group_scope_assigns_one_id_to_all_missing_samples():
    samples = _samples(None, None, None)

    sglang_rollout._assign_missing_session_ids(samples, "group")

    assert len({sample.session_id for sample in samples}) == 1
    assert samples[0].session_id


@pytest.mark.unit
def test_group_scope_inherits_one_explicit_session_id():
    samples = _samples(None, "caller-id", None)

    sglang_rollout._assign_missing_session_ids(samples, "group")

    assert [sample.session_id for sample in samples] == ["caller-id", "caller-id", "caller-id"]


@pytest.mark.unit
def test_group_scope_rejects_conflicting_explicit_session_ids():
    with pytest.raises(ValueError, match="at most one explicit non-empty session_id"):
        sglang_rollout._assign_missing_session_ids(_samples("first", "second"), "group")


@pytest.mark.unit
def test_session_id_scope_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported rollout session ID scope"):
        sglang_rollout._assign_missing_session_ids(_samples(None), "rollout")


@pytest.mark.unit
def test_session_id_survives_serialization_and_retry():
    samples = _samples(None, None)

    sglang_rollout._assign_missing_session_ids(samples, "group")
    restored = [Sample.from_dict(sample.to_dict()) for sample in samples]
    before_retry = [sample.session_id for sample in restored]
    sglang_rollout._assign_missing_session_ids(restored, "group")

    assert [sample.session_id for sample in restored] == before_retry


@pytest.mark.unit
def test_group_scope_preserves_sample_order():
    samples = _samples(None, None, None)
    order = [id(sample) for sample in samples]

    sglang_rollout._assign_missing_session_ids(samples, "group")

    assert [id(sample) for sample in samples] == order


@pytest.mark.unit
def test_group_scope_flows_through_group_generate_to_http_with_one_routing_key(monkeypatch):
    captured = []

    async def fake_post(url, payload, headers=None):
        captured.append(headers)
        return {"meta_info": {"finish_reason": {"type": "stop"}}, "text": "response"}

    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    async def fake_rm(args, sample):
        return 0.0

    monkeypatch.setattr(sglang_rollout, "async_rm", fake_rm)
    args = Namespace(
        ci_test=False,
        rollout_session_id_scope="group",
        sglang_enable_deterministic_inference=False,
        group_rm=False,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        custom_generate_function_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
        use_rollout_routing_replay=False,
        sglang_speculative_algorithm=False,
    )
    group = _samples(None, None, None)
    for sample in group:
        sample.tokens = [11, 12]
    sampling_params = {"max_new_tokens": 4, "temperature": 0.0}

    result = asyncio.run(sglang_rollout.generate_and_rm_group(args, group, sampling_params))

    assert result == group
    assert len(captured) == 3
    assert {headers["X-SMG-Routing-Key"] for headers in captured} == {group[0].session_id}
    assert group[0].session_id and all(sample.session_id == group[0].session_id for sample in group)


@pytest.mark.unit
@pytest.mark.parametrize("scope", ["sample", "group"])
def test_empty_group_is_a_noop(scope):
    samples: list[Sample] = []

    sglang_rollout._assign_missing_session_ids(samples, scope)

    assert samples == []


@pytest.mark.unit
def test_default_scope_preserves_per_sample_sampling_seeds(monkeypatch):
    captured_sampling_params = []

    class FakeGroupGenerateState:
        def __init__(self, args) -> None:
            self.aborted = False
            self.group_sampling_seeds = [101, 102]

    async def fake_generate_and_rm(args, sample, sampling_params, evaluation=False):
        captured_sampling_params.append(sampling_params)
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGroupGenerateState)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
    args = Namespace(
        rollout_session_id_scope="sample",
        sglang_enable_deterministic_inference=True,
        group_rm=False,
    )
    group = _samples(None, None)
    sampling_params = {"temperature": 0.5}

    result = asyncio.run(sglang_rollout.generate_and_rm_group(args, group, sampling_params))

    assert result == group
    assert len({sample.session_id for sample in group}) == 2
    assert captured_sampling_params == [
        {"temperature": 0.5, "sampling_seed": 101},
        {"temperature": 0.5, "sampling_seed": 102},
    ]
    assert sampling_params == {"temperature": 0.5}


class _FakeGenerateState:
    def __init__(self, args) -> None:
        self.aborted = False
        self.semaphore = asyncio.Semaphore(3)
        self.tokenizer = None
        self.processor = None

    @contextmanager
    def dp_rank_context(self):
        yield 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "router_policy, expected_headers",
    [
        ("consistent_hashing", {"X-SMG-Routing-Key": "caller-id"}),
        ("round_robin", None),
        ("cache_aware", None),
    ],
)
def test_generate_sends_routing_header_only_for_consistent_hashing(monkeypatch, router_policy, expected_headers):
    captured = {}

    async def fake_post(url, payload, headers=None):
        captured.update(url=url, payload=payload, headers=headers)
        return {"meta_info": {"finish_reason": {"type": "stop"}}, "text": "response"}

    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    args = Namespace(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy=router_policy,
        use_rollout_routing_replay=False,
        sglang_speculative_algorithm=False,
    )
    sample = Sample(prompt="prompt", tokens=[11, 12], session_id="caller-id")
    sampling_params = {"max_new_tokens": 4, "temperature": 0.5}

    asyncio.run(sglang_rollout.generate(args, sample, sampling_params))

    assert captured["url"] == "http://127.0.0.1:30000/generate"
    assert captured["headers"] == expected_headers
    assert captured["payload"] == {"sampling_params": sampling_params, "return_logprob": True, "input_ids": [11, 12]}
    assert sample.tokens == [11, 12]
    assert sampling_params == {"max_new_tokens": 4, "temperature": 0.5}
