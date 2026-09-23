"""Sampler heads survive the real rollout-manager, DP and microbatch boundaries."""

import asyncio
import sys
import types
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from test_score_centering import args, meta

from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def no_gpu_server_imports(monkeypatch):
    deployment = types.ModuleType("slime.backends.sglang_utils.deployment")
    deployment.start_rollout_servers = lambda *args: None
    monkeypatch.setitem(sys.modules, deployment.__name__, deployment)
    if "sglang_router" not in sys.modules:
        monkeypatch.setitem(sys.modules, "sglang_router", SimpleNamespace(__version__="0.3.0"))


def manager(**overrides):
    from slime.ray.rollout import RolloutManager

    cls = RolloutManager.__ray_metadata__.modified_class
    result = cls.__new__(cls)
    result.args = args(**overrides)
    result.custom_convert_samples_to_train_data_func = None
    result._post_process_rewards = lambda samples: ([1.0] * len(samples), [1.0] * len(samples))
    return result


def samples():
    result = []
    for i in range(2):
        sample = Sample(index=i, tokens=[9])
        sample.append_response_tokens(args(), tokens=[3], log_probs=[-0.5], meta_info=meta())
        result.append(sample)
    return result


def test_topk_training_transport_and_microbatch_order(monkeypatch):
    packed = types.ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = object
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: None
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    from slime.backends.megatron_utils.data import DataIterator

    data = samples()
    data[1].rollout_topk_token_ids[0] = [5, 6, 7]
    batch = manager()._convert_samples_to_train_data(data)
    tensorize_rollout_data_for_training(batch)
    assert batch["rollout_topk_token_ids"][0].dtype == torch.int32
    assert batch["rollout_topk_log_probs"][0].dtype == torch.float32
    iterator = DataIterator(batch, micro_batch_indices=[[1], [0]])
    keys = ["rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"]
    assert iterator.get_next(keys)["rollout_topk_token_ids"][0].tolist() == [[5, 6, 7]]
    assert iterator.get_next(keys)["rollout_topk_token_ids"][0].tolist() == [[3, 1, 4]]


@pytest.mark.parametrize("field", ["rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"])
def test_missing_sampler_metadata_rejected_by_manager(field):
    data = samples()
    setattr(data[1], field, None)
    with pytest.raises(ValueError, match="Score centering"):
        manager()._convert_samples_to_train_data(data)


def test_generate_requests_sampler_topk(monkeypatch):
    from slime.rollout import sglang_rollout as rollout

    a = args(sglang_router_ip="localhost", sglang_router_port=1234, use_rollout_routing_replay=False, ci_test=False)
    monkeypatch.setattr(rollout, "GenerateState", lambda _: SimpleNamespace(tokenizer=None, processor=None))
    monkeypatch.setattr(rollout, "_prepare_prompt_ids", lambda *_: [9])
    captured = []

    async def post(url, payload, **kwargs):
        captured.append((payload, kwargs))
        info = meta()
        info["output_token_logprobs"] = [[-0.5, 3, None]]
        info["finish_reason"] = {"type": "stop"}
        return {"text": "x", "meta_info": info}

    monkeypatch.setattr(rollout, "post", post)
    sample = asyncio.run(rollout.generate(a, Sample(prompt="test"), {"max_new_tokens": 8}))
    payload, kwargs = captured[0]
    assert payload["top_logprobs_num"] == 3
    assert payload["return_logprob"] is True
    assert kwargs["score_centering_top_k"] == 3
    assert sample.rollout_topk_token_ids.tolist() == [[3, 1, 4]]


def test_streaming_score_centering_rejected():
    from slime.rollout.sglang_streaming_rollout import generate_streaming

    with pytest.raises(ValueError, match="streaming"):
        asyncio.run(generate_streaming(args(), Sample(), {}))


@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_dp_transport_keeps_heads_aligned(monkeypatch, transport):
    from slime.ray import rollout

    mgr = manager(rollout_data_transport=transport, global_batch_size=2)
    mgr.train_parallel_config = {"dp_size": 2}
    monkeypatch.setattr(rollout, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    captured = []

    def put(data, **kwargs):
        captured.append(kwargs)
        return data

    monkeypatch.setattr(rollout.ray, "put", put)
    data = samples()
    data[1].rollout_topk_token_ids[0] = [5, 6, 7]
    refs = mgr._split_train_data_by_dp(mgr._convert_samples_to_train_data(data))
    assert refs[0].inner["rollout_topk_token_ids"][0].tolist() == [[5, 6, 7]]
    assert refs[1].inner["rollout_topk_token_ids"][0].tolist() == [[3, 1, 4]]
    assert captured == ([{"_tensor_transport": "nixl"}] * 2 if transport == "nixl" else [{}, {}])


def test_evaluation_preserves_training_score_centering(monkeypatch):
    from contextlib import nullcontext
    from slime.rollout import sglang_rollout as rollout

    a = args(partial_rollout=False, group_rm=True, custom_generate_function_path=None)
    state = SimpleNamespace(
        semaphore=asyncio.Semaphore(1),
        aborted=False,
        active_server_generations=0,
        dp_rank_context=lambda: nullcontext(),
    )
    flags = []
    state_args = []

    def get_state(received):
        state_args.append(received)
        return state

    async def generate(received, sample, params):
        flags.append(received.use_score_centering)
        return sample

    async def hooks(received, sample, **kwargs):
        return sample

    monkeypatch.setattr(rollout, "GenerateState", get_state)
    monkeypatch.setattr(rollout, "generate", generate)
    monkeypatch.setattr(rollout, "apply_rollout_sample_hooks", hooks)

    async def run():
        await rollout.generate_and_rm(a, Sample(), {"temperature": 0}, evaluation=True)
        await rollout.generate_and_rm(a, Sample(), {"temperature": 0.8})

    asyncio.run(run())
    assert flags == [False, True]
    assert all(value is a for value in state_args)
    assert a.use_score_centering


def test_spilled_heads_survive_buffer_and_debug_dump_lifetimes(tmp_path):
    from slime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data
    from slime.utils.routed_experts import cleanup_routed_experts_rollout, link_routed_experts_for_rollout
    from slime.utils.score_centering import spill_sampler_topk, validate_sampler_topk
    from slime.utils.tensor_store import DiskTensorRef

    a = args(rollout_routed_experts_store_dir=str(tmp_path))
    sample = samples()[0]
    spill_sampler_topk(a, sample, 1)
    assert isinstance(sample.rollout_topk_token_ids, DiskTensorRef)
    link_routed_experts_for_rollout(a, sample, 2)
    path = str(tmp_path / "debug.pt")
    save_debug_rollout_data(path, [sample], rollout_id=2, evaluation=False)
    cleanup_routed_experts_rollout(a, 1)
    validate_sampler_topk(sample, 3)
    cleanup_routed_experts_rollout(a, 2)
    restored = load_debug_rollout_data(path, rollout_id=2)[0]
    validate_sampler_topk(restored, 3)
    restored.append_response_tokens(a, tokens=[3], log_probs=[-0.5], meta_info=meta())
    assert restored.rollout_topk_token_ids.tolist() == [[3, 1, 4]] * 2


@pytest.mark.parametrize("disk", [False, True])
def test_training_metrics_ignore_sampler_head_payloads(monkeypatch, tmp_path, disk):
    from megatron.core import mpu
    from slime.observability import train_metric_utils as metrics
    from slime.utils.tensor_store import DiskTensorRef

    for name, value in {
        "get_tensor_model_parallel_rank": 0,
        "is_pipeline_last_stage": True,
        "get_context_parallel_world_size": 1,
        "get_data_parallel_world_size": 1,
    }.items():
        monkeypatch.setattr(mpu, name, lambda *a, _value=value, **kw: _value, raising=False)
    reported = []
    monkeypatch.setattr(metrics, "gather_log_data", lambda name, args, rollout_id, data: reported.append(data))
    batch = manager()._convert_samples_to_train_data(samples())
    tensorize_rollout_data_for_training(batch)
    batch.update(total_lengths=[2, 2], global_batch_sizes=[2])
    if disk:
        for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
            batch[key] = [DiskTensorRef.write(x, tmp_path / f"{key}_{i}") for i, x in enumerate(batch[key])]
    metrics.log_rollout_data(
        0, args(ci_test=False, log_multi_turn=False, log_passrate=False, log_correct_samples=False), batch
    )
    assert "rollout_topk_token_ids" not in reported[0]
    assert "rollout_topk_log_probs" not in reported[0]
    assert "rollout_log_probs" in reported[0]


@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_exact_top_p_transport_and_microbatch(monkeypatch, transport):
    import numpy as np
    from test_score_centering import binary_top_p_meta
    from slime.ray import rollout

    packed = types.ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = object
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: None
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    from slime.backends.megatron_utils.data import DataIterator

    mgr = manager(rollout_top_p=0.9, rollout_data_transport=transport, global_batch_size=2)
    mgr.train_parallel_config = {"dp_size": 2}
    monkeypatch.setattr(rollout, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    monkeypatch.setattr(rollout.ray, "put", lambda data, **kwargs: data)
    samples = []
    for i in range(2):
        sample = Sample(index=i, tokens=[9])
        sample.append_response_tokens(
            mgr.args, tokens=[4, 2], log_probs=[float(np.log(0.7)), 0.0], meta_info=binary_top_p_meta()
        )
        if i == 1:
            sample.append_response_tokens(mgr.args, tokens=[8], trainable=False)
        samples.append(sample)
    batch = mgr._convert_samples_to_train_data(samples)
    refs = mgr._split_train_data_by_dp(batch)
    assert refs[0].inner["rollout_top_p_token_offsets"][0].tolist() == [0, 2, 3, 3]
    for ref in refs:
        tensorize_rollout_data_for_training(ref.inner)
        iterator = DataIterator(ref.inner, micro_batch_indices=[[0]])
        data = iterator.get_next(["rollout_top_p_log_probs", "rollout_top_p_token_ids", "rollout_top_p_token_offsets"])
        assert data["rollout_top_p_log_probs"][0].dtype == torch.float32
        torch.testing.assert_close(data["rollout_top_p_log_probs"][0].exp(), torch.tensor([0.3, 0.7, 1.0]))
        assert data["rollout_top_p_token_ids"][0].tolist() == [1, 4, 2]


def test_generate_requests_complete_top_p_probabilities(monkeypatch):
    import numpy as np
    from test_score_centering import binary_top_p_meta
    from slime.rollout import sglang_rollout as rollout

    a = args(
        rollout_top_p=0.9,
        sglang_router_ip="localhost",
        sglang_router_port=1234,
        use_rollout_routing_replay=False,
        ci_test=False,
    )
    monkeypatch.setattr(rollout, "GenerateState", lambda _: SimpleNamespace(tokenizer=None, processor=None))
    monkeypatch.setattr(rollout, "_prepare_prompt_ids", lambda *_: [9])

    async def post(url, payload, **kwargs):
        assert payload["sampling_params"]["custom_params"]["return_top_p_log_probs"]
        assert "top_logprobs_num" not in payload
        assert kwargs["score_centering_top_k"] == 0
        info = binary_top_p_meta()
        info["finish_reason"] = {"type": "stop"}
        return {"text": "x", "meta_info": info}

    monkeypatch.setattr(rollout, "post", post)
    sample = asyncio.run(rollout.generate(a, Sample(prompt="test"), {"max_new_tokens": 8}))
    np.testing.assert_allclose(np.exp(sample.rollout_top_p_log_probs), [0.3, 0.7, 1.0], rtol=1e-6)
    assert sample.rollout_top_p_token_ids.tolist() == [1, 4, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
