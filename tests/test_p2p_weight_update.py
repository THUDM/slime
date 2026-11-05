"""Ownership and failure boundaries of the P2P updater; no Megatron install needed."""

import importlib.util
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NUM_GPUS = 0


@pytest.fixture
def update(monkeypatch):
    # Mock only framework dependencies, and execute the production updater.
    # This keeps the ownership and transaction tests in the CPU CI matrix.
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    common = ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.all_gather_param = Mock()
    common.named_params_and_buffers = Mock()
    monkeypatch.setitem(sys.modules, common.__name__, common)
    conversion = ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    conversion.convert_to_hf = Mock()
    monkeypatch.setitem(sys.modules, conversion.__name__, conversion)
    name = "slime.backends.megatron_utils.update_weight._p2p_under_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "slime/backends/megatron_utils/update_weight/update_weight_from_p2p.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("expert_tp", [1, 2])
def test_each_dense_and_expert_tensor_has_one_source_with_complete_tp_participation(update, monkeypatch, expert_tp):
    dense = {
        "module.module.embedding.word_embeddings.weight": 400,
        "module.module.decoder.layers.0.self_attention.q_a.weight": 100,
        "module.module.decoder.layers.0.self_attention.kv_a.weight": 50,
        "module.module.decoder.layers.1.self_attention.q.weight": 100,
        "module.module.decoder.layers.2.mlp.linear_fc1.weight": 200,
    }
    emitted, gathers = defaultdict(list), defaultdict(list)
    location = [None]

    def gather(name, tensor):
        gathers[name].append(location[0])
        return tensor

    monkeypatch.setattr(update, "all_gather_param", gather)
    monkeypatch.setattr(update, "convert_to_hf", lambda args, model, name, tensor, quant: [(name, tensor)])
    for dp in range(2):
        for tp in range(2):
            location[0] = (dp, tp)
            expert = f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{tp if expert_tp == 1 else dp}"
            params = [(name, torch.empty(size)) for name, size in dense.items()] + [(expert, torch.empty(100))]
            monkeypatch.setattr(update, "named_params_and_buffers", lambda *args, params=params: iter(params))
            monkeypatch.setattr(
                update,
                "mpu",
                SimpleNamespace(
                    get_data_parallel_rank=lambda dp=dp, **kwargs: dp,
                    get_data_parallel_world_size=lambda **kwargs: 2,
                    get_expert_data_parallel_rank=lambda dp=dp: dp if expert_tp == 1 else 0,
                    get_expert_tensor_parallel_rank=lambda tp=tp: 0 if expert_tp == 1 else tp,
                    get_tensor_model_parallel_rank=lambda tp=tp: tp,
                ),
            )
            for unit in update.iter_owned_hf_tensors(None, None, None, None):
                for name, _ in unit:
                    emitted[name].append((dp, tp))
    assert set(emitted) == set(dense) | {
        f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{i}" for i in range(2)
    }
    assert Counter(len(senders) for senders in emitted.values()) == {1: len(emitted)}
    for name in dense:
        assert sorted(tp for _, tp in gathers[name]) == [0, 1]
        assert len({dp for dp, _ in gathers[name]}) == 1
    q, kv = (f"module.module.decoder.layers.0.self_attention.{projection}.weight" for projection in ("q_a", "kv_a"))
    assert emitted[q] == emitted[kv]
    assert {dp for name in dense for dp, _ in emitted[name]} == {0, 1}
    for name in set(emitted) - set(dense):
        assert len(gathers[name]) == expert_tp


def make_updater(update):
    args = SimpleNamespace(
        flush_cache_interval=1,
        update_weight_start_version=0,
        update_weight_p2p_timeout=1,
        rollout_num_gpus_per_engine=2,
    )
    return update.UpdateWeightFromP2P(args, [], lambda: {}, model_name="qwen3moe", quantization_config=None)


@pytest.mark.parametrize(
    "failure_at", [None, "pause_generation", "flush_cache", "load", "post_process_weights", "set_weight_version"]
)
def test_version_commit_and_resume_require_all_prior_phases_to_succeed(update, monkeypatch, failure_at):
    updater = make_updater(update)
    updater.quantization_config = {"quant_method": "compressed-tensors"}
    actions = []

    def action(name, **kwargs):
        actions.append(name)
        if name == failure_at:
            raise RuntimeError("injected failure")

    monkeypatch.setattr(updater, "_coordinator_call", action)
    monkeypatch.setattr(updater, "_send_weights", lambda version: action("load"))
    if failure_at:
        with pytest.raises(RuntimeError, match="injected failure"):
            updater.update_weights()
        assert updater.weight_version == 0
        assert "continue_generation" not in actions
        with pytest.raises(RuntimeError, match="restart"):
            updater.update_weights()
    else:
        updater.update_weights()
        assert actions == [
            "pause_generation",
            "flush_cache",
            "post_process_weights",
            "load",
            "post_process_weights",
            "set_weight_version",
            "continue_generation",
        ]
        assert updater.weight_version == 1


def test_engine_rejection_is_broadcast_to_training_peers(update, monkeypatch):
    updater = make_updater(update)
    updater.rollout_engines = [SimpleNamespace(set_weight_version=SimpleNamespace(remote=lambda **kwargs: "ref"))]
    broadcasts = []
    monkeypatch.setattr(update.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update.dist, "broadcast_object_list", lambda result, **kwargs: broadcasts.append(result[:]))
    monkeypatch.setattr(update, "get_gloo_group", lambda: None)
    monkeypatch.setattr(update.ray, "get", lambda *args, **kwargs: [{"success": False, "message": "rejected"}])
    with pytest.raises(RuntimeError, match="rejected"):
        updater._coordinator_call("set_weight_version", new_version="1")
    assert "rejected" in broadcasts[0][0]


@pytest.mark.parametrize("reject_decode_commit", [False, True])
def test_pd_neither_role_resumes_before_both_versions_are_committed(update, monkeypatch, reject_decode_commit):
    updater = make_updater(update)
    actions = []
    versions = {}

    def call(role, method, **kwargs):
        actions.append((role, method))
        if method == "set_weight_version":
            if role == "decode" and reject_decode_commit:
                return {"success": False, "message": "decode rejected version"}
            versions[role] = kwargs["new_version"]
        if method == "continue_generation":
            assert versions == {"prefill": "1", "decode": "1"}
        return {"success": True}

    def engine(role):
        return SimpleNamespace(
            **{
                method: SimpleNamespace(remote=lambda method=method, **kwargs: (role, method, kwargs))
                for method in ("pause_generation", "flush_cache", "set_weight_version", "continue_generation")
            }
        )

    updater.connect_rollout_engines([engine("prefill"), engine("decode")], None, engine_gpu_counts=[2, 2])
    monkeypatch.setattr(update.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(update, "get_gloo_group", lambda: None)
    monkeypatch.setattr(
        update.ray, "get", lambda refs, **kwargs: [call(role, method, **kw) for role, method, kw in refs]
    )

    def load(version):
        assert set(actions) == {
            (role, method) for role in ("prefill", "decode") for method in ("pause_generation", "flush_cache")
        }

    monkeypatch.setattr(updater, "_send_weights", load)
    if reject_decode_commit:
        with pytest.raises(RuntimeError, match="decode rejected version"):
            updater.update_weights()
        assert not any(method == "continue_generation" for _, method in actions)
    else:
        updater.update_weights()
        assert actions[-2:] == [("prefill", "continue_generation"), ("decode", "continue_generation")]


@pytest.mark.parametrize("config", [{"pp_size": 2}, {"tp_size": 1}])
def test_unsupported_serving_topology_fails_before_transfer(update, config):
    updater = make_updater(update)
    with pytest.raises(ValueError, match="PP=1"):
        updater.connect_rollout_engines([object()], None, engine_gpu_counts=[2], engine_parallel_configs=[config])


def test_abort_monitor_propagates_peer_failure_and_aborts_all_training_groups(update, monkeypatch):
    store = torch.distributed.HashStore()
    aborted = []
    monkeypatch.setattr(update._AbortWatch, "_abort_group", staticmethod(aborted.append))
    watch = update._AbortWatch(store, "update-1", ["dense-tp", "expert-tp"], [], 2)
    try:
        store.compare_set("update-1", "", "receiver failed")
        deadline = time.monotonic() + 2
        while len(aborted) != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert set(aborted) == {"dense-tp", "expert-tp"}
        with pytest.raises(RuntimeError, match="receiver failed"):
            watch.check()
    finally:
        watch.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
