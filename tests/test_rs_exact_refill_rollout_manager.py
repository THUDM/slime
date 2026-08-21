import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _snapshot_modules(prefixes):
    return {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }


def _restore_modules(prefixes, snapshot):
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


# The routine CPU job intentionally omits the SGLang CUDA runtime. These are
# import-only serving edges; every RolloutManager method under test stays real.
_STUB_PREFIXES = ("sglang", "slime.backends.sglang_utils", "slime.ray.rollout")
_STUB_SNAPSHOT = None
if "sglang.srt.constants" not in sys.modules and not _module_available("sglang"):
    _STUB_SNAPSHOT = _snapshot_modules(_STUB_PREFIXES)
    _sglang = _stub_module("sglang")
    _sglang_srt = _stub_module("sglang.srt")
    _sglang_constants = _stub_module(
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
        GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    )
    _sglang.srt = _sglang_srt
    _sglang_srt.constants = _sglang_constants
    _stub_module(
        "slime.backends.sglang_utils.sglang_engine",
        SGLangEngine=type("SGLangEngine", (), {}),
    )

try:
    import slime.ray.rollout as rollout_module
    from slime.ray.rollout import RolloutManager
    from slime.utils.rs_refill import run_rs_batch_refill
    from slime.utils.types import Sample
finally:
    if _STUB_SNAPSHOT is not None:
        _restore_modules(_STUB_PREFIXES, _STUB_SNAPSHOT)
        for _parent_name, _attribute, _module_name in (
            ("slime.backends", "sglang_utils", "slime.backends.sglang_utils"),
            ("slime.ray", "rollout", "slime.ray.rollout"),
        ):
            _parent = sys.modules.get(_parent_name)
            if _parent is None:
                continue
            if _module_name in _STUB_SNAPSHOT:
                setattr(_parent, _attribute, _STUB_SNAPSHOT[_module_name])
            else:
                _parent.__dict__.pop(_attribute, None)

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, name, function, events):
        self.name = name
        self.function = function
        self.events = events

    def remote(self, *args):
        self.events.append((self.name, args))
        return self.function(*args)


class _LifecycleActor:
    def __init__(self, reports, cache_refs):
        self.reports = reports
        self.cache_refs = cache_refs
        self.events = []
        self.cache_live = True

    def async_score_rs_candidates(self, rollout_id, candidates):
        self.events.append(("score", rollout_id, candidates))
        if self.reports and isinstance(self.reports[0], list):
            return self.reports
        return [self.reports]

    def async_take_rs_candidate_log_probs(self, rollout_id, selected):
        self.events.append(("take", rollout_id, list(selected)))
        if isinstance(self.cache_refs, list):
            self.cache_live = False
        return self.cache_refs

    def async_discard_rs_candidate_log_probs(self, rollout_id):
        self.events.append(("discard", rollout_id))
        self.cache_live = False
        return True


class _RoundLifecycleActor:
    def __init__(self, report_rounds, cache_rounds):
        self.report_rounds = list(report_rounds)
        self.cache_rounds = list(cache_rounds)
        self.events = []
        self.current_cache = None

    def async_score_rs_candidates(self, rollout_id, candidates):
        self.events.append(("score", rollout_id, candidates))
        self.current_cache = self.cache_rounds.pop(0)
        reports = self.report_rounds.pop(0)
        if reports and isinstance(reports[0], list):
            return reports
        return [reports]

    def async_take_rs_candidate_log_probs(self, rollout_id, selected):
        self.events.append(("take", rollout_id, list(selected)))
        cache = self.current_cache
        self.current_cache = None
        return [{index: cache[index] for index in selected}]

    def async_discard_rs_candidate_log_probs(self, rollout_id):
        self.events.append(("discard", rollout_id))
        self.current_cache = None
        return True


def _sample(index, group_index, *, weight_version="7", loss_mask=None):
    if loss_mask is None:
        loss_mask = [1, 1]
    elif not isinstance(loss_mask, torch.Tensor):
        loss_mask = list(loss_mask)
    return Sample(
        index=index,
        group_index=group_index,
        rollout_id=index,
        tokens=[100 + index, 200 + index, 300 + index],
        response_length=len(loss_mask),
        loss_mask=loss_mask,
        weight_versions=[weight_version],
        rollout_log_probs=[-0.1] * len(loss_mask),
    )


def _reports(groups, *, policy_version="8"):
    return [
        {
            "sample_index": sample.index,
            "group_index": sample.group_index,
            "valid_tokens": sum(sample.loss_mask),
            "gate_passed": True,
            "policy_version": policy_version,
            "candidate_cache_bytes": sample.response_length * 4,
        }
        for group in groups
        for sample in group
    ]


def _real_lifecycle_manager(groups, *, cache_limit=1 << 20):
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(
        rollout_batch_size=len(groups),
        n_samples_per_prompt=len(groups[0]),
        rs_refill_max_rounds=2,
        rs_refill_rpc_timeout_seconds=123.0,
        rs_refill_max_candidate_cache_bytes=cache_limit,
        save_debug_rollout_data=None,
    )
    manager.train_parallel_config = {
        "dp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    manager._pending_rs_batches = {
        7: {
            "accepted": [],
            "unscored": groups,
            "initial_candidate_count": len(groups),
            "round": 0,
            "seen_sample_indices": set(),
            "seen_group_indices": set(),
            "seen_rollout_ids": {sample.rollout_id for group in groups for sample in group},
            "awaiting_log_prob_indices": None,
            "awaiting_log_prob_bytes": None,
            "proximal_log_probs_by_sample_index": {},
            "retained_logprob_cache_bytes": 0,
            "accepted_mask_fingerprints": {},
            "metrics": {"rollout/source_groups": len(groups)},
            "initial_generation_seconds": 0.5,
        }
    }
    conversions = []
    splits = []
    debug_saves = []

    def convert(samples, preflight=False):
        conversions.append((preflight, [sample.index for sample in samples]))
        return {
            "rollout_ids": [sample.rollout_id for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

    def split(data, global_batch_size=None):
        splits.append(global_batch_size)
        return {**data, "split_global_batch_size": global_batch_size}

    manager._convert_samples_to_train_data = convert
    manager._split_train_data_by_dp = split
    return manager, conversions, splits, debug_saves


def _manager_rpc(manager, events, snapshots):
    def finalize(rollout_id, coordinator_seconds):
        pending = manager._pending_rs_batches[rollout_id]
        snapshots.append(
            (
                "finalize",
                pending["awaiting_log_prob_indices"],
                sorted(pending["proximal_log_probs_by_sample_index"]),
            )
        )
        return manager.finalize_rs_batch(rollout_id, coordinator_seconds)

    def abort(rollout_id):
        pending = manager._pending_rs_batches[rollout_id]
        awaiting = pending["awaiting_log_prob_indices"]
        snapshots.append(
            (
                "abort",
                None if awaiting is None else list(awaiting),
                sorted(pending["proximal_log_probs_by_sample_index"]),
            )
        )
        return manager.abort_rs_batch(rollout_id)

    return SimpleNamespace(
        prepare_rs_candidate_data=_RemoteMethod("prepare", manager.prepare_rs_candidate_data, events),
        apply_rs_candidate_reports=_RemoteMethod("apply", manager.apply_rs_candidate_reports, events),
        store_rs_accepted_log_probs=_RemoteMethod("store", manager.store_rs_accepted_log_probs, events),
        generate_rs_replacement_candidates=_RemoteMethod(
            "generate", manager.generate_rs_replacement_candidates, events
        ),
        finalize_rs_batch=_RemoteMethod("finalize", finalize, events),
        abort_rs_batch=_RemoteMethod("abort", abort, events),
    )


def test_rollout_manager_generates_exact_initial_and_aligned_replacement_counts():
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(
        ci_test=False,
        use_fault_tolerance=False,
        rollout_batch_size=8,
        n_samples_per_prompt=2,
        rs_refill_max_rounds=2,
    )
    manager.train_parallel_config = {
        "dp_size": 2,
        "vpp_size": 2,
        "microbatch_group_size_per_vp_stage": 4,
    }
    manager._pending_rs_batches = {}
    manager.health_monitoring_resume = lambda: None
    requested_counts = []

    def generate(_rollout_id, group_count, *, known_rollout_ids=None):
        requested_counts.append((group_count, known_rollout_ids))
        return [[SimpleNamespace(), SimpleNamespace()] for _ in range(group_count)], {}, set()

    manager._call_rollout_for_group_count = generate

    manager._generate_rs_candidates(7)
    assert requested_counts == [(8, None)]

    manager._pending_rs_batches[7] = {
        "accepted": [[SimpleNamespace()] for _ in range(4)],
        "unscored": [],
        "round": 0,
        "awaiting_log_prob_indices": None,
        "awaiting_log_prob_bytes": None,
        "seen_rollout_ids": {"initial"},
        "metrics": {},
    }
    manager.generate_rs_replacement_candidates(7)

    assert requested_counts == [(8, None), (4, {"initial"})]
    assert manager._pending_rs_batches[7]["round"] == 1


def test_rollout_manager_rejects_an_unaligned_final_effective_batch():
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    manager.args = SimpleNamespace(rs_batch_refill=True, rollout_batch_size=5, n_samples_per_prompt=2)
    topology = {"dp_size": 2, "vpp_size": 2, "microbatch_group_size_per_vp_stage": 4}

    with pytest.raises(ValueError, match=r"effective sample count.*10, alignment = 8"):
        manager.set_train_parallel_config(topology)

    assert not hasattr(manager, "train_parallel_config")


def test_rollout_manager_real_lifecycle_applies_stores_and_finalizes(monkeypatch):
    groups = [
        [_sample(0, 0), _sample(1, 0)],
        [_sample(10, 1, loss_mask=[1, 0]), _sample(11, 1, loss_mask=[1, 0])],
    ]
    manager, conversions, splits, debug_saves = _real_lifecycle_manager(groups, cache_limit=32)
    cache = {
        sample.index: torch.tensor([sample.index + 0.25, sample.index + 0.5]) for group in groups for sample in group
    }
    actor = _LifecycleActor(
        _reports(groups),
        [{0: cache[0], 10: cache[10]}, {1: cache[1], 11: cache[11]}],
    )
    rpc_events = []
    state_snapshots = []
    rollout_logs = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)

    manager_timeouts = []

    def manager_get(value, *, timeout=None):
        manager_timeouts.append(timeout)
        return value

    monkeypatch.setattr(rollout_module.ray, "get", manager_get)
    monkeypatch.setattr(rollout_module.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(
        rollout_module,
        "save_debug_rollout_data",
        lambda path_template, samples, *, rollout_id, evaluation: debug_saves.append(
            (path_template, rollout_id, evaluation, [sample.index for sample in samples])
        ),
    )
    monkeypatch.setattr(
        rollout_module,
        "log_rollout_data",
        lambda rollout_id, args, samples, metrics, elapsed: rollout_logs.append(
            (rollout_id, [sample.index for sample in samples], dict(metrics), elapsed)
        ),
    )

    result = run_rs_batch_refill(
        actor,
        manager_rpc,
        7,
        resolve=lambda value, **_kwargs: value,
        clock=iter([0.0, 1.0, 2.0, 3.0]).__next__,
    )

    assert [name for name, _ in rpc_events] == ["prepare", "apply", "store", "finalize"]
    assert actor.events[0] == (
        "score",
        7,
        {
            "rollout_ids": [0, 1, 10, 11],
            "sample_indices": [0, 1, 10, 11],
            "split_global_batch_size": 4,
        },
    )
    assert actor.events[1] == ("take", 7, [0, 1, 10, 11])
    assert state_snapshots == [("finalize", None, [0, 1, 10, 11])]
    assert conversions == [
        (True, [0, 1, 10, 11]),
        (False, [0, 1, 10, 11]),
    ]
    assert splits == [4, None]
    assert debug_saves == [(None, 7, False, [0, 1, 10, 11])]
    assert result["sample_indices"] == [0, 1, 10, 11]
    assert result["split_global_batch_size"] is None
    for actual, index in zip(result["rs_preflight_log_probs"], [0, 1, 10, 11], strict=True):
        torch.testing.assert_close(actual, cache[index])

    assert 7 not in manager._pending_rs_batches
    assert actor.cache_live is False
    assert [event[0] for event in actor.events] == ["score", "take"]
    assert len(rollout_logs) == 1
    rollout_id, sample_indices, metrics, elapsed = rollout_logs[0]
    assert rollout_id == 7
    assert sample_indices == [0, 1, 10, 11]
    assert elapsed == 3.5
    assert metrics["rollout/rs_refill/initial_policy_staleness"] == 1
    assert metrics["rollout/rs_refill/candidate_groups"] == 2
    assert metrics["rollout/rs_refill/scored_groups"] == 2
    assert metrics["rollout/rs_refill/rejected_groups"] == 0
    assert metrics["rollout/rs_refill/surplus_groups"] == 0
    assert metrics["rollout/rs_refill/rounds"] == 0
    assert metrics["rollout/rs_refill/accepted_groups"] == 2
    assert metrics["rollout/rs_refill/scored_trainable_tokens"] == 6
    assert metrics["rollout/rs_refill/aggregate_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_aggregate_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_actor_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/logprob_cache_limit_bytes"] == 32
    assert metrics["rollout/rs_refill/selected_logprob_transfer_bytes"] == 32
    assert metrics["rollout/rs_refill/retained_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_retained_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/effective_trainable_tokens"] == 6
    assert metrics["rollout/rs_refill/gate_acceptance_rate"] == 1
    assert metrics["rollout/rs_refill/selection_utilization"] == 1
    assert manager_timeouts == [123.0, 123.0]


def test_rollout_manager_real_lifecycle_refills_a_rejected_group_and_accepts_tensor_masks(monkeypatch):
    initial = [
        [_sample(0, 0), _sample(1, 0)],
        [_sample(10, 1), _sample(11, 1)],
    ]
    replacement = [
        [
            _sample(20, 2, weight_version="8", loss_mask=torch.tensor([1, 0])),
            _sample(21, 2, weight_version="8", loss_mask=[1, 0]),
        ]
    ]
    initial_reports = _reports(initial)
    for report in initial_reports:
        if report["group_index"] == 1:
            report["gate_passed"] = False
    cache_rounds = [
        {
            sample.index: torch.full((sample.response_length,), float(sample.index))
            for group in initial
            for sample in group
        },
        {
            sample.index: torch.full((sample.response_length,), float(sample.index))
            for group in replacement
            for sample in group
        },
    ]
    actor = _RoundLifecycleActor([initial_reports, _reports(replacement)], cache_rounds)
    manager, conversions, splits, debug_saves = _real_lifecycle_manager(initial)
    generated = []

    def generate(rollout_id, group_count, *, known_rollout_ids=None):
        generated.append((rollout_id, group_count, set(known_rollout_ids)))
        return replacement, {"rollout/replacement_marker": 1}, {20, 21}

    manager._call_rollout_for_group_count = generate
    rpc_events = []
    state_snapshots = []
    rollout_logs = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)
    manager_timeouts = []

    def manager_get(value, *, timeout=None):
        manager_timeouts.append(timeout)
        return value

    monkeypatch.setattr(rollout_module.ray, "get", manager_get)
    monkeypatch.setattr(rollout_module.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(
        rollout_module,
        "save_debug_rollout_data",
        lambda path_template, samples, *, rollout_id, evaluation: debug_saves.append(
            (path_template, rollout_id, evaluation, [sample.index for sample in samples])
        ),
    )
    monkeypatch.setattr(
        rollout_module,
        "log_rollout_data",
        lambda rollout_id, args, samples, metrics, elapsed: rollout_logs.append(
            (rollout_id, [sample.index for sample in samples], dict(metrics), elapsed)
        ),
    )

    result = run_rs_batch_refill(
        actor,
        manager_rpc,
        7,
        resolve=lambda value, **_kwargs: value,
        clock=iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).__next__,
    )

    assert [name for name, _ in rpc_events] == [
        "prepare",
        "apply",
        "store",
        "generate",
        "prepare",
        "apply",
        "store",
        "finalize",
    ]
    assert [event[0] for event in actor.events] == ["score", "take", "score", "take"]
    assert generated == [(7, 1, {0, 1, 10, 11})]
    assert result["sample_indices"] == [0, 1, 20, 21]
    assert [tensor.tolist() for tensor in result["rs_preflight_log_probs"]] == [
        [0.0, 0.0],
        [1.0, 1.0],
        [20.0, 20.0],
        [21.0, 21.0],
    ]
    assert conversions == [
        (True, [0, 1, 10, 11]),
        (True, [20, 21]),
        (False, [0, 1, 20, 21]),
    ]
    assert splits == [4, 2, None]
    assert debug_saves == [(None, 7, False, [0, 1, 20, 21])]
    assert state_snapshots == [("finalize", None, [0, 1, 20, 21])]
    assert manager_timeouts == [123.0, 123.0, 123.0, 123.0]
    metrics = rollout_logs[0][2]
    assert metrics["rollout/rs_refill/rounds"] == 1
    assert metrics["rollout/rs_refill/scored_groups"] == 3
    assert metrics["rollout/rs_refill/rejected_groups"] == 1
    assert metrics["rollout/rs_refill/generated_replacement_groups"] == 1
    assert metrics["rollout/rs_refill/effective_trainable_tokens"] == 6
    assert metrics["rollout/rs_refill/aggregate_candidate_logprob_cache_bytes"] == 48
    assert metrics["rollout/rs_refill/peak_aggregate_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_actor_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/selected_logprob_transfer_bytes"] == 32
    assert metrics["rollout/rs_refill/retained_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_retained_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/replacement_round_1/rollout/replacement_marker"] == 1
    assert 7 not in manager._pending_rs_batches


def test_rollout_manager_real_lifecycle_aborts_after_cache_rpc_failure(monkeypatch):
    groups = [
        [_sample(0, 0), _sample(1, 0)],
        [_sample(10, 1), _sample(11, 1)],
    ]
    manager, conversions, splits, debug_saves = _real_lifecycle_manager(groups)
    failed_cache_ref = object()
    actor = _LifecycleActor(_reports(groups), failed_cache_ref)
    rpc_events = []
    state_snapshots = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)

    def get(value, *, timeout=None):
        assert timeout == 123.0
        if value is failed_cache_ref:
            raise RuntimeError("cache RPC failed")
        return value

    monkeypatch.setattr(rollout_module.ray, "get", get)
    monkeypatch.setattr(rollout_module.time, "perf_counter", lambda: 100.0)

    with pytest.raises(RuntimeError, match="cache RPC failed"):
        run_rs_batch_refill(
            actor,
            manager_rpc,
            7,
            resolve=lambda value, **_kwargs: value,
            clock=iter([0.0, 1.0, 2.0]).__next__,
        )

    assert [name for name, _ in rpc_events] == ["prepare", "apply", "store", "abort"]
    assert actor.events == [
        (
            "score",
            7,
            {
                "rollout_ids": [0, 1, 10, 11],
                "sample_indices": [0, 1, 10, 11],
                "split_global_batch_size": 4,
            },
        ),
        ("take", 7, [0, 1, 10, 11]),
        ("discard", 7),
    ]
    assert state_snapshots == [("abort", [0, 1, 10, 11], [])]
    assert 7 not in manager._pending_rs_batches
    assert manager.abort_rs_batch(7) is False
    assert actor.cache_live is False
    assert conversions == [(True, [0, 1, 10, 11])]
    assert splits == [4]
    assert debug_saves == []


def test_rollout_manager_rejects_aggregate_selected_cache_before_actor_transfer(monkeypatch):
    groups = [
        [_sample(0, 0), _sample(1, 0)],
        [_sample(10, 1), _sample(11, 1)],
    ]
    manager, _, _, _ = _real_lifecycle_manager(groups, cache_limit=16)
    reports = _reports(groups)
    actor = _LifecycleActor([reports[:2], reports[2:]], [{}, {}])
    rpc_events = []
    state_snapshots = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, *, timeout=None: value)

    with pytest.raises(RuntimeError, match=r"before transfer.*required=32, limit=16"):
        run_rs_batch_refill(
            actor,
            manager_rpc,
            7,
            resolve=lambda value, **_kwargs: value,
            clock=iter([0.0, 1.0, 2.0]).__next__,
        )

    assert [name for name, _ in rpc_events] == ["prepare", "apply", "abort"]
    assert [event[0] for event in actor.events] == ["score", "discard"]
    assert state_snapshots == [("abort", None, [])]
    assert 7 not in manager._pending_rs_batches


def test_rollout_manager_rejects_cross_round_retained_cache_before_second_transfer(monkeypatch):
    initial = [
        [_sample(0, 0), _sample(1, 0)],
        [_sample(10, 1), _sample(11, 1)],
    ]
    replacement = [[_sample(20, 2, weight_version="8"), _sample(21, 2, weight_version="8")]]
    initial_reports = _reports(initial)
    for report in initial_reports:
        if report["group_index"] == 1:
            report["gate_passed"] = False
    cache_rounds = [
        {sample.index: torch.zeros(sample.response_length) for group in initial for sample in group},
        {sample.index: torch.zeros(sample.response_length) for group in replacement for sample in group},
    ]
    replacement_reports = _reports(replacement)
    actor = _RoundLifecycleActor(
        [[initial_reports[:2], initial_reports[2:]], [replacement_reports]],
        cache_rounds,
    )
    manager, _, _, _ = _real_lifecycle_manager(initial, cache_limit=24)
    manager._call_rollout_for_group_count = lambda *_args, **_kwargs: (replacement, {}, {20, 21})
    rpc_events = []
    state_snapshots = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, *, timeout=None: value)

    with pytest.raises(RuntimeError, match=r"before transfer.*retained=16, incoming=16.*limit=24"):
        run_rs_batch_refill(
            actor,
            manager_rpc,
            7,
            resolve=lambda value, **_kwargs: value,
            clock=iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).__next__,
        )

    assert [name for name, _ in rpc_events] == ["prepare", "apply", "store", "generate", "prepare", "apply", "abort"]
    assert [event[0] for event in actor.events] == ["score", "take", "score", "discard"]
    assert state_snapshots == [("abort", None, [0, 1])]
    assert 7 not in manager._pending_rs_batches


@pytest.mark.parametrize(
    "invalid_tensor",
    [
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros((2, 1), dtype=torch.float32),
        torch.zeros(4, dtype=torch.float32)[::2],
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32, device="meta"),
        [0.0, 0.0],
    ],
    ids=["float64", "int32", "two-dimensional", "non-contiguous", "wrong-length", "non-cpu", "non-tensor"],
)
def test_rollout_manager_rejects_invalid_transferred_cache_contract(monkeypatch, invalid_tensor):
    groups = [[_sample(0, 0), _sample(1, 0)]]
    manager, _, _, _ = _real_lifecycle_manager(groups)
    cache = {sample.index: invalid_tensor for sample in groups[0]}
    actor = _LifecycleActor(_reports(groups), [cache])
    rpc_events = []
    state_snapshots = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, *, timeout=None: value)

    with pytest.raises(RuntimeError, match=r"contiguous one-dimensional CPU float32 tensors"):
        run_rs_batch_refill(
            actor,
            manager_rpc,
            7,
            resolve=lambda value, **_kwargs: value,
            clock=iter([0.0, 1.0, 2.0, 3.0]).__next__,
        )

    assert [name for name, _ in rpc_events] == ["prepare", "apply", "store", "abort"]
    assert [event[0] for event in actor.events] == ["score", "take", "discard"]
    assert state_snapshots == [("abort", [0, 1], [])]
    assert 7 not in manager._pending_rs_batches


def test_rollout_manager_rejects_underreported_cache_before_actor_transfer(monkeypatch):
    groups = [[_sample(0, 0), _sample(1, 0)]]
    reports = _reports(groups)
    reports[0]["candidate_cache_bytes"] = 0
    manager, _, _, _ = _real_lifecycle_manager(groups)
    actor = _LifecycleActor(reports, [{sample.index: torch.zeros(2) for sample in groups[0]}])
    rpc_events = []
    state_snapshots = []
    manager_rpc = _manager_rpc(manager, rpc_events, state_snapshots)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, *, timeout=None: value)

    with pytest.raises(RuntimeError, match=r"reported=0, expected=8"):
        run_rs_batch_refill(
            actor,
            manager_rpc,
            7,
            resolve=lambda value, **_kwargs: value,
            clock=iter([0.0, 1.0, 2.0]).__next__,
        )

    assert [name for name, _ in rpc_events] == ["prepare", "apply", "abort"]
    assert [event[0] for event in actor.events] == ["score", "discard"]
    assert state_snapshots == [("abort", None, [])]
    assert 7 not in manager._pending_rs_batches


def test_rollout_manager_reports_distinct_aggregate_and_per_actor_cache_peaks(monkeypatch):
    groups = [
        [_sample(0, 0), _sample(1, 0)],
        [_sample(10, 1), _sample(11, 1)],
    ]
    reports = _reports(groups)
    manager, _, _, _ = _real_lifecycle_manager(groups, cache_limit=32)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, *, timeout=None: value)

    result = manager.apply_rs_candidate_reports(7, [reports[:2], reports[2:]], 0.5)

    assert result["complete"] is True
    metrics = manager._pending_rs_batches[7]["metrics"]
    assert metrics["rollout/rs_refill/aggregate_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_aggregate_candidate_logprob_cache_bytes"] == 32
    assert metrics["rollout/rs_refill/peak_actor_candidate_logprob_cache_bytes"] == 16


@pytest.mark.parametrize("invalid_bytes", [True, -1, 1.5, None])
def test_rollout_manager_rejects_invalid_candidate_cache_byte_reports(monkeypatch, invalid_bytes):
    groups = [[_sample(0, 0), _sample(1, 0)]]
    reports = _reports(groups)
    reports[0]["candidate_cache_bytes"] = invalid_bytes
    manager, _, _, _ = _real_lifecycle_manager(groups)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value, *, timeout=None: value)

    with pytest.raises(RuntimeError, match="non-negative integers"):
        manager.apply_rs_candidate_reports(7, [reports], 0.5)

    pending = manager._pending_rs_batches[7]
    assert pending["accepted"] == []
    assert pending["awaiting_log_prob_indices"] is None
    assert pending["awaiting_log_prob_bytes"] is None
    assert pending["proximal_log_probs_by_sample_index"] == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
