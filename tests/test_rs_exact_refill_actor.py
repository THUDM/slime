"""CPU contracts for actor-local exact RS refill scoring and cache reuse."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import actor as actor_module
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.ray.actor_group import RayTrainGroup

NUM_GPUS = 0


def _make_args(**overrides):
    values = {
        "advantage_estimator": "grpo",
        "calculate_per_token_loss": False,
        "compute_advantages_and_returns": True,
        "custom_pg_loss_reducer_function_path": None,
        "custom_tis_function_path": None,
        "entropy_coef": 0.0,
        "eps_clip": 0.2,
        "eps_clip_c": None,
        "eps_clip_high": 0.2,
        "get_mismatch_metrics": True,
        "keep_old_actor": False,
        "kl_coef": 0.0,
        "loss_type": "policy_loss",
        "ref_update_interval": None,
        "rollout_top_p": 1.0,
        "rs_batch_refill": True,
        "rs_refill_max_candidate_cache_bytes": 1 << 20,
        "rs_level": "sequence",
        "rs_lower_bound": 0.5,
        "rs_upper_bound": 2.0,
        "rs_veto_threshold": None,
        "save_debug_train_data": None,
        "tis_lower_bound": None,
        "tis_upper_bound": 2.0,
        "use_critic": False,
        "use_kl_loss": False,
        "use_opd": False,
        "use_opsm": False,
        "use_rollout_logprobs": False,
        "use_rollout_routing_replay": False,
        "use_routing_replay": False,
        "use_tis": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _WeightsBackuper:
    def __init__(self):
        self.backup_tags = {"actor"}
        self.backups = []
        self.restores = []

    def backup(self, tag):
        self.backups.append(tag)

    def restore(self, tag):
        self.restores.append(tag)


def _make_actor(args=None):
    actor = object.__new__(MegatronTrainRayActor)
    actor.args = args or _make_args()
    actor._active_model_tag = "actor"
    actor._rs_candidate_log_probs = {}
    actor.weights_backuper = _WeightsBackuper()
    actor.weight_updater = SimpleNamespace(weight_version=9, pop_metrics=lambda: {})
    actor.prof = SimpleNamespace(step=lambda **_kwargs: None)
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor.rollout_data_postprocess = None
    return actor


def _patch_last_pipeline_rank(monkeypatch):
    monkeypatch.setattr(actor_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(actor_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)


def test_preflight_scores_and_transfers_only_selected_cache_once(monkeypatch):
    actor = _make_actor()
    train_log_probs = [
        torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True),
    ]
    rollout_data = {
        "group_indices": [3, 4],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "num_microbatches": [2],
        "rollout_log_probs": [torch.zeros(2), torch.zeros(2)],
        "sample_indices": [10, 11],
    }
    actor._get_rollout_data = lambda _ref: rollout_data
    actor.compute_log_prob = lambda *_args, **_kwargs: {"log_probs": train_log_probs}
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data)
    _patch_last_pipeline_rank(monkeypatch)

    reports = actor.score_rs_candidates(27, object())

    assert reports == [
        {
            "sample_index": 10,
            "group_index": 3,
            "valid_tokens": 2,
            "gate_passed": True,
            "policy_version": "9",
            "candidate_cache_bytes": 8,
        },
        {
            "sample_index": 11,
            "group_index": 4,
            "valid_tokens": 2,
            "gate_passed": False,
            "policy_version": "9",
            "candidate_cache_bytes": 8,
        },
    ]
    cached = actor._rs_candidate_log_probs[27]
    assert set(cached) == {10, 11}
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in cached.values())
    assert all(not value.requires_grad for value in cached.values())

    selected = actor.take_rs_candidate_log_probs(27, [10, 99])

    assert set(selected) == {10}
    torch.testing.assert_close(selected[10], torch.zeros(2))
    assert 27 not in actor._rs_candidate_log_probs
    with pytest.raises(RuntimeError, match="No RS candidate logprob cache"):
        actor.take_rs_candidate_log_probs(27, [10])


def test_preflight_metadata_mismatch_does_not_publish_cache(monkeypatch):
    actor = _make_actor()
    rollout_data = {
        "group_indices": [],
        "loss_masks": [torch.ones(1)],
        "num_microbatches": [1],
        "rollout_log_probs": [torch.zeros(1)],
        "sample_indices": [5],
    }
    actor._get_rollout_data = lambda _ref: rollout_data
    actor.compute_log_prob = lambda *_args, **_kwargs: {"log_probs": [torch.zeros(1)]}
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data)
    _patch_last_pipeline_rank(monkeypatch)

    with pytest.raises(RuntimeError, match="logprob/group count mismatch"):
        actor.score_rs_candidates(12, object())

    assert 12 not in actor._rs_candidate_log_probs


def test_preflight_rejects_candidate_cache_before_pinned_allocation(monkeypatch):
    actor = _make_actor(_make_args(rs_refill_max_candidate_cache_bytes=7))
    rollout_data = {
        "group_indices": [3],
        "loss_masks": [torch.ones(2)],
        "num_microbatches": [1],
        "rollout_log_probs": [torch.zeros(2)],
        "sample_indices": [10],
    }
    actor._get_rollout_data = lambda _ref: rollout_data
    actor.compute_log_prob = lambda *_args, **_kwargs: {"log_probs": [torch.zeros(2)]}
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data)
    _patch_last_pipeline_rank(monkeypatch)

    with pytest.raises(RuntimeError, match=r"required=8, limit=7"):
        actor.score_rs_candidates(27, object())

    assert 27 not in actor._rs_candidate_log_probs


def test_duplicate_take_request_preserves_cache_for_cleanup(monkeypatch):
    actor = _make_actor()
    actor._rs_candidate_log_probs[4] = {2: torch.zeros(1)}
    _patch_last_pipeline_rank(monkeypatch)

    with pytest.raises(ValueError, match="must be unique"):
        actor.take_rs_candidate_log_probs(4, [2, 2])

    assert 4 in actor._rs_candidate_log_probs
    actor.discard_rs_candidate_log_probs(4)
    actor.discard_rs_candidate_log_probs(4)
    assert actor._rs_candidate_log_probs == {}


def _patch_train_actor_dependencies(monkeypatch, train_calls):
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda data: data)
    monkeypatch.setattr(actor_module, "compute_advantages_and_returns", lambda _args, _data: None)
    monkeypatch.setattr(actor_module.train_metric_utils, "log_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module.train_metric_utils, "log_perf_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module.train_data_utils, "save_debug_train_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "inverse_timer", lambda _name: nullcontext())
    monkeypatch.setattr(actor_module, "timer", lambda _name: nullcontext())
    monkeypatch.setattr(actor_module, "train", lambda *args, **kwargs: train_calls.append((args, kwargs)))


def _training_batch():
    return {
        "global_batch_sizes": [1],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int)],
        "num_microbatches": [1],
        "response_lengths": [2],
        "rs_preflight_log_probs": [torch.tensor([-0.2, -0.3])],
    }


def test_train_actor_reuses_preflight_log_probs_without_recompute(monkeypatch):
    actor = _make_actor()
    train_calls = []
    _patch_train_actor_dependencies(monkeypatch, train_calls)
    actor.compute_log_prob = lambda *_args, **_kwargs: pytest.fail("proximal logprobs were recomputed")
    rollout_data = _training_batch()
    cached = rollout_data["rs_preflight_log_probs"]

    actor.train_actor(8, rollout_data)

    assert rollout_data["log_probs"] is cached
    assert "rs_preflight_log_probs" not in rollout_data
    assert len(train_calls) == 1
    assert train_calls[0][0][4] is rollout_data


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("response_lengths", "response lengths changed after proximal preflight"),
        ("loss_masks", "changed samples accepted by preflight"),
    ],
)
def test_train_actor_rejects_post_preflight_batch_mutation(monkeypatch, mutation, error):
    actor = _make_actor()
    train_calls = []
    _patch_train_actor_dependencies(monkeypatch, train_calls)
    actor.compute_log_prob = lambda *_args, **_kwargs: pytest.fail("proximal logprobs were recomputed")

    def mutate(_args, _rollout_id, data):
        if mutation == "response_lengths":
            data["response_lengths"][0] = 1
        else:
            data["loss_masks"][0][0] = 0

    actor.rollout_data_postprocess = mutate

    with pytest.raises(RuntimeError, match=error):
        actor.train_actor(8, _training_batch())

    assert train_calls == []


@pytest.mark.parametrize(
    ("enabled", "rollout_data", "error"),
    [
        (True, {}, "missing its proximal logprob cache"),
        (True, {"rs_preflight_log_probs": [], "log_probs": []}, "conflicting proximal logprob fields"),
        (
            True,
            {
                "loss_masks": [torch.ones(1)],
                "response_lengths": [1],
                "rs_preflight_log_probs": [],
            },
            "proximal cache/sample count mismatch",
        ),
        (
            True,
            {
                "loss_masks": [torch.ones(2)],
                "response_lengths": [2],
                "rs_preflight_log_probs": [torch.zeros(1)],
            },
            "proximal cache shape mismatch",
        ),
        (False, {"rs_preflight_log_probs": []}, "while --rs-batch-refill is disabled"),
    ],
)
def test_train_actor_rejects_invalid_preflight_cache_contract(enabled, rollout_data, error):
    actor = _make_actor(_make_args(rs_batch_refill=enabled))

    with pytest.raises(RuntimeError, match=error):
        actor.train_actor(1, rollout_data)


def test_policy_loss_routes_rs_refill_through_shared_gate(monkeypatch):
    args = _make_args(custom_tis_function_path="must-not-load")
    cached_log_probs = [torch.zeros(2)]
    loss_masks = [torch.ones(2)]
    batch = {
        "advantages": [torch.ones(2)],
        "log_probs": cached_log_probs,
        "loss_masks": loss_masks,
        "response_lengths": [2],
        "rollout_log_probs": [torch.zeros(2)],
        "rollout_mask_sums": torch.tensor([2.0]),
        "total_lengths": [3],
        "unconcat_tokens": [torch.zeros(3, dtype=torch.long)],
    }
    calls = {}

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            None,
            {"entropy": [torch.zeros(2)], "log_probs": [torch.full((2,), 0.1)]},
        ),
    )
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: lambda value: value.mean())
    monkeypatch.setattr(loss_module, "load_function", lambda _path: pytest.fail("custom TIS was loaded"))

    def apply_gate(**kwargs):
        calls["gate"] = kwargs
        return kwargs["pg_loss"], [mask.clone() for mask in kwargs["loss_masks"]], {}

    def validate_masks(original, *candidates):
        calls["validation"] = (original, candidates)

    monkeypatch.setattr(loss_module, "apply_rs_refill_tis", apply_gate)
    monkeypatch.setattr(loss_module, "validate_final_rs_masks", validate_masks)

    loss_module.policy_loss_function(
        args,
        batch,
        torch.zeros((1, 2, 2), requires_grad=True),
        lambda value: value.mean(),
    )

    assert calls["gate"]["train_log_probs"] is cached_log_probs
    assert calls["gate"]["loss_masks"] is loss_masks
    original, candidates = calls["validation"]
    assert len(candidates) == 2
    assert candidates[0] is loss_masks
    assert original[0] is not loss_masks[0]
    torch.testing.assert_close(original[0], loss_masks[0])


class _RemoteMethod:
    def __init__(self, actor_index, method, calls):
        self.actor_index = actor_index
        self.method = method
        self.calls = calls

    def remote(self, *args):
        call = (self.actor_index, self.method, args)
        self.calls.append(call)
        return call


class _ActorHandle:
    def __init__(self, actor_index, calls):
        self.score_rs_candidates = _RemoteMethod(actor_index, "score", calls)
        self.take_rs_candidate_log_probs = _RemoteMethod(actor_index, "take", calls)
        self.discard_rs_candidate_log_probs = _RemoteMethod(actor_index, "discard", calls)


def test_actor_group_fans_out_exact_refill_rpcs():
    calls = []
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = [_ActorHandle(0, calls), _ActorHandle(1, calls)]
    rollout_ref = object()

    score_refs = group.async_score_rs_candidates(17, rollout_ref)
    take_refs = group.async_take_rs_candidate_log_probs(17, [2, 5])
    discard_refs = group.async_discard_rs_candidate_log_probs(17)

    assert score_refs == calls[:2]
    assert take_refs == calls[2:4]
    assert discard_refs == calls[4:]
    assert calls == [
        (0, "score", (17, rollout_ref)),
        (1, "score", (17, rollout_ref)),
        (0, "take", (17, [2, 5])),
        (1, "take", (17, [2, 5])),
        (0, "discard", (17,)),
        (1, "discard", (17,)),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
