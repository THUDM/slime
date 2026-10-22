"""CPU tests for FlashREINFORCE (NVIDIA-NeMo/labs-molt#116).

Covers the rollout-batch-mean advantage (``--advantage-estimator flash_reinforce``) and the
binary-KL sequence trust region (``binary_kl_trust_region_function``), including its
context-parallel reduction.
"""

import argparse
import math
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import _cp_dist_helpers
import pytest
import torch
from test_score_centering import args as score_centering_args

from slime.utils.types import Sample

NUM_GPUS = 0

TRUST_REGION = "slime.backends.megatron_utils.loss.binary_kl_trust_region_function"


def args(**overrides):
    values = dict(
        use_score_centering=False,
        advantage_estimator="flash_reinforce",
        rollout_temperature=1.0,
        use_tis=True,
        custom_tis_function_path=TRUST_REGION,
        tis_binary_kl_threshold=5e-3,
    )
    return score_centering_args(**(values | overrides))


@pytest.fixture(autouse=True)
def no_gpu_server_imports(monkeypatch):
    deployment = types.ModuleType("slime.backends.sglang_utils.deployment")
    deployment.start_rollout_servers = lambda *args: None
    monkeypatch.setitem(sys.modules, deployment.__name__, deployment)
    if "sglang_router" not in sys.modules:
        monkeypatch.setitem(sys.modules, "sglang_router", SimpleNamespace(__version__="0.3.0"))


# --------------------------------- advantage ---------------------------------


def post_process(rewards, **overrides):
    from slime.data.batch_builder import BatchBuilder

    values = dict(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="flash_reinforce",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=1,
        rollout_batch_size=len(rewards),
        reward_key=None,
    )
    builder = BatchBuilder(Namespace(**(values | overrides)))
    return builder._post_process_rewards([Sample(index=i, reward=r) for i, r in enumerate(rewards)])


def constant_rewards(args, samples):
    return [9.0] * len(samples), [7.0] * len(samples)


def test_flash_reinforce_centers_the_whole_batch_without_std():
    raw, centered = post_process([1.0, 0.0, 0.0, 0.0])
    assert raw == [1.0, 0.0, 0.0, 0.0]
    assert centered == pytest.approx([0.75, -0.25, -0.25, -0.25])


def test_flash_reinforce_ignores_prompt_groups():
    rewards = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    _, centered = post_process(rewards, n_samples_per_prompt=4, rollout_batch_size=2)
    assert centered == pytest.approx([r - 0.5 for r in rewards])


def test_flash_reinforce_single_outcome_batch_is_zero():
    _, centered = post_process([1.0, 1.0, 1.0])
    assert centered == [0.0, 0.0, 0.0]


def test_flash_reinforce_respects_disable_rewards_normalization():
    raw, rewards = post_process([1.0, 0.0], rewards_normalization=False)
    assert rewards == raw == [1.0, 0.0]


def test_custom_reward_post_process_replaces_centering():
    raw, rewards = post_process([1.0, 0.0], custom_reward_post_process_path="test_flash_reinforce.constant_rewards")
    assert (raw, rewards) == ([9.0, 9.0], [7.0, 7.0])


GROUP_REWARDS = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
GROUP_CENTERED = [0.75, -0.25, -0.25, -0.25, 0.25, 0.25, -0.75, 0.25]


@pytest.mark.parametrize(
    "estimator,std,expected",
    [
        ("grpo", True, [x / (0.5 + 1e-6) for x in GROUP_CENTERED]),
        ("grpo", False, GROUP_CENTERED),
        ("gspo", True, [x / (0.5 + 1e-6) for x in GROUP_CENTERED]),
        ("reinforce_plus_plus_baseline", True, GROUP_CENTERED),
        ("ppo", True, GROUP_REWARDS),
    ],
)
def test_other_estimators_are_unchanged(estimator, std, expected):
    _, rewards = post_process(
        GROUP_REWARDS,
        advantage_estimator=estimator,
        grpo_std_normalization=std,
        n_samples_per_prompt=4,
        rollout_batch_size=2,
    )
    assert rewards == pytest.approx(expected)


def test_advantages_broadcast_the_centered_reward(monkeypatch):
    from megatron.core import mpu

    from slime.backends.megatron_utils.loss import compute_advantages_and_returns

    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    a = Namespace(
        advantage_estimator="flash_reinforce",
        use_rollout_logprobs=False,
        kl_coef=0.0,
        custom_advantage_function_path=None,
        use_opd=False,
        normalize_advantages=False,
    )
    rollout_data = dict(rewards=[0.75, -0.25], rollout_log_probs=[torch.zeros(3), torch.zeros(2)])
    compute_advantages_and_returns(a, rollout_data)
    for advantage, reward, length in zip(rollout_data["advantages"], [0.75, -0.25], [3, 2], strict=True):
        torch.testing.assert_close(advantage, torch.full((length,), reward))


# --------------------------------- arguments ---------------------------------


def test_cli_parses_estimator_and_threshold(monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module

    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)
    default = parser.parse_args(["--rollout-batch-size", "1"])
    assert default.tis_binary_kl_threshold == 5e-3
    configured = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--advantage-estimator",
            "flash_reinforce",
            "--tis-binary-kl-threshold",
            "inf",
        ]
    )
    assert configured.advantage_estimator == "flash_reinforce"
    assert configured.tis_binary_kl_threshold == math.inf


@pytest.mark.parametrize("threshold", [0.0, -1.0, math.nan])
def test_validation_rejects_non_positive_threshold(monkeypatch, threshold):
    from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

    module = load_slime_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match="tis-binary-kl-threshold"):
        module.slime_validate_args(make_slime_validate_args(tis_binary_kl_threshold=threshold))


def test_validation_accepts_flash_reinforce_recipe(monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

    module = load_slime_arguments_module(monkeypatch)
    a = make_slime_validate_args(
        advantage_estimator="flash_reinforce",
        use_tis=True,
        custom_tis_function_path=TRUST_REGION,
        tis_binary_kl_threshold=math.inf,
        num_steps_per_rollout=1,
        rollout_batch_size=4,
        n_samples_per_prompt=1,
    )
    module.slime_validate_args(a)
    assert a.global_batch_size == 4


# ------------------------------- trust region -------------------------------


def trust_region(a, train, rollout, masks=None, pg_loss=None):
    """Run the callback on full sequences (no context parallelism)."""
    from slime.backends.megatron_utils.loss import binary_kl_trust_region_function

    train = [torch.tensor(x, dtype=torch.float32) for x in train]
    rollout = [torch.tensor(x, dtype=torch.float32) for x in rollout]
    masks = [torch.ones(len(x), dtype=torch.int) for x in train] if masks is None else masks
    lengths = [len(x) for x in train]
    if pg_loss is None:
        pg_loss = torch.ones(sum(lengths))
    return binary_kl_trust_region_function(
        a,
        pg_loss=pg_loss,
        train_log_probs=train,
        rollout_log_probs=rollout,
        loss_masks=masks,
        total_lengths=[n + 4 for n in lengths],
        response_lengths=lengths,
    )


FAR = ([0.0, math.log(0.5)], [-math.log(3.0), math.log(0.5)])  # pi=[1, .5] vs mu=[1/3, .5]


def test_far_off_token_rejects_the_whole_sequence():
    pg_loss, masks, metrics = trust_region(args(tis_binary_kl_threshold=3e-3), [FAR[0]], [FAR[1]])
    torch.testing.assert_close(pg_loss, torch.zeros(2))
    assert masks[0].tolist() == [1, 1]  # rejection is a zero weight, not a mask
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.ones(2))
    torch.testing.assert_close(metrics["tis"], torch.tensor([3.0, 1.0]))


def test_trust_region_is_two_sided():
    pg_loss, _, metrics = trust_region(args(tis_binary_kl_threshold=3e-3), [FAR[1]], [FAR[0]])
    torch.testing.assert_close(pg_loss, torch.zeros(2))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.ones(2))


def test_close_sequence_keeps_its_unclipped_importance_weight():
    pi, mu = math.log(0.5), math.log(0.505)
    pg_loss, _, metrics = trust_region(args(tis_binary_kl_threshold=3e-3), [[pi, pi]], [[mu, mu]])
    torch.testing.assert_close(pg_loss, torch.full((2,), 0.5 / 0.505))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(2))
    assert metrics["tis_binary_kl"].max() < 1e-4


def test_infinite_threshold_disables_the_gate():
    pg_loss, _, _ = trust_region(args(tis_binary_kl_threshold=math.inf), [FAR[0]], [FAR[1]])
    torch.testing.assert_close(pg_loss, torch.tensor([3.0, 1.0]))


def test_masked_tokens_do_not_enter_the_sequence_mean():
    pi, mu = math.log(0.5), math.log(0.505)
    train, rollout = [[pi, 0.0, pi]], [[mu, math.log(1e-9), mu]]
    masks = [torch.tensor([1, 0, 1], dtype=torch.int)]
    pg_loss, _, metrics = trust_region(args(tis_binary_kl_threshold=3e-3), train, rollout, masks)
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(3))
    torch.testing.assert_close(pg_loss[[0, 2]], torch.full((2,), 0.5 / 0.505))


def test_sequence_without_loss_tokens_is_kept():
    masks = [torch.zeros(2, dtype=torch.int)]
    _, _, metrics = trust_region(args(tis_binary_kl_threshold=3e-3), [FAR[0]], [FAR[1]], masks)
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.zeros(2))


@pytest.mark.parametrize("bad", [math.nan, -math.inf])
def test_non_finite_rollout_log_probs_are_rejected_with_finite_weights(bad):
    pg_loss, _, metrics = trust_region(args(), [[0.0, math.log(0.5)]], [[bad, math.log(0.5)]])
    assert torch.isfinite(pg_loss).all()
    torch.testing.assert_close(pg_loss, torch.zeros(2))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.ones(2))
    assert torch.isfinite(metrics["tis"]).all()


def test_only_the_offending_sequence_is_rejected():
    pi, mu = math.log(0.5), math.log(0.505)
    pg_loss, _, metrics = trust_region(args(tis_binary_kl_threshold=3e-3), [[pi, pi], FAR[0]], [[mu, mu], FAR[1]])
    torch.testing.assert_close(pg_loss, torch.tensor([0.5 / 0.505, 0.5 / 0.505, 0.0, 0.0]))
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_mismatch_metrics_alone_leave_the_loss_untouched():
    pg_loss, _, metrics = trust_region(args(use_tis=False, tis_binary_kl_threshold=3e-3), [FAR[0]], [FAR[1]])
    torch.testing.assert_close(pg_loss, torch.ones(2))
    assert set(metrics) == {"tis", "tis_abs", "tis_binary_kl", "tis_seq_reject_frac"}


# ------------------------------- policy loss -------------------------------


def policy_batch():
    """Two responses; the second is pushed off-policy so the trust region drops it."""
    torch.manual_seed(3)
    total, response = [8, 8], [3, 2]
    tokens = [torch.randint(0, 12, (t,)) for t in total]
    logits = torch.randn(1, sum(total), 12)
    current = target_log_probs(logits, tokens, total, response)
    return logits, dict(
        total_lengths=total,
        response_lengths=response,
        unconcat_tokens=tokens,
        loss_masks=[torch.tensor([1, 0, 1]), torch.tensor([1, 1])],
        rollout_mask_sums=[torch.tensor(2), torch.tensor(2)],
        # Sample 0 is nearly on-policy; sample 1 claims its first token was sampled with mu=0.9.
        rollout_log_probs=[current[0] + 1e-3, torch.stack([torch.tensor(math.log(0.9)), current[1][1]])],
        advantages=[torch.full((3,), 0.5), torch.full((2,), -0.5)],
    )


def target_log_probs(logits, tokens, total, response):
    out, offset = [], 0
    for t, r, tok in zip(total, response, tokens, strict=True):
        rows = logits.squeeze(0)[offset + t - r - 1 : offset + t - 1].float()
        out.append(rows.log_softmax(-1).gather(-1, tok[-r:, None]).squeeze(-1))
        offset += t
    return out


def reference_loss(logits, batch, threshold):
    loss = logits.new_zeros(())
    current = target_log_probs(logits, batch["unconcat_tokens"], batch["total_lengths"], batch["response_lengths"])
    for i, logp in enumerate(current):
        mu, mask = batch["rollout_log_probs"][i], batch["loss_masks"][i].float()
        with torch.no_grad():
            ratio = (logp - mu).clamp(-30, 30).exp()
            p, q = mu.exp().clamp(1e-6, 1 - 1e-6), logp.exp().clamp(1e-6, 1 - 1e-6)
            kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
            keep = float((kl * mask).sum() / mask.sum().clamp_min(1) <= threshold)
        term = -batch["advantages"][i] * logp * ratio * keep
        loss = loss + (term * mask).sum() / batch["rollout_mask_sums"][i]
    return loss


def test_policy_loss_gradient_matches_reference(monkeypatch):
    from megatron.core import mpu

    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import policy_loss_function

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    logits, batch = policy_batch()
    logits.requires_grad_()
    a = args()
    reduce = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    loss, metrics = policy_loss_function(a, batch, logits, reduce)
    actual = torch.autograd.grad(loss, logits)[0]
    expected = torch.autograd.grad(reference_loss(logits, batch, a.tis_binary_kl_threshold), logits)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    # The second response (logit rows 13..14) is rejected: exactly zero gradient.
    assert actual[0, 13:15].count_nonzero() == 0
    assert actual[0, 4:7].count_nonzero() > 0
    # Old log-probs are the detached current ones, so the PPO ratio is exactly 1.
    assert metrics["ppo_kl"].item() == 0.0
    torch.testing.assert_close(metrics["tis_seq_reject_frac"], torch.tensor(1.0))


def test_rejection_keeps_entropy_and_ppo_kl_metrics(monkeypatch):
    from megatron.core import mpu

    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import policy_loss_function

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    logits, batch = policy_batch()
    reduce = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    _, gated = policy_loss_function(args(), batch, logits, reduce)
    _, open_gate = policy_loss_function(args(tis_binary_kl_threshold=math.inf), batch, logits, reduce)
    for key in ("entropy_loss", "ppo_kl", "pg_clipfrac"):
        torch.testing.assert_close(gated[key], open_gate[key])


# ------------------------------ context parallel ------------------------------

# Sample 1 (T=8, R=1) puts its only response token on CP rank 0, so rank 1 owns no
# token of it but must still join the all-reduce.
CP_TOTALS, CP_RESPONSES = [8, 8, 8], [3, 1, 5]


def cp_inputs():
    torch.manual_seed(5)
    train = [-0.7 - torch.randn(r).abs() for r in CP_RESPONSES]  # probabilities below 0.5
    rollout = [x + 0.002 * torch.randn(len(x)) for x in train]  # nearly on-policy
    train[2][1], rollout[2][1] = math.log(0.1), math.log(0.9)  # one far-off token in sequence 2
    masks = [
        torch.tensor([1, 1, 0], dtype=torch.int),
        torch.tensor([1], dtype=torch.int),
        torch.ones(5, dtype=torch.int),
    ]
    return train, rollout, masks


def cp_worker(rank, world_size, port, recompute):
    import torch.distributed as dist
    from megatron.core import mpu
    from torch.utils.checkpoint import checkpoint

    torch.set_num_threads(1)
    group = _cp_dist_helpers.init_worker_process_group(rank, world_size, port)
    _cp_dist_helpers.stub_megatron_in_worker(world_size, rank)
    mpu.get_context_parallel_group = lambda: group
    try:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp
        from slime.backends.megatron_utils.loss import binary_kl_trust_region_function

        train, rollout, masks = cp_inputs()
        a = args(tis_binary_kl_threshold=5e-3)

        def local(values):
            return [slice_log_prob_with_cp(x, t, r) for x, t, r in zip(values, CP_TOTALS, CP_RESPONSES, strict=True)]

        # Single-process reference on full sequences, sliced to this rank afterwards.
        _cp_dist_helpers.stub_megatron_in_worker(1, 0)
        full_pg, _, full_metrics = binary_kl_trust_region_function(
            a,
            pg_loss=torch.ones(sum(CP_RESPONSES)),
            train_log_probs=train,
            rollout_log_probs=rollout,
            loss_masks=masks,
            total_lengths=CP_TOTALS,
            response_lengths=CP_RESPONSES,
        )
        _cp_dist_helpers.stub_megatron_in_worker(world_size, rank)
        expected_pg = torch.cat(local(list(full_pg.split(CP_RESPONSES))))
        expected_reject = torch.cat(local(list(full_metrics["tis_seq_reject_frac"].split(CP_RESPONSES))))
        assert expected_reject.sum() > 0  # the off-policy sequence is rejected

        local_train, local_rollout = local(train), local(rollout)
        pg_loss = torch.ones(sum(len(x) for x in local_train), requires_grad=True)

        def gate(pg):
            out, _, metrics = binary_kl_trust_region_function(
                a,
                pg_loss=pg,
                train_log_probs=local_train,
                rollout_log_probs=local_rollout,
                loss_masks=masks,
                total_lengths=CP_TOTALS,
                response_lengths=CP_RESPONSES,
            )
            return out, metrics["tis_seq_reject_frac"]

        if recompute:
            # --recompute-loss-function reruns the callback (and its all-reduce) in backward.
            gated, reject = checkpoint(gate, pg_loss, use_reentrant=False)
        else:
            gated, reject = gate(pg_loss)
        gated.sum().backward()
        torch.testing.assert_close(gated.detach(), expected_pg)
        torch.testing.assert_close(reject, expected_reject)
        torch.testing.assert_close(pg_loss.grad, expected_pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("recompute", [False, True])
def test_context_parallel_gate_matches_full_sequences(recompute):
    torch.multiprocessing.spawn(cp_worker, args=(2, _cp_dist_helpers.free_port(), recompute), nprocs=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
