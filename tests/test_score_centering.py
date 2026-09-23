"""Equation-level and distributed integration checks for arXiv:2609.20807."""

from argparse import Namespace

import _cp_dist_helpers
import pytest
import torch

from slime.utils.score_centering import (
    get_score_centering_is_config,
    importance_weights,
    score_centering_correction,
    score_centering_request,
    validate_sampler_topk,
    validate_score_centering_args,
)
from slime.utils.types import Sample

NUM_GPUS = 0


def args(mode="none", **overrides):
    values = dict(
        pg_loss_type=None,
        use_score_centering=True,
        score_centering_top_k=3,
        use_tis=mode != "none",
        tis_clip_low=0.5 if mode == "mis" else 0.0,
        tis_clip=5.0 if mode == "mis" else 2.0,
        custom_tis_function_path="slime.backends.megatron_utils.loss.icepop_function" if mode == "mis" else None,
        rollout_temperature=0.8,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        allgather_cp=False,
        log_probs_chunk_size=2,
        entropy_coef=0.0,
        use_kl_loss=False,
        use_rollout_logprobs=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        opd_full_vocab=False,
        opd_teacher_top_k=0,
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        use_unbiased_kl=False,
    )
    return Namespace(**(values | overrides))


def dense_reference(logits, ids, q_head, targets, sampled_q, advantage, mode, temperature=1.0, low=None, high=None):
    """Explicitly reconstruct every tail token, then sum weighted scores over V."""
    logp = (logits.float() / temperature).log_softmax(-1)
    with torch.no_grad():
        p = logp.exp()
        rho = (1 - q_head.exp().sum(-1)).clamp_min(1e-6) / (1 - p.gather(-1, ids).sum(-1)).clamp_min(1e-6)
        q = p * rho[:, None]
        q.scatter_(-1, ids, q_head.exp())

        # Independent weight implementation for the oracle.
        def weight(r):
            if mode == "none":
                return torch.ones_like(r)
            if mode == "tis":
                return torch.maximum(
                    torch.minimum(r, torch.tensor(2.0 if high is None else high)),
                    torch.tensor(0.0 if low is None else low),
                )
            return torch.where((r >= (0.5 if low is None else low)) & (r <= (5 if high is None else high)), r, 0)

        coeff = q * weight(p / q)
        w = weight((logp.gather(-1, targets[:, None]).squeeze(-1) - sampled_q).exp())
    return -advantage * (w * logp.gather(-1, targets[:, None]).squeeze(-1) - (coeff * logp).sum(-1))


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_topk_gradient_matches_full_reconstructed_distribution(mode):
    torch.manual_seed(71)
    logits = (torch.randn(5, 11) * 2).requires_grad_()
    q = (torch.randn(5, 11) * 2).log_softmax(-1).requires_grad_()
    q_head, ids = q.topk(3)
    targets = torch.tensor([0, 7, 2, 10, 4])  # includes actions outside the head
    adv = torch.tensor([1.0, -2.0, 0.5, 0.0, -1.0])
    logp = logits.log_softmax(-1)
    correction, _, _ = score_centering_correction(logp.gather(-1, ids), q_head, mode=mode)
    sampled = logp.gather(-1, targets[:, None]).squeeze(-1)
    sampled_q = q.gather(-1, targets[:, None]).squeeze(-1)
    w = importance_weights((sampled - sampled_q).exp(), mode).detach()
    loss = (-adv * (w * sampled - correction)).sum()
    ref = dense_reference(logits, ids, q_head, targets, sampled_q, adv, mode).sum()
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], torch.autograd.grad(ref, logits)[0], atol=2e-6, rtol=2e-6
    )
    assert torch.autograd.grad(loss, q, allow_unused=True)[0] is None


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_constant_reward_has_zero_expected_gradient(mode):
    # Full vocabulary head: expectation over all sampled actions cancels drift.
    logits = torch.tensor([1.0, -1.0, 0.2, 2.0], requires_grad=True)
    q = torch.tensor([0.15, 0.55, 0.25, 0.05])
    logp = logits.log_softmax(-1)
    correction, _, _ = score_centering_correction(logp[None], q.log()[None], mode=mode)
    w = importance_weights((logp - q.log()).exp(), mode).detach()
    expected_loss = (q * (-w * logp + correction)).sum()
    torch.testing.assert_close(
        torch.autograd.grad(expected_loss, logits)[0], torch.zeros_like(logits), atol=2e-7, rtol=0
    )


def test_on_policy_correction_zero_and_tiny_tails_finite():
    p = torch.tensor([[0.0, -90.0, -95.0]], requires_grad=True)
    correction, _, _ = score_centering_correction(p, p.detach())
    assert correction.item() == 0
    correction, _, _ = score_centering_correction(p, torch.tensor([[-0.01, -8.0, -9.0]]), mode="tis")
    correction.sum().backward()
    assert torch.isfinite(correction).all() and torch.isfinite(p.grad).all()


def make_batch():
    torch.manual_seed(8)
    total, response = [8, 8], [3, 2]
    q = [torch.randn(r, 12).log_softmax(-1) for r in response]
    heads = [x.topk(3) for x in q]
    tokens = [torch.randint(0, 12, (t,)) for t in total]
    return dict(
        total_lengths=total,
        response_lengths=response,
        unconcat_tokens=tokens,
        loss_masks=[torch.tensor([1, 0, 1]), torch.tensor([1, 1])],
        rollout_mask_sums=[torch.tensor(2), torch.tensor(2)],
        rollout_topk_token_ids=[x.indices for x in heads],
        rollout_topk_log_probs=[x.values for x in heads],
        rollout_log_probs=[
            x.gather(-1, t[-r:, None]).squeeze(-1) for x, t, r in zip(q, tokens, response, strict=True)
        ],
        advantages=[torch.tensor([1.0, -3.0, -1.0]), torch.tensor([2.0, -0.5])],
    )


def reference_batch(logits, batch, a):
    loss = logits.new_zeros(())
    offset = 0
    for i, (t, r) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        rows = logits.squeeze(0)[offset + t - r - 1 : offset + t - 1]
        term = dense_reference(
            rows,
            batch["rollout_topk_token_ids"][i],
            batch["rollout_topk_log_probs"][i],
            batch["unconcat_tokens"][i][-r:],
            batch["rollout_log_probs"][i],
            batch["advantages"][i],
            ("mis" if a.custom_tis_function_path else "tis") if a.use_tis else "none",
            a.rollout_temperature,
            a.tis_clip_low,
            a.tis_clip,
        )
        mask = batch["loss_masks"][i]
        loss = loss + (term * mask).sum() / mask.sum()
        offset += t
    return loss


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_policy_loss_gradient_matches_dense_reference(mode, dtype, monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import policy_loss_function

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    batch = make_batch()
    logits = torch.randn(1, 16, 12, dtype=dtype).requires_grad_()
    a = args(mode=mode)
    reduce = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    loss, metrics = policy_loss_function(a, batch, logits.float(), reduce)
    ref = reference_batch(logits, batch, a)
    # Head-only scalar differs from dense CE by a zero-gradient multiple of sum p log p.
    actual_grad = torch.autograd.grad(loss, logits)[0]
    expected_grad = torch.autograd.grad(ref, logits)[0]
    tol = 0.008 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual_grad, expected_grad, atol=tol, rtol=tol)
    assert set(metrics) >= {"loss", "pg_loss", "sc_correction", "sc_sampler_head_mass"}
    assert actual_grad[0, 5].count_nonzero() == 0  # masked response position


def distributed_worker(rank, world_size, port, layout, mode, disk=False):
    import torch.distributed as dist
    from megatron.core import mpu

    torch.set_num_threads(1)
    group = _cp_dist_helpers.init_worker_process_group(rank, world_size, port)
    is_tp = layout == "tp"
    _cp_dist_helpers.stub_megatron_in_worker(1 if is_tp else world_size, 0 if is_tp else rank)
    # Every rank must create the singleton groups in the same order.
    singles = [dist.new_group([i]) for i in range(world_size)]
    mpu.get_tensor_model_parallel_group = lambda: group if is_tp else singles[rank]
    mpu.get_context_parallel_group = lambda: group
    try:
        batch = make_batch()
        full_logits = torch.randn(16, 12)
        a = args(mode=mode, allgather_cp=layout == "allgather")
        if is_tp:
            local = full_logits[:, rank * 6 : (rank + 1) * 6]
        elif layout == "allgather":
            local = full_logits[rank * 8 : (rank + 1) * 8]
        else:
            row_ids = torch.cat(
                [
                    torch.tensor([i + rank * 2, i + rank * 2 + 1, i + (3 - rank) * 2, i + (3 - rank) * 2 + 1])
                    for i in [0, 8]
                ]
            )
            local = full_logits[row_ids]
        local = local.clone().requires_grad_()
        original = dict(batch)
        import tempfile
        from pathlib import Path
        from slime.utils.tensor_store import DiskTensorRef

        with tempfile.TemporaryDirectory() as directory:
            if disk:
                for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
                    batch[key] = [
                        DiskTensorRef.write(
                            value.int() if key.endswith("ids") else value, Path(directory) / f"{key}_{i}.safetensors"
                        )
                        for i, value in enumerate(batch[key])
                    ]
            _check_distributed_batch(
                rank,
                layout,
                is_tp,
                batch,
                original,
                full_logits,
                local,
                a,
                None if is_tp or layout == "allgather" else row_ids,
                disk,
            )
    finally:
        dist.destroy_process_group()


def _check_distributed_batch(rank, layout, is_tp, batch, original, full_logits, local, a, row_ids, disk):
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
    from slime.backends.megatron_utils.loss import policy_loss_function

    if not is_tp:
        fields = ["advantages", "rollout_log_probs"]
        if layout != "allgather" and not disk:
            fields += ["rollout_topk_token_ids", "rollout_topk_log_probs"]
        for key in fields:
            batch[key] = [
                slice_log_prob_with_cp(x, t, r)
                for x, t, r in zip(batch[key], batch["total_lengths"], batch["response_lengths"], strict=True)
            ]
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    loss, _ = policy_loss_function(a, batch, local.unsqueeze(0), reducer)
    ref_logits = full_logits.clone().requires_grad_()
    ref = reference_batch(ref_logits.unsqueeze(0), original, a)
    loss.backward()
    ref.backward()
    if is_tp:
        expected = ref_logits.grad[:, rank * 6 : (rank + 1) * 6]
    elif layout == "allgather":
        expected = ref_logits.grad[rank * 8 : (rank + 1) * 8]
    else:
        expected = ref_logits.grad[row_ids]
    torch.testing.assert_close(local.grad, expected, atol=2e-6, rtol=2e-6)
    # Training dumps must restore integer heads without losing CP rows or precision.
    from slime.observability.train_data_utils import restore_context_parallel_fields_to_cpu

    restored = restore_context_parallel_fields_to_cpu(
        {
            key: value
            for key, value in batch.items()
            if key in ("total_lengths", "response_lengths", "rollout_topk_token_ids", "rollout_topk_log_probs")
        },
        lambda *args: pytest.fail("Sampler heads must use their own integer-preserving gather"),
        keep_restored=True,
        allgather_cp=layout == "allgather",
    )
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
        for actual, reference in zip(restored[key], original[key], strict=True):
            if disk:
                actual = actual.load()
            torch.testing.assert_close(actual, reference.to(actual.dtype))


@pytest.mark.parametrize("layout", ["tp", "zigzag", "allgather"])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_distributed_gradients(layout, mode):
    torch.multiprocessing.spawn(distributed_worker, args=(2, _cp_dist_helpers.free_port(), layout, mode), nprocs=2)


def meta(n=1):
    from test_score_centering_binary import binary_meta

    return binary_meta([[3, 1, 4]] * n, [[-0.5, -2.0, -3.0]] * n)


def test_sample_append_resume_tool_tokens_and_serialization():
    a = args()
    s = Sample(tokens=[9])
    s.append_response_tokens(a, tokens=[3], log_probs=[-0.5], meta_info=meta())
    s.append_response_tokens(a, tokens=[8, 9], trainable=False)
    s = Sample.from_dict(s.to_dict())
    s.append_response_tokens(a, tokens=[1], log_probs=[-2.0], meta_info=meta())
    validate_sampler_topk(s, 3)
    assert s.loss_mask == [1, 0, 0, 1]
    assert s.rollout_topk_token_ids[0].tolist() == s.rollout_topk_token_ids[-1].tolist() == [3, 1, 4]
    assert len(s.rollout_topk_log_probs) == s.response_length == 4


def test_missing_topk_fails_before_token_append():
    s = Sample(tokens=[9])
    with pytest.raises(ValueError, match="requires binary"):
        s.append_response_tokens(args(), tokens=[3], log_probs=[-0.5])
    assert s.tokens == [9]
    assert s.response_length == 0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(pg_loss_type="ppo"),
        dict(rollout_top_p=0.9),
        dict(rollout_top_k=8),
        dict(rollout_temperature=0),
        dict(score_centering_top_k=0),
        dict(train_backend="fsdp"),
        dict(custom_generate_function_path="slime.rollout.sglang_streaming_rollout.generate_streaming"),
        dict(use_opd=True),
        dict(loss_type="sft_loss"),
        dict(use_tis=True, tis_clip_low=6),
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        validate_score_centering_args(args(**overrides))


def test_request_and_configuration():
    a = args()
    validate_score_centering_args(a)
    assert score_centering_request(a, {}) == {"return_logprob": True, "top_logprobs_num": 3}
    assert score_centering_request(args(use_score_centering=False), {}) == {}


@pytest.mark.parametrize(
    "params", [{"temperature": 0.6}, {"min_p": 0.1}, {"repetition_penalty": 1.1}, {"regex": "[0-9]+"}]
)
def test_request_rejects_distribution_overrides(params):
    with pytest.raises(ValueError):
        score_centering_request(args(), params)


def test_empty_sample_topk_shape():
    s = Sample(tokens=[0])
    validate_sampler_topk(s, 3)
    assert s.rollout_topk_token_ids.shape == (0, 3)


@pytest.mark.parametrize(
    "ids,logps",
    [
        ([[1, 1, 2]], [[-1.0, -2.0, -3.0]]),
        ([[1, 2, 3]], [[0.0, 0.0, 0.0]]),
        ([[1, 2, 3]], [[float("nan"), -2.0, -3.0]]),
    ],
)
def test_loaded_sampler_data_is_validated(ids, logps):
    s = Sample(tokens=[0, 1], response_length=1, rollout_topk_token_ids=ids, rollout_topk_log_probs=logps)
    with pytest.raises(ValueError):
        validate_sampler_topk(s, 3)


def test_cli_defaults_and_overrides(monkeypatch):
    import argparse
    from test_megatron_argument_validation import load_slime_arguments_module

    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)
    default = parser.parse_args(["--rollout-batch-size", "1"])
    assert default.pg_loss_type is None
    assert default.use_score_centering is False
    assert default.score_centering_top_k == 128
    configured = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--use-score-centering",
            "--pg-loss-type",
            "reinforce",
            "--score-centering-top-k",
            "32",
            "--use-tis",
            "--custom-tis-function-path",
            "slime.backends.megatron_utils.loss.icepop_function",
            "--tis-clip-low",
            "0.4",
            "--tis-clip",
            "4",
        ]
    )
    validate_score_centering_args(configured)
    assert configured.pg_loss_type == "reinforce"
    assert configured.tis_clip_low == 0.4
    assert configured.tis_clip == 4
    assert configured.score_centering_top_k == 32


@pytest.mark.parametrize("sc", [False, True])
@pytest.mark.parametrize("tis", [False, True])
@pytest.mark.parametrize("pg_loss_type", [None, "reinforce"])
def test_independent_switches_preserve_losses_and_gradients(sc, tis, pg_loss_type, monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils import loss as loss_module
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    a = args(
        pg_loss_type=pg_loss_type,
        use_score_centering=sc,
        use_tis=tis,
        tis_clip_low=0.3,
        tis_clip=1.5,
        opd_full_vocab=False,
        opd_teacher_top_k=0,
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
    )
    validate_score_centering_args(a)
    batch = make_batch()
    # Deliberately distinguish stored old policy from current policy. SC must
    # use current/sampler weights; the old PPO+TIS path must keep old/sampler.
    batch["log_probs"] = [x + 0.7 for x in batch["rollout_log_probs"]]
    batch["rollout_mask_sums"] = [x.sum() for x in batch["loss_masks"]]
    logits = torch.randn(1, 16, 12).requires_grad_()
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    calls = []
    original_tis = loss_module.vanilla_tis_function

    def tracked_tis(**kwargs):
        calls.append(1)
        return original_tis(**kwargs)

    monkeypatch.setattr(loss_module, "vanilla_tis_function", tracked_tis)
    loss, _ = loss_module.policy_loss_function(a, batch, logits, reducer)
    assert len(calls) == int(tis)
    ref_logits = logits.detach().clone().requires_grad_()
    if sc:
        ref_loss = reference_batch(ref_logits, batch, a)
    else:
        ref_loss = ref_logits.new_zeros(())
        offset = 0
        for i, (t, r) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
            logp = (ref_logits[0, offset + t - r - 1 : offset + t - 1] / a.rollout_temperature).log_softmax(-1)
            logp = logp.gather(-1, batch["unconcat_tokens"][i][-r:, None]).squeeze(-1)
            ratio = (logp - batch["log_probs"][i]).exp()
            adv, mask = batch["advantages"][i], batch["loss_masks"][i]
            term = (
                -adv * logp
                if pg_loss_type == "reinforce"
                else torch.maximum(-adv * ratio, -adv * ratio.clamp(0.8, 1.2))
            )
            if tis:
                train_logp = logp.detach() if pg_loss_type == "reinforce" else batch["log_probs"][i]
                term = term * (train_logp - batch["rollout_log_probs"][i]).exp().clamp(0.3, 1.5)
            ref_loss = ref_loss + (term * mask).sum() / mask.sum()
            offset += t
        torch.testing.assert_close(loss, ref_loss, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], torch.autograd.grad(ref_loss, ref_logits)[0], atol=2e-6, rtol=2e-6
    )


@pytest.mark.parametrize("sc", [False, True])
def test_reinforce_shares_entropy_and_reference_kl(sc, monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import policy_loss_function

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    a = args(pg_loss_type="reinforce", use_score_centering=sc, mode="tis")
    batch = make_batch()
    batch["ref_log_probs"] = [x - 0.2 for x in batch["rollout_log_probs"]]
    logits = torch.randn(1, 16, 12).requires_grad_()
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    base, _ = policy_loss_function(a, batch, logits, reducer)
    a.entropy_coef, a.use_kl_loss, a.kl_loss_coef, a.kl_loss_type = 0.13, True, 0.27, "k2"
    actual, metrics = policy_loss_function(a, batch, logits, reducer)
    entropy, kl = logits.new_zeros(()), logits.new_zeros(())
    offset = 0
    for i, (t, r) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        logp = (logits[0, offset + t - r - 1 : offset + t - 1] / a.rollout_temperature).log_softmax(-1)
        sampled = logp.gather(-1, batch["unconcat_tokens"][i][-r:, None]).squeeze(-1)
        mask = batch["loss_masks"][i]
        entropy = entropy + (-(logp.exp() * logp).sum(-1) * mask).sum() / mask.sum()
        kl = kl + (0.5 * (sampled - batch["ref_log_probs"][i]).square() * mask).sum() / mask.sum()
        offset += t
    expected = base - a.entropy_coef * entropy + a.kl_loss_coef * kl
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics["entropy_loss"], entropy.detach())
    torch.testing.assert_close(metrics["kl_loss"], kl.detach())
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, "ppo"),
        ({"advantage_estimator": "cispo"}, "cispo"),
        ({"advantage_estimator": "gspo"}, "ppo"),
        ({"pg_loss_type": "reinforce"}, "reinforce"),
        ({"use_score_centering": True}, "reinforce"),
    ],
)
def test_pg_objective_defaults(overrides, expected):
    from slime.utils.ppo_utils import get_pg_loss_type

    a = args(**({"use_score_centering": False} | overrides))
    assert get_pg_loss_type(a) == expected


@pytest.mark.parametrize("estimator", ["gspo", "cispo"])
def test_reinforce_rejects_conflicting_objectives(estimator):
    from slime.utils.ppo_utils import get_pg_loss_type

    with pytest.raises(ValueError, match="cannot combine"):
        get_pg_loss_type(args(use_score_centering=False, pg_loss_type="reinforce", advantage_estimator=estimator))


@pytest.mark.parametrize(
    "path,mode",
    [
        (None, "tis"),
        ("slime.backends.megatron_utils.loss.vanilla_tis_function", "tis"),
        ("slime.backends.megatron_utils.loss.icepop_function", "mis"),
    ],
)
def test_score_centering_uses_shared_is_config(path, mode):
    a = args(use_tis=True, custom_tis_function_path=path, tis_clip_low=0.4, tis_clip=3)
    validate_score_centering_args(a)
    assert get_score_centering_is_config(a) == dict(mode=mode, low=0.4, high=3)
    a.use_tis = False
    assert get_score_centering_is_config(a) == dict(mode="none")


def test_unknown_tis_callback_rejected_only_when_composing():
    a = args(use_tis=True, custom_tis_function_path="my_module.sequence_mask")
    with pytest.raises(ValueError, match="custom loss/mask callbacks"):
        validate_score_centering_args(a)
    a.use_tis = False
    validate_score_centering_args(a)
    a.use_score_centering, a.use_tis = False, True
    validate_score_centering_args(a)


def test_full_argument_validation_accepts_sc_tis_with_rollout_logprobs(monkeypatch):
    from test_megatron_argument_validation import load_slime_arguments_module, make_slime_validate_args

    module = load_slime_arguments_module(monkeypatch)
    a = make_slime_validate_args(
        use_score_centering=True,
        score_centering_top_k=128,
        use_tis=True,
        tis_clip_low=0.0,
        tis_clip=2.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        use_rollout_logprobs=True,
        loss_type="policy_loss",
        eval_resume_step=None,
    )
    module.slime_validate_args(a)
    assert a.use_tis and a.use_score_centering and a.use_rollout_logprobs


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_shared_weight_rule_preserves_legacy_callbacks(mode):
    from slime.backends.megatron_utils.loss import icepop_function, vanilla_tis_function

    a = args(tis_clip_low=0.5, tis_clip=2.0)
    ratios = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
    old, sampler = ratios.log() - 3, torch.full_like(ratios, -3)
    actual_ratios = (old - sampler).exp()
    pg = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], requires_grad=True)
    masks = [torch.ones(5)]
    fn = vanilla_tis_function if mode == "tis" else icepop_function
    loss, returned_masks, metrics = fn(
        a, pg_loss=pg, train_log_probs=[old], rollout_log_probs=[sampler], loss_masks=masks
    )
    expected_weights = (
        actual_ratios.clamp(0.5, 2.0)
        if mode == "tis"
        else torch.where((actual_ratios >= 0.5) & (actual_ratios <= 2.0), actual_ratios, 0)
    )
    torch.testing.assert_close(loss, pg * expected_weights, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(loss.sum(), pg)[0], expected_weights, atol=0, rtol=0)
    torch.testing.assert_close(metrics["tis_clipfrac"], (expected_weights != actual_ratios).float(), atol=0, rtol=0)
    assert returned_masks is masks


@pytest.mark.parametrize("layout", ["tp", "zigzag", "allgather"])
def test_distributed_disk_gradients(layout):
    torch.multiprocessing.spawn(
        distributed_worker, args=(2, _cp_dist_helpers.free_port(), layout, "none", True), nprocs=2
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
