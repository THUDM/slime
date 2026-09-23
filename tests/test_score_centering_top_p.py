"""Exact SC on replay supports, checked against dense masked distributions."""

import _cp_dist_helpers
import numpy as np
import pytest
import torch
from test_score_centering import args, make_batch

from slime.utils.ppo_utils import calculate_ragged_log_probs
from slime.utils.score_centering import score_centering_request, validate_score_centering_args

NUM_GPUS = 0


def top_p_batch(top_p=None):
    batch = make_batch()
    batch.pop("rollout_topk_token_ids")
    batch.pop("rollout_topk_log_probs")
    batch.update(rollout_top_p_token_ids=[], rollout_top_p_token_offsets=[], rollout_top_p_log_probs=[])
    for i, r in enumerate(batch["response_lengths"]):
        ids, offsets, logps, sampled = [], [0], [], []
        for row, target in enumerate(batch["unconcat_tokens"][i][-r:]):
            if top_p is None:
                # Ragged sets, including a singleton and a masked environment row.
                support = target[None] if row == 0 else torch.unique(torch.cat((torch.arange(7), target[None])))
                q = torch.randn(len(support)).log_softmax(0)
            else:
                # Known nucleus boundaries: .9 retains one token, .95 retains
                # three, including the token that crosses the threshold.
                probs = torch.tensor([0.91, 0.03, 0.02, 0.01, 0.01, 0.01, 0.002, 0.002, 0.002, 0.001, 0.001, 0.002])
                ordered_ids = (torch.arange(12) + target) % 12
                sorted_probs, order = probs.sort(descending=True)
                keep = sorted_probs.cumsum(0) - sorted_probs <= top_p
                support = ordered_ids[order[keep]]
                q = (sorted_probs[keep] / sorted_probs[keep].sum()).log()
            sampled.append(q[support == target][0])
            if batch["loss_masks"][i][row]:
                ids.extend(support.tolist())
                logps.extend(q.tolist())
            offsets.append(len(ids))
        batch["rollout_top_p_token_ids"].append(torch.tensor(ids, dtype=torch.int32))
        batch["rollout_top_p_token_offsets"].append(torch.tensor(offsets, dtype=torch.int32))
        batch["rollout_top_p_log_probs"].append(torch.tensor(logps))
        batch["rollout_log_probs"][i] = torch.stack(sampled)
    return batch


def weight(ratio, a):
    if not a.use_tis:
        return torch.ones_like(ratio)
    if a.custom_tis_function_path:
        return torch.where((ratio >= a.tis_clip_low) & (ratio <= a.tis_clip), ratio, 0)
    return ratio.clamp(a.tis_clip_low, a.tis_clip)


def reference(logits, batch, a):
    result = logits.sum() * 0
    position = 0
    for i, (total, response) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        offsets = batch["rollout_top_p_token_offsets"][i]
        for row in range(response):
            if not batch["loss_masks"][i][row]:
                continue
            start, end = offsets[row : row + 2]
            ids = batch["rollout_top_p_token_ids"][i][start:end].long()
            q = batch["rollout_top_p_log_probs"][i][start:end].exp()
            values = logits[0, position + total - response - 1 + row].float() / a.rollout_temperature
            keep = torch.zeros_like(values, dtype=torch.bool).scatter_(0, ids, True)
            logp = values.masked_fill(~keep, -torch.inf).log_softmax(0)
            coeff = (q * weight(logp[ids].exp() / q, a)).detach()
            target = batch["unconcat_tokens"][i][-response + row]
            w = weight((logp[target] - batch["rollout_log_probs"][i][row]).exp(), a).detach()
            loss = -batch["advantages"][i][row] * (w * logp[target] - (coeff * logp[ids]).sum())
            result = result + loss / batch["loss_masks"][i].sum()
        position += total
    return result


@pytest.mark.parametrize("top_p", [0.9, 0.95])
def test_request_selects_complete_support(top_p):
    a = args(rollout_top_p=top_p)
    validate_score_centering_args(a)
    params = {"custom_params": {"other": 1}}
    assert score_centering_request(a, params) == {"return_logprob": True}
    assert params["top_p"] == top_p
    assert params["custom_params"] == {"other": 1, "return_top_p_token_ids": True, "return_top_p_log_probs": True}
    with pytest.raises(ValueError, match="configured"):
        score_centering_request(a, {"top_p": 0.8})


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_p", [0.9, 0.95])
def test_exact_loss_gradient(mode, dtype, top_p, monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import policy_loss_function

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    batch = top_p_batch(top_p=top_p) if top_p == 0.95 else top_p_batch()
    if top_p == 0.95:
        for offsets, mask in zip(batch["rollout_top_p_token_offsets"], batch["loss_masks"], strict=True):
            assert offsets.diff().tolist() == (3 * mask).tolist()
    a = args(mode=mode, rollout_top_p=top_p)
    logits = torch.randn(1, 16, 12, dtype=dtype).requires_grad_()
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    loss, metrics = policy_loss_function(a, batch, logits.float(), reducer)
    expected = reference(logits, batch, a)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    tol = 0.004 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, logits)[0], atol=tol, rtol=tol
    )
    assert metrics["sc_sampler_head_mass"] == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_singleton_support_has_zero_correction_and_policy_gradient(mode, dtype, monkeypatch):
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import (
        get_log_probs_and_entropy,
        get_rollout_top_p_logprob_kwargs,
        get_score_centering_terms,
        policy_loss_function,
    )

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    batch = top_p_batch()
    for i, response in enumerate(batch["response_lengths"]):
        batch["rollout_top_p_token_ids"][i] = batch["unconcat_tokens"][i][-response:].int()
        batch["rollout_top_p_token_offsets"][i] = torch.arange(response + 1, dtype=torch.int32)
        batch["rollout_top_p_log_probs"][i] = torch.zeros(response)
        batch["rollout_log_probs"][i] = torch.zeros(response)
    a = args(mode=mode, rollout_top_p=0.95)
    # Large differences in the unmasked logits must not affect singleton rows.
    logits = (100 * torch.randn(1, 16, 12, dtype=dtype)).requires_grad_()
    _, sampled = get_log_probs_and_entropy(
        logits.float(),
        args=a,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        **get_rollout_top_p_logprob_kwargs(a, batch),
    )
    for logp in sampled["log_probs"]:
        torch.testing.assert_close(logp, torch.zeros_like(logp), atol=0, rtol=0)
    terms = get_score_centering_terms(a, batch, logits.float())
    torch.testing.assert_close(terms["sc_correction"], torch.zeros(5), atol=0, rtol=0)
    for key in ("sc_sampler_head_mass", "sc_train_head_mass"):
        torch.testing.assert_close(terms[key], torch.ones(5), atol=0, rtol=0)
    correction_grad = torch.autograd.grad(terms["sc_correction"].sum(), logits)[0]
    torch.testing.assert_close(correction_grad, torch.zeros_like(logits), atol=0, rtol=0)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
    )
    loss, _ = policy_loss_function(a, batch, logits.float(), reducer)
    assert loss.item() == 0
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad, torch.zeros_like(logits), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_exact_correction_cancels_expected_drift(mode):
    a = args(mode=mode)
    logits = torch.tensor([[1.0, -2, 0.3, 0.8]], requires_grad=True)
    ids, offsets = torch.tensor([0, 2, 3]), torch.tensor([0, 3])
    q = torch.tensor([0.1, 0.3, 0.6])
    logp = calculate_ragged_log_probs(logits, ids, offsets, None, 0.8)
    w = weight(logp.exp() / q, a).detach()
    correction = (q * w * logp).sum()
    expected_loss = (q * (-w * logp + correction)).sum()
    torch.testing.assert_close(
        torch.autograd.grad(expected_loss, logits)[0], torch.zeros_like(logits), atol=2e-7, rtol=0
    )


def distributed_worker(rank, world_size, port, layout, mode):
    import torch.distributed as dist
    from megatron.core import mpu

    torch.set_num_threads(1)
    group = _cp_dist_helpers.init_worker_process_group(rank, world_size, port)
    is_tp = layout == "tp"
    _cp_dist_helpers.stub_megatron_in_worker(1 if is_tp else world_size, 0 if is_tp else rank)
    singles = [dist.new_group([i]) for i in range(world_size)]
    mpu.get_tensor_model_parallel_group = lambda: group if is_tp else singles[rank]
    mpu.get_context_parallel_group = lambda: group
    mpu.get_tensor_model_parallel_rank = lambda: rank if is_tp else 0
    try:
        from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
        from slime.backends.megatron_utils.loss import policy_loss_function

        batch = top_p_batch()
        full_logits = torch.randn(16, 12)
        a = args(mode=mode, rollout_top_p=0.9, allgather_cp=layout == "allgather")
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
        if not is_tp:
            for key in ["advantages", "rollout_log_probs"]:
                batch[key] = [
                    slice_log_prob_with_cp(x, t, r)
                    for x, t, r in zip(batch[key], batch["total_lengths"], batch["response_lengths"], strict=True)
                ]
        reducer = get_sum_of_sample_mean(
            batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], batch["rollout_mask_sums"]
        )
        loss, _ = policy_loss_function(a, batch, local.unsqueeze(0), reducer)
        ref_logits = full_logits.clone().requires_grad_()
        expected_loss = reference(ref_logits.unsqueeze(0), original, a)
        loss.backward()
        expected_loss.backward()
        expected = (
            ref_logits.grad[:, rank * 6 : (rank + 1) * 6]
            if is_tp
            else ref_logits.grad[rank * 8 : (rank + 1) * 8] if layout == "allgather" else ref_logits.grad[row_ids]
        )
        torch.testing.assert_close(local.grad, expected, atol=2e-6, rtol=2e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["tp", "zigzag", "allgather"])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_distributed_exact_gradients(layout, mode):
    torch.multiprocessing.spawn(distributed_worker, args=(2, _cp_dist_helpers.free_port(), layout, mode), nprocs=2)


def binary_top_p_meta():
    import pybase64

    def encode(values, dtype):
        return pybase64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode()

    return dict(
        top_p_token_ids=encode([1, 4, 2], "<i4"),
        top_p_token_offsets=encode([0, 2, 3], "<i4"),
        top_p_log_probs=encode(np.log([0.3, 0.7, 1.0]), "<f4"),
        output_token_logprobs=[[float(np.log(0.7)), 4, None], [0.0, 2, None]],
    )


def test_binary_resume_and_masked_environment():
    import json
    from slime.utils.score_centering import decode_score_centering_response, validate_sampler_top_p
    from slime.utils.types import Sample

    a = args(rollout_top_p=0.9)
    sample = Sample(tokens=[8])
    sample.append_response_tokens(a, tokens=[9], trainable=False)
    for _ in range(2):
        meta = decode_score_centering_response(json.dumps({"meta_info": binary_top_p_meta()}), 0)["meta_info"]
        sample.append_response_tokens(a, tokens=[4, 2], log_probs=[float(np.log(0.7)), 0], meta_info=meta)
        sample.append_response_tokens(a, tokens=[9], trainable=False)
        sample = Sample.from_dict(sample.to_dict())
    assert sample.loss_mask == [0, 1, 1, 0, 1, 1, 0]
    assert sample.rollout_top_p_token_offsets.tolist() == [0, 0, 2, 3, 3, 5, 6, 6]
    validate_sampler_top_p(
        sample.rollout_top_p_token_ids,
        sample.rollout_top_p_token_offsets,
        sample.rollout_top_p_log_probs,
        sample.response_length,
        sample.loss_mask,
        sample.tokens[-sample.response_length :],
        sample.rollout_log_probs,
    )
    assert sample.rollout_topk_token_ids is None


@pytest.mark.parametrize("corruption", ["missing", "partial", "duplicate", "offsets", "nan", "sampled"])
def test_invalid_support_rejected(corruption):
    from slime.utils.score_centering import validate_sampler_top_p

    ids, offsets, q = [1, 2], [0, 2], np.log([0.3, 0.7])
    if corruption == "missing":
        q = None
    elif corruption == "partial":
        q = np.log([0.2, 0.4])
    elif corruption == "duplicate":
        ids = [1, 1]
    elif corruption == "offsets":
        offsets = [0, 1]
    elif corruption == "nan":
        q[0] = np.nan
    with pytest.raises(ValueError):
        validate_sampler_top_p(
            ids, offsets, q, 1, tokens=[3] if corruption == "sampled" else [2], sampled_logps=[float(np.log(0.7))]
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
