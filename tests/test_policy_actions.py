"""Native GRPO/OPD integration for chronological multi-codebook trajectories."""

from argparse import Namespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.policy_actions import (
    build_policy_action_view,
    store_old_action_scores,
    validate_action_policy_args,
)

NUM_GPUS = 0


def arguments(**overrides):
    defaults = dict(
        custom_megatron_log_prob_function_path="test.extractor",
        policy_stop_loss_weight=1.0,
        advantage_estimator="grpo",
        loss_type="policy_loss",
        pg_loss_type=None,
        use_rollout_logprobs=False,
        use_opd=False,
        opd_type="sglang",
        opd_kl_coef=0.4,
        kl_coef=0.0,
        custom_advantage_function_path=None,
        normalize_advantages=False,
        use_score_centering=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        allgather_cp=False,
        recompute_loss_function=False,
    )
    return Namespace(**(defaults | overrides))


def sample(rows=5, frames=2, terminal=True, advantage=1.0, codebooks=2):
    start = rows - frames - 1
    code_mask = torch.zeros(rows, codebooks, dtype=torch.bool)
    code_mask[start : start + frames] = True
    stop_mask = torch.zeros(rows, dtype=torch.bool)
    stop_mask[start : start + frames + int(terminal)] = True
    stops = torch.zeros(rows, dtype=torch.long)
    if terminal:
        stops[rows - 1] = 1
    # Different scores per action detect accidental broadcasting and summing.
    old_codes = -torch.arange(1, rows * codebooks + 1).reshape(rows, codebooks).float() / 10 - 1
    old_stop = -torch.arange(1, rows + 1).float() / 10 - 1
    mm = {
        "policy_code_mask": code_mask,
        "policy_stop_mask": stop_mask,
        "policy_stop_actions": stops,
        "policy_old_code_logprobs": old_codes,
        "policy_old_stop_logprobs": old_stop,
        "policy_rollout_code_logprobs": old_codes - 0.03,
        "policy_rollout_stop_logprobs": old_stop - 0.07,
        "policy_group_advantage": torch.tensor([advantage]),
    }
    delta = torch.linspace(-0.3, 0.35, rows * (codebooks + 1)).reshape(rows, codebooks + 1)
    output = (torch.cat((old_codes, old_stop[:, None]), dim=-1) + delta).requires_grad_()
    return {"mm": mm, "output": output, "rows": rows, "frames": frames}


def batch_for(samples, padding=0):
    mm = {key: torch.cat([s["mm"][key] for s in samples]) for key in samples[0]["mm"]}
    outputs = torch.cat([s["output"] for s in samples])
    if padding:
        outputs = torch.cat((outputs, torch.full((padding, outputs.shape[-1]), 99.0)))
    batch = {
        "multimodal_train_inputs": mm,
        "total_lengths": [s["rows"] for s in samples],
        "response_lengths": [s["frames"] for s in samples],
        "loss_masks": [torch.ones(s["frames"]) for s in samples],
        "rollout_mask_sums": torch.tensor([float(s["frames"]) for s in samples]),
        "advantages": [torch.full((s["frames"],), s["mm"]["policy_group_advantage"].item()) for s in samples],
        "log_probs": [torch.zeros(s["frames"]) for s in samples],
    }
    return batch, outputs[None]


def extract(args, batch, output):
    codes, stops, summaries = [], [], []
    offset = 0
    for rows, frames in zip(batch["total_lengths"], batch["response_lengths"], strict=True):
        part = output[0, offset : offset + rows]
        codes.append(part[:, :-1])
        stops.append(part[:, -1])
        start = rows - frames - 1
        summaries.append(part[start : start + frames, :-1].sum(-1))
        offset += rows
    return {"log_probs": summaries, "policy_code_logprobs": codes, "policy_stop_logprobs": stops}


@pytest.fixture(autouse=True)
def native_eager(monkeypatch):
    from slime.utils import misc

    monkeypatch.setattr(misc, "load_function", lambda path: extract)
    # Exercise the production operator body without CPU CI compilation cost.
    native = loss_module.compute_policy_loss
    monkeypatch.setattr(loss_module, "compute_policy_loss", getattr(native, "__wrapped__", native))
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1, raising=False)


def run_policy(args, samples, padding=0):
    batch, output = batch_for(samples, padding)
    value, metrics = loss_module.policy_loss_function(args, batch, output, lambda x: x.sum())
    return value, metrics


def independent_reference(args, samples):
    result = samples[0]["output"].new_zeros(())
    for s in samples:
        mm = s["mm"]
        for column, kind in ((slice(None, -1), "code"), (-1, "stop")):
            current = s["output"][:, column][mm[f"policy_{kind}_mask"]]
            if current.numel() == 0:
                continue
            source = "rollout" if args.use_rollout_logprobs else "old"
            old = mm[f"policy_{source}_{kind}_logprobs"][mm[f"policy_{kind}_mask"]]
            adv = mm["policy_group_advantage"].expand_as(current)
            if args.use_opd:
                teacher = mm[f"policy_teacher_{kind}_logprobs"][mm[f"policy_{kind}_mask"]]
                adv = adv - args.opd_kl_coef * (old - teacher)
            ratio = (current - old).exp()
            clipped = ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip_high)
            objective = torch.maximum(-adv * ratio, -adv * clipped).mean()
            result = result + objective * (args.policy_stop_loss_weight if kind == "stop" else 1)
    return result


@pytest.mark.parametrize("frames,terminal", [(2, True), (2, False), (0, True)])
def test_native_policy_actions_and_gradients(frames, terminal):
    args = arguments(policy_stop_loss_weight=0.7)
    s = sample(frames=frames, terminal=terminal, advantage=-1.2)
    value, metrics = run_policy(args, [s], padding=7)
    ref = independent_reference(args, [s])
    torch.testing.assert_close(value, ref)
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.autograd.grad(ref, s["output"])[0])
    assert "entropy_loss" not in metrics  # Selected logp is not distribution entropy.
    assert s["frames"] == frames  # No fabricated RVQ or expanded response_length.


def test_native_opd_is_applied_to_selected_stop_and_code(monkeypatch):
    args = arguments(use_opd=True)
    samples = [sample(advantage=0.0), sample(rows=3, frames=0, advantage=0.0)]
    for s in samples:
        for kind in ("code", "stop"):
            old = s["mm"][f"policy_old_{kind}_logprobs"]
            s["mm"][f"policy_teacher_{kind}_logprobs"] = old + torch.linspace(-0.8, 0.9, old.numel()).reshape_as(old)
    calls = []
    native = loss_module.apply_opd_kl_to_advantages

    def spy(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(loss_module, "apply_opd_kl_to_advantages", spy)
    value, metrics = run_policy(args, samples)
    torch.testing.assert_close(value, independent_reference(args, samples))
    assert calls == [True]
    assert "opd_reverse_kl" in metrics


def test_packed_and_microbatch_gradients_are_equal():
    args = arguments(policy_stop_loss_weight=1.4)
    samples = [sample(advantage=0.4), sample(rows=4, frames=2, terminal=False, advantage=-1.2), sample(rows=3, frames=0, advantage=0.9)]
    packed, _ = run_policy(args, samples, padding=4)
    separate = sum(run_policy(args, [s])[0] for s in samples)
    torch.testing.assert_close(packed, separate)
    targets = [s["output"] for s in samples]
    for pgrad, sgrad in zip(torch.autograd.grad(packed, targets), torch.autograd.grad(separate, targets), strict=True):
        torch.testing.assert_close(pgrad, sgrad)


def test_prism_extractor_integrates_with_native_policy_loss(monkeypatch):
    from slime.utils import misc
    from slime_plugins.models.prism import extract_policy_log_probs

    monkeypatch.setattr(misc, "load_function", lambda path: extract_policy_log_probs)
    args = arguments()
    samples = [sample(), sample(rows=3, frames=0, advantage=-0.5)]
    value, _ = run_policy(args, samples, padding=4)
    torch.testing.assert_close(value, independent_reference(args, samples))


def test_existing_megatron_wrapper_preserves_global_sample_mean(monkeypatch):
    args = arguments()
    samples = [sample(advantage=0.4), sample(rows=3, frames=0, advantage=-0.8)]
    packed_batch, packed_output = batch_for(samples)
    packed, _, _ = loss_module.loss_function(args, packed_batch, 1, 2, packed_output)
    micros = []
    for s in samples:
        batch, output = batch_for([s])
        micros.append(loss_module.loss_function(args, batch, 2, 2, output)[0])
    # Megatron averages the microbatch losses at the end of its schedule.
    torch.testing.assert_close(packed, sum(micros) / 2)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    shards = []
    for s in samples:
        batch, output = batch_for([s])
        shards.append(loss_module.loss_function(args, batch, 1, 2, output)[0])
    # DDP averages rank gradients; both shards still contribute one sample.
    torch.testing.assert_close(packed, sum(shards) / 2)


def test_old_scores_are_detached_and_frozen_across_updates():
    s = sample()
    batch, output = batch_for([s])
    scores = extract(arguments(), batch, output)
    rollout = {"multimodal_train_inputs": [s["mm"]]}
    store_old_action_scores(rollout, scores)
    old = s["mm"]["policy_old_code_logprobs"].clone()
    assert not s["mm"]["policy_old_code_logprobs"].requires_grad
    with torch.no_grad():
        s["output"].add_(0.2)
    torch.testing.assert_close(s["mm"]["policy_old_code_logprobs"], old)


def test_zero_frame_gets_native_grpo_scalar_and_skips_frame_opd():
    args = arguments(use_opd=True)
    samples = [sample(advantage=0.0), sample(rows=3, frames=0, advantage=0.0)]
    rollout = {
        "multimodal_train_inputs": [s["mm"] for s in samples],
        "log_probs": [torch.zeros(s["frames"]) for s in samples],
        "loss_masks": [torch.ones(s["frames"]) for s in samples],
        "rewards": [1.3, -0.4],
        "response_lengths": [s["frames"] for s in samples],
        "total_lengths": [s["rows"] for s in samples],
    }
    # No frame teacher_log_probs: action teacher scores live on a distinct axis.
    loss_module.compute_advantages_and_returns(args, rollout)
    torch.testing.assert_close(samples[1]["mm"]["policy_group_advantage"], torch.tensor([-0.4]))
    assert rollout["advantages"][1].shape == (0,)


def test_rollout_source_is_configurable():
    args = arguments(use_rollout_logprobs=True)
    s = sample()
    value, _ = run_policy(args, [s])
    torch.testing.assert_close(value, independent_reference(args, [s]))


def test_missing_teacher_or_frozen_old_fails():
    s = sample()
    with pytest.raises(ValueError, match="Missing selected teacher"):
        run_policy(arguments(use_opd=True), [s])
    del s["mm"]["policy_old_code_logprobs"]
    with pytest.raises(ValueError, match="Missing selected old"):
        run_policy(arguments(), [s])


@pytest.mark.parametrize("config", [dict(use_tis=True), dict(advantage_estimator="gspo"), dict(normalize_advantages=True), dict(kl_coef=0.1), dict(context_parallel_size=2)])
def test_unadapted_token_features_fail_explicitly(config):
    with pytest.raises(ValueError):
        validate_action_policy_args(arguments(**config))


def test_entropy_requires_actual_distribution_scores():
    s = sample()
    batch, output = batch_for([s])
    with pytest.raises(ValueError, match="actual per-action entropy"):
        build_policy_action_view(arguments(entropy_coef=0.01), batch, extract(arguments(), batch, output))


@pytest.mark.parametrize("stop_weight", [0.0, 0.7, 1.0, 3.0])
def test_clip_fraction_counts_real_actions_independently_of_stop_weight(stop_weight):
    s = sample(advantage=1.0)
    mm = s["mm"]
    with torch.no_grad():
        s["output"].copy_(torch.cat((mm["policy_old_code_logprobs"], mm["policy_old_stop_logprobs"][:, None]), -1))
        s["output"].add_(torch.tensor(1.5).log())
    _, metrics = run_policy(arguments(policy_stop_loss_weight=stop_weight), [s])
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["ppo_kl"], -torch.tensor(1.5).log())


@pytest.mark.parametrize("frames", [0, 2])
def test_removed_trajectory_has_no_code_or_terminal_stop_gradient(frames):
    s = sample(frames=frames)
    s["mm"]["policy_sample_mask"] = torch.zeros(1, dtype=torch.bool)
    value, metrics = run_policy(arguments(), [s])
    torch.testing.assert_close(value, torch.tensor(0.0))
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.zeros_like(s["output"]))
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(0.0))


def test_native_all_zero_frame_mask_removes_terminal_stop_too():
    s = sample()
    batch, output = batch_for([s])
    batch["loss_masks"][0].zero_()
    value, _ = loss_module.policy_loss_function(arguments(), batch, output, lambda x: x.sum())
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.zeros_like(s["output"]))


def test_partial_frame_mask_is_rejected_until_action_semantics_are_defined():
    s = sample()
    batch, output = batch_for([s])
    batch["loss_masks"][0][0] = 0
    with pytest.raises(ValueError, match="Partial frame loss masks"):
        loss_module.policy_loss_function(arguments(), batch, output, lambda x: x.sum())
