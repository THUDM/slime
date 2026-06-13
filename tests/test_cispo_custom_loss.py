from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.cispo.cispo_loss import (
    _get_cispo_config,
    _get_sampling_log_probs,
    _validate_supported_options,
    cispo_loss_function,
    compute_cispo_loss,
)

NUM_GPUS = 0


def test_cispo_loss_matches_detached_ratio_formula():
    target_log_probs = torch.tensor([0.0, 0.5, -1.0], requires_grad=True)
    sampling_log_probs = torch.tensor([0.0, -0.5, -1.0])
    advantages = torch.tensor([1.0, -2.0, 0.25])
    ppo_kl = sampling_log_probs - target_log_probs

    loss, clipfrac = compute_cispo_loss(ppo_kl, target_log_probs, advantages, 0.0, 1.5)

    expected_ratio = torch.exp(target_log_probs.detach() - sampling_log_probs)
    expected_clipped_ratio = torch.clamp(expected_ratio, min=0.0, max=1.5)
    expected_loss = -expected_clipped_ratio * advantages * target_log_probs.detach()

    assert torch.allclose(loss.detach(), expected_loss)
    assert torch.equal(clipfrac, torch.tensor([0.0, 1.0, 0.0]))

    loss.sum().backward()
    assert torch.allclose(target_log_probs.grad, -expected_clipped_ratio * advantages)


def test_cispo_config_reads_loss_config_namespace():
    args = Namespace(loss_config={"cispo": {"clip_low_threshold": 0.1, "clip_high_threshold": 3.0}})
    assert _get_cispo_config(args) == pytest.approx((0.1, 3.0))


def test_cispo_config_requires_yaml_thresholds():
    with pytest.raises(KeyError, match="clip_low_threshold"):
        _get_cispo_config(Namespace())


def test_cispo_sampling_log_probs_fall_back_without_rollout_log_probs():
    old_log_probs = [torch.tensor([0.1, 0.2])]
    target_log_probs = [torch.tensor([0.3, 0.4])]

    assert (
        _get_sampling_log_probs(
            Namespace(use_rollout_logprobs=True),
            {"log_probs": old_log_probs},
            target_log_probs,
        )
        is old_log_probs
    )


def test_cispo_validation_rejects_tis():
    with pytest.raises(ValueError, match="use-tis"):
        _validate_supported_options(Namespace(use_tis=True))


def test_cispo_validation_allows_mismatch_metrics_and_opsm():
    _validate_supported_options(Namespace(use_tis=False, get_mismatch_metrics=True, use_opsm=True))


def test_cispo_custom_loss_reports_opd_reverse_kl(monkeypatch):
    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {
            "log_probs": [torch.tensor([0.1, 0.2], requires_grad=True)],
            "entropy": [torch.tensor([0.0, 0.0])],
        }

    monkeypatch.setattr("examples.cispo.cispo_loss.get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    args = Namespace(
        entropy_coef=0.0,
        loss_config={"cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}},
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([0.0, 0.0])],
        "opd_reverse_kl": [torch.tensor([0.3, 0.5])],
        "response_lengths": [2],
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    _, reported_loss = cispo_loss_function(args, batch, torch.zeros(1), lambda x: x.mean())

    assert reported_loss["opd_reverse_kl"] == pytest.approx(torch.tensor(0.4))


def test_cispo_custom_loss_supports_kl_loss(monkeypatch):
    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {
            "log_probs": [torch.tensor([0.1, 0.2], requires_grad=True)],
            "entropy": [torch.tensor([0.0, 0.0])],
        }

    monkeypatch.setattr("examples.cispo.cispo_loss.get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    args = Namespace(
        entropy_coef=0.0,
        kl_loss_coef=2.0,
        kl_loss_type="k1",
        loss_config={"cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}},
        use_kl_loss=True,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_tis=False,
        use_unbiased_kl=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([0.0, 0.0])],
        "ref_log_probs": [torch.tensor([0.0, 0.0])],
        "response_lengths": [2],
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    _, reported_loss = cispo_loss_function(args, batch, torch.zeros(1), lambda x: x.mean())

    assert reported_loss["kl_loss"].item() == pytest.approx(0.15)
    assert reported_loss["loss"].item() == pytest.approx(
        reported_loss["pg_loss"].item() + 2.0 * reported_loss["kl_loss"].item()
    )


def test_cispo_custom_loss_supports_opsm(monkeypatch):
    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {
            "log_probs": [torch.tensor([0.1, 0.2], requires_grad=True)],
            "entropy": [torch.tensor([0.0, 0.0])],
        }

    def fake_compute_opsm_mask(**kwargs):
        return torch.tensor([0.0, 1.0]), torch.tensor(0.5)

    monkeypatch.setattr("examples.cispo.cispo_loss.get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr("examples.cispo.cispo_loss.all_gather_with_cp", lambda x, *args, **kwargs: x)
    monkeypatch.setattr("examples.cispo.cispo_loss.compute_opsm_mask", fake_compute_opsm_mask)

    args = Namespace(
        entropy_coef=0.0,
        loss_config={"cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}},
        use_kl_loss=False,
        use_opsm=True,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([0.0, 0.0])],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "response_lengths": [2],
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    _, reported_loss = cispo_loss_function(args, batch, torch.zeros(1), lambda x: x.mean())

    expected_pg_loss = -(torch.exp(torch.tensor(0.2)) * torch.tensor(0.2)) / 2
    assert reported_loss["pg_loss"].item() == pytest.approx(expected_pg_loss.item())
    assert reported_loss["opsm_clipfrac"].item() == pytest.approx(0.5)


def test_cispo_custom_loss_reports_rollout_metric_like_policy_loss(monkeypatch):
    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {
            "log_probs": [torch.tensor([0.1, 0.2], requires_grad=True)],
            "entropy": [torch.tensor([0.0, 0.0])],
        }

    monkeypatch.setattr("examples.cispo.cispo_loss.get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    args = Namespace(
        entropy_coef=0.0,
        loss_config={"cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}},
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=True,
        use_tis=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([10.0, 10.0])],
        "response_lengths": [2],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2])],
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    _, reported_loss = cispo_loss_function(args, batch, torch.zeros(1), lambda x: x.mean())

    assert reported_loss["train_rollout_logprob_abs_diff"].item() == pytest.approx(0.0)


def test_cispo_custom_loss_uses_custom_pg_loss_reducer(monkeypatch):
    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {
            "log_probs": [torch.tensor([0.1, 0.2], requires_grad=True)],
            "entropy": [torch.tensor([0.0, 0.0])],
        }

    def fake_load_function(path):
        assert path == "tests.custom_reducer"

        def custom_reducer(total_lengths, response_lengths, loss_masks, calculate_per_token_loss):
            return lambda x: x.sum()

        return custom_reducer

    monkeypatch.setattr("examples.cispo.cispo_loss.get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr("examples.cispo.cispo_loss.load_function", fake_load_function)

    args = Namespace(
        calculate_per_token_loss=False,
        custom_pg_loss_reducer_function_path="tests.custom_reducer",
        entropy_coef=0.0,
        get_mismatch_metrics=False,
        loss_config={"cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}},
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([0.0, 0.0])],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "response_lengths": [2],
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    _, reported_loss = cispo_loss_function(args, batch, torch.zeros(1), lambda x: x.mean())

    expected = -(torch.exp(torch.tensor([0.1, 0.2])) * torch.tensor([0.1, 0.2])).sum()
    assert reported_loss["pg_loss"].item() == pytest.approx(expected.item())


def test_cispo_custom_loss_reports_mismatch_metrics(monkeypatch):
    def fake_get_log_probs_and_entropy(*args, **kwargs):
        return None, {
            "log_probs": [torch.tensor([0.1, 0.2], requires_grad=True)],
            "entropy": [torch.tensor([0.0, 0.0])],
        }

    def fake_load_function(path):
        assert path == "tests.mismatch"

        def mismatch_func(**kwargs):
            return None, kwargs["loss_masks"], {"mis_metric": torch.tensor([0.2, 0.4])}

        return mismatch_func

    def fake_get_sum_of_sample_mean(*args, **kwargs):
        return lambda x: x.mean()

    monkeypatch.setattr("examples.cispo.cispo_loss.get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr("examples.cispo.cispo_loss.load_function", fake_load_function)
    monkeypatch.setattr("examples.cispo.cispo_loss.get_sum_of_sample_mean", fake_get_sum_of_sample_mean)

    args = Namespace(
        calculate_per_token_loss=False,
        custom_pg_loss_reducer_function_path=None,
        custom_tis_function_path="tests.mismatch",
        entropy_coef=0.0,
        get_mismatch_metrics=True,
        loss_config={"cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}},
        qkv_format="thd",
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([0.0, 0.0])],
        "loss_masks": [torch.tensor([1.0, 1.0])],
        "response_lengths": [2],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2])],
        "rollout_mask_sums": [torch.tensor(2.0)],
        "total_lengths": [2],
        "unconcat_tokens": [torch.tensor([1, 2])],
    }

    _, reported_loss = cispo_loss_function(args, batch, torch.zeros(1), lambda x: x.mean())

    assert reported_loss["mis_metric"].item() == pytest.approx(0.3)
    assert "ois" in reported_loss


def test_cispo_custom_loss_function_is_loadable():
    assert callable(cispo_loss_function)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
