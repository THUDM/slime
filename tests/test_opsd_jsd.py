"""Unit tests for the OPSD (on-policy self-distillation) full-vocab JSD core.

These run on CPU with a single (no-op) process group, exercising the same code
path used under tensor/vocab parallelism (where the all-reduces become real).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from slime.utils.ppo_utils import compute_vocab_parallel_jsd


def _reference_jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, beta: float) -> torch.Tensor:
    """Independent dense reference using F.kl_div, mirroring OPSD's GOLD trainer.

    M = (1 - beta) * student + beta * teacher
    JSD = beta * KL(teacher || M) + (1 - beta) * KL(student || M)
    Endpoints: beta=0 -> KL(teacher || student); beta=1 -> KL(student || teacher).
    """
    s = F.log_softmax(student_logits, dim=-1)
    t = F.log_softmax(teacher_logits, dim=-1)
    if beta == 0.0:
        # F.kl_div(input, target, log_target) = sum target * (log target - input) = KL(target || exp(input))
        return F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
    if beta == 1.0:
        return F.kl_div(t, s, reduction="none", log_target=True).sum(-1)
    m = torch.logsumexp(torch.stack([s + math.log(1 - beta), t + math.log(beta)]), dim=0)
    kl_teacher = F.kl_div(m, t, reduction="none", log_target=True)
    kl_student = F.kl_div(m, s, reduction="none", log_target=True)
    return (beta * kl_teacher + (1 - beta) * kl_student).sum(-1)


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 1.0])
def test_jsd_value_matches_reference(beta):
    torch.manual_seed(0)
    student = torch.randn(7, 23, dtype=torch.float64)
    teacher = torch.randn(7, 23, dtype=torch.float64)
    got = compute_vocab_parallel_jsd(student, teacher, beta, process_group=None)
    expected = _reference_jsd(student, teacher, beta)
    assert torch.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 1.0])
def test_jsd_gradient_only_through_student(beta):
    torch.manual_seed(1)
    teacher = torch.randn(5, 17, dtype=torch.float64)

    student = torch.randn(5, 17, dtype=torch.float64, requires_grad=True)
    compute_vocab_parallel_jsd(student, teacher, beta, process_group=None).sum().backward()

    student_ref = student.detach().clone().requires_grad_(True)
    _reference_jsd(student_ref, teacher, beta).sum().backward()

    assert torch.allclose(student.grad, student_ref.grad, atol=1e-9)


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_jsd_temperature_matches_reference(temperature):
    torch.manual_seed(3)
    student = torch.randn(6, 19, dtype=torch.float64)
    teacher = torch.randn(6, 19, dtype=torch.float64)
    got = compute_vocab_parallel_jsd(student, teacher, 0.5, process_group=None, temperature=temperature)
    expected = _reference_jsd(student / temperature, teacher / temperature, 0.5)
    assert torch.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_jsd_non_negative_and_zero_for_identical(beta):
    torch.manual_seed(2)
    logits = torch.randn(4, 11)
    assert (compute_vocab_parallel_jsd(logits, logits.clone(), beta, process_group=None) >= -1e-6).all()
    assert torch.allclose(
        compute_vocab_parallel_jsd(logits, logits.clone(), beta, process_group=None),
        torch.zeros(4),
        atol=1e-6,
    )
