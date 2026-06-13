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
    """Dense reference following TRL's GOLD/GKD generalized JSD."""
    s = F.log_softmax(student_logits, dim=-1)
    t = F.log_softmax(teacher_logits, dim=-1)
    if beta == 0.0:
        return (t.exp() * (t - s)).sum(-1)
    if beta == 1.0:
        return (s.exp() * (s - t)).sum(-1)
    m = torch.logsumexp(torch.stack([s + math.log(beta), t + math.log(1 - beta)]), dim=0)
    return (beta * (t.exp() * (t - m)) + (1 - beta) * (s.exp() * (s - m))).sum(-1)


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
