"""CPU unit tests for boxed_ prefix handling in ``async_rm``.

``rm_type="boxed_math"`` used to pre-strip ``\\boxed{}`` and then hand the
bare string to ``grade_answer_verl``, which looks for ``\\boxed`` again and
returns False. Types that extract boxed answers themselves (math / dapo /
deepscaler) must skip that pre-extract; types that do not (f1, gpqa, ...)
still get it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.rm_hub import async_rm
from slime.utils.types import Sample


NUM_GPUS = 0


def _args(rm_type: str):
    return SimpleNamespace(custom_rm_path=None, rm_type=rm_type)


def _sample(response: str, label: str) -> Sample:
    return Sample(response=response, label=label, metadata={})


def _reward(rm_type: str, response: str, label: str):
    return asyncio.run(async_rm(_args(rm_type), _sample(response, label)))


@pytest.mark.unit
def test_boxed_math_scores_correct_boxed_answer():
    assert _reward("boxed_math", r"Answer: \boxed{5}", "5") == 1


@pytest.mark.unit
def test_boxed_math_scores_wrong_boxed_answer_zero():
    assert _reward("boxed_math", r"Answer: \boxed{5}", "6") == 0


@pytest.mark.unit
def test_math_still_requires_boxed_wrapper():
    """Bare ``rm_type="math"`` must keep requiring ``\\boxed``; do not
    loosen ``grade_answer_verl`` to accept unboxed strings."""
    assert _reward("math", "5", "5") == 0


@pytest.mark.unit
def test_boxed_math_with_no_boxed_content_is_zero():
    assert _reward("boxed_math", "just 5", "5") == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
