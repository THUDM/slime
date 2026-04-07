from __future__ import annotations

from examples.MOPD.code_verifier import compute_score


async def reward_func(args, sample, **kwargs):
    if sample.status in {sample.Status.TRUNCATED, sample.Status.ABORTED, sample.Status.FAILED}:
        return 0.0

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    try:
        score, _ = compute_score(sample.response or "", metadata, continuous=True, max_partial_cases=10)
        return float(score)
    except Exception:
        return 0.0
