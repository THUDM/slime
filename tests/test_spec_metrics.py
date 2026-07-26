"""CPU unit tests for ``slime.ray.rollout._compute_spec_metrics``.

``Sample.SpecInfo`` deliberately accumulates the four raw counters across
partial-rollout turns ("cannot directly use spec info from sglang because of
partial rollout") precisely so a correct batch statistic can be formed. The
batch metric must therefore pool those counters — ``Σaccept / Σdraft`` and
``Σcompletion / Σverify`` — the same way ``_compute_prefix_cache_metrics``
pools ``Σcached / Σprompt`` five lines below it.

The old implementation averaged the *per-sample ratios* instead, which

  * weights a 10-token response the same as a 4k-token one, and
  * counts samples whose counters were never populated (aborted / partial
    rollout: ``SpecInfo.add`` only runs on terminal meta info) as hard 0.0s,

both of which only ever bias the report downward — anyone tuning
``--sglang-speculative-*`` off these numbers optimizes against a deflated
signal.
"""

from __future__ import annotations

import sys
import types


# ``slime.ray.rollout`` imports sglang / sglang_router / wandb / transformers at
# module level; none are installed in the CPU CI env and none are touched by
# the pure functions under test. Stub just enough for the import to resolve
# (same approach as tests/test_agent/test_agent_rollout_cpu.py).
def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub("sglang_router", __version__="0.2.3")
_stub("sglang_router.launch_router", RouterArgs=object, launch_router=lambda *a, **k: None)
_stub("sglang")
_stub("sglang.srt")
_stub(
    "sglang.srt.constants",
    GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
    GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
    GPU_MEMORY_TYPE_WEIGHTS="weights",
)
_stub("sglang.srt.server_args", ServerArgs=object)
_stub("sglang.srt.utils", kill_process_tree=lambda *a, **k: None)
_stub("wandb")
_stub(
    "transformers",
    **{n: type(n, (), {}) for n in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin")},
)

import pytest

from slime.ray.rollout import _compute_spec_metrics
from slime.utils.types import Sample


NUM_GPUS = 0


class _SpecArgs:
    sglang_speculative_algorithm = "EAGLE"


def _sample(accept: int, draft: int, verify: int, completion: int) -> Sample:
    sample = Sample(index=0, prompt="p")
    sample.spec_info.spec_accept_token_num = accept
    sample.spec_info.spec_draft_token_num = draft
    sample.spec_info.spec_verify_ct = verify
    sample.spec_info.completion_token_num = completion
    return sample


@pytest.mark.unit
def test_disabled_without_speculative_algorithm():
    args = types.SimpleNamespace(sglang_speculative_algorithm=None)
    assert _compute_spec_metrics(args, [_sample(1, 2, 1, 2)]) == {}


@pytest.mark.unit
def test_pools_counters_across_samples():
    # 7 tiny responses + 1 long one: an unweighted mean of per-sample ratios
    # would report accept_rate (7*0.125 + 0.5)/8 = 0.172, accept_length
    # (7*1.0 + 3.0)/8 = 1.25 — both far below the pooled truth.
    samples = [_sample(1, 8, 1, 1) for _ in range(7)] + [_sample(2000, 4000, 1000, 3000)]

    metrics = _compute_spec_metrics(_SpecArgs(), samples)

    assert metrics["spec_accept_rate"] == pytest.approx((7 * 1 + 2000) / (7 * 8 + 4000))
    assert metrics["spec_accept_length"] == pytest.approx((7 * 1 + 3000) / (7 * 1 + 1000))


@pytest.mark.unit
def test_samples_without_counters_do_not_dilute():
    # SpecInfo.add only runs on terminal meta info, so aborted / partial
    # samples legitimately carry all-zero counters. They must not drag the
    # batch statistic toward zero.
    scored = [_sample(30, 50, 10, 22) for _ in range(6)]
    unscored = [_sample(0, 0, 0, 0) for _ in range(2)]

    metrics = _compute_spec_metrics(_SpecArgs(), scored + unscored)

    assert metrics["spec_accept_rate"] == pytest.approx(30 / 50)
    assert metrics["spec_accept_length"] == pytest.approx(22 / 10)


@pytest.mark.unit
def test_no_counters_at_all_reports_nothing():
    # All-aborted batch (or empty): no fake zeros, and no ZeroDivisionError.
    assert _compute_spec_metrics(_SpecArgs(), [_sample(0, 0, 0, 0)]) == {}
    assert _compute_spec_metrics(_SpecArgs(), []) == {}
