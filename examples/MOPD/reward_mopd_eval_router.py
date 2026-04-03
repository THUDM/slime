"""MOPD eval reward router — dispatches by domain to the correct verifier.

math       → reward_deepmath_mathverify  (math-verify)
code       → reward_code_execution       (code sandbox)
tool/stem/structured → reward_multidomain_v1 (tool-call / mcqa / json-schema)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make jl_workspace math/code reward modules importable.
_AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
_JL_MATH_DIR = _AVALANCHE_ROOT / "jl_workspace" / "experiment" / "math"
_JL_CODE_DIR = _AVALANCHE_ROOT / "jl_workspace" / "experiment" / "code"
for _p in (_JL_MATH_DIR, _JL_CODE_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from reward_code_execution import reward_func as _code_reward  # noqa: E402
from reward_deepmath_mathverify import reward_func as _math_reward  # noqa: E402
from reward_multidomain_v1 import reward_func as _mdv1_reward  # noqa: E402


def _infer_domain(sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("domain") or "").strip().lower()


async def reward_func(args, sample, **kwargs):
    domain = _infer_domain(sample)
    if domain == "math":
        return await _math_reward(args, sample, **kwargs)
    if domain == "code":
        return await _code_reward(args, sample, **kwargs)
    # tool, stem, structured — handled by multidomain_v1 via reward_type
    return await _mdv1_reward(args, sample, **kwargs)
