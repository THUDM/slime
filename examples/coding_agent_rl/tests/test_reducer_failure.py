"""Tests for §4.2 generate.py fan-out + U4 fail-soft reducer wrapper.

Covers SPEC §7.1 entry `test_reducer_failure.py`:
  * import path bad -> _load_reducer falls back to default + logs error
  * reducer raises -> sample marked abort
  * reducer returns None / non-list -> sample marked abort
  * default uniform fan-out splits reward by K
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Minimal Sample stand-in so we can import generate.py without slime.utils
# ---------------------------------------------------------------------------
class _Status:
    ABORTED = "ABORTED"
    COMPLETED = "COMPLETED"


class _FakeSample:
    Status = _Status

    def __init__(self):
        self.tokens = []
        self.response = ""
        self.response_length = 0
        self.loss_mask = []
        self.reward = 0.0
        self.status = ""
        self.metadata: dict | None = None
        self.session_id = "sid"
        self.label = "lbl"
        self.prompt = "p"


# Inject our fake Sample BEFORE importing generate.py
_fake_types = types.ModuleType("slime.utils.types")
_fake_types.Sample = _FakeSample
sys.modules.setdefault("slime.utils.types", _fake_types)

_fake_misc = types.ModuleType("slime.utils.misc")


class _SingletonMeta(type):
    _inst: dict = {}

    def __call__(cls, *a, **kw):
        if cls not in cls._inst:
            cls._inst[cls] = super().__call__(*a, **kw)
        return cls._inst[cls]


_fake_misc.SingletonMeta = _SingletonMeta
sys.modules.setdefault("slime.utils.misc", _fake_misc)

_fake_proc = types.ModuleType("slime.utils.processing_utils")
_fake_proc.load_tokenizer = lambda *a, **kw: None
sys.modules.setdefault("slime.utils.processing_utils", _fake_proc)

# Load generate.py as a stand-alone module (bypass the package-level
# `from . import middleware, sandbox` relative import that would otherwise
# require the heavy slime trainer deps).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_gen_under_test",
    str(Path(__file__).resolve().parents[1] / "generate.py"),
)
gen = _ilu.module_from_spec(_spec)
# Stub the relative imports BEFORE exec_module runs.
import middleware as _mw_stub  # noqa: E402
import sandbox as _sb_stub  # noqa: E402
sys.modules["_gen_under_test.middleware"] = _mw_stub
sys.modules["_gen_under_test.sandbox"] = _sb_stub
_pkg = types.ModuleType("_gen_pkg")
_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
_pkg.middleware = _mw_stub
_pkg.sandbox = _sb_stub
sys.modules["_gen_pkg"] = _pkg
_spec2 = _ilu.spec_from_file_location(
    "_gen_pkg.generate",
    str(Path(__file__).resolve().parents[1] / "generate.py"),
)
gen = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(gen)


class _DummyTok:
    def decode(self, ids, skip_special_tokens=False):
        return "decoded"


class _Args:
    swe_segment_reducer_path = None


def test_default_uniform_fan_out_splits_reward() -> None:
    segs = [
        ([1, 2], [10, 11], [1, 1], {"segment_kind": "subagent", "completed_turns": 3,
                                      "finish_reason": "end_turn", "num_aborts": 0,
                                      "tito_masked_turns": 0}),
        ([1, 2], [20, 21], [1, 1], {"segment_kind": "final", "completed_turns": 5,
                                      "finish_reason": "stop", "num_aborts": 0,
                                      "tito_masked_turns": 0}),
    ]
    proto = _FakeSample()
    proto.metadata = {"foo": "bar"}
    out = gen._default_uniform_fan_out(segs, 1.0, proto, _DummyTok(), "inst-1")
    assert len(out) == 2
    assert out[0].reward == 0.5
    assert out[1].reward == 0.5
    assert out[0].metadata["segment_kind"] == "subagent"
    assert out[1].metadata["segment_kind"] == "final"
    assert out[0].metadata["instance_id"] == "inst-1"
    assert out[0].metadata["segment_idx"] == 0
    assert out[1].metadata["segment_idx"] == 1
    assert out[0].metadata["num_segments"] == 2
    assert out[0].metadata["foo"] == "bar"  # carry-over


def test_load_reducer_bad_path_falls_back() -> None:
    class Args:
        swe_segment_reducer_path = "nonexistent.module.fn"
    fn = gen._load_reducer(Args())
    assert fn is gen._default_uniform_fan_out


def test_load_reducer_not_callable_falls_back() -> None:
    fake = types.ModuleType("fake_mod")
    fake.not_a_fn = 42
    sys.modules["fake_mod"] = fake

    class Args:
        swe_segment_reducer_path = "fake_mod.not_a_fn"
    fn = gen._load_reducer(Args())
    assert fn is gen._default_uniform_fan_out


def test_load_reducer_no_path_default() -> None:
    class Args:
        pass
    fn = gen._load_reducer(Args())
    assert fn is gen._default_uniform_fan_out


def _make_state_with(reducer):
    class S:
        pass
    s = S()
    s.segment_reducer = reducer  # instance attr, not class -> no self-binding
    s.tokenizer = _DummyTok()
    return s


def test_fan_out_reducer_raises_marks_abort() -> None:
    def bad(*a, **kw):
        raise RuntimeError("intentional")
    state = _make_state_with(bad)
    proto = _FakeSample()
    out = gen._fan_out_with_fail_soft(state, [], 1.0, proto, "inst-x")
    assert len(out) == 1
    assert out[0].status == _Status.ABORTED
    assert "reducer_failure" in out[0].metadata["abort_reason"]


def test_fan_out_reducer_returns_none_marks_abort() -> None:
    def returns_none(*a, **kw):
        return None
    state = _make_state_with(returns_none)
    out = gen._fan_out_with_fail_soft(state, [], 0.5, _FakeSample(), "inst-y")
    assert len(out) == 1
    assert out[0].status == _Status.ABORTED


def test_fan_out_reducer_returns_non_list_marks_abort() -> None:
    def returns_str(*a, **kw):
        return "not a list"
    state = _make_state_with(returns_str)
    out = gen._fan_out_with_fail_soft(state, [], 0.5, _FakeSample(), "inst-z")
    assert len(out) == 1
    assert out[0].status == _Status.ABORTED


def test_fan_out_happy_path_passthrough() -> None:
    segs = [([1], [10], [1], {"segment_kind": "final", "completed_turns": 1,
                                "finish_reason": "stop", "num_aborts": 0,
                                "tito_masked_turns": 0})]
    state = _make_state_with(gen._default_uniform_fan_out)
    out = gen._fan_out_with_fail_soft(state, segs, 1.0, _FakeSample(), "inst-h")
    assert len(out) == 1
    assert out[0].status == _Status.COMPLETED
    assert out[0].reward == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
