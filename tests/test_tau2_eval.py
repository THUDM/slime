import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU2_DIR = ROOT / "examples" / "tau-bench" / "tau2"
sys.path.insert(0, str(TAU2_DIR))

httpx = types.ModuleType("httpx")


class _DummyTimeout:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _DummyAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        pass


httpx.Timeout = _DummyTimeout
httpx.AsyncClient = _DummyAsyncClient
sys.modules["httpx"] = httpx

transformers = types.ModuleType("transformers")


class _DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return None


transformers.AutoTokenizer = _DummyAutoTokenizer
sys.modules["transformers"] = transformers

import eval as tau2_eval


def test_eval_num_samples_guard():
    parser = tau2_eval._build_arg_parser()
    args = parser.parse_args(
        [
            "--hf-checkpoint",
            "dummy",
            "--sglang-url",
            "http://localhost:30000/generate",
            "--output",
            "eval.json",
            "--num-samples",
            "0",
        ]
    )
    with pytest.raises(SystemExit):
        tau2_eval._validate_args(args, parser)
