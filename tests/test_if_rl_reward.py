from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.types import Sample


def _load_reward_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "if_rl" / "reward_ifrl.py"
    spec = importlib.util.spec_from_file_location("reward_ifrl_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward_ifrl = _load_reward_module()


def _score_sample(sample: Sample) -> float:
    return asyncio.run(reward_ifrl.reward_func(None, sample))


def test_reward_requires_all_instructions_to_pass():
    sample = Sample(
        prompt="Return a response that contains alpha and ends with omega.",
        response="alpha appears here but the ending is wrong",
        status=Sample.Status.COMPLETED,
        metadata={
            "prompt_text": "Return a response that contains alpha and ends with omega.",
            "instruction_id_list": [
                "keywords:existence",
                "last_word:last_word_answer",
            ],
            "kwargs": [
                {"keywords": ["alpha"]},
                {"last_word": "omega"},
            ],
        },
    )

    assert _score_sample(sample) == 0.0


def test_reward_is_zero_for_truncated_samples():
    sample = Sample(
        prompt="Include alpha in the answer.",
        response="alpha",
        status=Sample.Status.TRUNCATED,
        metadata={
            "prompt_text": "Include alpha in the answer.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["alpha"]}],
        },
    )

    assert _score_sample(sample) == 0.0
