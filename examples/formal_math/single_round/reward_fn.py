import asyncio
import logging
import re
from types import SimpleNamespace

from kimina_client import SnippetStatus

try:
    import kimina_wrapper
except ImportError:
    from . import kimina_wrapper

logger = logging.getLogger(__name__)

_TIMEOUT = 60


class RewardFn:
    def __init__(self):
        self._verifier = kimina_wrapper.KiminaServerAndClientCluster()

    async def __call__(self, args, sample, **kwargs):
        try:
            code, code_error_cat = _assemble_code(prompt=sample.prompt, response=sample.response)
            if code is None:
                return dict(reward_value=0.0, reward_cat=code_error_cat)

            resp = await self._verifier.check(snips=code, timeout=_TIMEOUT, show_progress=False)
            result = _single(resp.results)
            analysis = result.analyze()
            is_valid = analysis.status == SnippetStatus.valid

            return dict(
                reward_value=float(is_valid),
                reward_cat="success" if is_valid else f"lean_{analysis.status.value}",
                lean_result=result.model_dump(),
                extracted_code=code,
            )
        except Exception as e:
            logger.warning(f"Error in RewardFn: {e=} {sample.prompt=} {sample.response=}")
            return dict(reward_value=0.0, reward_cat="python_error", error_details=str(e))


def _single(arr):
    assert len(arr) == 1, f"{arr=}"
    return arr[0]


def _assemble_code(prompt: str, response: str) -> tuple[str | None, str | None]:
    prompt_code_block = _extract_last_full_code_block(prompt)
    assert prompt_code_block is not None

    response_code_block = _extract_last_full_code_block(response)
    if response_code_block is None:
        return None, "no_code"

    question_code = prompt_code_block[: prompt_code_block.index(":=")]
    answer_code = _extract_answer_code_from_response_code_block(response_code_block)

    return question_code + answer_code, None


def _extract_answer_code_from_response_code_block(response_code_block: str):
    haystack = ":="
    if haystack in response_code_block:
        return response_code_block[response_code_block.index(haystack) :]
    else:
        # leanabell prover style: only output the proof code
        return ":= by\n" + response_code_block


def _extract_last_full_code_block(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1] if matches else None


_REWARD_FN: RewardFn | None = None


async def reward_fn(*args, **kwargs):
    global _REWARD_FN
    if _REWARD_FN is None:
        _REWARD_FN = RewardFn()
    return await _REWARD_FN(*args, **kwargs)


if __name__ == "__main__":
    # Run this UT with:
    # python examples/formal_math/single_round/reward_fn.py

    test_pairs = [
        # from stoney0062/Leanabell-Prover-Traindata-SFT
        (
            """
Hello

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A pirate is counting his loot in base 6. He has $4532_6$ dollars worth of silver, $1254_6$ dollars worth of pearls, and $654_6$ dollars worth of exotic spices. What is the total dollar amount of his loot? Express your answer in base 10. Show that it is 1636.-/
theorem pirate_loot_total (silver pearls spices : ℕ) (h_silver : silver = 4 * 6 ^ 3 + 5 * 6 ^ 2 + 3 * 6 + 2)
(h_pears : pearls = 1 * 6 ^ 3 + 2 * 6 ^ 2 + 5 * 6 + 4) (h_spices : spices = 6 * 6 ^ 2 + 5 * 6 + 4) :
silver + pearls + spices = 1636 := by
  sorry
```
    """,
            """
```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A pirate is counting his loot in base 6. He has $4532_6$ dollars worth of silver, $1254_6$ dollars worth of pearls, and $654_6$ dollars worth of exotic spices. What is the total dollar amount of his loot? Express your answer in base 10. Show that it is 1636.-/
theorem pirate_loot_total (silver pearls spices : ℕ) (h_silver : silver = 4 * 6 ^ 3 + 5 * 6 ^ 2 + 3 * 6 + 2)
(h_pears : pearls = 1 * 6 ^ 3 + 2 * 6 ^ 2 + 5 * 6 + 4) (h_spices : spices = 6 * 6 ^ 2 + 5 * 6 + 4) :
silver + pearls + spices = 1636 := by
  rw [h_silver, h_pears, h_spices]
  norm_num
  <;> linarith
```
    """,
        ),
        # from leanabell SFT output
        # TODO: provide a correct example
        # (
        #     '''<|im_start|>user\nComplete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- Evaluate $\\log_327$. Show that it is 3.-/\ntheorem mathd_algebra_484 : Real.log 27 / Real.log 3 = 3 := by\n  sorry\n```\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.<|im_end|>\n<|im_start|>assistant\n''',
        #     ''''```lean\n  /-\n  To evaluate \\(\\log_3 27\\), we start by expressing 27 as a power of 3. We know that \\(27 = 3^2\\). Therefore, we can rewrite the expression as:\n  \\[\n  \\log_3 27 = \\log_3 (3^2) = 3\n  \\]\n  This completes the proof.\n  -/\n  -- Use the definition of Real.logb to rewrite the expression.\n  simp [Real.logb, show (3 : ℝ) = 2 by norm_num, show (27 : ℝ) = 3 ^ 2 by norm_num]\n  -- Simplify the expression using the properties of logarithms and arithmetic operations.\n  <;> simp [Real.logb, show (3 : ℝ) = 2 by norm_num, show (27 : ℝ) = 3 ^ 2 by norm_num]\n  -- Normalize the numerical values to confirm the result.\n  <;> norm_num\n  -- Ensure the final result is correct by checking reflexivity.\n  <;> rfl\n``` <|im_end|>''',
        # ),
    ]

    import ray

    ray.init()

    for test_prompt, test_response in test_pairs:
        output = asyncio.run(reward_fn(None, SimpleNamespace(prompt=test_prompt, response=test_response)))
        print(f"{output=}")
        assert output["reward_value"] == 1.0
