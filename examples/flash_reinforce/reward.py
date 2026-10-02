"""Reward of the FlashREINFORCE recipe: molt's math grader, called the way molt's MathEnv calls it.

Use with ``--custom-rm-path reward.math_reward`` and this directory on ``PYTHONPATH``.
The label is the dataset's ``reward_model`` dict (``--label-key reward_model``).
"""

from math_grader import score_response


async def math_reward(args, sample, **kwargs) -> float:
    # Grade prompt + response like molt; the grader prefers an answer found in the response.
    return score_response(sample.prompt + sample.response, sample.prompt, sample.label)["reward"]
