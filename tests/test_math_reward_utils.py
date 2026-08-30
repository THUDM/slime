from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


def test_deepscaler_reward_fallback_without_markers():
    response = "Answer: \\boxed{42}"
    assert get_deepscaler_rule_based_reward(response, "42") == 1
