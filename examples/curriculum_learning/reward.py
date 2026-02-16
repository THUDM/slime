from slime.utils.types import Sample
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


def compute_format_verify_reward(checkers, instruction, response):
    if not checkers:
        return 0.0

    satisfied = 0
    for checker_src in checkers:
        try:
            local_ns = {}
            exec(checker_src, {"re": __import__("re")}, local_ns)
            if local_ns["check_following"](instruction, response):
                satisfied += 1
        except Exception:
            pass

    return satisfied / len(checkers)


async def async_rm_math_if(args, sample: Sample):

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label

    if rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)

    elif rm_type == "format_verify":
        functions = metadata.get("functions", [])
        reward_value = compute_format_verify_reward(
            functions,
            sample.prompt, 
            response.split('</think>')[-1],
        )
        return reward_value

    elif rm_type == "ifbench":
        from slime.rollout.rm_hub.ifbench import compute_ifbench_reward
        return compute_ifbench_reward(response, label, metadata=metadata)
    
    else:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
