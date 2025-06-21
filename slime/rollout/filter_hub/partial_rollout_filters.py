from slime.utils.types import Sample


__all__ = ["valid_partial_sample"]


def valid_partial_sample(args, sample: Sample, **kwargs):
    if (sample.response and 
        len(sample.response.strip()) < args.partial_rollout_min_response_length):
        return False
    if (sample.response_length and sample.response_length < args.partial_rollout_min_tokens):
        return False
    return True
