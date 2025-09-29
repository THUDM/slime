from slime.utils.http_utils import post
from slime.utils.types import Sample


async def generate_with_slime_router(args, sample: Sample, sampling_params) -> Sample:
    """Generate using SlimeRouter with text-based workflow"""

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Build full text (prompt + existing response)
    if isinstance(sample.prompt, str):
        full_text = sample.prompt + sample.response
    else:
        # Handle list of dicts format (chat format)
        # For now, just convert to simple string - this might need refinement
        full_text = str(sample.prompt) + sample.response

    # Adjust max_new_tokens based on existing response length
    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= sample.response_length

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for SlimeRouter (text-based)
    payload = {
        "text": full_text,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    # Call SlimeRouter /generate endpoint
    output = await post(url, payload)

    # Extract generated text and update sample
    generated_text = output.get("text", "")
    sample.response += generated_text
    # Don't update response_length here - it will be calculated from actual tokens later

    # Get token IDs and logprobs using SlimeRouter's /retrieve_from_text
    retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
    retrieve_payload = {"text": sample.prompt + sample.response, "return_logp": True}

    retrieve_output = await post(retrieve_url, retrieve_payload)

    # Update sample with retrieved token information
    if "tokens" in retrieve_output:
        sample.tokens = retrieve_output["tokens"]

        # Calculate response_length from actual tokens
        # Get prompt tokens to determine response length
        if hasattr(sample, "prompt_tokens") and sample.prompt_tokens:
            prompt_token_count = len(sample.prompt_tokens)
        else:
            # Fallback: tokenize prompt to get prompt token count
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            prompt_tokens = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
            prompt_token_count = len(prompt_tokens)

        # Calculate response_length as the difference between total and prompt tokens
        sample.response_length = len(sample.tokens) - prompt_token_count

    if "logp" in retrieve_output:
        # For SlimeRouter, we get the full logprobs - need to extract only response ones
        full_logprobs = retrieve_output["logp"]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []

        # Get the full token sequence to determine prompt vs response split
        full_tokens = retrieve_output.get("tokens", [])

        # Calculate prompt token count (this should match the original prompt)
        if hasattr(sample, "prompt_tokens") and sample.prompt_tokens:
            prompt_token_count = len(sample.prompt_tokens)
        else:
            # Fallback: tokenize prompt to get prompt token count
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            prompt_tokens = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
            prompt_token_count = len(prompt_tokens)

        # Extract only the response log_probs (skip prompt part)
        response_logprobs = full_logprobs[prompt_token_count:] if len(full_logprobs) > prompt_token_count else []

        # Ensure we only add logprobs for the actual response tokens
        if len(response_logprobs) > 0:
            sample.rollout_log_probs.extend(response_logprobs)

    # Handle weight version if available
    if "meta_info" in output and "weight_version" in output["meta_info"]:
        sample.weight_versions.append(output["meta_info"]["weight_version"])

    # Set finish reason based on output
    if "meta_info" in output and "finish_reason" in output["meta_info"]:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
    else:
        # Default to completed if no finish_reason provided
        sample.status = Sample.Status.COMPLETED

    return sample
