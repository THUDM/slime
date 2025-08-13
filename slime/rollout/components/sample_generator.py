from slime.utils.http_utils import post
from slime.utils.types import Sample


async def generate_one_sample_vanilla(args, tokenizer, sample: Sample, sampling_params) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(
            tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        )

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload - shared structure
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": args.use_token_output,
    }

    if args.use_token_output:
        # Token-based mode: use tokens directly
        if len(sample.response) > 0:
            input_token_ids = sample.tokens
        else:
            # First turn: initialize with prompt tokens
            prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
            input_token_ids = prompt_token_ids
            # Initialize sample.tokens with prompt for subsequent turns
            if not sample.tokens:  # Only set if empty
                sample.tokens = prompt_token_ids
        payload["input_ids"] = input_token_ids
    else:
        # String-based mode: original implementation
        input_text = sample.prompt + sample.response
        payload["text"] = input_text

    output = await post(url, payload, use_http2=args.use_http2)

    if args.use_token_output:
        # Extract new response tokens
        assert (
            "meta_info" in output and "output_token_logprobs" in output["meta_info"]
        ), "output_token_logprobs is not in the output"
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += tokenizer.decode(new_response_tokens, skip_special_tokens=False)
    else:
        # String-based processing
        sample.response += output["text"]
        prompt_tokens_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        response_token_ids = tokenizer(sample.response, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample
