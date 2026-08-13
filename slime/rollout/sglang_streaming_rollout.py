"""Streaming sglang rollout (example).

Drop-in alternative to :func:`slime.rollout.sglang_rollout.generate` that
consumes sglang's SSE stream incrementally instead of awaiting one final JSON
response. The win is on **abort**: every chunk we receive lands directly on
``sample`` (tokens, response text, log-probs), so when a partial-rollout
recycling or weight-update abort fires mid-generation, the partial state is
already on the sample — we don't depend on ``/abort_request`` returning the
collected text.

Wire it in as the per-sample generate function::

    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \\
    --custom-generate-function-path slime.rollout.sglang_streaming_rollout.generate_streaming

This generator selects request abort, so Slime cancels each active streaming
HTTP request instead of using server-wide abort.

Request cancellation only preserves metadata received before disconnect.
SGLang emits top-p and routed-expert replay data on its terminal chunk, so use
server abort when those features must survive a partial rollout.

The outer rollout loop (semaphore, dp_rank balancing, abort orchestration,
partial-rollout buffer hand-off) is still owned by ``sglang_rollout``; this
file only replaces the inner HTTP call.

Both cumulative and incremental SGLang streams are accepted. The latter is used
when the server enables incremental streaming output.
"""

import json
import logging
from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, _prepare_prompt_ids
from slime.rollout.streaming_utils import SGLangStreamAccumulator
from slime.utils import http_utils
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span
from slime.utils.types import Sample

__all__ = ["generate_streaming"]

logger = logging.getLogger(__name__)


async def generate_streaming(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Streaming counterpart to :func:`slime.rollout.sglang_rollout.generate`.

    Applies each SSE chunk onto ``sample`` so an abort that cuts the stream
    still leaves a coherent partial sample behind.
    """
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in (
        Sample.Status.PENDING,
        Sample.Status.ABORTED,
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    payload: dict[str, Any] = {
        "sampling_params": sampling_params,
        "return_logprob": True,
        "stream": True,
    }
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}

    # Preserve the pre-call state so both stream formats also work when
    # resuming a partial rollout. A terminal full top-p snapshot may replay
    # the current call from this base to keep metadata aligned.
    base_tokens = list(sample.tokens)
    base_response = sample.response or ""
    base_response_length = sample.response_length
    base_log_probs = None if sample.rollout_log_probs is None else list(sample.rollout_log_probs)
    base_top_p_token_ids = sample.rollout_top_p_token_ids
    base_top_p_token_offsets = sample.rollout_top_p_token_offsets
    base_routed_experts = sample.rollout_routed_experts
    base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None

    last_meta_info: dict[str, Any] = {}
    stream = SGLangStreamAccumulator(
        output_mode="incremental" if args.sglang_incremental_streaming_output else "cumulative"
    )

    client = http_utils._http_client
    assert client is not None, "http client not initialized; call init_http_client first"

    try:
        with trace_span(
            sample, "sglang_generate_stream", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}
        ) as span:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:") :].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("sglang_streaming: skipping non-JSON chunk: %r", data_str[:120])
                        continue

                    update = stream.add(chunk)
                    last_meta_info = update.meta_info

                    if update.replace_call_state:
                        sample.tokens = list(base_tokens)
                        sample.response = base_response
                        sample.response_length = base_response_length
                        sample.rollout_log_probs = None if base_log_probs is None else list(base_log_probs)
                        sample.rollout_top_p_token_ids = base_top_p_token_ids
                        sample.rollout_top_p_token_offsets = base_top_p_token_offsets
                        sample.rollout_routed_experts = base_routed_experts
                        sample.loss_mask = None if base_loss_mask is None else list(base_loss_mask)

                    sample.append_response_tokens(
                        args,
                        tokens=update.tokens,
                        log_probs=update.log_probs,
                        trainable=True,
                        meta_info=last_meta_info,
                        text=None,
                        update_terminal_info=bool(last_meta_info.get("finish_reason")),
                    )

                    if state.aborted:
                        break

            if last_meta_info.get("finish_reason"):
                span.update(build_sglang_meta_trace_attrs(last_meta_info))
    finally:
        sample.response = base_response + stream.response_text(
            lambda token_ids: state.tokenizer.decode(
                token_ids,
                skip_special_tokens=sampling_params.get("skip_special_tokens", True),
            )
        )

    if state.aborted and not last_meta_info.get("finish_reason"):
        sample.status = Sample.Status.ABORTED
    elif not last_meta_info.get("finish_reason"):
        raise RuntimeError("SGLang streaming response ended without a terminal finish_reason.")

    return sample


generate_streaming.abort_mode = "request"
