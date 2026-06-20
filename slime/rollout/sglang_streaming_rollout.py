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

The outer rollout loop (semaphore, dp_rank balancing, abort orchestration,
partial-rollout buffer hand-off) is still owned by ``sglang_rollout``; this
file only replaces the inner HTTP call.

sglang's default streaming output is cumulative — server-side
``state.output_token_logprobs`` accumulates and every chunk references the
full list-so-far (see ``tokenizer_manager.py``). If anyone ever flips
``--incremental-streaming-output`` on the sglang server, the text/output_ids
deltas will need different handling here.
"""

import json
import logging
from argparse import Namespace
from typing import Any

import numpy as np
import pybase64

from slime.rollout.sglang_rollout import (
    GenerateState,
    _extract_rollout_top_p_token_data,
    _merge_rollout_top_p_token_data,
    _prepare_prompt_ids,
)
from slime.utils import http_utils
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span
from slime.utils.types import Sample

__all__ = ["generate_streaming"]

logger = logging.getLogger(__name__)


async def generate_streaming(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Streaming counterpart to :func:`slime.rollout.sglang_rollout.generate`.

    Writes the cumulative state from each SSE chunk onto ``sample`` so an
    abort that cuts the stream still leaves a coherent partial sample behind.
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

    # Snapshot pre-call sample state. sglang's SSE chunks are cumulative
    # *within this call*; on each chunk we rebuild the post-call view of the
    # sample = prior state + chunk delta. That way a mid-stream break leaves
    # the sample exactly at the boundary of the last chunk we observed.
    base_tokens = list(sample.tokens)
    base_response = sample.response or ""
    base_response_length = sample.response_length
    base_log_probs = list(sample.rollout_log_probs or [])
    base_top_p_token_ids = list(sample.rollout_top_p_token_ids or [])
    base_top_p_token_offsets = list(sample.rollout_top_p_token_offsets or [0])
    base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None

    last_meta_info: dict[str, Any] = {}
    call_tokens: list[int] = []
    call_log_probs: list[float] = []
    call_text: str = ""

    client = http_utils._http_client
    assert client is not None, "http client not initialized; call init_http_client first"

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

                meta = chunk.get("meta_info") or {}
                last_meta_info = meta

                call_text = chunk.get("text", call_text)
                if "output_token_logprobs" in meta:
                    call_tokens = [item[1] for item in meta["output_token_logprobs"]]
                    call_log_probs = [item[0] for item in meta["output_token_logprobs"]]

                # Surface partial state on the sample immediately. If the
                # outer abort path cuts us, whatever we've written so far is
                # what survives — no /abort_request round-trip needed.
                sample.tokens = base_tokens + call_tokens
                sample.response = base_response + call_text
                sample.response_length = base_response_length + len(call_tokens)
                sample.rollout_log_probs = base_log_probs + call_log_probs
                top_p_data = _extract_rollout_top_p_token_data(meta, expected_num_tokens=len(call_tokens))
                if top_p_data is not None:
                    sample.rollout_top_p_token_ids, sample.rollout_top_p_token_offsets = (
                        _merge_rollout_top_p_token_data(
                            base_top_p_token_ids,
                            base_top_p_token_offsets,
                            *top_p_data,
                        )
                    )
                if base_loss_mask is not None:
                    assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
                    sample.loss_mask = base_loss_mask + [1] * len(call_tokens)

                if state.aborted:
                    break

        if last_meta_info.get("finish_reason"):
            span.update(build_sglang_meta_trace_attrs(last_meta_info))

    # MoE routing replay (when requested) ships in the terminal chunk.
    if "routed_experts" in last_meta_info:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(last_meta_info["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    if last_meta_info.get("finish_reason"):
        sample.update_from_meta_info(args, last_meta_info)
    elif state.aborted:
        sample.status = Sample.Status.ABORTED

    return sample
