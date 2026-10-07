"""Score Centering, arXiv:2609.20807, equations (9), (12), and (14)."""

import math
import os

import numpy as np
import torch

from slime.data.tensor import TensorRef
from slime.utils.ppo_utils import get_pg_loss_type, importance_weights

SAMPLER_TOPK_FIELDS = ("rollout_topk_token_ids", "rollout_topk_log_probs")


def get_score_centering_is_config(args):
    """Use the existing TIS switch and bounds for equation (14).

    Arbitrary loss/mask callbacks cannot supply weights for every candidate
    token and the modeled tail. Only the built-in token-wise rules compose.
    """
    if not getattr(args, "use_tis", False):
        return dict(mode="none")
    path = getattr(args, "custom_tis_function_path", None)
    if path in (None, "slime.backends.megatron_utils.loss.vanilla_tis_function"):
        mode = "tis"
    elif path == "slime.backends.megatron_utils.loss.icepop_function":
        mode = "mis"
    else:
        raise ValueError(
            "Score centering with --use-tis supports the built-in TIS and icepop_function only; "
            "custom loss/mask callbacks cannot be applied to the head and tail scores."
        )
    low, high = args.tis_clip_low, args.tis_clip
    if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high <= 0 or low > high:
        raise ValueError("Score centering IS bounds require finite 0 <= tis_clip_low <= tis_clip and tis_clip > 0.")
    return dict(mode=mode, low=low, high=high)


def score_centering_correction(train_head_logp, sampler_head_logp, *, mode="none", low=None, high=None, eps=1e-6):
    """Return the differentiable head-only correction and detached head masses.

    Inputs are full-vocabulary-normalized log probabilities [tokens, k], not
    probabilities renormalized over the head. Tail masses are floored as in
    appendix A.3. Only the final multiplication by train_head_logp has a gradient.
    """
    with torch.no_grad():
        p, q = train_head_logp.float().exp(), sampler_head_logp.float().exp()
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * importance_weights(rho.reciprocal(), mode, low, high)
        weights = importance_weights((train_head_logp.float() - sampler_head_logp.float()).exp(), mode, low, high)
        residual = q * weights - alpha.unsqueeze(-1) * p
    return (residual * train_head_logp).sum(-1), q_mass, p_mass


def score_centering_request(args, sampling_params):
    """Extra SGLang /generate fields; validate the actual request's distribution."""
    if not getattr(args, "use_score_centering", False):
        return {}
    # SGLang's omitted-temperature default can differ from the trainer's.
    temperature = sampling_params.setdefault("temperature", args.rollout_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Score centering requires stochastic sampling (temperature > 0).")
    if temperature != args.rollout_temperature:
        raise ValueError("Score centering requires the configured rollout temperature on every training request.")
    top_p = sampling_params.setdefault("top_p", getattr(args, "rollout_top_p", 1.0))
    if not 0 < top_p <= 1 or top_p != getattr(args, "rollout_top_p", 1.0):
        raise ValueError("Score centering requires the configured rollout top_p in (0, 1] on every request.")
    if (temperature != 1 or top_p < 1) and os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in (
        "1",
        "true",
    ):
        raise ValueError(
            "Score centering requires temperature-scaled, post-truncation sampler logprobs; "
            "unset SGLANG_RETURN_ORIGINAL_LOGPROB."
        )
    for key, default in (("top_k", -1), ("min_p", 0.0)):
        if sampling_params.get(key, default) != default:
            raise ValueError(f"Score centering requires {key}={default}; only top-p truncation is supported.")
    for key, default in (("repetition_penalty", 1.0), ("presence_penalty", 0.0), ("frequency_penalty", 0.0)):
        if sampling_params.get(key, default) != default:
            raise ValueError(f"Score centering requires {key}={default}.")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "logit_bias"):
        if sampling_params.get(key):
            raise ValueError(f"Score centering does not support constrained sampling ({key}).")
    if top_p < 1:
        sampling_params.setdefault("custom_params", {}).update(
            return_top_p_token_ids=True, return_top_p_log_probs=True
        )
        return {"return_logprob": True}
    return {"return_logprob": True, "top_logprobs_num": args.score_centering_top_k}


def validate_score_centering_args(args):
    if not getattr(args, "use_score_centering", False):
        return
    get_pg_loss_type(args)
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("Score centering is supported by the Megatron backend only.")
    if (
        getattr(args, "custom_generate_function_path", None)
        == "slime.rollout.sglang_streaming_rollout.generate_streaming"
    ):
        raise ValueError("Score centering does not support streaming rollout.")
    if args.rollout_top_p == 1 and args.score_centering_top_k < 1:
        raise ValueError("--score-centering-top-k must be positive.")
    get_score_centering_is_config(args)
    if args.loss_type != "policy_loss":
        raise ValueError("--use-score-centering requires --loss-type policy_loss.")
    for key in ("use_opd", "use_opsm", "get_mismatch_metrics", "use_unbiased_kl"):
        if getattr(args, key, False):
            raise ValueError(f"--use-score-centering cannot combine with --{key.replace('_', '-')}.")
    if getattr(args, "custom_pg_loss_reducer_function_path", None):
        raise ValueError("Score centering does not support a custom PG reducer.")
    if getattr(args, "rollout_logprob_server_url", None):
        raise ValueError(
            "Score centering requires original sampler probabilities, not a rollout logprob server's recomputation."
        )
    if args.advantage_estimator in ("gspo", "cispo"):
        raise ValueError("Score centering uses a REINFORCE objective; select grpo or a REINFORCE advantage estimator.")
    score_centering_request(
        args, {"temperature": args.rollout_temperature, "top_p": args.rollout_top_p, "top_k": args.rollout_top_k}
    )


def _validate_head_arrays(chunk_ids, chunk_logps):
    if (
        (chunk_ids < 0).any()
        or (chunk_ids > np.iinfo(np.int32).max).any()
        or (np.diff(np.sort(chunk_ids, axis=-1), axis=-1) == 0).any()
    ):
        raise ValueError("Sampler top-k token ids must be distinct and nonnegative int32 integers.")
    if not np.isfinite(chunk_logps).all() or (chunk_logps > 0).any():
        raise ValueError("Sampler top-k logprobs must be finite nonpositive values.")


def extract_sampler_topk(meta_info, count, k):
    """Decode binary sampler heads, or reuse locally validated compact arrays."""
    if "score_centering_topk" in meta_info:
        ids, logps = meta_info["score_centering_topk"]
        if ids.shape != (count, k) or logps.shape != (count, k):
            raise ValueError(f"Sampler top-k tensors must have shape {(count, k)}.")
        return ids, logps

    import pybase64

    required = ("output_topk_token_ids", "output_topk_log_probs", "output_topk_shape")
    if any(key not in meta_info for key in required):
        raise ValueError("Score centering requires binary sampler top-k metadata for every generated token.")
    if meta_info["output_topk_shape"] != [count, k]:
        raise ValueError(f"Sampler top-k tensors must have shape {(count, k)}.")
    ids = np.frombuffer(pybase64.b64decode(meta_info["output_topk_token_ids"]), dtype="<i4").reshape(count, k).copy()
    logps = np.frombuffer(pybase64.b64decode(meta_info["output_topk_log_probs"]), dtype="<f4").reshape(count, k).copy()
    for start in range(0, count, 1024):
        _validate_head_arrays(ids[start : start + 1024], logps[start : start + 1024].astype(np.float64))
    return ids, logps


def decode_score_centering_response(content, k):
    """Run in the HTTP worker thread, before Ray serializes the response.

    k > 0 decodes fixed-width heads; k == 0 decodes complete top-p supports.
    Replace binary payloads with compact arrays before Ray transport.
    """
    import json

    output = json.loads(content)
    meta = output.get("meta_info", {})
    count = len(meta.get("output_token_logprobs", []))
    if k == 0:
        # Exact top-p SC uses the complete ragged sampler distribution.
        meta.pop("score_centering_top_p", None)
        if count:
            meta["score_centering_top_p"] = extract_sampler_top_p(meta, count)
            for key in ("top_p_token_ids", "top_p_token_offsets", "top_p_log_probs"):
                meta.pop(key, None)
        return output
    if count:
        # Only locally validated compact arrays may take the fast path.
        meta.pop("score_centering_topk", None)
        meta["score_centering_topk"] = extract_sampler_topk(meta, count, k)
        for key in ("output_topk_token_ids", "output_topk_log_probs", "output_topk_shape"):
            meta.pop(key, None)
    return output


def extract_sampler_top_p(meta_info, count):
    """Decode the complete normalized sampler distribution on the replay mask."""
    if "score_centering_top_p" in meta_info:
        return meta_info["score_centering_top_p"]
    import pybase64

    from slime.utils.types import _extract_rollout_top_p_token_data

    data = _extract_rollout_top_p_token_data(meta_info, expected_num_tokens=count)
    if data is None or "top_p_log_probs" not in meta_info:
        raise ValueError("Top-p score centering requires complete sampler top-p ids, offsets, and logprobs.")
    ids, offsets = (value.numpy() for value in data)
    logps = np.frombuffer(pybase64.b64decode(meta_info["top_p_log_probs"]), dtype="<f4").copy()
    validate_sampler_top_p(ids, offsets, logps, count)
    return ids, offsets, logps


def validate_sampler_top_p(ids, offsets, logps, count, loss_mask=None, tokens=None, sampled_logps=None):
    if ids is None or offsets is None or logps is None:
        raise ValueError("Top-p score centering requires complete sampler top-p ids, offsets, and logprobs.")
    # Supports can be large; keep them shared and read only bounded ranges.
    ids = ids if isinstance(ids, TensorRef) else np.asarray(ids)
    logps = logps if isinstance(logps, TensorRef) else np.asarray(logps)
    trusted = all(isinstance(value, TensorRef) and value.validated for value in (ids, offsets, logps))
    if trusted:
        # Immutable, previously checked captures need no payload reads when
        # republishing a collection or a cross-rollout buffer snapshot.
        if (
            len(ids.shape) != 1
            or logps.shape != ids.shape
            or offsets.shape != (count + 1,)
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(offsets.dtype, np.integer)
        ):
            raise ValueError("Invalid top-p score-centering ids/logprobs/offsets.")

        # Published samples already established one-dimensional token fields.
        # Inspect list lengths, not np.shape(list), which traverses every token.
        def shape(value):
            return (len(value),) if isinstance(value, (list, tuple)) else np.shape(value)

        if loss_mask is not None and shape(loss_mask) != (count,):
            raise ValueError("Top-p loss mask must align with the response length.")
        if tokens is not None and (shape(tokens) != (count,) or shape(sampled_logps) != (count,)):
            raise ValueError("Top-p sampled tokens and logprobs must align with the response length.")
        return
    offsets = np.asarray(offsets.load() if isinstance(offsets, TensorRef) else offsets)
    if (
        len(ids.shape) != 1
        or logps.shape != ids.shape
        or offsets.shape != (count + 1,)
        or not np.issubdtype(ids.dtype, np.integer)
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets[0] != 0
        or offsets[-1] != len(ids)
        or (offsets[1:] < offsets[:-1]).any()
    ):
        raise ValueError("Invalid top-p score-centering ids/logprobs/offsets.")
    offsets = offsets.astype(np.int64, copy=False)
    mask = np.ones(count, dtype=bool) if loss_mask is None else np.asarray(loss_mask, dtype=bool)
    if mask.shape != (count,):
        raise ValueError("Top-p loss mask must align with the response length.")
    if tokens is not None:
        tokens, sampled_logps = np.asarray(tokens), np.asarray(sampled_logps)
        if tokens.shape != (count,) or sampled_logps.shape != (count,):
            raise ValueError("Top-p sampled tokens and logprobs must align with the response length.")

    # Per-token NumPy calls block the rollout event loop at high concurrency.
    # Reduce whole ragged rows together, bounding temporary arrays by both row
    # and candidate counts (a single wider support is kept intact).
    first = 0
    while first < count:
        stop = min(
            count,
            first + 4096,
            max(first + 1, np.searchsorted(offsets, int(offsets[first]) + 2**20, side="right") - 1),
        )
        start, end = offsets[first], offsets[stop]
        lengths = np.diff(offsets[first : stop + 1])
        row_ids = np.repeat(np.arange(stop - first, dtype=np.int64), lengths)
        chunk_ids = np.asarray(ids[int(start) : int(end)])
        chunk_logps = np.asarray(logps[int(start) : int(end)]).astype(np.float64)
        if (chunk_ids < 0).any() or (chunk_ids > np.iinfo(np.int32).max).any():
            raise ValueError("Sampler top-p token ids must be distinct and nonnegative int32 integers.")
        keys = np.sort(row_ids * 2**31 + chunk_ids.astype(np.int64, copy=False))
        if (keys[1:] == keys[:-1]).any():
            raise ValueError("Sampler top-p token ids must be distinct and nonnegative int32 integers.")
        if not np.isfinite(chunk_logps).all() or (chunk_logps > 0).any():
            raise ValueError("Sampler top-p logprobs must be finite nonpositive values.")
        nonempty = lengths > 0
        if (mask[first:stop] & ~nonempty).any():
            raise ValueError("Missing top-p support for a trainable token.")
        active = nonempty | mask[first:stop]
        if tokens is not None:
            selected = chunk_ids == tokens[first:stop][row_ids]
            matched_rows = row_ids[selected]
            if (
                not np.array_equal(matched_rows, np.flatnonzero(active))
                or not np.isclose(
                    chunk_logps[selected], sampled_logps[first:stop][matched_rows], rtol=1e-4, atol=1e-5
                ).all()
            ):
                raise ValueError("Top-p sampler distribution must include the sampled token with its rollout logprob.")
        first = stop


def validate_sampler_topk(sample, k):
    """Validate arrays before publication; subsequent transport checks only references."""
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if sample.response_length == 0 and ids is None and logps is None:
        sample.rollout_topk_token_ids = torch.empty((0, k), dtype=torch.int32)
        sample.rollout_topk_log_probs = torch.empty((0, k), dtype=torch.float32)
        return
    if ids is None or logps is None:
        raise ValueError("Score centering requires rollout_topk_token_ids and rollout_topk_log_probs on every sample.")
    shape = (sample.response_length, k)
    disk = [isinstance(value, TensorRef) for value in (ids, logps)]
    if any(disk):
        if not all(disk):
            raise ValueError("Sampler top-k ids and logprobs must both be disk references.")
        for value, dtype, key in zip((ids, logps), ("int32", "float32"), SAMPLER_TOPK_FIELDS, strict=True):
            if tuple(value.shape) != shape or value.dtype != dtype or value.nbytes != math.prod(shape) * 4:
                raise ValueError(f"Sampler top-k disk reference must have shape {shape} and dtype {dtype}.")
            if value.kind != key:
                raise ValueError(f"Unexpected sampler top-k disk reference kind: {value.kind}")
            value.validate()
        if ids.validated and logps.validated:
            return
    else:
        ids, logps = np.asarray(ids), np.asarray(logps)
        if tuple(ids.shape) != shape or tuple(logps.shape) != shape:
            raise ValueError(f"Sampler top-k tensors must have shape {shape}.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Sampler top-k ids must be distinct nonnegative integers.")
    for start in range(0, sample.response_length, 1024):
        _validate_head_arrays(
            np.asarray(ids[start : start + 1024]), np.asarray(logps[start : start + 1024], dtype=np.float64)
        )
