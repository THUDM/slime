"""Shape-only rollout inspection; no tensor payloads or token sequences are read."""

import numpy as np


_CAPTURE_FIELDS = (
    "rollout_routed_experts",
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
    "rollout_top_p_log_probs",
)


def describe_sample(sample):
    tensors = {}
    for name in _CAPTURE_FIELDS:
        value = getattr(sample, name, None)
        if value is None:
            continue
        # Captures normally arrive as TensorRefs. Raw arrays at initial
        # publication have their own shape; no load/materialization is needed.
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if name == "rollout_routed_experts" and isinstance(value, (list, tuple)) and value:
            shapes = [tuple(part.shape) for part in value]
            if any(part[1:] != shapes[0][1:] for part in shapes):
                raise ValueError("Routed-experts chunks have different layer/expert shapes")
            shape = (sum(part[0] for part in shapes), *shapes[0][1:])
            dtype = value[0].dtype
        elif shape is None:
            array = np.asarray(value)
            shape, dtype = array.shape, array.dtype
        tensors[name] = {"shape": list(shape), "dtype": str(dtype).removeprefix("torch.")}
    return {
        "index": sample.index.item() if isinstance(sample.index, np.generic) else sample.index,
        "group_index": sample.group_index.item() if isinstance(sample.group_index, np.generic) else sample.group_index,
        "status": sample.status.value,
        "tokens": len(sample.tokens),
        "response_length": sample.response_length,
        "loss_mask": len(sample.loss_mask) if sample.loss_mask is not None else None,
        "rollout_log_probs": len(sample.rollout_log_probs) if sample.rollout_log_probs is not None else None,
        "tensors": tensors,
    }


def validate_sample_metadata(metadata, args):
    """Check dimensions/dtypes against this run, trusting published contents."""
    for sample in metadata:
        count = sample["response_length"]
        if not 0 <= count <= sample["tokens"]:
            raise ValueError("Response length exceeds the token count")
        for field in ("loss_mask", "rollout_log_probs"):
            if sample[field] is not None and sample[field] != count:
                raise ValueError(f"{field} length does not match the response length")
        tensors = sample["tensors"]
        routes = tensors.get("rollout_routed_experts")
        if getattr(args, "use_rollout_routing_replay", False) and routes is not None:
            expected = [max(0, sample["tokens"] - 1), int(args.num_layers), args.moe_router_topk]
            if routes["shape"] != expected or not np.issubdtype(np.dtype(routes["dtype"]), np.integer):
                raise ValueError("Routed-experts shape/dtype does not match tokens and model")
        if not getattr(args, "use_score_centering", False):
            continue
        top_p = getattr(args, "rollout_top_p", 1.0) < 1
        prefix = "rollout_top_p" if top_p else "rollout_topk"
        ids, logps = tensors.get(prefix + "_token_ids"), tensors.get(prefix + "_log_probs")
        if ids is None or logps is None:
            raise ValueError("Missing sampler capture metadata")
        if (
            ids["shape"] != logps["shape"]
            or not np.issubdtype(np.dtype(ids["dtype"]), np.integer)
            or not np.issubdtype(np.dtype(logps["dtype"]), np.floating)
        ):
            raise ValueError("Sampler ids/logprobs shape/dtype mismatch")
        if top_p:
            offsets = tensors.get(prefix + "_token_offsets")
            if (
                len(ids["shape"]) != 1
                or offsets is None
                or offsets["shape"] != [count + 1]
                or not np.issubdtype(np.dtype(offsets["dtype"]), np.integer)
            ):
                raise ValueError("Sampler top-p offsets/shape mismatch")
        elif ids["shape"] != [count, args.score_centering_top_k]:
            raise ValueError("Sampler top-k shape mismatch")
