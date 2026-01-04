"""Data invariant validation utilities - for testing only.

These utilities verify that the data invariants hold throughout the training pipeline.
They should NOT be used in production code.
"""


def validate_sample_invariants(sample, context: str = ""):
    """Validate Sample object data invariants.

    Args:
        sample: Sample object to validate
        context: Context string for error messages

    Raises:
        ValueError: If any invariant is violated
    """
    errors = []

    if sample.response_length is not None and sample.response_length <= 0:
        errors.append(f"response_length must be > 0, got {sample.response_length}")

    if sample.loss_mask is not None and sample.response_length is not None:
        if len(sample.loss_mask) != sample.response_length:
            errors.append(
                f"len(loss_mask)={len(sample.loss_mask)} != " f"response_length={sample.response_length}"
            )

    if sample.tokens is not None and sample.response_length is not None:
        if len(sample.tokens) < sample.response_length:
            errors.append(f"len(tokens)={len(sample.tokens)} < " f"response_length={sample.response_length}")

    if errors:
        raise ValueError(f"[{context}] Sample invariant violations:\n" + "\n".join(errors))


def validate_rollout_data(rollout_data: dict, context: str = ""):
    """Validate rollout_data dictionary invariants.

    Args:
        rollout_data: Rollout data dictionary to validate
        context: Context string for error messages

    Raises:
        ValueError: If any invariant is violated
    """
    errors = []

    response_lengths = rollout_data.get("response_lengths", [])
    loss_masks = rollout_data.get("loss_masks", [])

    if len(response_lengths) != len(loss_masks):
        errors.append(f"len(response_lengths)={len(response_lengths)} != " f"len(loss_masks)={len(loss_masks)}")

    for i, (rl, lm) in enumerate(zip(response_lengths, loss_masks)):
        if hasattr(lm, "__len__") and len(lm) != rl:
            errors.append(f"Sample {i}: len(loss_mask)={len(lm)} != response_length={rl}")

    # Validate advantages/returns if present
    for key in ["advantages", "returns"]:
        if key in rollout_data:
            data = rollout_data[key]
            for i, (item, rl) in enumerate(zip(data, response_lengths)):
                if hasattr(item, "__len__") and len(item) != rl:
                    errors.append(f"Sample {i}: len({key})={len(item)} != response_length={rl}")

    if errors:
        raise ValueError(f"[{context}] Rollout data invariant violations:\n" + "\n".join(errors))
