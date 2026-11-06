"""Deterministic training seeds derived from persisted global sample occurrence IDs."""


def training_sample_seed(base_seed: int, sample_index: int) -> int:
    # Do not wrap: a wrap would silently reuse streams in a sufficiently long run.
    if type(base_seed) is not int or type(sample_index) is not int:
        raise ValueError("Sample seed mode requires integer base seed and sample.index")
    if base_seed < 0 or sample_index < 0:
        raise ValueError("Sample seed mode requires nonnegative seed and sample.index")
    seed = base_seed + sample_index
    if seed >= 2**31:
        raise ValueError("Sample seed mode exceeded the signed 32-bit seed range")
    return seed
