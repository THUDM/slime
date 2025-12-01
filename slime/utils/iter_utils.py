from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch


# details: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
def group_by(iterable, key=None):
    """Similar to itertools.groupby, but do not require iterable to be sorted"""
    ret = defaultdict(list)
    for item in iterable:
        ret[key(item) if key is not None else item].append(item)
    return dict(ret)


# TODO fsdp can also use this
def chunk_named_params_by_size(named_params: Iterable[tuple[str, torch.Tensor]], chunk_size: int):
    return _chunk_by_size(
        named_params,
        compute_size=lambda named_weight: named_weight[1].nbytes,
        chunk_size=chunk_size,
    )


def _chunk_by_size(objects: Iterable[Any], compute_size: Callable[[Any], int], chunk_size: int):
    bucket: list[Any] = []
    bucket_size = 0

    for obj in objects:
        obj_size = compute_size(obj)

        if bucket and (bucket_size + obj_size) >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0

        bucket.append(obj)
        bucket_size += obj_size

    if bucket:
        yield bucket
