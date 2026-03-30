import logging
import re

try:
    import ray
except ModuleNotFoundError:  # pragma: no cover
    ray = None

from .timer import Timer

__all__ = ["load_hf_dataset", "get_minimum_num_micro_batch_size", "process_rollout_data"]

logger = logging.getLogger(__name__)


def _parse_generalized_path(path: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", path)) is None:
        return path, None

    start = int(x) if (x := m.group("start")) != "" else None
    end = int(x) if (x := m.group("end")) != "" else None
    return m.group("real_path"), slice(start, end)


def load_hf_dataset(path: str):
    import datasets as hf_datasets

    real_path, row_slice = _parse_generalized_path(path)
    if real_path.endswith(".jsonl"):
        dataset = hf_datasets.load_dataset("json", data_files=real_path, split="train")
    elif real_path.endswith(".parquet"):
        dataset = hf_datasets.load_dataset("parquet", data_files=real_path, split="train")
    else:
        raise ValueError(f"Unsupported file format: {real_path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info("load_hf_dataset path=%s applying slice row_slice=%s", real_path, row_slice)
        dataset = dataset.select(range(*row_slice.indices(len(dataset))))
    return dataset


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)
    return len(batches)


def process_rollout_data(args, rollout_data_refs, dp_rank, dp_size):
    assert len(rollout_data_refs) == dp_size
    if ray is None:
        raise ModuleNotFoundError("ray is required to process rollout data")
    rollout_data = ray.get(rollout_data_refs[dp_rank])
    Timer().seq_lens = [len(s.tokens) for s in rollout_data]
    return rollout_data
