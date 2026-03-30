import abc
import logging
import os

import torch

from slime.utils.data import load_hf_dataset
from slime.utils.misc import load_function

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_examples(self, num_prompts: int) -> list[dict]:
        """Return num_prompts raw dataset examples (dicts)."""

    @abc.abstractmethod
    def add_examples(self, examples: list[dict]):
        """Re-queue examples (e.g. aborted prompts) back into the source."""

    @abc.abstractmethod
    def save(self, rollout_id):
        """Save the state of the data source."""

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """Load the state of the data source."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Length of the data source."""


class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args
        self.epoch_id = 0
        self.sample_offset = 0
        self.requeue: list[dict] = []

        if args.rollout_global_dataset and args.prompt_data is not None:
            self.base_dataset = load_hf_dataset(args.prompt_data)
            self.dataset = self._dataset_for_epoch(self.epoch_id)
        else:
            self.base_dataset = None
            self.dataset = None

    def _dataset_for_epoch(self, epoch_id):
        if self.base_dataset is None:
            return None
        if not self.args.rollout_shuffle:
            return self.base_dataset
        return self.base_dataset.shuffle(seed=self.args.rollout_seed + epoch_id)

    def get_examples(self, num_prompts: int) -> list[dict]:
        examples = self._pop_requeue(num_prompts)
        remaining = num_prompts - len(examples)
        if remaining <= 0:
            return examples

        if self.dataset is None:
            return examples + [{} for _ in range(remaining)]

        if self.sample_offset + remaining <= len(self.dataset):
            examples.extend(self.dataset[idx] for idx in range(self.sample_offset, self.sample_offset + remaining))
            self.sample_offset += remaining
            return examples

        examples.extend(self.dataset[idx] for idx in range(self.sample_offset, len(self.dataset)))
        remaining = num_prompts - len(examples)
        self.epoch_id += 1
        self.dataset = self._dataset_for_epoch(self.epoch_id)
        examples.extend(self.dataset[idx] for idx in range(remaining))
        self.sample_offset = remaining
        return examples

    def _pop_requeue(self, num_prompts: int) -> list[dict]:
        if not self.requeue or num_prompts <= 0:
            return []
        num_to_pop = min(len(self.requeue), num_prompts)
        examples = self.requeue[:num_to_pop]
        del self.requeue[:num_to_pop]
        return examples

    def add_examples(self, examples: list[dict]):
        if examples:
            self.requeue.extend(examples)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "requeue": self.requeue,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return
        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load data source state from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.requeue = state_dict.get("requeue", [])
        self.dataset = self._dataset_for_epoch(self.epoch_id)

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer: list[dict] = []
        if args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(args.buffer_filter_path)

    def get_examples(self, num_prompts: int) -> list[dict]:
        examples = self._get_from_buffer(num_prompts)
        remaining = num_prompts - len(examples)
        if remaining > 0:
            examples += super().get_examples(remaining)
        return examples

    def _get_from_buffer(self, num_prompts: int) -> list[dict]:
        if not self.buffer or num_prompts <= 0:
            return []
        return self.buffer_filter(self.args, None, self.buffer, num_prompts)

    def add_examples(self, examples: list[dict]):
        if examples:
            self.buffer.extend(examples)

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[dict], num_prompts: int) -> list[dict]:
    n = min(len(buffer), num_prompts)
    result = buffer[:n]
    del buffer[:n]
    return result
