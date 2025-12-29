import abc
import copy
import logging
import os
from pathlib import Path

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        # Delayed initialization: dataset will be created in set_train_parallel_config()
        # Reason: DP config is not available until TrainRayActor.init() completes
        self._dataset = None
        self._tokenizer = None
        self._processor = None
        self._dp_size = None
        self._use_hf_datasets = getattr(args, "use_hf_datasets", False)

        # Prepare tokenizer/processor if using global dataset
        # These are needed early for dump_details
        if args.rollout_global_dataset:
            self._tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            self._processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                self._tokenizer.save_pretrained(Path(d) / "tokenizer")
                if self._processor:
                    self._processor.save_pretrained(Path(d) / "processor")

    def set_train_parallel_config(self, config: dict):
        """Called by RolloutManager after receiving DP config from TrainRayActor.

        This triggers lazy initialization of the dataset with DP information.

        Args:
            config: Configuration dict containing dp_size, etc.
        """
        self._dp_size = config.get("dp_size", 1)

        # Lazy initialization of dataset (only if not already created)
        if self._dataset is None and self.args.rollout_global_dataset:
            logger.info(f"Initializing dataset with dp_size={self._dp_size}")
            self._create_dataset()

    def _create_dataset(self):
        """Create dataset with DP awareness.

        This method selects the appropriate dataset implementation based on args:
        - HF Datasets (streaming mode) if --use-hf-datasets is set
        - Legacy Dataset otherwise

        Note: RolloutManager is a single instance, so we do NOT shard by dp_rank.
        Data sharding happens in RolloutManager._split_train_data_by_dp().
        """
        if self._use_hf_datasets:
            # Use HuggingFace Datasets streaming mode
            from slime.utils.hf_dataset import HFIterableDatasetAdapter

            logger.info("Creating HFIterableDatasetAdapter (streaming mode)")
            self._dataset = HFIterableDatasetAdapter(
                path=self.args.prompt_data,
                tokenizer=self._tokenizer,
                processor=self._processor,
                max_length=self.args.rollout_max_prompt_len,
                prompt_key=self.args.input_key,
                label_key=self.args.label_key,
                tool_key=self.args.tool_key,
                metadata_key=self.args.metadata_key,
                multimodal_keys=self.args.multimodal_keys,
                seed=self.args.rollout_seed,
                apply_chat_template=self.args.apply_chat_template,
                apply_chat_template_kwargs=self.args.apply_chat_template_kwargs,
                dp_size=self._dp_size or 1,
                buffer_size=getattr(self.args, "hf_dataset_buffer_size", 1000),
                shuffle_buffer_size=getattr(self.args, "hf_dataset_shuffle_buffer", 10000),
                num_proc=getattr(self.args, "hf_dataset_num_proc", 8),
            )

            # Apply initial shuffle if requested
            if self.args.rollout_shuffle:
                self._dataset.shuffle(self.epoch_id)

        else:
            # Use legacy Dataset implementation
            logger.info("Creating legacy Dataset")
            self._dataset = Dataset(
                self.args.prompt_data,
                tokenizer=self._tokenizer,
                processor=self._processor,
                max_length=self.args.rollout_max_prompt_len,
                prompt_key=self.args.input_key,
                multimodal_keys=self.args.multimodal_keys,
                label_key=self.args.label_key,
                metadata_key=self.args.metadata_key,
                tool_key=self.args.tool_key,
                apply_chat_template=self.args.apply_chat_template,
                apply_chat_template_kwargs=self.args.apply_chat_template_kwargs,
                seed=self.args.rollout_seed,
            )

            # Apply initial shuffle if requested
            if self.args.rollout_shuffle:
                self._dataset.shuffle(self.epoch_id)

    @property
    def dataset(self):
        """Accessor for dataset with auto-initialization fallback.

        This ensures backward compatibility with code that directly accesses self.dataset.
        """
        if self._dataset is None and self.args.rollout_global_dataset:
            # Fallback: Initialize with dp_size=1 if set_train_parallel_config was not called
            logger.warning(
                "Dataset accessed before set_train_parallel_config was called. "
                "Initializing with dp_size=1. This may indicate a bug."
            )
            self._dp_size = 1
            self._create_dataset()
        return self._dataset

    def get_samples(self, num_samples):
        # Mixed mode: auto-detect dataset type using duck typing
        if self.dataset is None:
            # Case 1: No dataset (--disable-rollout-global-dataset)
            prompt_samples = [Sample() for _ in range(num_samples)]

        elif hasattr(self.dataset, "get_next_batch"):
            # Case 2: HF adapters - use streaming interface
            # Note: HF adapters handle epoch switching internally
            prompt_samples = self.dataset.get_next_batch(num_samples)

            # Sync epoch_id from HF adapter (it handles epoch switching internally)
            if hasattr(self.dataset, "epoch_id"):
                self.epoch_id = self.dataset.epoch_id

        else:
            # Case 3: Legacy Dataset - use array access
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                # Handle epoch boundary
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples

        # Common processing: wrap prompt_samples into groups
        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }

        # Save HF adapter state if using HF Datasets
        if self.dataset is not None and hasattr(self.dataset, "get_checkpoint_state"):
            state_dict["hf_adapter_state"] = self.dataset.get_checkpoint_state()

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

        logger.info(f"load metadata from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        # Restore dataset state based on type (mixed mode)
        if self.dataset is not None:
            if hasattr(self.dataset, "load_checkpoint_state"):
                # HF adapters: use dedicated checkpoint API
                hf_state = state_dict.get("hf_adapter_state")
                if hf_state:
                    logger.info(
                        f"Restoring HF adapter state: epoch={hf_state.get('epoch_id')}, consumed={hf_state.get('consumed_count')}"
                    )
                    self.dataset.load_checkpoint_state(hf_state)  # type: ignore[attr-defined]
            elif self.args.rollout_shuffle:
                # Legacy Dataset: manual shuffle
                self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
