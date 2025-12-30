"""HuggingFace Datasets integration for Slime.

This module provides streaming dataset adapter using HuggingFace Datasets library,
enabling efficient loading of large-scale datasets (100GB+) without exhausting memory.

Key Features:
- Streaming mode (HFIterableDatasetAdapter): Zero memory overhead, suitable for 100GB+ datasets
- Unified interface compatible with legacy Dataset class
- Reproducible shuffling with epoch-based seeds (via HF's set_epoch)
- Checkpoint support using HF's native state_dict/load_state_dict
- PyTorch DataLoader integration for multi-process prefetching

Architecture Note:
- These adapters are used by RolloutDataSource (single instance)
- They generate global data (not sharded by dp_rank)
- Data sharding happens in RolloutManager._split_train_data_by_dp()
"""

import json
import logging

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class HFDatasetAdapterBase:
    """Base class for HF Dataset adapters.

    Defines the common interface that both streaming and cached adapters must implement.
    This ensures compatibility with RolloutDataSource regardless of the underlying mode.
    """

    def get_next_batch(self, num_samples: int) -> list[Sample]:
        """Get next batch of samples (sequential consumption).

        Args:
            num_samples: Number of samples to fetch

        Returns:
            List of Sample objects
        """
        raise NotImplementedError

    def shuffle(self, new_epoch_id: int):
        """Shuffle dataset for new epoch with reproducible seed.

        Args:
            new_epoch_id: Epoch ID used to derive shuffle seed
        """
        raise NotImplementedError

    def get_checkpoint_state(self) -> dict:
        """Get current state for checkpoint.

        Returns:
            Dictionary containing epoch_id, consumed_count, etc.
        """
        raise NotImplementedError

    def load_checkpoint_state(self, state: dict):
        """Load state from checkpoint.

        Args:
            state: State dictionary saved by get_checkpoint_state()
        """
        raise NotImplementedError

    # NOTE: __len__ and __getitem__ are NOT implemented
    # Reason: Streaming datasets cannot determine length after filtering
    # Slime's data consumption is sequential, not random access


class HFStreamingDatasetWrapper(TorchIterableDataset):
    """Thin wrapper around HF IterableDataset for PyTorch DataLoader compatibility.

    This wrapper leverages HF's native capabilities:
    - state_dict() / load_state_dict() for checkpoint save/resume
    - set_epoch() for automatic reshuffling (effective_seed = seed + epoch)
    - shuffle(seed, buffer_size) for fast approximate shuffling

    We only add __len__() support via known dataset_size.

    Args:
        hf_dataset: HF IterableDataset (already with .shuffle() applied if needed)
        dataset_size: Known dataset size for __len__() support
    """

    def __init__(self, hf_dataset: HFIterableDataset, dataset_size: int):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        return self.dataset_size

    def set_epoch(self, epoch: int):
        """Delegate to HF's set_epoch() - triggers automatic reshuffle."""
        self.hf_dataset.set_epoch(epoch)

    def state_dict(self) -> dict:
        """Delegate to HF's state_dict() - returns shard + example position."""
        return self.hf_dataset.state_dict()

    def load_state_dict(self, state_dict: dict):
        """Delegate to HF's load_state_dict() - resumes from checkpoint."""
        self.hf_dataset.load_state_dict(state_dict)

    def __iter__(self):
        """Iterate up to dataset_size samples."""
        count = 0
        for sample in self.hf_dataset:
            yield sample
            count += 1
            if count >= self.dataset_size:
                break


def create_streaming_dataloader(
    hf_dataset: HFIterableDataset,
    dataset_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    seed: int = 42,
    shuffle_buffer_size: int = 10000,
    do_shuffle: bool = True,
) -> tuple[DataLoader, HFStreamingDatasetWrapper]:
    """Create DataLoader from HF streaming dataset with known size.

    Args:
        hf_dataset: HF IterableDataset
        dataset_size: Known dataset size for __len__() and epoch tracking
        num_workers: DataLoader workers (0 for single-process)
        prefetch_factor: Prefetch factor per worker
        seed: Random seed for shuffling
        shuffle_buffer_size: Buffer size for approximate shuffling
        do_shuffle: Whether to enable shuffling

    Returns:
        Tuple of (DataLoader, HFStreamingDatasetWrapper)
    """
    # Apply shuffle at creation time (with buffer_size for fast approximate shuffle)
    if do_shuffle:
        hf_dataset = hf_dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    wrapper = HFStreamingDatasetWrapper(hf_dataset, dataset_size)

    dataloader = DataLoader(
        wrapper,
        batch_size=None,  # Return individual samples (we batch ourselves)
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return dataloader, wrapper


class HFIterableDatasetAdapter(HFDatasetAdapterBase):
    """Streaming mode HF Dataset adapter using PyTorch DataLoader.

    This adapter enables loading and processing large datasets (100GB+) without
    loading everything into memory. It uses HuggingFace's streaming mode combined
    with PyTorch DataLoader for multi-process prefetching.

    Uses HF's native checkpoint support:
    - state_dict() / load_state_dict() for efficient save/resume
    - set_epoch() for automatic reshuffling (effective_seed = seed + epoch)
    - shuffle(seed, buffer_size) for fast approximate shuffling

    Key Design Decisions:
    - No .shard(dp_rank): RolloutManager is a single instance, generates global data
    - dataset_size is required for __len__() support and epoch tracking
    - PyTorch DataLoader handles prefetching (replaces custom threading)
    - Sequential consumption: No random access, only get_next_batch()

    Args:
        path: Dataset path (local JSONL/Parquet or HF hub)
        dataset_size: Known dataset size (required for epoch tracking)
        tokenizer: HuggingFace tokenizer
        processor: Optional multimodal processor
        max_length: Max prompt length for filtering
        prompt_key: Key for prompt in raw data (default: "text")
        label_key: Key for label in raw data
        tool_key: Key for tools in raw data
        metadata_key: Key for metadata (default: "metadata")
        multimodal_keys: Mapping of multimodal types to keys
        seed: Random seed for shuffle (default: 42)
        apply_chat_template: Whether to apply chat template (default: False)
        apply_chat_template_kwargs: Additional kwargs for chat template
        dp_size: Data parallel size (NOT used for sharding, only buffer sizing)
        num_workers: Number of DataLoader workers (default: 4)
        prefetch_factor: Prefetch factor per worker (default: 2)
        shuffle_buffer_size: Buffer size for HF shuffle (default: 10000)
        do_shuffle: Whether to enable shuffling (default: True)
    """

    def __init__(
        self,
        path: str,
        dataset_size: int,
        tokenizer,
        processor,
        max_length: int | None,
        *,
        prompt_key: str = "text",
        label_key: str | None = None,
        tool_key: str | None = None,
        metadata_key: str = "metadata",
        multimodal_keys: dict | None = None,
        seed: int = 42,
        apply_chat_template: bool = False,
        apply_chat_template_kwargs: dict | None = None,
        dp_size: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        shuffle_buffer_size: int = 10000,
        do_shuffle: bool = True,
    ):
        self.path = path
        self.dataset_size = dataset_size
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.label_key = label_key
        self.tool_key = tool_key
        self.metadata_key = metadata_key
        self.multimodal_keys = multimodal_keys
        self.seed = seed
        self.apply_chat_template = apply_chat_template
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.dp_size = dp_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_buffer_size = shuffle_buffer_size
        self.do_shuffle = do_shuffle

        # State tracking
        self.epoch_id = 0
        self.consumed_count = 0  # Samples consumed in current epoch
        self.global_consumed_count = 0  # Total samples consumed across all epochs

        # Load and process HF dataset
        hf_dataset = self._load_and_process_dataset()

        # Create DataLoader with wrapper
        self.dataloader, self._wrapper = create_streaming_dataloader(
            hf_dataset,
            dataset_size=dataset_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            do_shuffle=do_shuffle,
        )
        self._iter = None

        logger.info(
            f"HFIterableDatasetAdapter initialized: "
            f"path={path}, dataset_size={dataset_size}, "
            f"num_workers={num_workers}, shuffle_buffer={shuffle_buffer_size}"
        )

    def __len__(self) -> int:
        return self.dataset_size

    def _load_and_process_dataset(self) -> HFIterableDataset:
        """Load base dataset and apply processing pipeline.

        Returns:
            Processed HF IterableDataset ready for iteration
        """
        logger.info(f"Loading dataset from {self.path} (streaming mode)")

        # Determine file type and load
        if self.path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=self.path, split="train", streaming=True)
        elif self.path.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=self.path, split="train", streaming=True)
        else:
            # Try as HF dataset name
            try:
                dataset = load_dataset(self.path, split="train", streaming=True)
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset from {self.path}. "
                    f"Supported formats: .jsonl, .parquet, or HuggingFace dataset name. "
                    f"Error: {e}"
                ) from e

        # Apply preprocessing (map + filter)
        dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=128,
        )

        # Filter out invalid samples
        dataset = dataset.filter(lambda x: x["is_valid"])

        return dataset

    def _preprocess_function(self, examples: dict) -> dict:
        """Preprocess function for HF .map().

        This function processes a batch of raw samples and converts them to Sample objects.
        Samples that are too long are filtered out by marking is_valid=False.

        Args:
            examples: Batch of raw data from HF dataset

        Returns:
            Processed batch with 'samples' and 'is_valid' fields
        """
        from slime.utils.data import _build_messages, _should_skip_prompt

        processed_samples = []
        is_valid_list = []

        batch_size = len(examples[list(examples.keys())[0]])  # Get batch size from first key

        for idx in range(batch_size):
            # Extract single example
            data = {k: v[idx] for k, v in examples.items()}

            try:
                # Build messages
                as_conversation = self.apply_chat_template
                prompt = _build_messages(data, self.prompt_key, as_conversation, self.multimodal_keys)

                # Handle metadata
                metadata = data.get(self.metadata_key) or {}

                # Handle tools
                tools = None
                if self.tool_key is not None and self.tool_key in data:
                    tools = data[self.tool_key]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    elif isinstance(tools, np.ndarray):
                        tools = tools.tolist()
                    assert isinstance(tools, list), f"tools must be a list, got {type(tools)}"
                    metadata["tools"] = tools

                # Apply chat template
                if self.apply_chat_template:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=True,
                        **self.apply_chat_template_kwargs,
                    )
                else:
                    formatted_prompt = prompt

                # Handle multimodal (optional, skip if not using multimodal)
                multimodal_inputs = None
                if self.processor:
                    # Note: Multimodal support is not prioritized in Phase 1
                    logger.warning("Multimodal support is experimental in streaming mode")
                    try:
                        from qwen_vl_utils import process_vision_info

                        assert isinstance(prompt, list), "prompt must be a list when processor is not None"
                        images, videos = process_vision_info(prompt)
                        multimodal_inputs = {"images": images, "videos": videos}
                    except Exception as e:
                        logger.warning(f"Failed to process multimodal input: {e}, skipping sample")
                        is_valid_list.append(False)
                        processed_samples.append(None)
                        continue

                # Filter by length
                if _should_skip_prompt(
                    formatted_prompt, self.tokenizer, self.processor, self.max_length, multimodal_inputs
                ):
                    is_valid_list.append(False)
                    processed_samples.append(None)
                    continue

                # Create Sample object
                sample = Sample(
                    prompt=formatted_prompt,
                    label=data[self.label_key] if self.label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )

                processed_samples.append(sample)
                is_valid_list.append(True)

            except Exception as e:
                logger.warning(f"Failed to preprocess sample: {e}, skipping")
                is_valid_list.append(False)
                processed_samples.append(None)
                continue

        return {"samples": processed_samples, "is_valid": is_valid_list}

    def get_next_batch(self, num_samples: int) -> list[Sample]:
        """Get next batch of samples using DataLoader.

        This is the main consumption interface. StopIteration naturally propagates
        to the main thread, enabling clean epoch transitions.

        Args:
            num_samples: Number of samples to fetch

        Returns:
            List of Sample objects
        """
        if self._iter is None:
            self._iter = iter(self.dataloader)

        samples = []
        for _ in range(num_samples):
            try:
                # Get processed sample from DataLoader
                sample_data = next(self._iter)
                # Extract Sample object from the processed dict
                sample = sample_data["samples"]
                if sample is None:
                    continue  # Skip invalid (should be filtered, but just in case)

                samples.append(sample)
                self.consumed_count += 1
                self.global_consumed_count += 1

            except StopIteration:
                # Epoch ended - clean transition in main thread!
                logger.info(f"Epoch {self.epoch_id} completed ({self.consumed_count} samples)")
                self.epoch_id += 1
                self.consumed_count = 0
                self._wrapper.set_epoch(self.epoch_id)  # Triggers reshuffle in HF
                self._iter = iter(self.dataloader)

                # Get sample from new epoch
                try:
                    sample_data = next(self._iter)
                    sample = sample_data["samples"]
                    if sample is not None:
                        samples.append(sample)
                        self.consumed_count += 1
                        self.global_consumed_count += 1
                except StopIteration:
                    logger.warning("New epoch iterator immediately exhausted")
                    break

        return samples

    def shuffle(self, new_epoch_id: int):
        """Shuffle for new epoch.

        This is called by RolloutDataSource when starting a new epoch.
        Delegates to HF's set_epoch() which handles reshuffling automatically.

        Args:
            new_epoch_id: New epoch ID
        """
        if self.epoch_id == new_epoch_id:
            return

        logger.info(f"Shuffling for epoch {new_epoch_id} (current epoch: {self.epoch_id})")
        self.epoch_id = new_epoch_id
        self.consumed_count = 0
        self._wrapper.set_epoch(new_epoch_id)  # HF handles reshuffling
        self._iter = iter(self.dataloader)

    def get_checkpoint_state(self) -> dict:
        """Get state for checkpoint using HF's native state_dict.

        Returns:
            State dictionary containing epoch_id, consumed_count, and HF state
        """
        return {
            "epoch_id": self.epoch_id,
            "consumed_count": self.consumed_count,
            "global_consumed_count": self.global_consumed_count,
            "hf_state_dict": self._wrapper.state_dict(),  # HF's native state
        }

    def load_checkpoint_state(self, state: dict):
        """Load state from checkpoint using HF's native load_state_dict.

        Args:
            state: State dictionary saved by get_checkpoint_state()
        """
        self.epoch_id = state.get("epoch_id", 0)
        self.consumed_count = state.get("consumed_count", 0)
        self.global_consumed_count = state.get("global_consumed_count", 0)

        # Restore HF iterator state (handles shard + example position)
        if "hf_state_dict" in state:
            self._wrapper.load_state_dict(state["hf_state_dict"])

        self._wrapper.set_epoch(self.epoch_id)
        self._iter = iter(self.dataloader)

        logger.info(
            f"Loaded checkpoint: epoch={self.epoch_id}, "
            f"consumed={self.consumed_count}, "
            f"global_consumed={self.global_consumed_count} "
            f"(using HF native checkpoint)"
        )

    def __iter__(self):
        """Iterate over dataset (for compatibility).

        Note: Prefer get_next_batch() for production use.
        """
        if self._iter is None:
            self._iter = iter(self.dataloader)

        while True:
            try:
                sample_data = next(self._iter)
                sample = sample_data["samples"]
                if sample is None:
                    continue
                self.consumed_count += 1
                self.global_consumed_count += 1
                yield sample
            except StopIteration:
                # Epoch ended
                self.epoch_id += 1
                self.consumed_count = 0
                self._wrapper.set_epoch(self.epoch_id)
                self._iter = iter(self.dataloader)
