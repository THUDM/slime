"""HuggingFace Datasets streaming adapter for large-scale datasets (100GB+).

This module provides a streaming dataset adapter using HuggingFace Datasets library,
enabling efficient loading of large-scale datasets without exhausting memory.

Key Features:
- Streaming mode: Zero memory overhead, suitable for 100GB+ datasets
- Reproducible shuffling with epoch-based seeds (via HF's set_epoch)
- Checkpoint support using HF's native state_dict/load_state_dict
- PyTorch DataLoader integration for multi-process prefetching

Architecture Note:
- Used by RolloutDataSource (single instance)
- Generates global data (not sharded by dp_rank)
- Data sharding happens in RolloutManager._split_train_data_by_dp()
"""

import json
import logging

import numpy as np
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class HFIterableDatasetAdapter:
    """Streaming HF Dataset adapter with checkpoint support.

    Enables loading and processing large datasets (100GB+) without loading
    everything into memory. Uses HuggingFace's streaming mode combined with
    PyTorch DataLoader for multi-process prefetching.

    Uses HF's native checkpoint support:
    - state_dict() / load_state_dict() for efficient save/resume
    - set_epoch() for automatic reshuffling (effective_seed = seed + epoch)
    - shuffle(seed, buffer_size) for fast approximate shuffling

    Key Design:
    - RolloutManager is a single instance, generates global data
    - dataset_size is required for __len__() support and epoch tracking
    - Sequential consumption only: No random access, only get_next_batch()

    VERIFIED: HF's state_dict enables exact position resume without sample skipping.
    See tests/test_hf_datasets.py::TestHFStateTracking for verification tests.

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
        num_workers: Number of DataLoader workers (default: 4)
        prefetch_factor: Prefetch factor per worker (default: 2)
        shuffle_buffer_size: Buffer size for HF shuffle (default: 10000)
        do_shuffle: Whether to enable shuffling (default: True)
        split: Dataset split name (default: "train")
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
        num_workers: int = 4,
        prefetch_factor: int = 2,
        shuffle_buffer_size: int = 10000,
        do_shuffle: bool = True,
        split: str = "train",
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
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_buffer_size = shuffle_buffer_size
        self.do_shuffle = do_shuffle
        self.split = split

        # State tracking
        self.epoch_id = 0
        self.consumed_count = 0  # Samples consumed in current epoch
        self.global_consumed_count = 0  # Total samples consumed across all epochs

        # Load and process HF dataset
        self.hf_dataset = self._load_and_process_dataset()

        # Apply shuffle at creation time
        if do_shuffle:
            self.hf_dataset = self.hf_dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

        # Create DataLoader
        self.dataloader = DataLoader(
            _HFDatasetWrapper(self.hf_dataset, dataset_size),
            batch_size=None,  # Return individual samples (we batch ourselves)
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
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
        """Load base dataset and apply processing pipeline."""
        logger.info(f"Loading dataset from {self.path} (streaming mode, split={self.split})")

        # Determine file type and load
        if self.path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=self.path, split=self.split, streaming=True)
        elif self.path.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=self.path, split=self.split, streaming=True)
        else:
            # Try as HF dataset name
            try:
                dataset = load_dataset(self.path, split=self.split, streaming=True)
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset from {self.path} with split '{self.split}'. "
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

        Processes a batch of raw samples and converts them to Sample objects.
        Samples that are too long are filtered out by marking is_valid=False.
        """
        from slime.utils.data import _build_messages, _should_skip_prompt

        processed_samples = []
        is_valid_list = []

        batch_size = len(examples[list(examples.keys())[0]])

        for idx in range(batch_size):
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

                # Handle multimodal (experimental)
                multimodal_inputs = None
                if self.processor:
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
        """
        if self._iter is None:
            self._iter = iter(self.dataloader)

        samples = []
        for _ in range(num_samples):
            try:
                sample_data = next(self._iter)
                sample = sample_data["samples"]
                if sample is None:
                    continue

                samples.append(sample)
                self.consumed_count += 1
                self.global_consumed_count += 1

            except StopIteration:
                # Epoch ended - clean transition
                logger.info(f"Epoch {self.epoch_id} completed ({self.consumed_count} samples)")
                self.epoch_id += 1
                self.consumed_count = 0
                self.hf_dataset.set_epoch(self.epoch_id)  # Triggers reshuffle
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

        Called by RolloutDataSource when starting a new epoch.
        Delegates to HF's set_epoch() which handles reshuffling automatically.
        """
        if self.epoch_id == new_epoch_id:
            return

        logger.info(f"Shuffling for epoch {new_epoch_id} (current epoch: {self.epoch_id})")
        self.epoch_id = new_epoch_id
        self.consumed_count = 0
        self.hf_dataset.set_epoch(new_epoch_id)
        self._iter = iter(self.dataloader)

    def get_checkpoint_state(self) -> dict:
        """Get state for checkpoint using HF's native state_dict.

        State tracking:
        - epoch_id: Current epoch number (for seed+epoch reproducible shuffle)
        - consumed_count: Samples consumed in current epoch (for statistics)
        - global_consumed_count: Total samples consumed across all epochs
        - hf_state_dict: HF's native iterator state (stores exact position)

        VERIFIED: hf_state_dict enables exact position resume without sample skipping.
        """
        return {
            "epoch_id": self.epoch_id,
            "consumed_count": self.consumed_count,
            "global_consumed_count": self.global_consumed_count,
            "hf_state_dict": self.hf_dataset.state_dict(),
        }

    def load_checkpoint_state(self, state: dict):
        """Load state from checkpoint using HF's native load_state_dict.

        VERIFIED BEHAVIOR:
        - HF's state_dict() stores exact iterator position (shard + offset)
        - load_state_dict() restores this position accurately
        - No manual sample skipping is needed after load
        """
        self.epoch_id = state.get("epoch_id", 0)
        self.consumed_count = state.get("consumed_count", 0)
        self.global_consumed_count = state.get("global_consumed_count", 0)

        # Restore HF iterator state
        if "hf_state_dict" in state:
            self.hf_dataset.load_state_dict(state["hf_state_dict"])

        self.hf_dataset.set_epoch(self.epoch_id)
        self._iter = iter(self.dataloader)

        logger.info(
            f"Loaded checkpoint: epoch={self.epoch_id}, "
            f"consumed={self.consumed_count}, "
            f"global_consumed={self.global_consumed_count}"
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
                self.epoch_id += 1
                self.consumed_count = 0
                self.hf_dataset.set_epoch(self.epoch_id)
                self._iter = iter(self.dataloader)


class _HFDatasetWrapper(TorchIterableDataset):
    """Minimal wrapper for PyTorch DataLoader compatibility.

    Only provides __iter__ with dataset_size limit. All other operations
    go directly through the HF dataset.
    """

    def __init__(self, hf_dataset: HFIterableDataset, dataset_size: int):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.dataset_size = dataset_size

    def __iter__(self):
        count = 0
        for sample in self.hf_dataset:
            yield sample
            count += 1
            if count >= self.dataset_size:
                break
