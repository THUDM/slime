"""HuggingFace Datasets integration for Slime.

This module provides streaming dataset adapter using HuggingFace Datasets library,
enabling efficient loading of large-scale datasets (100GB+) without exhausting memory.

Key Features:
- Streaming mode (HFIterableDatasetAdapter): Zero memory overhead, suitable for 100GB+ datasets
- Unified interface compatible with legacy Dataset class
- Reproducible shuffling with epoch-based seeds
- Checkpoint support for resumable training
- Prefetch buffer for high throughput

Architecture Note:
- These adapters are used by RolloutDataSource (single instance)
- They generate global data (not sharded by dp_rank)
- Data sharding happens in RolloutManager._split_train_data_by_dp()
"""

import json
import logging
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np
from datasets import IterableDataset, load_dataset

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


class HFIterableDatasetAdapter(HFDatasetAdapterBase):
    """Streaming mode HF Dataset adapter (streaming=True).

    This adapter enables loading and processing large datasets (100GB+) without
    loading everything into memory. It uses HuggingFace's streaming mode combined
    with a prefetch buffer to ensure training throughput.

    Key Design Decisions:
    - No .shard(dp_rank): RolloutManager is a single instance, generates global data
    - Prefetch buffer size = base_buffer_size * dp_size (needs data for all ranks)
    - Epoch-based shuffle: Compatible with buffer-based shuffle (10K buffer default)
    - Sequential consumption: No random access, only get_next_batch()

    Args:
        path: Dataset path (local JSONL/Parquet or HF hub)
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
        buffer_size: Base prefetch buffer size (actual = base * dp_size)
        shuffle_buffer_size: Buffer size for HF shuffle (default: 10000)
        num_proc: Number of parallel workers for preprocessing (default: 8)
    """

    def __init__(
        self,
        path: str,
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
        buffer_size: int = 1000,
        shuffle_buffer_size: int = 10000,
        num_proc: int = 8,
    ):
        self.path = path
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
        self.base_buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_proc = num_proc

        # State tracking
        self.epoch_id = 0
        self.consumed_count = 0  # Samples consumed in current epoch
        self.global_consumed_count = 0  # Total samples consumed across all epochs

        # Prefetch components
        self._prefetch_queue: Queue | None = None
        self._prefetch_thread: Thread | None = None
        self._stop_event: Event | None = None
        self._current_iterator: Iterator | None = None

        # Load base dataset (but don't start prefetch yet)
        # Prefetch will be started on first get_next_batch() call
        self._base_dataset = self._load_base_dataset()

        logger.info(
            f"HFIterableDatasetAdapter initialized: "
            f"path={path}, dp_size={dp_size}, "
            f"buffer_size={self.base_buffer_size}*{dp_size}={self.base_buffer_size * dp_size}, "
            f"shuffle_buffer={shuffle_buffer_size}"
        )

    def _load_base_dataset(self) -> IterableDataset:
        """Load base dataset from HuggingFace Datasets.

        Returns:
            IterableDataset instance
        """
        logger.info(f"Loading dataset from {self.path} (streaming mode)")

        # Determine file type
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

        # NOTE: We do NOT call .shard(dp_rank) here!
        # Reason: RolloutManager is a single instance without dp_rank
        # It generates global data for all ranks, then splits in _split_train_data_by_dp()

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

    def _create_epoch_iterator(self) -> Iterator:
        """Create iterator for current epoch.

        This handles:
        1. Shuffling with epoch-specific seed
        2. Preprocessing with parallel workers
        3. Skipping already consumed samples (for checkpoint resume)

        Returns:
            Iterator over processed samples
        """
        dataset = self._base_dataset

        # Shuffle with epoch-specific seed for reproducibility
        shuffle_seed = self.seed + self.epoch_id
        logger.info(
            f"Epoch {self.epoch_id}: Shuffling with seed {shuffle_seed}, buffer_size={self.shuffle_buffer_size}"
        )
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=self.shuffle_buffer_size)

        # Apply preprocessing
        # NOTE: For IterableDataset, .map() processes on-the-fly, no pre-computation
        # NOTE: For IterableDataset, column_names may be None before iteration
        # We skip remove_columns to avoid issues; the filter() will keep only needed columns
        dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=100,  # Process 100 raw samples at a time
        )

        # Filter out invalid samples
        dataset = dataset.filter(lambda x: x["is_valid"])

        # Create iterator
        iterator = iter(dataset)

        # Skip already consumed samples (for checkpoint resume)
        if self.consumed_count > 0:
            logger.info(f"Skipping {self.consumed_count} already consumed samples for checkpoint resume")
            skipped = 0
            try:
                for _ in range(self.consumed_count):
                    next(iterator)
                    skipped += 1
            except StopIteration:
                logger.warning(
                    f"Dataset exhausted while skipping (skipped {skipped}/{self.consumed_count}), "
                    f"resetting to start of next epoch"
                )
                self.consumed_count = 0
                self.epoch_id += 1
                return self._create_epoch_iterator()

        return iterator

    def _prefetch_worker(self):
        """Background thread that prefetches samples into queue."""
        logger.info("Prefetch worker started")

        while not self._stop_event.is_set():
            try:
                # Get next sample from current iterator
                # NOTE: After .map(batched=True) + .filter(), iterator yields single records
                # not batches. Each record has {"samples": single_Sample, "is_valid": bool}
                sample_data = next(self._current_iterator)

                # Extract the single Sample object (not a list!)
                sample = sample_data["samples"]
                if sample is None:
                    continue  # Skip invalid samples (should be filtered out, but just in case)

                if self._stop_event.is_set():
                    break

                # Block until queue has space
                try:
                    self._prefetch_queue.put(sample, timeout=1.0)
                except Exception:
                    if not self._stop_event.is_set():
                        raise

            except StopIteration:
                # End of epoch, restart iterator
                logger.info(f"Epoch {self.epoch_id} completed, starting epoch {self.epoch_id + 1}")
                self.epoch_id += 1
                self.consumed_count = 0
                self._current_iterator = self._create_epoch_iterator()

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Prefetch worker error: {e}", exc_info=True)
                break

        logger.info("Prefetch worker stopped")

    def start_prefetch(self):
        """Start background prefetch thread."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            logger.warning("Prefetch already started")
            return

        # Calculate actual buffer size: base * dp_size
        # Reason: Need to generate data for all DP ranks
        actual_buffer_size = self.base_buffer_size * self.dp_size

        self._prefetch_queue = Queue(maxsize=actual_buffer_size)
        self._stop_event = Event()
        self._current_iterator = self._create_epoch_iterator()

        self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True, name="HFDatasetPrefetch")
        self._prefetch_thread.start()

        logger.info(
            f"Started prefetch with buffer size {actual_buffer_size} (base={self.base_buffer_size} * dp_size={self.dp_size})"
        )

    def stop_prefetch(self):
        """Stop background prefetch thread."""
        if self._prefetch_thread is None:
            return

        logger.info("Stopping prefetch...")
        self._stop_event.set()

        # Give thread time to stop gracefully
        self._prefetch_thread.join(timeout=5.0)
        if self._prefetch_thread.is_alive():
            logger.warning("Prefetch thread did not stop gracefully")

        # Clear queue
        if self._prefetch_queue is not None:
            while not self._prefetch_queue.empty():
                try:
                    self._prefetch_queue.get_nowait()
                except Empty:
                    break

        self._prefetch_thread = None
        self._prefetch_queue = None
        logger.info("Prefetch stopped")

    def get_next_batch(self, num_samples: int) -> list[Sample]:
        """Get next batch of samples (sequential consumption).

        This is the main consumption interface, replacing the old __getitem__.

        Args:
            num_samples: Number of samples to fetch

        Returns:
            List of Sample objects
        """
        # Lazy start of prefetch on first call
        if self._prefetch_queue is None:
            self.start_prefetch()

        samples = []
        for _ in range(num_samples):
            try:
                sample = self._prefetch_queue.get(timeout=30.0)
                samples.append(sample)
                self.consumed_count += 1
                self.global_consumed_count += 1
            except Empty:
                logger.warning(
                    f"Prefetch queue timeout after getting {len(samples)}/{num_samples} samples. "
                    f"This may indicate slow data preprocessing or insufficient buffer size."
                )
                break

        return samples

    def shuffle(self, new_epoch_id: int):
        """Shuffle for new epoch.

        This is called by RolloutDataSource when starting a new epoch.

        Args:
            new_epoch_id: New epoch ID
        """
        if self.epoch_id == new_epoch_id:
            return

        logger.info(f"Shuffling for epoch {new_epoch_id} (current epoch: {self.epoch_id})")

        # Stop current prefetch
        self.stop_prefetch()

        # Update epoch and reset consumed count
        self.epoch_id = new_epoch_id
        self.consumed_count = 0

        # Restart prefetch with new epoch
        self.start_prefetch()

    def get_checkpoint_state(self) -> dict:
        """Get state for checkpoint.

        Returns:
            State dictionary containing epoch_id, consumed_count, etc.
        """
        return {
            "epoch_id": self.epoch_id,
            "consumed_count": self.consumed_count,
            "global_consumed_count": self.global_consumed_count,
        }

    def load_checkpoint_state(self, state: dict):
        """Load state from checkpoint.

        Args:
            state: State dictionary saved by get_checkpoint_state()
        """
        self.epoch_id = state.get("epoch_id", 0)
        self.consumed_count = state.get("consumed_count", 0)
        self.global_consumed_count = state.get("global_consumed_count", 0)

        logger.info(
            f"Loaded checkpoint: epoch={self.epoch_id}, "
            f"consumed={self.consumed_count}, "
            f"global_consumed={self.global_consumed_count}"
        )

    def __iter__(self):
        """Iterate over dataset (for compatibility).

        Note: Prefer get_next_batch() for production use.
        """
        if self._prefetch_queue is None:
            self.start_prefetch()

        while True:
            try:
                sample = self._prefetch_queue.get(timeout=30.0)
                self.consumed_count += 1
                self.global_consumed_count += 1
                yield sample
            except Empty:
                logger.warning("Iterator exhausted (prefetch queue timeout)")
                break

    def __del__(self):
        """Cleanup: Stop prefetch thread when object is destroyed."""
        self.stop_prefetch()
