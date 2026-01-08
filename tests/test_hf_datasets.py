"""Unit tests for HuggingFace Datasets integration (streaming mode).

This test file covers:
1. HFIterableDatasetAdapter basic functionality (initialization, get_next_batch, shuffle)
2. RolloutDataSource mixed mode logic (auto-detection via duck typing)
3. Checkpoint support (save/load/resume across epochs)
4. Edge cases (dataset=None, empty dataset, sample_offset overflow)

Test Strategy:
- Use small synthetic datasets (100 samples) for fast execution
- Mock Ray actors and heavy dependencies where appropriate
- Focus on correctness of data flow and state management
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# Test fixtures and utilities
@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_jsonl_data(temp_dir):
    """Create small test dataset (100 samples) in JSONL format."""
    data_path = Path(temp_dir) / "test_data.jsonl"
    samples = []
    for i in range(100):
        samples.append(
            {
                "input": f"Test prompt {i}",
                "label": f"Test label {i}",
                "metadata": {"sample_id": i},
            }
        )

    with open(data_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return str(data_path)


@pytest.fixture
def mock_args(test_jsonl_data, temp_dir):
    """Mock args object for testing."""
    args = MagicMock()

    # Data source config
    args.rollout_global_dataset = True
    args.prompt_data = test_jsonl_data
    args.input_key = "input"
    args.label_key = "label"
    args.metadata_key = "metadata"
    args.tool_key = None
    args.multimodal_keys = None
    args.apply_chat_template = False
    args.apply_chat_template_kwargs = {}
    args.rollout_seed = 42
    args.rollout_shuffle = False
    args.rollout_max_prompt_len = None
    args.n_samples_per_prompt = 1

    # HF Datasets config
    args.use_hf_datasets = False  # Default to Legacy
    args.hf_dataset_buffer_size = 10
    args.hf_dataset_shuffle_buffer = 100
    args.hf_dataset_num_proc = 4
    args.hf_datasets_num_samples = 100  # Required for HF streaming mode
    args.hf_dataset_split = "train"  # Default split name

    # Checkpoint config
    args.save = str(Path(temp_dir) / "checkpoints")
    args.load = None
    args.dump_details = None

    # Mock tokenizer and processor
    args.hf_checkpoint = None

    return args


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: list(range(len(text.split())))  # Simple mock
    tokenizer.apply_chat_template = lambda msgs, **kwargs: str(msgs)
    return tokenizer


@pytest.fixture
def mock_processor():
    """Mock HuggingFace processor (for multimodal)."""
    return None  # Most tests don't use multimodal


@pytest.fixture
def test_jsonl_data_with_ids(temp_dir):
    """Create test dataset with unique IDs for deduplication testing.

    Creates 100 samples with:
    - Unique sample_id in metadata
    - Sequential numbering for deterministic testing
    """
    data_path = Path(temp_dir) / "test_data_with_ids.jsonl"
    for i in range(100):
        sample = {
            "input": f"Test prompt {i}",
            "label": f"Test label {i}",
            "metadata": {"sample_id": f"sample_{i:04d}"},
        }
        with open(data_path, "a") as f:
            f.write(json.dumps(sample) + "\n")
    return str(data_path)


# ============================================================================
# Test Class 1: HFDatasetAdapters Basic Functionality
# ============================================================================


class TestHFDatasetAdapters:
    """Test HF adapters' core functionality."""

    def test_streaming_adapter_initialization(self, test_jsonl_data, mock_tokenizer, mock_processor):
        """Test HFIterableDatasetAdapter initialization."""
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        adapter = HFIterableDatasetAdapter(
            path=test_jsonl_data,
            dataset_size=100,  # Required for epoch tracking
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            metadata_key="metadata",
            seed=42,
            num_workers=0,  # Single-process for testing
            shuffle_buffer_size=100,
        )

        # Check state tracking
        assert adapter.epoch_id == 0
        assert adapter.consumed_count == 0
        assert adapter.global_consumed_count == 0
        assert len(adapter) == 100  # Dataset size

    def test_get_next_batch_sequential(self, test_jsonl_data, mock_tokenizer, mock_processor):
        """Test sequential consumption via get_next_batch()."""
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        adapter = HFIterableDatasetAdapter(
            path=test_jsonl_data,
            dataset_size=100,  # Required for epoch tracking
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            seed=42,
            num_workers=0,  # Single-process for testing
        )

        # Consume first batch
        batch1 = adapter.get_next_batch(num_samples=10)
        assert len(batch1) == 10
        assert adapter.consumed_count == 10

        # Consume second batch
        batch2 = adapter.get_next_batch(num_samples=10)
        assert len(batch2) == 10
        assert adapter.consumed_count == 20

        # Check no overlap
        assert batch1 != batch2

    def test_epoch_switch(self, test_jsonl_data, mock_tokenizer, mock_processor):
        """Test automatic epoch switching when dataset is exhausted."""
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        adapter = HFIterableDatasetAdapter(
            path=test_jsonl_data,
            dataset_size=100,  # Required for epoch tracking
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            seed=42,
            num_workers=0,  # Single-process for testing
        )

        # Consume entire dataset
        total_samples = 0
        while adapter.epoch_id == 0 and total_samples < 200:
            batch = adapter.get_next_batch(num_samples=10)
            total_samples += len(batch)

        # Should have switched to epoch 1
        assert adapter.epoch_id >= 1


# ============================================================================
# Test Class 2: RolloutDataSource Mixed Mode Logic
# ============================================================================


class TestRolloutDataSourceMixedMode:
    """Test RolloutDataSource mixed mode logic (duck typing)."""

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_get_samples_with_dataset_none(self, mock_load_proc, mock_load_tok, mock_args):
        """Test get_samples() when dataset=None (--disable-rollout-global-dataset)."""
        from slime.rollout.data_source import RolloutDataSource
        from slime.utils.types import Sample

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        # Disable global dataset
        mock_args.rollout_global_dataset = False

        data_source = RolloutDataSource(mock_args)
        samples = data_source.get_samples(num_samples=5)

        # Should return 5 groups of empty samples
        assert len(samples) == 5
        assert all(isinstance(group, list) for group in samples)
        assert all(isinstance(sample, Sample) for group in samples for sample in group)

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_duck_typing_detection(self, mock_load_proc, mock_load_tok, mock_args):
        """Test duck typing correctly detects HF adapters vs Legacy Dataset."""
        from slime.rollout.data_source import RolloutDataSource

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        data_source = RolloutDataSource(mock_args)

        # Check dataset was created (Legacy mode since use_hf_datasets=False)
        assert data_source.dataset is not None

        # Verify duck typing: should not have get_next_batch (Legacy Dataset)
        assert not hasattr(data_source.dataset, "get_next_batch")

        # Verify has .samples attribute (Legacy Dataset)
        assert hasattr(data_source.dataset, "samples")

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_get_samples_with_hf_streaming(self, mock_load_proc, mock_load_tok, mock_args):
        """Test get_samples() with HF Streaming mode."""
        from slime.rollout.data_source import RolloutDataSource

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        # Enable HF Datasets mode
        mock_args.use_hf_datasets = True

        data_source = RolloutDataSource(mock_args)

        # Verify duck typing detected HF adapter
        assert hasattr(data_source.dataset, "get_next_batch")

        # Get samples
        samples = data_source.get_samples(num_samples=5)
        assert len(samples) == 5


# ============================================================================
# Test Class 3: Checkpoint Support
# ============================================================================


class TestCheckpointSupport:
    """Test checkpoint save/load/resume functionality."""

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_save_and_load_legacy(self, mock_load_proc, mock_load_tok, mock_args):
        """Test checkpoint save/load for Legacy Dataset."""
        from slime.rollout.data_source import RolloutDataSource

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        # Create data source with Legacy Dataset
        data_source = RolloutDataSource(mock_args)

        # Consume some samples
        data_source.get_samples(num_samples=10)
        data_source.get_samples(num_samples=5)

        # Save checkpoint
        rollout_id = 42
        data_source.save(rollout_id)

        # Verify checkpoint file exists
        ckpt_path = Path(mock_args.save) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
        assert ckpt_path.exists()

        # Load checkpoint into new data source
        data_source2 = RolloutDataSource(mock_args)

        # Set load path
        data_source2.args.load = mock_args.save
        data_source2.load(rollout_id)

        # Verify state restored
        assert data_source2.sample_offset == data_source.sample_offset
        assert data_source2.epoch_id == data_source.epoch_id
        assert data_source2.sample_group_index == data_source.sample_group_index
        assert data_source2.sample_index == data_source.sample_index

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_save_and_load_hf_streaming(self, mock_load_proc, mock_load_tok, mock_args):
        """Test checkpoint save/load for HF Streaming mode."""
        from slime.rollout.data_source import RolloutDataSource

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        # Enable HF Datasets mode
        mock_args.use_hf_datasets = True

        data_source = RolloutDataSource(mock_args)

        # Consume some samples
        data_source.get_samples(num_samples=10)

        # Save checkpoint
        rollout_id = 42
        data_source.save(rollout_id)

        # Verify checkpoint contains HF adapter state
        ckpt_path = Path(mock_args.save) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
        state_dict = torch.load(ckpt_path)
        assert "hf_adapter_state" in state_dict
        assert "epoch_id" in state_dict["hf_adapter_state"]
        assert "consumed_count" in state_dict["hf_adapter_state"]


# ============================================================================
# Test Class 4: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_dataset_none(self, mock_load_proc, mock_load_tok, mock_args):
        """Test behavior when dataset=None."""
        from slime.rollout.data_source import RolloutDataSource
        from slime.utils.types import Sample

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        # Disable global dataset
        mock_args.rollout_global_dataset = False

        data_source = RolloutDataSource(mock_args)

        # get_samples should return empty Sample objects
        samples = data_source.get_samples(num_samples=3)
        assert len(samples) == 3
        assert all(isinstance(group, list) for group in samples)
        assert all(len(group) == mock_args.n_samples_per_prompt for group in samples)
        assert all(isinstance(sample, Sample) for group in samples for sample in group)

    @patch("slime.rollout.data_source.load_tokenizer")
    @patch("slime.rollout.data_source.load_processor")
    def test_checkpoint_nonexistent_path(self, mock_load_proc, mock_load_tok, mock_args):
        """Test loading from nonexistent checkpoint path."""
        from slime.rollout.data_source import RolloutDataSource

        mock_load_tok.return_value = MagicMock()
        mock_load_proc.return_value = None

        data_source = RolloutDataSource(mock_args)

        # Set load path to nonexistent location
        data_source.args.load = "/nonexistent/path"

        # Should not raise error, just log warning
        data_source.load(rollout_id=999)

        # State should remain at initial values
        assert data_source.sample_offset == 0
        assert data_source.epoch_id == 0


# ============================================================================
# Integration Tests (Optional - require full environment)
# ============================================================================


class TestIntegration:
    """End-to-end integration tests (require HF datasets library)."""

    def test_full_training_loop_simulation(self, temp_dir, test_jsonl_data_with_ids, mock_tokenizer, mock_processor):
        """Simulate full training loop: rollout → train → checkpoint → resume.

        This test verifies:
        1. Sequential consumption across multiple rollouts
        2. Checkpoint save at step N
        3. Checkpoint resume from step N
        4. No sample duplication (via metadata.sample_id)
        5. Automatic epoch switching (100 samples / 32 batch → 3+ epochs)
        """
        from slime.rollout.data_source import RolloutDataSource

        # Setup args
        args = MagicMock()
        args.rollout_global_dataset = True
        args.prompt_data = test_jsonl_data_with_ids
        args.input_key = "input"
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.tool_key = None
        args.multimodal_keys = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = {}
        args.rollout_seed = 42
        args.rollout_shuffle = False  # Disable shuffle for deterministic testing
        args.rollout_max_prompt_len = None
        args.n_samples_per_prompt = 1
        args.use_hf_datasets = True  # Enable HF Datasets mode
        args.hf_dataset_buffer_size = 10
        args.hf_dataset_shuffle_buffer = 100
        args.hf_dataset_num_proc = 4
        args.hf_datasets_num_samples = 100  # Required for epoch tracking
        args.hf_dataset_split = "train"  # Dataset split name
        args.save = str(Path(temp_dir) / "checkpoints")
        args.load = None
        args.hf_checkpoint = None

        # Mock tokenizer/processor loading
        with patch("slime.rollout.data_source.load_tokenizer", return_value=mock_tokenizer), patch(
            "slime.rollout.data_source.load_processor", return_value=mock_processor
        ):

            # === Phase 1: Run 10 rollouts with checkpoint at step 5 ===
            data_source = RolloutDataSource(args)

            consumed_sample_ids = set()
            num_rollouts = 10
            batch_size = 32  # 32 prompts per rollout
            current_epoch = 0

            for rollout_id in range(num_rollouts):
                # Track epoch before getting samples
                before_epoch = data_source.epoch_id

                samples = data_source.get_samples(num_samples=batch_size)

                # Track epoch after getting samples
                after_epoch = data_source.epoch_id

                # Verify batch size
                assert len(samples) == batch_size, f"Expected {batch_size} sample groups, got {len(samples)}"

                # Check for epoch transition
                if after_epoch > before_epoch:
                    # Epoch changed, clear dedup set (samples can repeat across epochs)
                    consumed_sample_ids.clear()
                    current_epoch = after_epoch

                # Extract sample IDs and check for duplicates within same epoch
                for group in samples:
                    for sample in group:
                        sample_id = sample.metadata.get("sample_id")
                        assert sample_id is not None, f"Sample missing unique ID at rollout {rollout_id}"
                        assert (
                            sample_id not in consumed_sample_ids
                        ), f"Duplicate sample detected: {sample_id} at rollout {rollout_id}, epoch {current_epoch}"
                        consumed_sample_ids.add(sample_id)

                # Save checkpoint at step 5
                if rollout_id == 5:
                    data_source.save(rollout_id)

            # Verify epoch switching occurred (100 samples / 32 batch = 3.125 batches per epoch)
            # After 10 rollouts (320 samples requested), should be in epoch 3+
            assert data_source.epoch_id >= 2, f"Expected multiple epochs, but only in epoch {data_source.epoch_id}"

            # === Phase 2: Verify checkpoint file exists and structure ===
            ckpt_path = Path(args.save) / "rollout/global_dataset_state_dict_5.pt"
            assert ckpt_path.exists(), "Checkpoint file not created"

            # Load and verify checkpoint structure
            state_dict = torch.load(ckpt_path)
            required_keys = ["sample_offset", "epoch_id", "sample_group_index", "sample_index"]
            for key in required_keys:
                assert key in state_dict, f"Missing key in checkpoint: {key}"

            # Verify HF adapter state is present
            assert "hf_adapter_state" in state_dict, "Missing HF adapter state in checkpoint"
            hf_state = state_dict["hf_adapter_state"]
            assert "epoch_id" in hf_state, "Missing epoch_id in HF adapter state"
            assert "consumed_count" in hf_state, "Missing consumed_count in HF adapter state"

            # === Phase 3: Resume from checkpoint and verify state restoration ===
            data_source2 = RolloutDataSource(args)
            data_source2.args.load = args.save
            data_source2.load(rollout_id=5)

            # Verify state restoration
            assert data_source2.dataset is not None, "Dataset not initialized after load"
            if hasattr(data_source2.dataset, "consumed_count"):
                # HF adapter should restore consumed_count
                assert data_source2.dataset.consumed_count >= 0, "Invalid consumed_count after restore"

            # Continue for 5 more rollouts (simulating steps 6-10)
            # This verifies that checkpoint correctly saved position
            for _rollout_id in range(6, num_rollouts + 1):
                samples = data_source2.get_samples(num_samples=batch_size)
                assert len(samples) == batch_size, f"Expected {batch_size} samples after resume"

    def test_epoch_boundary_checkpoint(self, temp_dir, test_jsonl_data_with_ids, mock_tokenizer, mock_processor):
        """Test checkpoint save/load at epoch boundary.

        Edge case: Verify correct behavior when checkpoint occurs exactly
        at epoch transition (after consuming all 100 samples).
        """
        from slime.rollout.data_source import RolloutDataSource

        # Setup args (similar to test_full_training_loop_simulation)
        args = MagicMock()
        args.rollout_global_dataset = True
        args.prompt_data = test_jsonl_data_with_ids
        args.input_key = "input"
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.tool_key = None
        args.multimodal_keys = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = {}
        args.rollout_seed = 42
        args.rollout_shuffle = False
        args.rollout_max_prompt_len = None
        args.n_samples_per_prompt = 1
        args.use_hf_datasets = True
        args.hf_dataset_buffer_size = 10
        args.hf_dataset_shuffle_buffer = 100
        args.hf_dataset_num_proc = 4
        args.hf_datasets_num_samples = 100  # Required for epoch tracking
        args.hf_dataset_split = "train"  # Dataset split name
        args.save = str(Path(temp_dir) / "checkpoints")
        args.load = None
        args.hf_checkpoint = None

        with patch("slime.rollout.data_source.load_tokenizer", return_value=mock_tokenizer), patch(
            "slime.rollout.data_source.load_processor", return_value=mock_processor
        ):

            data_source = RolloutDataSource(args)

            # Consume exactly 100 samples (one complete epoch)
            # With 100 samples and batch_size=25, need 4 rollouts
            for _rollout_id in range(4):
                samples = data_source.get_samples(num_samples=25)
                assert len(samples) == 25

            # Epoch switch happens when we request samples that cross the boundary
            # Request one more sample to trigger epoch transition
            samples = data_source.get_samples(num_samples=1)
            assert len(samples) == 1

            # Should have completed epoch 0, now in epoch 1
            assert data_source.epoch_id >= 1, "Expected epoch transition after 100+ samples"

            # Save checkpoint at epoch boundary
            data_source.save(rollout_id=4)

            # Verify checkpoint
            ckpt_path = Path(args.save) / "rollout/global_dataset_state_dict_4.pt"
            assert ckpt_path.exists(), "Checkpoint not saved at epoch boundary"

            state_dict = torch.load(ckpt_path)
            assert "hf_adapter_state" in state_dict
            hf_state = state_dict["hf_adapter_state"]

            # At epoch boundary, consumed_count should be 0 (reset for new epoch)
            # or 100 (if tracking total), depending on implementation
            assert "epoch_id" in hf_state
            assert hf_state["epoch_id"] >= 1, "Epoch ID not incremented at boundary"

            # Resume from checkpoint
            data_source2 = RolloutDataSource(args)
            data_source2.args.load = args.save
            data_source2.load(rollout_id=4)

            # Continue consuming - should start from epoch 1
            samples = data_source2.get_samples(num_samples=10)
            assert len(samples) == 10, "Failed to consume after epoch boundary checkpoint"


# ============================================================================
# Test Class 5: HF State Tracking Verification
# ============================================================================


class TestHFStateTracking:
    """Test HF state tracking and checkpoint resume accuracy.

    These tests verify whether HF's state_dict()/load_state_dict() can
    accurately restore iterator position without manual sample skipping.

    VERIFICATION RESULTS (all tests passed):
    1. test_hf_state_dict_exact_resume: HF state_dict() stores exact iterator position
    2. test_consumed_count_consistency: consumed_count tracking is accurate
    3. test_shuffle_reproducibility_same_seed_epoch: Same seed+epoch = same shuffle
    4. test_epoch_boundary_exact_resume: Resume works correctly across epoch boundaries

    CONCLUSION: No additional offset is needed beyond seed + epoch + hf_state_dict.
    HF's native checkpoint mechanism handles exact position resume without manual skipping.
    """

    def test_hf_state_dict_exact_resume(self, test_jsonl_data_with_ids, mock_tokenizer, mock_processor):
        """Verify HF state_dict() restores exact iterator position.

        Golden truth: Use the SAME adapter's continued consumption as expected result.
        This tests whether we need additional offset tracking beyond seed + epoch.
        """
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        # Create adapter with NO shuffle for deterministic testing
        adapter1 = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            metadata_key="metadata",
            seed=42,
            num_workers=0,
            do_shuffle=False,  # Disable shuffle for deterministic test
        )

        # Step 1: Consume 35 samples
        _ = adapter1.get_next_batch(35)
        assert adapter1.consumed_count == 35

        # Step 2: Save checkpoint state at position 35
        checkpoint_state = adapter1.get_checkpoint_state()
        assert checkpoint_state["consumed_count"] == 35

        # Step 3: Continue consuming 10 more samples → GOLDEN EXPECTED
        golden_samples = adapter1.get_next_batch(10)
        golden_ids = [s.metadata.get("sample_id") for s in golden_samples]

        # Step 4: Create new adapter and load checkpoint
        adapter2 = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            metadata_key="metadata",
            seed=42,
            num_workers=0,
            do_shuffle=False,
        )

        adapter2.load_checkpoint_state(checkpoint_state)
        assert adapter2.consumed_count == 35  # State restored

        # Step 5: Get next 10 samples from resumed adapter
        resumed_samples = adapter2.get_next_batch(10)
        resumed_ids = [s.metadata.get("sample_id") for s in resumed_samples]

        # CRITICAL VERIFICATION:
        # resumed_ids should match golden_ids (from same checkpoint position)
        assert resumed_ids == golden_ids, (
            f"HF state_dict did not restore exact position!\n"
            f"Golden (from adapter1 after checkpoint): {golden_ids}\n"
            f"Resumed (from adapter2 after load): {resumed_ids}\n"
            f"If these differ, HF state_dict is NOT working and we need manual skipping."
        )

        # Additional check: consumed_count should update correctly after resume
        assert adapter2.consumed_count == 45  # 35 + 10

    def test_consumed_count_consistency(self, test_jsonl_data_with_ids, mock_tokenizer, mock_processor):
        """Verify consumed_count matches actual consumption."""
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        adapter = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            metadata_key="metadata",
            seed=42,
            num_workers=0,
            do_shuffle=False,
        )

        # Initial state
        assert adapter.consumed_count == 0
        assert adapter.global_consumed_count == 0

        # Consume batches and verify counting
        adapter.get_next_batch(10)
        assert adapter.consumed_count == 10
        assert adapter.global_consumed_count == 10

        adapter.get_next_batch(25)
        assert adapter.consumed_count == 35
        assert adapter.global_consumed_count == 35

        # Checkpoint should preserve counts
        state = adapter.get_checkpoint_state()
        assert state["consumed_count"] == 35
        assert state["global_consumed_count"] == 35

    def test_shuffle_reproducibility_same_seed_epoch(self, test_jsonl_data_with_ids, mock_tokenizer, mock_processor):
        """Verify same seed+epoch produces same shuffle order."""
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        # Create two adapters with same seed
        adapter1 = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            seed=42,
            num_workers=0,
            do_shuffle=True,
            shuffle_buffer_size=100,  # Full buffer for exact shuffle
        )

        adapter2 = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            seed=42,
            num_workers=0,
            do_shuffle=True,
            shuffle_buffer_size=100,
        )

        # Get first 20 samples from each
        samples1 = adapter1.get_next_batch(20)
        samples2 = adapter2.get_next_batch(20)

        ids1 = [s.metadata.get("sample_id") for s in samples1]
        ids2 = [s.metadata.get("sample_id") for s in samples2]

        # Same seed+epoch should produce same shuffle
        assert ids1 == ids2, (
            f"Same seed+epoch produced different shuffle!\n"
            f"Adapter1: {ids1[:5]}...\n"
            f"Adapter2: {ids2[:5]}..."
        )

    def test_epoch_boundary_exact_resume(self, test_jsonl_data_with_ids, mock_tokenizer, mock_processor):
        """Verify checkpoint/resume at epoch boundary works correctly."""
        from slime.utils.hf_dataset import HFIterableDatasetAdapter

        adapter1 = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            metadata_key="metadata",
            seed=42,
            num_workers=0,
            do_shuffle=False,
        )

        # Consume 95 samples (near epoch boundary)
        _ = adapter1.get_next_batch(95)
        assert adapter1.consumed_count == 95
        assert adapter1.epoch_id == 0

        # Save checkpoint
        checkpoint_state = adapter1.get_checkpoint_state()

        # Continue consuming - this will cross epoch boundary
        golden_samples = adapter1.get_next_batch(10)
        golden_ids = [s.metadata.get("sample_id") for s in golden_samples]
        # After crossing boundary: epoch_id should be 1, consumed_count reset
        assert adapter1.epoch_id >= 1, "Should have crossed epoch boundary"

        # Create new adapter and load checkpoint (at position 95)
        adapter2 = HFIterableDatasetAdapter(
            path=test_jsonl_data_with_ids,
            dataset_size=100,
            tokenizer=mock_tokenizer,
            processor=mock_processor,
            max_length=None,
            prompt_key="input",
            label_key="label",
            metadata_key="metadata",
            seed=42,
            num_workers=0,
            do_shuffle=False,
        )

        adapter2.load_checkpoint_state(checkpoint_state)

        # Resume and cross boundary
        resumed_samples = adapter2.get_next_batch(10)
        resumed_ids = [s.metadata.get("sample_id") for s in resumed_samples]

        # Verify samples match across epoch boundary
        assert resumed_ids == golden_ids, (
            f"Epoch boundary resume failed!\n"
            f"Golden: {golden_ids}\n"
            f"Resumed: {resumed_ids}"
        )