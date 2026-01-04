"""Tests for data invariant validation utilities."""

import pytest

from tests.utils.data_validation import validate_rollout_data, validate_sample_invariants


class MockSample:
    """Mock Sample object for testing."""

    def __init__(self):
        self.tokens = None
        self.response_length = None
        self.loss_mask = None


class TestSampleInvariants:
    """Tests for validate_sample_invariants."""

    def test_valid_sample(self):
        """Valid sample should pass validation."""
        sample = MockSample()
        sample.tokens = [1, 2, 3, 4, 5]
        sample.response_length = 3
        sample.loss_mask = [1, 1, 1]
        validate_sample_invariants(sample)  # Should not raise

    def test_none_values_pass(self):
        """Sample with None values should pass validation."""
        sample = MockSample()
        validate_sample_invariants(sample)  # Should not raise

    def test_invalid_response_length(self):
        """Zero or negative response_length should fail."""
        sample = MockSample()
        sample.response_length = 0
        with pytest.raises(ValueError, match="response_length must be > 0"):
            validate_sample_invariants(sample)

    def test_invalid_loss_mask_length(self):
        """loss_mask length != response_length should fail."""
        sample = MockSample()
        sample.tokens = [1, 2, 3, 4, 5]
        sample.response_length = 3
        sample.loss_mask = [1, 1]  # Wrong length
        with pytest.raises(ValueError, match="loss_mask"):
            validate_sample_invariants(sample)

    def test_tokens_shorter_than_response(self):
        """tokens length < response_length should fail."""
        sample = MockSample()
        sample.tokens = [1, 2]
        sample.response_length = 5
        sample.loss_mask = [1, 1, 1, 1, 1]
        with pytest.raises(ValueError, match="tokens"):
            validate_sample_invariants(sample)

    def test_context_in_error_message(self):
        """Context should appear in error message."""
        sample = MockSample()
        sample.response_length = 0
        with pytest.raises(ValueError, match="test_context"):
            validate_sample_invariants(sample, context="test_context")


class TestRolloutDataInvariants:
    """Tests for validate_rollout_data."""

    def test_valid_rollout_data(self):
        """Valid rollout data should pass validation."""
        rollout_data = {
            "response_lengths": [3, 4],
            "loss_masks": [[1, 1, 1], [1, 1, 1, 1]],
        }
        validate_rollout_data(rollout_data)  # Should not raise

    def test_valid_rollout_data_with_advantages(self):
        """Valid rollout data with advantages should pass."""
        rollout_data = {
            "response_lengths": [3, 4],
            "loss_masks": [[1, 1, 1], [1, 1, 1, 1]],
            "advantages": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]],
            "returns": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        }
        validate_rollout_data(rollout_data)  # Should not raise

    def test_mismatched_list_lengths(self):
        """response_lengths and loss_masks with different lengths should fail."""
        rollout_data = {
            "response_lengths": [3, 4, 5],  # 3 items
            "loss_masks": [[1, 1, 1], [1, 1, 1, 1]],  # 2 items
        }
        with pytest.raises(ValueError, match="response_lengths.*loss_masks"):
            validate_rollout_data(rollout_data)

    def test_mismatched_loss_mask_length(self):
        """loss_mask with wrong length should fail."""
        rollout_data = {
            "response_lengths": [3, 4],
            "loss_masks": [[1, 1], [1, 1, 1, 1]],  # First one wrong
        }
        with pytest.raises(ValueError, match="Sample 0"):
            validate_rollout_data(rollout_data)

    def test_mismatched_advantages_length(self):
        """advantages with wrong length should fail."""
        rollout_data = {
            "response_lengths": [3, 4],
            "loss_masks": [[1, 1, 1], [1, 1, 1, 1]],
            "advantages": [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4]],  # First one wrong
        }
        with pytest.raises(ValueError, match="Sample 0.*advantages"):
            validate_rollout_data(rollout_data)

    def test_empty_data(self):
        """Empty rollout data should pass."""
        rollout_data = {
            "response_lengths": [],
            "loss_masks": [],
        }
        validate_rollout_data(rollout_data)  # Should not raise

    def test_context_in_error_message(self):
        """Context should appear in error message."""
        rollout_data = {
            "response_lengths": [3],
            "loss_masks": [[1, 1]],  # Wrong length
        }
        with pytest.raises(ValueError, match="test_context"):
            validate_rollout_data(rollout_data, context="test_context")
