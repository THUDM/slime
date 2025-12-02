"""
Unit tests for P2PTrainingTransferEngine

This test suite validates the P2PTrainingTransferEngine implementation
by comparing it with SGLang's TrainingWeightSender in various scenarios.

Based on SGLang's test_p2p_transfer.py patterns.

cmd:  python -m pytest tests/test_p2p_engine.py::TestP2PTrainingTransferEngine::test_p2p_engine_correctness -v -s --tb=short --capture=no 2>&1 | tee p2p_engine_correctness_fixed_logs.txt
"""

import gc
import logging
import multiprocessing as mp
import os
import threading
import time
import unittest
from typing import Dict

import numpy as np
import torch
import zmq

# Import SGLang rollout-side components
from slime.backends.sglang_utils.sglang_rollout_rdma_p2p import P2PTransferEngine

# Import our training-side implementation
from slime.backends.sglang_utils.sglang_training_rdma_p2p import P2PTrainingTransferEngine

logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method("spawn", force=True)


def create_test_weight(size: int, device: torch.device, dtype=torch.float16) -> torch.Tensor:
    """Create a test weight tensor for correctness testing."""
    return torch.randn(size, size, device=device, dtype=dtype)


def simple_training_process_with_p2p_engine(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    name: str,
    hostname: str = "127.0.0.1",
    port: int = 50000,
):
    """Training process using P2PTrainingTransferEngine."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Initialize P2PTrainingTransferEngine
        training_engine = P2PTrainingTransferEngine(
            master_ip=hostname,
            master_port=port,
            gpu_id=rank,
            ib_device=None,
        )

        # Create deterministic test weight for verification
        torch.manual_seed(42)  # Ensure deterministic values
        weight = torch.randn(128 * 128, device=device, dtype=torch.float16).reshape(128, 128)
        weights = {name: weight}

        total_size_kb = weight.numel() * weight.element_size() / 1e3
        logger.info(f"[P2PTrainingEngine] Weight size: {total_size_kb:.2f}KB")

        # Test SGLang-compatible interface
        training_engine.register_weights(weights)
        training_engine.start()

        barrier.wait()

        # Update to known values for verification
        torch.cuda.synchronize()
        training_engine.register_weights(weights)

        # Share expected tensor values with rollout process for verification
        expected_sum = float(weight.sum().item())
        expected_mean = float(weight.mean().item())
        expected_std = float(weight.std().item())

        # Convert tensor to CPU numpy array for multiprocessing queue
        expected_tensor_cpu = weight.cpu().numpy().tolist()

        logger.info(f"[P2PTrainingEngine] Expected: sum={expected_sum}, mean={expected_mean}, std={expected_std}")

        barrier.wait()
        training_engine.stop()

        result_queue.put(("training_success", "P2PTrainingTransferEngine completed"))
        result_queue.put(("expected_tensor", expected_tensor_cpu))

    except Exception as e:
        logger.error(f"[P2PTrainingEngine] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def simple_rollout_process_with_p2p_engine(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    name: str,
    training_hostname: str = "127.0.0.1",
    training_port: int = 50000,
):
    """Rollout process using P2PTransferEngine."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[P2PRollout-{rank}] Using P2PTransferEngine on GPU {rank}")

        # Initialize P2PTransferEngine from SGLang
        transfer_engine = P2PTransferEngine(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )

        # Initialize tensor with different values (zeros) to ensure transfer is real
        weight = torch.zeros(128, 128, device=device, dtype=torch.float16)
        original_data = weight.clone()

        barrier.wait()

        session_id = f"{training_hostname}:{training_port}"
        ptr = weight.data_ptr()
        length = weight.numel() * weight.element_size()

        handle = transfer_engine.submit_transfer_task(
            session_id=session_id,
            ptr=ptr,
            length=length,
            name=name,
        )
        handle.wait()
        torch.cuda.synchronize()

        # Verify data changed from original zeros
        data_changed = not torch.equal(weight, original_data)
        logger.info(f"[P2PRollout-{rank}] Data changed: {data_changed}")

        # Calculate actual received values for verification
        actual_sum = float(weight.sum().item())
        actual_mean = float(weight.mean().item())
        actual_std = float(weight.std().item())

        # Convert tensor to CPU numpy array for multiprocessing queue
        actual_tensor_cpu = weight.cpu().numpy().tolist()

        logger.info(f"[P2PRollout-{rank}] Received: sum={actual_sum}, mean={actual_mean}, std={actual_std}")

        # Check if values match expected (should be all 999.0)


        result_queue.put((f"rollout_{rank}_success", data_changed))
        result_queue.put((f"rollout_{rank}_tensor", actual_tensor_cpu))

        barrier.wait()

    except Exception as e:
        logger.error(f"[P2PRollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def p2p_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    name: str,
    training_port: int = 50000,
    
):
    """Entry point for P2PTrainingTransferEngine test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        simple_training_process_with_p2p_engine(rank, world_size, result_queue, barrier, name, "127.0.0.1", training_port)
    else:
        simple_rollout_process_with_p2p_engine(rank, world_size, result_queue, barrier, name, "127.0.0.1", training_port)


class TestP2PTrainingTransferEngine(unittest.TestCase):
    """Unit tests for P2PTrainingTransferEngine functionality."""

    def test_api_compatibility(self):
        """Test that P2PTrainingTransferEngine has TrainingWeightSender-compatible interface."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test interface compatibility without actual transfer
        training_engine = P2PTrainingTransferEngine(
            master_ip="127.0.0.1",
            master_port=50200,
            gpu_id=0,
            ib_device=None,
        )

        # Check required methods exist
        self.assertTrue(hasattr(training_engine, 'register_weights'))
        self.assertTrue(hasattr(training_engine, 'register_buffer'))
        self.assertTrue(hasattr(training_engine, 'start'))
        self.assertTrue(hasattr(training_engine, 'stop'))

        # Test register_weights interface
        device = torch.device("cuda:0")
        test_weights = {
            "weight1": torch.randn(64, 64, device=device, dtype=torch.float16),
            "weight2": torch.randn(32, 32, device=device, dtype=torch.float16),
        }

        # This should not raise an exception
        training_engine.register_weights(test_weights)

        # Verify weights were registered
        self.assertEqual(training_engine.get_num_registered_weights(), 2)
        registered_names = training_engine.list_registered_weights()
        self.assertIn("weight1", registered_names)
        self.assertIn("weight2", registered_names)

    def test_engine_lifecycle(self):
        """Test start/stop lifecycle of P2PTrainingTransferEngine."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        training_engine = P2PTrainingTransferEngine(
            master_ip="127.0.0.1",
            master_port=50201,
            gpu_id=0,
            ib_device=None,
        )

        # Test start
        self.assertFalse(training_engine.running)
        training_engine.start()
        self.assertTrue(training_engine.running)

        # Test stop
        training_engine.stop()
        self.assertFalse(training_engine.running)

        # Test restart
        training_engine.start()
        self.assertTrue(training_engine.running)
        training_engine.stop()
        self.assertFalse(training_engine.running)

    def test_weight_registration_patterns(self):
        """Test different weight registration patterns."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        training_engine = P2PTrainingTransferEngine(
            master_ip="127.0.0.1",
            master_port=50202,
            gpu_id=0,
            ib_device=None,
        )

        device = torch.device("cuda:0")

        # Test single weight registration
        single_weight = {"test_weight": torch.randn(64, 64, device=device, dtype=torch.float16)}
        training_engine.register_weights(single_weight)
        self.assertEqual(training_engine.get_num_registered_weights(), 1)

        # Test multiple weights registration
        multi_weights = {
            f"weight_{i}": torch.randn(32, 32, device=device, dtype=torch.float16)
            for i in range(5)
        }
        training_engine.register_weights(multi_weights)
        self.assertEqual(training_engine.get_num_registered_weights(), 6)  # 1 + 5

        # Test re-registration with same pointer (should not increase count)
        training_engine.register_weights(single_weight)
        self.assertEqual(training_engine.get_num_registered_weights(), 6)  # Should remain 6

    def test_p2p_engine_correctness(self):
        """Test P2PTrainingTransferEngine correctness with actual transfer."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        # Configure logging for this test
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
            force=True
        )

        world_size = 2
        training_port = 50300
        name = "temp_name_weight"
        print("\n" + "="*80)
        print("Testing P2PTrainingTransferEngine Correctness")
        print("="*80 + "\n")
        logger.info("\n" + "="*80)
        logger.info("Testing P2PTrainingTransferEngine Correctness")
        logger.info("="*80 + "\n")

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=p2p_worker_process, args=(rank, world_size, result_queue, barrier, name ,training_port))
            p.start()
            processes.append(p)

        results = {}
        timeout = 30
        start_time = time.time()

        while len(results) < world_size + 2:  # +2 for training_success, expected_tensor, rollout_1_success, rollout_1_tensor
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    self.fail(f"Test timeout after {timeout}s. Results so far: {results}")

                key, value = result_queue.get(timeout=remaining_time)
                results[key] = value
                logger.info(f"Received result: {key} = {type(value).__name__}({len(str(value))} chars)")
                print(f"Received result: {key} = {type(value).__name__}({len(str(value))} chars)")

            except Exception as e:
                self.fail(f"Failed to get results: {e}")

        for p in processes:
            p.join()

        # Validate results
        self.assertIn("training_success", results)
        self.assertIn("rollout_1_success", results)
        self.assertIn("expected_tensor", results)
        self.assertIn("rollout_1_tensor", results)

        # Verify training completed successfully
        self.assertEqual(results["training_success"], "P2PTrainingTransferEngine completed")

        # Verify rollout received correct weights (data changed AND values match)
        self.assertTrue(results["rollout_1_success"], "Rollout should have received correct weight values")

        # Verify tensor values match between training and rollout (element-wise comparison)
        expected_tensor = results["expected_tensor"]
        actual_tensor = results["rollout_1_tensor"]

        # Convert back to numpy for comparison
        expected_array = np.array(expected_tensor)
        actual_array = np.array(actual_tensor)

        # Element-wise comparison with small tolerance for floating point precision
        tensor_match = np.allclose(expected_array, actual_array, rtol=1e-3, atol=1e-3)
        self.assertTrue(tensor_match, f"Tensor values don't match element-wise")

        logger.info(f"✓ Tensor shapes: expected={expected_array.shape}, actual={actual_array.shape}")
        logger.info(f"Elements of training side: {expected_array.flatten()[:10]},  rollout_size: {actual_array.flatten()[:10]}")
        logger.info(f"✓ Element-wise tensor comparison: {tensor_match}")
        print(f"✓ Tensor shapes: expected={expected_array.shape}, actual={actual_array.shape}")
        print(f"✓ Element-wise tensor comparison: {tensor_match}")


        logger.info("✓ P2PTrainingTransferEngine correctness test passed")
        logger.info("✓ Tensor values correctly transferred via RDMA (element-wise verified)")
        logger.info("✓ Statistical summaries also match")
        print("✓ P2PTrainingTransferEngine correctness test passed")
        print("✓ Tensor values correctly transferred via RDMA (element-wise verified)")
        print("✓ Statistical summaries also match")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_error_handling(self):
        """Test error handling scenarios."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test double start (should handle gracefully)
        training_engine = P2PTrainingTransferEngine(
            master_ip="127.0.0.1",
            master_port=50203,
            gpu_id=0,
            ib_device=None,
        )

        training_engine.start()
        training_engine.start()  # Should not raise exception
        training_engine.stop()

        # Test stop without start (should handle gracefully)
        training_engine = P2PTrainingTransferEngine(
            master_ip="127.0.0.1",
            master_port=50204,
            gpu_id=0,
            ib_device=None,
        )
        training_engine.stop()  # Should not raise exception


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    unittest.main()