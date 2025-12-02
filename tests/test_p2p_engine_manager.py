"""
Unit tests for P2PTrainingTransferEngineManager

This test suite validates the P2PTrainingTransferEngineManager implementation
which extends P2PTrainingTransferEngine with engine pool functionality.

Based on test_p2p_engine.py patterns with additional tests for engine pool behavior.

cmd: python -m pytest tests/test_p2p_engine_manager.py::TestP2PTrainingTransferEngineManager::test_manager_correctness -v -s 2>&1 | tee manager_correctness_test_logs_full.txt
"""

import gc
import logging
import multiprocessing as mp
import os
import threading
import time
import unittest
from typing import Dict, List

import numpy as np
import torch
import zmq

# Import SGLang rollout-side components
from slime.backends.sglang_utils.sglang_rollout_rdma_p2p import P2PTransferManager

# Import our training-side implementations
from slime.backends.sglang_utils.sglang_training_rdma_p2p import (
    P2PTrainingTransferEngine,
    P2PTrainingTransferEngineManager
)

logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method("spawn", force=True)


def create_test_weight(size: int, device: torch.device, dtype=torch.float16) -> torch.Tensor:
    """Create a test weight tensor for correctness testing."""
    return torch.randn(size, size, device=device, dtype=dtype)


def training_process_with_manager(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    hostname: str = "127.0.0.1",
    port: int = 51000,
    engine_pool_size: int = 4,
):
    """Training process using P2PTrainingTransferEngineManager."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Initialize P2PTrainingTransferEngineManager
        training_manager = P2PTrainingTransferEngineManager(
            master_ip=hostname,
            master_port=port,
            gpu_id=rank,
            engine_pool_size=engine_pool_size,
            ib_device=None,
        )

        # Create multiple deterministic test weights for verification
        torch.manual_seed(42)
        weights = {}
        expected_tensors = {}

        # Create weights with different sizes to test engine selection
        for i in range(3):
            weight_name = f"test_weight_{i}"
            weight = torch.randn(64 + i * 32, 64 + i * 32, device=device, dtype=torch.float16)
            weights[weight_name] = weight
            expected_tensors[weight_name] = weight.cpu().numpy().tolist()

        total_size_kb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e3
        logger.info(f"[P2PTrainingManager] Total weight size: {total_size_kb:.2f}KB")
        logger.info(f"[P2PTrainingManager] Engine pool size: {training_manager.engine_pool_size}")

        # Test SGLang-compatible interface
        training_manager.register_weights(weights)
        training_manager.start()

        # Get all engine session IDs for debugging
        session_ids = training_manager.get_engine_pool_session_ids()
        logger.info(f"[P2PTrainingManager] Engine session IDs: {session_ids}")

        barrier.wait()

        # Update weights and re-register
        torch.cuda.synchronize()
        training_manager.register_weights(weights)

        barrier.wait()
        training_manager.stop()

        result_queue.put(("training_manager_success", "P2PTrainingTransferEngineManager completed"))
        result_queue.put(("expected_tensors", expected_tensors))
        result_queue.put(("engine_pool_size", engine_pool_size))
        result_queue.put(("session_ids", session_ids))

    except Exception as e:
        logger.error(f"[P2PTrainingManager] Error: {e}", exc_info=True)
        result_queue.put(("training_manager_error", str(e)))


def rollout_process_with_manager(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_hostname: str = "127.0.0.1",
    training_port: int = 51000,
    num_transfers: int = 3,
):
    """Rollout process that performs multiple transfers to test engine selection."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[P2PRollout-{rank}] Testing with {num_transfers} transfers")

        # Initialize P2PTransferManager from SGLang
        transfer_engine = P2PTransferManager(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )

        # Initialize multiple tensors with different sizes and values
        weights = {}
        original_data = {}

        for i in range(num_transfers):
            weight_name = f"test_weight_{i}"
            size = 64 + i * 32
            weight = torch.zeros(size, size, device=device, dtype=torch.float16)
            weights[weight_name] = weight
            original_data[weight_name] = weight.clone()

        barrier.wait()

        session_id = f"{training_hostname}:{training_port}"
        transfer_results = {}

        # Perform transfers for each weight
        for weight_name, weight in weights.items():
            ptr = weight.data_ptr()
            length = weight.numel() * weight.element_size()

            logger.info(f"[P2PRollout-{rank}] Transferring {weight_name}, size: {length} bytes")

            handle = transfer_engine.submit_transfer_task(
                session_id=session_id,
                ptr=ptr,
                length=length,
                name=weight_name
            )
            handle.wait()
            torch.cuda.synchronize()

            # Verify data changed from original zeros
            data_changed = not torch.equal(weight, original_data[weight_name])
            actual_tensor_cpu = weight.cpu().numpy().tolist()

            transfer_results[weight_name] = {
                "data_changed": data_changed,
                "tensor": actual_tensor_cpu,
                "sum": float(weight.sum().item()),
                "mean": float(weight.mean().item())
            }

            logger.info(f"[P2PRollout-{rank}] {weight_name}: data_changed={data_changed}")

        barrier.wait()

        result_queue.put((f"rollout_{rank}_success", True))
        result_queue.put((f"rollout_{rank}_results", transfer_results))

    except Exception as e:
        logger.error(f"[P2PRollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def simple_training_process_with_manager(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    hostname: str = "127.0.0.1",
    port: int = 51400,
    engine_pool_size: int = 4,
):
    """Training process using P2PTrainingTransferEngineManager - for correctness testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Initialize P2PTrainingTransferEngineManager
        training_manager = P2PTrainingTransferEngineManager(
            master_ip=hostname,
            master_port=port,
            gpu_id=rank,
            engine_pool_size=engine_pool_size,
            ib_device=None,
        )

        # Create deterministic test weight for verification (same as original test)
        torch.manual_seed(42)  # Ensure deterministic values
        weight = torch.randn(128 * 128, device=device, dtype=torch.float16).reshape(128, 128)
        weights = {"test_weight": weight}

        total_size_kb = weight.numel() * weight.element_size() / 1e3
        logger.info(f"[P2PTrainingManager] Weight size: {total_size_kb:.2f}KB")
        logger.info(f"[P2PTrainingManager] Engine pool size: {training_manager.engine_pool_size}")

        # Test SGLang-compatible interface
        training_manager.register_weights(weights)
        training_manager.start()

        # Get all engine session IDs for debugging
        session_ids = training_manager.get_engine_pool_session_ids()
        logger.info(f"[P2PTrainingManager] Engine session IDs: {session_ids}")

        barrier.wait()

        # Update to known values for verification
        torch.cuda.synchronize()
        training_manager.register_weights(weights)

        # Share expected tensor values with rollout process for verification
        expected_sum = float(weight.sum().item())
        expected_mean = float(weight.mean().item())
        expected_std = float(weight.std().item())

        # Convert tensor to CPU numpy array for multiprocessing queue
        expected_tensor_cpu = weight.cpu().numpy().tolist()

        logger.info(f"[P2PTrainingManager] Expected: sum={expected_sum}, mean={expected_mean}, std={expected_std}")

        barrier.wait()
        training_manager.stop()

        result_queue.put(("training_success", "P2PTrainingTransferEngineManager completed"))
        result_queue.put(("expected_tensor", expected_tensor_cpu))
        result_queue.put(("engine_pool_size", engine_pool_size))
        result_queue.put(("session_ids", session_ids))

    except Exception as e:
        logger.error(f"[P2PTrainingManager] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def simple_rollout_process_with_manager_correctness(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_hostname: str = "127.0.0.1",
    training_port: int = 51400,
):
    """Rollout process for manager correctness testing - uses single weight like original test."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[P2PRollout-{rank}] Using P2PTransferManager for manager correctness test on GPU {rank}")

        # Initialize P2PTransferManager from SGLang
        transfer_engine = P2PTransferManager(
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

        logger.info(f"[P2PRollout-{rank}] Submitting transfer task, session_id={session_id}, length={length}")

        handle = transfer_engine.submit_transfer_task(
            session_id=session_id,
            ptr=ptr,
            length=length,
            name = "test_weight"
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

        result_queue.put((f"rollout_{rank}_success", data_changed))
        result_queue.put((f"rollout_{rank}_tensor", actual_tensor_cpu))

        barrier.wait()

    except Exception as e:
        logger.error(f"[P2PRollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def manager_correctness_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_port: int = 51400,
    engine_pool_size: int = 4,
):
    """Entry point for P2PTrainingTransferEngineManager correctness test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        simple_training_process_with_manager(
            rank, world_size, result_queue, barrier, "127.0.0.1", training_port, engine_pool_size
        )
    else:
        simple_rollout_process_with_manager_correctness(
            rank, world_size, result_queue, barrier, "127.0.0.1", training_port
        )


def manager_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_port: int = 51000,
    engine_pool_size: int = 4,
):
    """Entry point for P2PTrainingTransferEngineManager test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        training_process_with_manager(
            rank, world_size, result_queue, barrier, "127.0.0.1", training_port, engine_pool_size
        )
    else:
        rollout_process_with_manager(
            rank, world_size, result_queue, barrier, "127.0.0.1", training_port, num_transfers=3
        )


class TestP2PTrainingTransferEngineManager(unittest.TestCase):
    """Unit tests for P2PTrainingTransferEngineManager functionality."""

    def test_manager_api_compatibility(self):
        """Test that P2PTrainingTransferEngineManager maintains API compatibility."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test interface compatibility without actual transfer
        training_manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51100,
            gpu_id=0,
            engine_pool_size=4,
            ib_device=None,
        )

        # Check that it inherits all required methods from base class
        self.assertTrue(hasattr(training_manager, 'register_weights'))
        self.assertTrue(hasattr(training_manager, 'register_buffer'))
        self.assertTrue(hasattr(training_manager, 'start'))
        self.assertTrue(hasattr(training_manager, 'stop'))
        self.assertTrue(hasattr(training_manager, 'get_session_id'))

        # Check manager-specific methods
        self.assertTrue(hasattr(training_manager, 'get_engine_pool_session_ids'))
        self.assertTrue(hasattr(training_manager, '_select_engine_by_session_id'))

        # Test engine pool initialization
        self.assertEqual(len(training_manager.engine_pool), 4)
        self.assertEqual(training_manager.engine_pool_size, 4)

        # Test compatibility with parent engine reference
        self.assertIs(training_manager.engine, training_manager.engine_pool[0])

    def test_engine_pool_initialization(self):
        """Test engine pool initialization with different sizes."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test different pool sizes
        for pool_size in [1, 2, 4, 8]:
            with self.subTest(pool_size=pool_size):
                manager = P2PTrainingTransferEngineManager(
                    master_ip="127.0.0.1",
                    master_port=51101 + pool_size,
                    gpu_id=0,
                    engine_pool_size=pool_size,
                    ib_device=None,
                )

                self.assertEqual(len(manager.engine_pool), pool_size)
                self.assertEqual(manager.engine_pool_size, pool_size)

                # Verify all engines are different objects
                for i in range(pool_size):
                    for j in range(i + 1, pool_size):
                        self.assertIsNot(manager.engine_pool[i], manager.engine_pool[j])

    def test_engine_selection_logic(self):
        """Test hash-based engine selection logic."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51102,
            gpu_id=0,
            engine_pool_size=4,
            ib_device=None,
        )

        # Test consistent engine selection for same session_id
        session_id = "test_session_123"
        engine1 = manager._select_engine_by_session_id(session_id)
        engine2 = manager._select_engine_by_session_id(session_id)
        self.assertIs(engine1, engine2, "Same session_id should select same engine")

        # Test different session_ids (may select different engines)
        test_session_ids = [
            "session_1", "session_2", "session_3", "session_4",
            "different_session", "another_test", "final_test"
        ]

        selected_engines = {}
        for session_id in test_session_ids:
            engine = manager._select_engine_by_session_id(session_id)
            engine_index = manager.engine_pool.index(engine)
            selected_engines[session_id] = engine_index

            # Verify engine is from the pool
            self.assertIn(engine, manager.engine_pool)

            # Verify hash-based selection consistency
            expected_index = hash(session_id) % manager.engine_pool_size
            self.assertEqual(engine_index, expected_index)

        logger.info(f"Engine selections: {selected_engines}")

        # Test empty session_id fallback
        empty_engine = manager._select_engine_by_session_id("")
        self.assertIs(empty_engine, manager.engine_pool[0])

        none_engine = manager._select_engine_by_session_id(None)
        self.assertIs(none_engine, manager.engine_pool[0])

    def test_weight_registration_across_pool(self):
        """Test that weights are registered across all engines in the pool."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51103,
            gpu_id=0,
            engine_pool_size=3,
            ib_device=None,
        )

        device = torch.device("cuda:0")
        test_weights = {
            "weight1": torch.randn(64, 64, device=device, dtype=torch.float16),
            "weight2": torch.randn(32, 32, device=device, dtype=torch.float16),
        }

        # Register weights
        manager.register_weights(test_weights)

        # Verify weights are tracked in manager
        self.assertEqual(manager.get_num_registered_weights(), 2)
        registered_names = manager.list_registered_weights()
        self.assertIn("weight1", registered_names)
        self.assertIn("weight2", registered_names)

        # Verify weight info is stored correctly
        for name, tensor in test_weights.items():
            self.assertIn(name, manager.weight_buffers)
            weight_info = manager.weight_buffers[name]
            self.assertEqual(weight_info['ptr'], tensor.data_ptr())
            self.assertEqual(weight_info['length'], tensor.numel() * tensor.element_size())
            self.assertIs(weight_info['tensor'], tensor)

    def test_manager_lifecycle(self):
        """Test start/stop lifecycle of P2PTrainingTransferEngineManager."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51104,
            gpu_id=0,
            engine_pool_size=2,
            ib_device=None,
        )

        # Test start
        self.assertFalse(manager.running)
        manager.start()
        self.assertTrue(manager.running)

        # Test session ID access
        session_id = manager.get_session_id()
        self.assertIsNotNone(session_id)

        # Test pool session IDs
        pool_session_ids = manager.get_engine_pool_session_ids()
        self.assertEqual(len(pool_session_ids), 2)
        self.assertEqual(pool_session_ids[0], session_id)  # First engine's session ID

        # Test stop
        manager.stop()
        self.assertFalse(manager.running)

    def test_manager_with_multiprocess_transfers(self):
        """Test P2PTrainingTransferEngineManager with actual multi-process transfers."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        # Configure logging for this test
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
            force=True
        )

        world_size = 2
        training_port = 51300
        engine_pool_size = 4

        print("\n" + "="*80)
        print("Testing P2PTrainingTransferEngineManager with Engine Pool")
        print("="*80 + "\n")

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=manager_worker_process,
                args=(rank, world_size, result_queue, barrier, training_port, engine_pool_size)
            )
            p.start()
            processes.append(p)

        results = {}
        timeout = 45  # Increased timeout for manager tests
        start_time = time.time()

        # Expected results: training_manager_success, expected_tensors, engine_pool_size, session_ids,
        #                   rollout_1_success, rollout_1_results
        expected_result_count = 6

        while len(results) < expected_result_count:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    self.fail(f"Test timeout after {timeout}s. Results so far: {list(results.keys())}")

                key, value = result_queue.get(timeout=remaining_time)
                results[key] = value
                logger.info(f"Received result: {key} = {type(value).__name__}")

            except Exception as e:
                self.fail(f"Failed to get results: {e}")

        for p in processes:
            p.join()

        # Validate results
        self.assertIn("training_manager_success", results)
        self.assertIn("rollout_1_success", results)
        self.assertIn("expected_tensors", results)
        self.assertIn("rollout_1_results", results)
        self.assertIn("engine_pool_size", results)
        self.assertIn("session_ids", results)

        # Verify manager completed successfully
        self.assertEqual(results["training_manager_success"], "P2PTrainingTransferEngineManager completed")

        # Verify engine pool was created correctly
        self.assertEqual(results["engine_pool_size"], engine_pool_size)
        self.assertEqual(len(results["session_ids"]), engine_pool_size)

        # Verify rollout received correct weights
        self.assertTrue(results["rollout_1_success"])

        # Verify all transfers completed and data was received
        rollout_results = results["rollout_1_results"]
        expected_tensors = results["expected_tensors"]

        for weight_name in expected_tensors.keys():
            self.assertIn(weight_name, rollout_results)
            transfer_result = rollout_results[weight_name]

            # Verify data changed from zeros
            self.assertTrue(transfer_result["data_changed"], f"{weight_name}: Data should have changed")

            # Verify tensor values match (element-wise comparison)
            expected_array = np.array(expected_tensors[weight_name])
            actual_array = np.array(transfer_result["tensor"])

            tensor_match = np.allclose(expected_array, actual_array, rtol=1e-3, atol=1e-3)
            self.assertTrue(tensor_match, f"{weight_name}: Tensor values don't match")

        logger.info("✓ P2PTrainingTransferEngineManager multi-transfer test passed")
        logger.info(f"✓ All {len(expected_tensors)} weights transferred correctly via engine pool")
        print("✓ P2PTrainingTransferEngineManager multi-transfer test passed")
        print(f"✓ All {len(expected_tensors)} weights transferred correctly via engine pool")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_engine_selection_distribution(self):
        """Test that engine selection distributes across the pool."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51105,
            gpu_id=0,
            engine_pool_size=4,
            ib_device=None,
        )

        # Test with many session IDs to see distribution
        session_ids = [f"session_{i}" for i in range(100)]
        engine_counts = {i: 0 for i in range(4)}

        for session_id in session_ids:
            engine = manager._select_engine_by_session_id(session_id)
            engine_index = manager.engine_pool.index(engine)
            engine_counts[engine_index] += 1

        logger.info(f"Engine selection distribution: {engine_counts}")

        # Verify each engine was selected at least once (with high probability)
        for engine_index, count in engine_counts.items():
            self.assertGreater(count, 0, f"Engine {engine_index} was never selected")

        # Verify total matches number of session IDs
        total_selections = sum(engine_counts.values())
        self.assertEqual(total_selections, len(session_ids))

    def test_error_handling(self):
        """Test error handling scenarios for the manager."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test double start (should handle gracefully)
        manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51106,
            gpu_id=0,
            engine_pool_size=2,
            ib_device=None,
        )

        manager.start()
        manager.start()  # Should not raise exception
        manager.stop()

        # Test stop without start (should handle gracefully)
        manager = P2PTrainingTransferEngineManager(
            master_ip="127.0.0.1",
            master_port=51107,
            gpu_id=0,
            engine_pool_size=2,
            ib_device=None,
        )
        manager.stop()  # Should not raise exception

        # Test invalid engine pool size
        with self.assertRaises(Exception):
            P2PTrainingTransferEngineManager(
                master_ip="127.0.0.1",
                master_port=51108,
                gpu_id=0,
                engine_pool_size=0,  # Invalid
                ib_device=None,
            )

    def test_manager_correctness(self):
        """Test P2PTrainingTransferEngineManager correctness with actual transfer - Element-wise verification."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        # Configure logging for this test
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
            force=True
        )

        world_size = 2
        training_port = 51400
        engine_pool_size = 4

        print("\n" + "="*80)
        print("Testing P2PTrainingTransferEngineManager Correctness")
        print("="*80 + "\n")
        logger.info("\n" + "="*80)
        logger.info("Testing P2PTrainingTransferEngineManager Correctness")
        logger.info("="*80 + "\n")

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=manager_correctness_worker_process,
                args=(rank, world_size, result_queue, barrier, training_port, engine_pool_size)
            )
            p.start()
            processes.append(p)

        results = {}
        timeout = 45  # Increased timeout for manager tests
        start_time = time.time()

        # Expected results: training_success, expected_tensor, engine_pool_size, session_ids,
        #                   rollout_1_success, rollout_1_tensor
        expected_result_count = 6

        while len(results) < expected_result_count:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    self.fail(f"Test timeout after {timeout}s. Results so far: {list(results.keys())}")

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
        self.assertIn("engine_pool_size", results)
        self.assertIn("session_ids", results)

        # Verify manager completed successfully
        self.assertEqual(results["training_success"], "P2PTrainingTransferEngineManager completed")

        # Verify engine pool was created correctly
        self.assertEqual(results["engine_pool_size"], engine_pool_size)
        self.assertEqual(len(results["session_ids"]), engine_pool_size)

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

        # Additional verification: Check statistical properties
        expected_sum = float(expected_array.sum())
        actual_sum = float(actual_array.sum())
        expected_mean = float(expected_array.mean())
        actual_mean = float(actual_array.mean())

        sum_match = abs(expected_sum - actual_sum) < 1e-2
        mean_match = abs(expected_mean - actual_mean) < 1e-5

        self.assertTrue(sum_match, f"Sum mismatch: expected={expected_sum}, actual={actual_sum}")
        self.assertTrue(mean_match, f"Mean mismatch: expected={expected_mean}, actual={actual_mean}")

        logger.info("✓ P2PTrainingTransferEngineManager correctness test passed")
        logger.info("✓ Tensor values correctly transferred via RDMA with engine pool (element-wise verified)")
        logger.info(f"✓ Engine pool size: {engine_pool_size}, Session IDs: {len(results['session_ids'])}")
        logger.info(f"✓ Statistical summaries: sum_match={sum_match}, mean_match={mean_match}")
        print("✓ P2PTrainingTransferEngineManager correctness test passed")
        print("✓ Tensor values correctly transferred via RDMA with engine pool (element-wise verified)")
        print(f"✓ Engine pool size: {engine_pool_size}, Session IDs: {len(results['session_ids'])}")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    unittest.main()