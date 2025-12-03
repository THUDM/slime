"""
Qwen32B Model Tests for P2P Transfer Engines

This test suite validates P2P weight transfer for Qwen32B model between
1 training engine and 2 rollout engines using realistic model weights.

Test configuration:
- 1 Training process using P2PTrainingTransferEngine (GPU 0)
- 2 Rollout processes using P2PTransferEngine (GPU 1, GPU 2)
- Qwen32B model weight simulation with realistic tensor sizes

cmd: 
# p2p
python -m pytest tests/test_p2p_engine_model_qwen2.py::TestQwen32BP2PTransfer::test_qwen32b_model_transfer -v -s --tb=short --capture=no
# baseline
python -m pytest tests/test_p2p_engine_model_qwen2.py::TestQwen32BP2PTransfer::test_qwen32b_model_transfer_baseline -v -s --tb=short --capture=no

Profiling:
P2P results saved to /root/slime/tests/p2p_timing_results.json
Distributed baseline results saved to /root/slime/tests/distributed_timing_results.json
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
import torch.distributed as dist
import zmq

# Import SGLang rollout-side components
from slime.backends.sglang_utils.sglang_rollout_rdma_p2p import P2PTransferEngine

# Import our training-side implementation
from slime.backends.sglang_utils.sglang_training_rdma_p2p import P2PTrainingTransferEngine

logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method("spawn", force=True)


def create_qwen32b_weights(device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """
    Create realistic Qwen32B model weights for testing.

    Qwen32B model has approximately:
    - Hidden size: 5120
    - Number of attention heads: 40
    - Number of layers: 64
    - Vocab size: 152064
    """
    torch.manual_seed(42)  # Ensure deterministic weights

    weights = {}

    # Embedding weights
    weights["embeddings.word_embeddings.weight"] = torch.randn(
        152064, 5120, device=device, dtype=dtype
    )

    # Layer weights (simulate a few key layers)
    for layer_idx in range(4):  # Test with 4 layers instead of full 64 for memory
        prefix = f"layers.{layer_idx}"

        # Attention weights
        weights[f"{prefix}.attention.q_proj.weight"] = torch.randn(
            5120, 5120, device=device, dtype=dtype
        )
        weights[f"{prefix}.attention.k_proj.weight"] = torch.randn(
            1024, 5120, device=device, dtype=dtype  # Key/Value have smaller dimension
        )
        weights[f"{prefix}.attention.v_proj.weight"] = torch.randn(
            1024, 5120, device=device, dtype=dtype
        )
        weights[f"{prefix}.attention.o_proj.weight"] = torch.randn(
            5120, 5120, device=device, dtype=dtype
        )

        # MLP weights
        weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
            13824, 5120, device=device, dtype=dtype  # FFN intermediate size
        )
        weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
            13824, 5120, device=device, dtype=dtype
        )
        weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
            5120, 13824, device=device, dtype=dtype
        )

        # Layer norm weights
        weights[f"{prefix}.input_layernorm.weight"] = torch.randn(
            5120, device=device, dtype=dtype
        )
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(
            5120, device=device, dtype=dtype
        )

    # Final layer norm and output projection
    weights["norm.weight"] = torch.randn(5120, device=device, dtype=dtype)
    weights["lm_head.weight"] = torch.randn(152064, 5120, device=device, dtype=dtype)

    # Calculate total size
    total_params = sum(w.numel() for w in weights.values())
    total_size_gb = total_params * 2 / (1024**3)  # float16 = 2 bytes per param

    logger.info(f"Created Qwen32B model weights: {len(weights)} tensors, "
                f"{total_params:,} parameters, {total_size_gb:.2f} GB")

    return weights


def qwen32b_training_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    weights: Dict[str, torch.Tensor],
    hostname: str = "127.0.0.1",
    port: int = 51000,
):
    """Training process using P2PTrainingTransferEngine with Qwen32B weights."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[QwenTraining-{rank}] Starting on GPU {rank}")

        # Initialize P2PTrainingTransferEngine
        training_engine = P2PTrainingTransferEngine(
            master_ip=hostname,
            master_port=port,
            gpu_id=rank,
            ib_device=None,
        )

        # Move weights to the correct device
        for name in weights:
            weights[name] = weights[name].to(device)

        # Calculate total weight info for logging
        total_size_mb = sum(
            w.numel() * w.element_size() for w in weights.values()
        ) / (1024 * 1024)

        logger.info(f"[QwenTraining-{rank}] Total model size: {total_size_mb:.2f} MB")

        # Register and start the training engine
        register_start_time = time.time()
        training_engine.register_weights(weights)
        training_engine.start()
        register_end_time = time.time()

        # Wait for all processes to be ready
        barrier.wait()

        # Update weights to known values for verification
        torch.cuda.synchronize()
        update_start_time = time.time()
        training_engine.register_weights(weights)
        update_end_time = time.time()

        # Prepare expected results for verification (save actual tensor data)
        expected_results = {}
        for name, weight in weights.items():
            expected_results[name] = {
                'sum': float(weight.sum().item()),
                'mean': float(weight.mean().item()),
                'std': float(weight.std().item()),
                'shape': list(weight.shape),
                'element_count': weight.numel(),
                'tensor_data': weight.cpu().detach().numpy()  # Convert to numpy for multiprocessing
            }

        logger.info(f"[QwenTraining-{rank}] Weights registered and ready for transfer")

        # Wait for rollout processes to complete
        barrier.wait()

        stop_start_time = time.time()
        training_engine.stop()
        stop_end_time = time.time()

        # Calculate timing results
        timing_results = {
            'register_and_start_time': register_end_time - register_start_time,
            'update_weights_time': update_end_time - update_start_time,
            'stop_engine_time': stop_end_time - stop_start_time,
            # 'total_time': stop_end_time - register_start_time
        }

        result_queue.put(("training_success", "Qwen32B training completed"))
        result_queue.put(("expected_weights", expected_results))
        result_queue.put(("training_timing", timing_results))

    except Exception as e:
        logger.error(f"[QwenTraining-{rank}] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def qwen32b_rollout_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    weights: Dict[str, torch.Tensor],
    training_hostname: str = "127.0.0.1",
    training_port: int = 51000,
):
    """Rollout process using P2PTransferEngine to receive Qwen32B weights."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[QwenRollout-{rank}] Starting on GPU {rank}")

        # Initialize P2PTransferEngine
        transfer_engine = P2PTransferEngine(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )

        # Move weights to the correct device and zero them out
        for name in weights:
            weights[name] = weights[name].to(device)
            weights[name].zero_()

        # Store original (zero) data for comparison
        original_data = {name: weight.clone() for name, weight in weights.items()}

        # Wait for training process to be ready
        barrier.wait()

        # Perform weight transfers
        session_id = f"{training_hostname}:{training_port}"
        handles = []

        logger.info(f"[QwenRollout-{rank}] Requesting {len(weights)} weight transfers")

        submit_start_time = time.time()
        for name, weight in weights.items():
            ptr = weight.data_ptr()
            length = weight.numel() * weight.element_size()

            handle = transfer_engine.submit_transfer_task(
                session_id=session_id,
                ptr=ptr,
                length=length,
                name=name
            )
            handles.append((name, handle))
        submit_end_time = time.time()

        # Wait for all transfers to complete
        logger.info(f"[QwenRollout-{rank}] Waiting for {len(handles)} transfers to complete")

        wait_start_time = time.time()
        for name, handle in handles:
            handle.wait()
            logger.debug(f"[QwenRollout-{rank}] Transfer completed for {name}")
        wait_end_time = time.time()

        torch.cuda.synchronize()
        sync_end_time = time.time()

        # Verify data changed from original zeros
        verify_start_time = time.time()
        weights_changed = {}
        received_results = {}

        for name, weight in weights.items():
            original = original_data[name]
            changed = not torch.equal(weight, original)
            weights_changed[name] = changed

            # Calculate statistics for verification (save actual tensor data)
            received_results[name] = {
                'sum': float(weight.sum().item()),
                'mean': float(weight.mean().item()),
                'std': float(weight.std().item()),
                'shape': list(weight.shape),
                'changed': changed,
                'element_count': weight.numel(),
                'tensor_data': weight.cpu().detach().numpy()  # Convert to numpy for multiprocessing
            }

        total_changed = sum(weights_changed.values())
        logger.info(f"[QwenRollout-{rank}] Weights changed: {total_changed}/{len(weights)}")
        verify_end_time = time.time()

        # Calculate timing results
        timing_results = {
            'submit_tasks_time': submit_end_time - submit_start_time,
            'wait_transfers_time': wait_end_time - wait_start_time,
            'sync_time': sync_end_time - wait_end_time,
            'total_transfer_time': sync_end_time - submit_start_time,
        }

        # Wait for all rollout processes to complete
        barrier.wait()

        result_queue.put((f"rollout_{rank}_success", total_changed == len(weights)))
        result_queue.put((f"rollout_{rank}_weights", received_results))
        result_queue.put((f"rollout_{rank}_timing", timing_results))

    except Exception as e:
        logger.error(f"[QwenRollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def qwen32b_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    weights: Dict[str, torch.Tensor],
    training_port: int = 51000,
):
    """Entry point for Qwen32B model test workers."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        # Training process
        qwen32b_training_process(rank, world_size, result_queue, barrier, weights, "127.0.0.1", training_port)
    else:
        # Rollout processes (rank 1 and 2)
        qwen32b_rollout_process(rank, world_size, result_queue, barrier, weights, "127.0.0.1", training_port)


def qwen32b_distributed_training_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    weights: Dict[str, torch.Tensor],
    master_addr: str = "127.0.0.1",
    master_port: str = "12355",
):
    """Training process using torch.distributed for weight transfer."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[DistTraining-{rank}] Starting on GPU {rank}")

        # Initialize distributed process group
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        init_start_time = time.time()
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        init_end_time = time.time()

        # Move weights to the correct device
        for name in weights:
            weights[name] = weights[name].to(device)

        # Calculate total weight info for logging
        total_size_mb = sum(
            w.numel() * w.element_size() for w in weights.values()
        ) / (1024 * 1024)

        logger.info(f"[DistTraining-{rank}] Total model size: {total_size_mb:.2f} MB")

        # Wait for all processes to be ready
        barrier.wait()

        # Broadcast weights to all processes
        logger.info(f"[DistTraining-{rank}] Broadcasting weights to all processes")
        broadcast_start_time = time.time()
        for name, weight in weights.items():
            dist.broadcast(weight, src=0)
        broadcast_end_time = time.time()

        # Prepare expected results for verification
        expected_results = {}
        for name, weight in weights.items():
            expected_results[name] = {
                'sum': float(weight.sum().item()),
                'mean': float(weight.mean().item()),
                'std': float(weight.std().item()),
                'shape': list(weight.shape),
                'element_count': weight.numel(),
                'tensor_data': weight.cpu().detach().numpy()
            }

        logger.info(f"[DistTraining-{rank}] Weights broadcasted successfully")

        # Wait for rollout processes to complete
        barrier.wait()

        destroy_start_time = time.time()
        dist.destroy_process_group()
        destroy_end_time = time.time()

        # Calculate timing results
        timing_results = {
            'init_process_group_time': init_end_time - init_start_time,
            'broadcast_time': broadcast_end_time - broadcast_start_time,
            'destroy_group_time': destroy_end_time - destroy_start_time,
            # 'total_time': destroy_end_time - init_start_time
        }

        result_queue.put(("training_success", "Qwen32B distributed training completed"))
        result_queue.put(("expected_weights", expected_results))
        result_queue.put(("training_timing", timing_results))

    except Exception as e:
        logger.error(f"[DistTraining-{rank}] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def qwen32b_distributed_rollout_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    weights: Dict[str, torch.Tensor],
    master_addr: str = "127.0.0.1",
    master_port: str = "12355",
):
    """Rollout process using torch.distributed to receive weights."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[DistRollout-{rank}] Starting on GPU {rank}")

        # Initialize distributed process group
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        init_start_time = time.time()
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        init_end_time = time.time()

        # Move weights to the correct device and zero them out
        for name in weights:
            weights[name] = weights[name].to(device)
            weights[name].zero_()

        # Store original (zero) data for comparison
        original_data = {name: weight.clone() for name, weight in weights.items()}

        # Wait for training process to be ready
        barrier.wait()

        logger.info(f"[DistRollout-{rank}] Receiving weights via broadcast")

        # Receive weights from rank 0 (training process)
        broadcast_start_time = time.time()
        for name, weight in weights.items():
            dist.broadcast(weight, src=0)
        broadcast_end_time = time.time()

        torch.cuda.synchronize()
        sync_end_time = time.time()

        # Verify data changed from original zeros
        weights_changed = {}
        received_results = {}

        for name, weight in weights.items():
            original = original_data[name]
            changed = not torch.equal(weight, original)
            weights_changed[name] = changed

            # Calculate statistics for verification
            received_results[name] = {
                'sum': float(weight.sum().item()),
                'mean': float(weight.mean().item()),
                'std': float(weight.std().item()),
                'shape': list(weight.shape),
                'changed': changed,
                'element_count': weight.numel(),
                'tensor_data': weight.cpu().detach().numpy()
            }

        total_changed = sum(weights_changed.values())
        logger.info(f"[DistRollout-{rank}] Weights changed: {total_changed}/{len(weights)}")

        # Wait for all rollout processes to complete
        barrier.wait()

        destroy_start_time = time.time()
        dist.destroy_process_group()
        destroy_end_time = time.time()

        # Calculate timing results
        timing_results = {
            'init_process_group_time': init_end_time - init_start_time,
            'broadcast_time': broadcast_end_time - broadcast_start_time,
            'sync_time': sync_end_time - broadcast_end_time,
            'destroy_group_time': destroy_end_time - destroy_start_time,
            'total_transfer_time': sync_end_time - broadcast_start_time,
            # 'total_time': destroy_end_time - init_start_time
        }

        result_queue.put((f"rollout_{rank}_success", total_changed == len(weights)))
        result_queue.put((f"rollout_{rank}_weights", received_results))
        result_queue.put((f"rollout_{rank}_timing", timing_results))

    except Exception as e:
        logger.error(f"[DistRollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def qwen32b_distributed_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    weights: Dict[str, torch.Tensor],
    master_addr: str = "127.0.0.1",
    master_port: str = "12355",
):
    """Entry point for distributed Qwen32B model test workers."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        # Training process
        qwen32b_distributed_training_process(rank, world_size, result_queue, barrier, weights, master_addr, master_port)
    else:
        # Rollout processes (rank 1 and 2)
        qwen32b_distributed_rollout_process(rank, world_size, result_queue, barrier, weights, master_addr, master_port)


class TestQwen32BP2PTransfer(unittest.TestCase):
    """Test suite for Qwen32B model P2P weight transfer."""

    def test_qwen32b_model_transfer(self):
        """Test P2P transfer with Qwen32B model weights between 1 training and 2 rollout engines."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
            force=True
        )

        world_size = 3  # 1 training + 2 rollout
        training_port = 51000

        print("\n" + "="*80)
        print("Testing Qwen32B Model P2P Weight Transfer")
        print("1 Training Engine + 2 Rollout Engines")
        print("="*80 + "\n")

        logger.info("Starting Qwen32B model P2P transfer test")

        # Create Qwen32B model weights on CPU first
        device = torch.device("cpu")
        weights = create_qwen32b_weights(device)

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=qwen32b_worker_process,
                args=(rank, world_size, result_queue, barrier, weights, training_port)
            )
            p.start()
            processes.append(p)

        # Collect results
        results = {}
        timeout = 120  # Longer timeout for large model
        start_time = time.time()

        # Expect: training_success, expected_weights, training_timing,
        #         rollout_1_success, rollout_1_weights, rollout_1_timing,
        #         rollout_2_success, rollout_2_weights, rollout_2_timing
        expected_results = 9

        while len(results) < expected_results:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    self.fail(f"Test timeout after {timeout}s. Results so far: {list(results.keys())}")

                key, value = result_queue.get(timeout=remaining_time)
                results[key] = value
                logger.info(f"Received result: {key}")
                print(f"Received result: {key}")

            except Exception as e:
                self.fail(f"Failed to get results: {e}")

        for p in processes:
            p.join()

        # Validate results
        self.assertIn("training_success", results)
        self.assertIn("expected_weights", results)
        self.assertIn("training_timing", results)
        self.assertIn("rollout_1_success", results)
        self.assertIn("rollout_1_weights", results)
        self.assertIn("rollout_1_timing", results)
        self.assertIn("rollout_2_success", results)
        self.assertIn("rollout_2_weights", results)
        self.assertIn("rollout_2_timing", results)

        # Verify training completed successfully
        self.assertEqual(results["training_success"], "Qwen32B training completed")

        # Verify both rollout processes received weights
        self.assertTrue(results["rollout_1_success"], "Rollout 1 should have received all weights")
        self.assertTrue(results["rollout_2_success"], "Rollout 2 should have received all weights")

        # Verify weight consistency between training and rollout processes
        expected_weights = results["expected_weights"]
        rollout_1_weights = results["rollout_1_weights"]
        rollout_2_weights = results["rollout_2_weights"]

        logger.info(f"Validating {len(expected_weights)} weight tensors")
        print(f"Validating {len(expected_weights)} weight tensors")

        weight_validation_count = 0
        for weight_name in expected_weights:
            if weight_name not in rollout_1_weights or weight_name not in rollout_2_weights:
                self.fail(f"Weight {weight_name} missing from rollout results")

            expected = expected_weights[weight_name]
            rollout_1 = rollout_1_weights[weight_name]
            rollout_2 = rollout_2_weights[weight_name]

            # Verify shapes match
            self.assertEqual(expected['shape'], rollout_1['shape'],
                           f"Shape mismatch for {weight_name}")
            self.assertEqual(expected['shape'], rollout_2['shape'],
                           f"Shape mismatch for {weight_name}")

            # Verify data changed (not zero anymore)
            self.assertTrue(rollout_1['changed'],
                          f"Weight {weight_name} not changed in rollout 1")
            self.assertTrue(rollout_2['changed'],
                          f"Weight {weight_name} not changed in rollout 2")


            # Verify exact tensor data matches (most important check)
            expected_numpy = expected['tensor_data']
            rollout_1_numpy = rollout_1['tensor_data']
            rollout_2_numpy = rollout_2['tensor_data']

            # Convert numpy arrays back to tensors for comparison
            expected_tensor = torch.from_numpy(expected_numpy)
            rollout_1_tensor = torch.from_numpy(rollout_1_numpy)
            rollout_2_tensor = torch.from_numpy(rollout_2_numpy)

            # Check training to rollout_1 exact match
            torch.testing.assert_close(expected_tensor, rollout_1_tensor, rtol=1e-6, atol=1e-8,
                                     msg=f"Exact tensor mismatch between training and rollout_1 for {weight_name}")

            # Check training to rollout_2 exact match
            torch.testing.assert_close(expected_tensor, rollout_2_tensor, rtol=1e-6, atol=1e-8,
                                     msg=f"Exact tensor mismatch between training and rollout_2 for {weight_name}")

            # Check rollout_1 to rollout_2 exact match
            torch.testing.assert_close(rollout_1_tensor, rollout_2_tensor, rtol=1e-8, atol=1e-10,
                                     msg=f"Exact tensor mismatch between rollout_1 and rollout_2 for {weight_name}")
            logger.info(f"#############################")
            logger.info(f"For weight {weight_name}: ")
            logger.info(f"Training side {weight_name}: {expected_tensor.flatten()[:10]}")
            logger.info(f"Rollout 1 {weight_name}: {rollout_1_tensor.flatten()[:10]}")
            logger.info(f"Rollout 2 {weight_name}: {rollout_2_tensor.flatten()[:10]}")
            weight_validation_count += 1

        logger.info(f"✓ Validated {weight_validation_count} weight tensors successfully")
        logger.info("✓ Qwen32B model P2P transfer test passed")
        logger.info("✓ All weights correctly transferred to both rollout engines")
        logger.info("✓ Exact tensor equality verified across all processes")

        print(f"✓ Validated {weight_validation_count} weight tensors successfully")
        print("✓ Qwen32B model P2P transfer test passed")
        print("✓ All weights correctly transferred to both rollout engines")
        print("✓ Exact tensor equality verified across all processes")

        # Save timing results
        timing_results = {
            'test_name': 'test_qwen32b_model_transfer',
            'method': 'P2P Transfer Engine',
            'training_timing': results.get("training_timing", {}),
            'rollout_1_timing': results.get("rollout_1_timing", {}),
            'rollout_2_timing': results.get("rollout_2_timing", {}),
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        import json
        with open('/root/slime/tests/p2p_timing_results.json', 'w') as f:
            json.dump(timing_results, f, indent=2)

        logger.info("Performance timing results saved to p2p_timing_results.json")
        print("Performance timing results saved to p2p_timing_results.json")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_qwen32b_model_transfer_baseline(self):
        """Test torch.distributed baseline transfer with Qwen32B model weights between 1 training and 2 rollout engines."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
            force=True
        )

        world_size = 3  # 1 training + 2 rollout
        master_port = "12355"

        print("\n" + "="*80)
        print("Testing Qwen32B Model Distributed Baseline Transfer")
        print("1 Training Engine + 2 Rollout Engines (torch.distributed)")
        print("="*80 + "\n")

        logger.info("Starting Qwen32B model distributed baseline transfer test")

        # Create Qwen32B model weights on CPU first
        device = torch.device("cpu")
        weights = create_qwen32b_weights(device)

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=qwen32b_distributed_worker_process,
                args=(rank, world_size, result_queue, barrier, weights, "127.0.0.1", master_port)
            )
            p.start()
            processes.append(p)

        # Collect results
        results = {}
        timeout = 120  # Longer timeout for large model
        start_time = time.time()

        # Expect: training_success, expected_weights, training_timing,
        #         rollout_1_success, rollout_1_weights, rollout_1_timing,
        #         rollout_2_success, rollout_2_weights, rollout_2_timing
        expected_results = 9

        while len(results) < expected_results:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    self.fail(f"Test timeout after {timeout}s. Results so far: {list(results.keys())}")

                key, value = result_queue.get(timeout=remaining_time)
                results[key] = value
                logger.info(f"Received result: {key}")
                print(f"Received result: {key}")

            except Exception as e:
                self.fail(f"Failed to get results: {e}")

        for p in processes:
            p.join()

        # Validate results
        self.assertIn("training_success", results)
        self.assertIn("expected_weights", results)
        self.assertIn("training_timing", results)
        self.assertIn("rollout_1_success", results)
        self.assertIn("rollout_1_weights", results)
        self.assertIn("rollout_1_timing", results)
        self.assertIn("rollout_2_success", results)
        self.assertIn("rollout_2_weights", results)
        self.assertIn("rollout_2_timing", results)

        # Verify training completed successfully
        self.assertEqual(results["training_success"], "Qwen32B distributed training completed")

        # Verify both rollout processes received weights
        self.assertTrue(results["rollout_1_success"], "Rollout 1 should have received all weights")
        self.assertTrue(results["rollout_2_success"], "Rollout 2 should have received all weights")

        # Verify weight consistency between training and rollout processes
        expected_weights = results["expected_weights"]
        rollout_1_weights = results["rollout_1_weights"]
        rollout_2_weights = results["rollout_2_weights"]

        logger.info(f"Validating {len(expected_weights)} weight tensors")
        print(f"Validating {len(expected_weights)} weight tensors")

        weight_validation_count = 0
        for weight_name in expected_weights:
            if weight_name not in rollout_1_weights or weight_name not in rollout_2_weights:
                self.fail(f"Weight {weight_name} missing from rollout results")

            expected = expected_weights[weight_name]
            rollout_1 = rollout_1_weights[weight_name]
            rollout_2 = rollout_2_weights[weight_name]

            # Verify shapes match
            self.assertEqual(expected['shape'], rollout_1['shape'],
                           f"Shape mismatch for {weight_name}")
            self.assertEqual(expected['shape'], rollout_2['shape'],
                           f"Shape mismatch for {weight_name}")

            # Verify data changed (not zero anymore)
            self.assertTrue(rollout_1['changed'],
                          f"Weight {weight_name} not changed in rollout 1")
            self.assertTrue(rollout_2['changed'],
                          f"Weight {weight_name} not changed in rollout 2")

            # Verify exact tensor data matches (most important check)
            expected_numpy = expected['tensor_data']
            rollout_1_numpy = rollout_1['tensor_data']
            rollout_2_numpy = rollout_2['tensor_data']

            # Convert numpy arrays back to tensors for comparison
            expected_tensor = torch.from_numpy(expected_numpy)
            rollout_1_tensor = torch.from_numpy(rollout_1_numpy)
            rollout_2_tensor = torch.from_numpy(rollout_2_numpy)

            # Check training to rollout_1 exact match
            torch.testing.assert_close(expected_tensor, rollout_1_tensor, rtol=1e-6, atol=1e-8,
                                     msg=f"Exact tensor mismatch between training and rollout_1 for {weight_name}")

            # Check training to rollout_2 exact match
            torch.testing.assert_close(expected_tensor, rollout_2_tensor, rtol=1e-6, atol=1e-8,
                                     msg=f"Exact tensor mismatch between training and rollout_2 for {weight_name}")

            # Check rollout_1 to rollout_2 exact match
            torch.testing.assert_close(rollout_1_tensor, rollout_2_tensor, rtol=1e-8, atol=1e-10,
                                     msg=f"Exact tensor mismatch between rollout_1 and rollout_2 for {weight_name}")

            logger.info(f"#############################")
            logger.info(f"For weight {weight_name}: ")
            logger.info(f"Training side {weight_name}: {expected_tensor.flatten()[:10]}")
            logger.info(f"Rollout 1 {weight_name}: {rollout_1_tensor.flatten()[:10]}")
            logger.info(f"Rollout 2 {weight_name}: {rollout_2_tensor.flatten()[:10]}")
            weight_validation_count += 1

        logger.info(f"✓ Validated {weight_validation_count} weight tensors successfully")
        logger.info("✓ Qwen32B model distributed baseline transfer test passed")
        logger.info("✓ All weights correctly transferred to both rollout engines")
        logger.info("✓ Exact tensor equality verified across all processes")

        print(f"✓ Validated {weight_validation_count} weight tensors successfully")
        print("✓ Qwen32B model distributed baseline transfer test passed")
        print("✓ All weights correctly transferred to both rollout engines")
        print("✓ Exact tensor equality verified across all processes")

        # Save timing results for baseline test
        timing_results = {
            'test_name': 'test_qwen32b_model_transfer_baseline',
            'method': 'torch.distributed baseline',
            'training_timing': results.get("training_timing", {}),
            'rollout_1_timing': results.get("rollout_1_timing", {}),
            'rollout_2_timing': results.get("rollout_2_timing", {}),
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        import json
        with open('/root/slime/tests/distributed_timing_results.json', 'w') as f:
            json.dump(timing_results, f, indent=2)

        logger.info("Performance timing results saved to distributed_timing_results.json")
        print("Performance timing results saved to distributed_timing_results.json")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_qwen32b_weight_creation(self):
        """Test Qwen32B weight creation utility."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda:0")
        weights = create_qwen32b_weights(device)

        # Verify expected weights are created
        self.assertIn("embeddings.word_embeddings.weight", weights)
        self.assertIn("layers.0.attention.q_proj.weight", weights)
        self.assertIn("layers.0.mlp.gate_proj.weight", weights)
        self.assertIn("norm.weight", weights)
        self.assertIn("lm_head.weight", weights)

        # Verify realistic tensor sizes
        embed_weight = weights["embeddings.word_embeddings.weight"]
        self.assertEqual(embed_weight.shape, (152064, 5120))

        q_proj_weight = weights["layers.0.attention.q_proj.weight"]
        self.assertEqual(q_proj_weight.shape, (5120, 5120))

        mlp_weight = weights["layers.0.mlp.gate_proj.weight"]
        self.assertEqual(mlp_weight.shape, (13824, 5120))

        # Verify all weights are on correct device and dtype
        for name, weight in weights.items():
            self.assertEqual(weight.device, device)
            self.assertEqual(weight.dtype, torch.float16)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    unittest.main()