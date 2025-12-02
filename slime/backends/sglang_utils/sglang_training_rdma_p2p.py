"""
P2P Training Transfer Engine for SLIME

This module implements the training-side counterpart to SGLang's P2P transfer system.
It provides classes for handling RDMA-based weight transfers from training processes
to rollout processes during RL training.

Based on SGLang's p2p_transfer.py implementation.
"""

import concurrent.futures
import logging
import os
import threading
import zmq
import torch

from collections import defaultdict
from queue import Queue
from typing import Dict, Optional

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.utils import get_int_env_var
from sglang.srt.utils.common import format_tcp_address

logger = logging.getLogger(__name__)


class P2PTrainingTransferEngine:
    """
    Training-side P2P transfer engine that handles RDMA write operations.

    This is the counterpart to SGLang's P2PTransferEngine. It:
    1. Listens for sync_status messages from rollout processes
    2. Performs RDMA transfers to send updated weights
    3. Sends completion confirmations back to rollout processes
    
    TODO(xinji1): should we put `P2PTrainingTransferEngine` to the `__init__`,
      or the `connect_rollout_engines`? Since the engine only needs
      the localhost as the host name.
    """

    def __init__(self, master_ip: str, master_port: int, gpu_id: int, ib_device: Optional[str] = None):
        """
        Initialize the training transfer engine.

        Args:
            master_ip: IP address to bind ZMQ socket. It should be the local ip.
            master_port: Port to bind ZMQ socket
            gpu_id: GPU device ID
            ib_device: InfiniBand device name (optional)
        """
        self.master_ip = master_ip
        self.master_port = master_port
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        self.running = False

        # Initialize the underlying Mooncake transfer engine
        self.engine = MooncakeTransferEngine(
            hostname=master_ip,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

        # Initialize ZMQ components
        self.context = None
        self.router_socket = None

        # Weight buffer registry: name -> {ptr, length, tensor}
        self.weight_buffers: Dict[str, Dict] = {}
        self.registration_lock = threading.Lock()

        # Task counter for generating unique task IDs
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()

        # Worker thread for handling messages
        self.worker_thread = None

        logger.info(
            f"P2PTrainingTransferEngine initialized on {master_ip}:{master_port}, GPU {gpu_id}"
        )

    def _generate_task_id(self) -> str:
        """Generate a unique task ID"""
        with self.task_counter_lock:
            task_id = f"training_task_{self.task_counter}"
            self.task_counter += 1
        return task_id

    def register_buffer(self, name: str, tensor) -> None:
        """
        Register a weight tensor for P2P transfer.

        Args:
            name: Name identifier for the weight
            tensor: PyTorch tensor to register
        """
        ptr = tensor.data_ptr()
        length = tensor.numel() * tensor.element_size()

        with self.registration_lock:
            # Check if we need to re-register (pointer changed)
            if name in self.weight_buffers:
                existing_ptr = self.weight_buffers[name]['ptr']
                if existing_ptr == ptr:
                    # Just update the tensor reference
                    self.weight_buffers[name]['tensor'] = tensor
                    logger.debug(f"Updated tensor reference for {name}")
                    return

            # Register new memory region with RDMA engine
            self.engine.register(ptr, length)
            self.weight_buffers[name] = {
                'ptr': ptr,
                'length': length,
                'tensor': tensor,
            }

            logger.info(f"Registered weight buffer: {name}, ptr={ptr:#x}, length={length}")

    def register_weights(self, weights_dict: Dict[str, torch.Tensor]) -> None:
        """
        SGLang-compatible method for registering multiple weight tensors.
        This is a wrapper around register_buffer to match TrainingWeightSender interface.

        Args:
            weights_dict: Dictionary mapping weight names to tensor objects
        """
        for name, tensor in weights_dict.items():
            self.register_buffer(name, tensor)

        logger.info(f"[P2PTrainingTransferEngine] Registered {len(weights_dict)} weight tensors")

    def start(self) -> None:
        """
        Start the training transfer engine.
        This creates the ZMQ socket and starts the worker thread.
        """
        if self.running:
            logger.warning("P2PTrainingTransferEngine is already running")
            return

        self.running = True

        # Initialize ZMQ components
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(format_tcp_address(self.master_ip, self.master_port))

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self.transfer_worker_training,
            daemon=True
        )
        self.worker_thread.start()

        logger.info(f"P2PTrainingTransferEngine started on {self.master_ip}:{self.master_port}")

    def stop(self) -> None:
        """
        Stop the training transfer engine.
        This terminates the worker thread and cleans up ZMQ resources.
        """
        if not self.running:
            return

        self.running = False

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

        # Clean up ZMQ resources
        if self.router_socket:
            self.router_socket.close()
        if self.context:
            self.context.term()

        logger.info("P2PTrainingTransferEngine stopped")

    def transfer_worker_training(self) -> None:
        """
        Worker thread that listens for sync_status messages from rollout processes.
        """
        logger.info("Training transfer worker started")

        while self.running:
            try:
                # Poll for incoming messages with timeout
                if self.router_socket.poll(timeout=100):  # 100ms timeout
                    frames = self.router_socket.recv_multipart()
                    if len(frames) < 2:
                        logger.debug("Received incomplete message frames")
                        continue

                    identity = frames[0]
                    rollout_message = None

                    # Parse the JSON message from frames
                    for i in range(1, len(frames)):
                        try:
                            if len(frames[i]) > 0:
                                rollout_message = zmq.utils.jsonapi.loads(frames[i])
                                logger.debug(f"Parsed JSON message from frame {i}: {rollout_message}")
                                break
                        except Exception as e:
                            logger.debug(f"Frame {i} is not valid JSON: {e}")
                            continue

                    if rollout_message is None:
                        logger.debug("No valid JSON message found in frames")
                        continue

                    # Check message type
                    msg_type = rollout_message.get("type", "")
                    if msg_type == "sync_status":
                        self._process_training_transfer_task(identity, rollout_message)
                    else:
                        logger.debug(f"Unknown message type: {msg_type}")

            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ error in training worker: {e}")
                break
            except Exception as e:
                if self.running:
                    logger.error(f"Unexpected error in training worker: {e}", exc_info=True)

        logger.info("Training transfer worker stopped")

    def _process_training_transfer_task(self, identity: bytes, rollout_message: dict) -> None:
        """
        Process a training transfer task from a rollout process.

        Args:
            identity: ZMQ identity of the requesting rollout process
            rollout_message: Parsed sync_status message from rollout
        """
        # Parse rollout_message
        rollout_transfer_session_id = rollout_message.get("transfer_session_id", "")
        rollout_ptr = rollout_message.get("ptr", 0)
        rollout_length = rollout_message.get("length", 0)
        task_id = rollout_message.get("task_id", "")
        rollout_weight_name = rollout_message.get("rollout_weight_name", "")

        logger.info(
            f"Processing transfer task: task_id={task_id}, "
            f"session_id={rollout_transfer_session_id}, ptr={rollout_ptr:#x}, length={rollout_length}"
        )

        try:
            weight_info = None
            weight_name = None

            with self.registration_lock:                
                weight_info = self.weight_buffers.get(rollout_weight_name, None)
                
            # Check if we found a matching weight
            if weight_info is None:
                available_names = [name for name in self.weight_buffers.keys()]
                error_msg = (
                    f"No weight found with name {rollout_weight_name}. "
                    f"All names: {available_names}"
                )
                logger.error(f"[Task {task_id}] {error_msg}")
                raise RuntimeError(error_msg)
            weight_name = rollout_weight_name
            src_ptr = weight_info['ptr']
            src_length = weight_info['length']

            # Verify length match
            if src_length != rollout_length:
                raise RuntimeError(f"Length mismatch: src={src_length}, dst={rollout_length}")

            # Perform RDMA transfer (similar to line 462 in SGLang test)
            logger.info(
                f"[Task {task_id}] Starting RDMA transfer: {weight_name}, "
                f"src_ptr={src_ptr:#x} -> dst_ptr={rollout_ptr:#x}, length={src_length}"
            )

            status = self.engine.transfer_sync(
                session_id=rollout_transfer_session_id,
                buffer=src_ptr,
                peer_buffer_address=rollout_ptr,
                length=src_length,
            )

            if status != 0:
                raise RuntimeError(f"RDMA transfer failed with status {status}")

            logger.info(
                f"[Task {task_id}] RDMA transfer completed: weight={weight_name}, "
                f"size={src_length/1e6:.2f}MB"
            )

            # Send success response
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "success",
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

            logger.info(f"[Task {task_id}] Sent success confirmation")

        except Exception as e:
            logger.error(f"[Task {task_id}] Transfer failed: {e}", exc_info=True)

            # Send failure response
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "failed",
                "error": str(e),
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

            logger.error(f"[Task {task_id}] Sent failure confirmation")

    def get_session_id(self):
        """Get the session ID from the underlying transfer engine"""
        return self.engine.get_session_id()

    def get_num_registered_weights(self) -> int:
        """Get the number of registered weight buffers"""
        with self.registration_lock:
            return len(self.weight_buffers)

    def list_registered_weights(self) -> list:
        """Get list of registered weight names"""
        with self.registration_lock:
            return list(self.weight_buffers.keys())


class P2PTrainingTransferEngineManager(P2PTrainingTransferEngine):
    """
    Manager version of P2PTrainingTransferEngine that maintains a pool of engines.

    Instead of using a single engine for all transfers, this manager:
    1. Initializes a pool of MooncakeTransferEngine instances
    2. Routes transfers to engines based on hash(p2p_session_id) % pool_size
    3. Maintains the same interface as the base P2PTrainingTransferEngine
    """

    def __init__(self, master_ip: str, master_port: int, gpu_id: int,
                 engine_pool_size: int = 4, ib_device: Optional[str] = None):
        """
        Initialize the training transfer engine manager.

        Args:
            master_ip: IP address to bind ZMQ socket
            master_port: Port to bind ZMQ socket
            gpu_id: GPU device ID
            engine_pool_size: Number of engines in the pool
            ib_device: InfiniBand device name (optional)
        """
        # Initialize parent class but don't create the single engine yet
        self.master_ip = master_ip
        self.master_port = master_port
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        self.running = False
        self.engine_pool_size = engine_pool_size

        # Initialize engine pool
        self.engine_pool: list = []
        for _ in range(engine_pool_size):
            engine = MooncakeTransferEngine(
                hostname=master_ip,
                gpu_id=gpu_id,
                ib_device=ib_device,
            )
            self.engine_pool.append(engine)

        # Override the single engine with the first one for compatibility
        self.engine = self.engine_pool[0]

        # Initialize ZMQ components
        self.context = None
        self.router_socket = None

        # Weight buffer registry: name -> {ptr, length, tensor}
        self.weight_buffers: Dict[str, Dict] = {}
        self.registration_lock = threading.Lock()

        # Task counter for generating unique task IDs
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()

        # Worker thread for handling messages
        self.worker_thread = None

        logger.info(
            f"P2PTrainingTransferEngineManager initialized on {master_ip}:{master_port}, "
            f"GPU {gpu_id}, pool_size={engine_pool_size}"
        )

    def register_buffer(self, name: str, tensor) -> None:
        """
        Register a weight tensor for P2P transfer across all engines in the pool.

        Args:
            name: Name identifier for the weight
            tensor: PyTorch tensor to register
        """
        ptr = tensor.data_ptr()
        length = tensor.numel() * tensor.element_size()

        with self.registration_lock:
            # Check if we need to re-register (pointer changed)
            if name in self.weight_buffers:
                existing_ptr = self.weight_buffers[name]['ptr']
                if existing_ptr == ptr:
                    # Just update the tensor reference
                    self.weight_buffers[name]['tensor'] = tensor
                    logger.debug(f"Updated tensor reference for {name}")
                    return

            # Register new memory region with ALL engines in the pool
            for i, engine in enumerate(self.engine_pool):
                engine.register(ptr, length)
                logger.debug(f"Registered with engine {i}: {name}, ptr={ptr:#x}, length={length}")

            self.weight_buffers[name] = {
                'ptr': ptr,
                'length': length,
                'tensor': tensor,
            }

            logger.info(f"Registered weight buffer across {len(self.engine_pool)} engines: "
                       f"{name}, ptr={ptr:#x}, length={length}")

    def _select_engine_by_session_id(self, session_id: str) -> MooncakeTransferEngine:
        """
        Select an engine from the pool based on session_id hash.

        Args:
            session_id: The p2p_session_id from the incoming message

        Returns:
            Selected MooncakeTransferEngine instance
        """
        if not session_id:
            # Fallback to first engine if session_id is empty
            logger.warning("Empty session_id, using engine 0")
            return self.engine_pool[0]

        engine_index = hash(session_id) % self.engine_pool_size
        selected_engine = self.engine_pool[engine_index]

        logger.debug(f"Selected engine {engine_index} for session_id={session_id}")
        return selected_engine

    def _process_training_transfer_task(self, identity: bytes, rollout_message: dict) -> None:
        """
        Process a training transfer task using the selected engine from the pool.

        Args:
            identity: ZMQ identity of the requesting rollout process
            rollout_message: Parsed sync_status message from rollout
        """
        # Parse rollout_message
        rollout_transfer_session_id = rollout_message.get("transfer_session_id", "")
        rollout_ptr = rollout_message.get("ptr", 0)
        rollout_length = rollout_message.get("length", 0)
        rollout_weight_name = rollout_message.get("rollout_weight_name", "")
        task_id = rollout_message.get("task_id", "")
        

        logger.info(
            f"Processing transfer task: task_id={task_id}, "
            f"session_id={rollout_transfer_session_id}, ptr={rollout_ptr:#x}, length={rollout_length}"
        )

        try:
            # Select engine based on session_id hash
            selected_engine = self._select_engine_by_session_id(rollout_transfer_session_id)

            weight_info = None
            weight_name = None

            with self.registration_lock:                
                weight_info = self.weight_buffers.get(rollout_weight_name, None)
                

            # Check if we found a matching weight
            if weight_info is None:
                available_names = [name for name in self.weight_buffers.keys()]
                error_msg = (
                    f"No weight found with name {rollout_weight_name}. "
                    f"All names: {available_names}"
                )
                logger.error(f"[Task {task_id}] {error_msg}")
                raise RuntimeError(error_msg)
            
            weight_name = rollout_weight_name
            src_ptr = weight_info['ptr']
            src_length = weight_info['length']

            # Verify length match
            if src_length != rollout_length:
                raise RuntimeError(f"Length mismatch: src={src_length}, dst={rollout_length}")

            # Perform RDMA transfer using the selected engine
            engine_index = self.engine_pool.index(selected_engine)
            logger.info(
                f"[Task {task_id}] Starting RDMA transfer with engine {engine_index}: {weight_name}, "
                f"src_ptr={src_ptr:#x} -> dst_ptr={rollout_ptr:#x}, length={src_length}"
            )

            status = selected_engine.transfer_sync(
                session_id=rollout_transfer_session_id,
                buffer=src_ptr,
                peer_buffer_address=rollout_ptr,
                length=src_length,
            )

            if status != 0:
                raise RuntimeError(f"RDMA transfer failed with status {status}")

            logger.info(
                f"[Task {task_id}] RDMA transfer completed using engine {engine_index}: "
                f"weight={weight_name}, size={src_length/1e6:.2f}MB"
            )

            # Send success response
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "success",
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

            logger.info(f"[Task {task_id}] Sent success confirmation")

        except Exception as e:
            logger.error(f"[Task {task_id}] Transfer failed: {e}", exc_info=True)

            # Send failure response
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "failed",
                "error": str(e),
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

            logger.error(f"[Task {task_id}] Sent failure confirmation")

    def get_session_id(self):
        """Get the session ID from the first engine (for compatibility)"""
        return self.engine_pool[0].get_session_id()

    def get_engine_pool_session_ids(self) -> list:
        """Get session IDs from all engines in the pool"""
        return [engine.get_session_id() for engine in self.engine_pool]

    def stop(self) -> None:
        """
        Stop the training transfer engine manager and clean up all engines.
        """
        if not self.running:
            return

        self.running = False

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

        # Clean up ZMQ resources
        if self.router_socket:
            self.router_socket.close()
        if self.context:
            self.context.term()

        logger.info(f"P2PTrainingTransferEngineManager stopped (pool_size={len(self.engine_pool)})")