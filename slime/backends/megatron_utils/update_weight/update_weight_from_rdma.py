import logging
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
from ray.actor import ActorHandle
from tqdm import tqdm

from .common import split_expert_and_non_expert_param_names
from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


class UpdateWeightFromRDMA(UpdateWeightFromRemote):
    """
    Update weights from RDMA using Transfer Engine.

    Similar to UpdateWeightFromNCCL but uses P2P RDMA transfer engine for the underlying weight transfer. Workflow
    consists of following steps:
    1. Based off the transfer plan, query the target rollout engines for remote session and weight info during connect_rollout_engines.
    2. Do TP-EP all-gather for bucketed weights on parameters needing transfer from local just as in NCCL case.
    3. Convert the gathered HF tensor into target shape and register them with Engine.
    4. Call engine to batch transfer weights for each transfer task.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        vocab_size: int,
    ) -> None:
        """
        Initialize. P2PTrainingTransferEngine created in connect_rollout_engines.
        Calls parent constructor and adds P2P RDMA specific attributes.
        """
        # Call parent constructor to initialize all base attributes
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            vocab_size=vocab_size,
        )

        # P2P RDMA specific initialization
        self.training_p2p_transfer_engine = None
        self.session_id = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Initialize P2PTrainingTransferEngine if serves as a source.
        """
        # Store rollout engines and lock
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._is_source = self.transfer_plan.is_source()

        # Initialize P2PTrainingTransferEngine on source rank
        if self._is_source:
            if self.training_p2p_transfer_engine is not None:
                self.training_p2p_transfer_engine.stop()
                self.session_id = None
                self.transfer_plan.clear_transfer_tasks()

            # Get master address and port for P2P communication
            local_ip = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]

            # Initialize P2PTrainingTransferEngine
            # self.training_p2p_transfer_engine = P2PTrainingTransferEngine(
            #     master_ip=local_ip,
            #     master_port=port,
            #     gpu_id=None,
            #     ib_device=None,
            # )
            self.training_p2p_transfer_engine.start()
            self.session_id = f"{local_ip}:{port}"
            logger.info(f"P2PTrainingTransferEngine started on {local_ip}:{port}")

            # Query Engine session and weight info from rollout instances according to the transfer plan
            self.remote_weight_infos_by_engine_and_rank = {}
            for target in self.transfer_plan.targets:
                if (target.engine_ind, target.engine_rank) not in self.remote_weight_infos_by_engine_and_rank:
                    self.remote_weight_infos_by_engine_and_rank[(target.engine_ind, target.engine_rank)] = ray.get(
                        self.rollout_engines[target.engine_ind].get_remote_instance_transfer_engine_info.remote(
                            rank=target.engine_rank
                        )["remote_instance_transfer_engine_info"]
                    )
                    logger.info(
                        f"Obtained remote session info from rollout engine {target.engine_ind} rank {target.engine_rank}"
                    )
                remote_session_id, remote_weight_info = self.remote_weight_infos_by_engine_and_rank[
                    (target.engine_ind, target.engine_rank)
                ]
                expert_params, non_expert_params = split_expert_and_non_expert_param_names(remote_weight_info.keys())
                self.transfer_plan.add_transfer_task(
                    session=remote_session_id,
                    remote_tensor_names=expert_params if target.group == "expert" else non_expert_params,
                    param_group=target.group,
                )

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Register weights with P2PTrainingTransferEngine and wait for transfers to complete.
        Based on lines 518-545 in SGLang test: register_weights pattern.
        Overrides parent method to use P2P RDMA instead of NCCL broadcast.
        """

        # TODO(jd): pin the memory for GPUs after resharding to avoid expensive registration.
        if not self._is_source or not converted_named_tensors:
            return

        # Lock the rollout engines to prevent concurrent operations (same as parent)
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            # Register all weights with the P2P training transfer engine
            # This follows the pattern from SGLang test lines 518-537
            for name, tensor in converted_named_tensors:
                self.training_p2p_transfer_engine.register_buffer(name, tensor)

            # Initiate weight transfer to all rollout engines as per the transfer plan
            refs = [
                engine.update_weights_from_distributed.remote(
                    names=[name for name, _ in converted_named_tensors],
                    dtypes=[param.dtype for _, param in converted_named_tensors],
                    shapes=[param.shape for _, param in converted_named_tensors],
                    group_name=self._group_name,
                    weight_version=str(self.weight_version),
                    session_id=f"{self.master_addr}:{self.master_port}",  # Pass P2P session info
                )
                for engine in self.rollout_engines
            ]

            # Wait for all P2P transfers to complete
            ray.get(refs)
            converted_named_tensors.clear()

        finally:
            # Release the lock (same as parent)
            ray.get(self.rollout_engine_lock.release.remote())
            if pbar:
                pbar.update(1)
