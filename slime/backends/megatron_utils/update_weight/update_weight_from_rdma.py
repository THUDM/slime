import logging
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
from ray.actor import ActorHandle

# from mooncake.engine import TransferEngine
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from tqdm import tqdm

from .common import register_memory_transfer_engine, split_expert_and_non_expert_param_names
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
    ) -> None:
        """
        Initialize transfer engine.
        """
        # Call parent constructor to initialize all base attributes
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            weight_update_mode="rdma",
        )

        self.transfer_engine = None

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
            # Get master address and port for P2P communication
            local_ip = ray._private.services.get_node_ip_address()
            self.transfer_engine = MooncakeTransferEngine(local_ip, None, None)
            logger.info(f"[RDMA] Transfer Engine initialized at port {self.transfer_engine.session_id}")
            # breakpoint()
            # self.transfer_engine = TransferEngine()
            # logger.info(f"[RDMA] Transfer Engine initialized at port {self.transfer_engine.get_rpc_port()}")

            # Query Engine session and weight info from rollout instances according to the transfer plan
            self.remote_weight_infos_by_session_id = {}
            targets_to_query = set((target.engine_ind, target.engine_rank) for target in self.transfer_plan.targets)
            targets_to_session_id = {}
            for engine_ind, engine_rank in targets_to_query:
                session_id, weights_info = ray.get(
                    self.rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
                )
                assert (
                    session_id is not None
                ), f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
                logger.info(
                    f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
                logger.info(list(weights_info.keys()))
                self.remote_weight_infos_by_session_id[session_id] = weights_info
                targets_to_session_id[(engine_ind, engine_rank)] = session_id

            # Associate transfer tasks based on obtained session and weight info
            for target in self.transfer_plan.targets:
                session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                expert_params, non_expert_params = split_expert_and_non_expert_param_names(
                    self.remote_weight_infos_by_session_id[session_id].keys()
                )
                params = expert_params if target.group == "expert" else non_expert_params
                self.transfer_plan.add_transfer_task(
                    session=session_id,
                    param_group=target.group,
                )
                logger.info(
                    f"Added transfer task for session {session_id} with {len(params)} tensors in group {target.group}."
                )

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=self.weight_version)
                for engine in self.rollout_engines
            ]
        )
        return

    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], session_id: str, pbar: tqdm | None = None
    ) -> None:
        """
        The RDMA P2P weight update is implemented as a single side write, meaning the trainer writes its weights directly to the rollout engines' memory.
        """

        if not self._is_source or not converted_named_tensors:
            return

        # Lock the rollout engines to prevent concurrent operations (same as parent)
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            # Features still missing for MVP:
            # TODO(jd): Implement resharding logic, right now it's not handled.
            # TODO: Some model may need target size handling like post_load_weights, currently not handled.
            # TODO(jd): Need a correctness test of the model weights similar to the test:https://github.com/sgl-project/sglang/pull/14997/changes#diff-6efab5fd819ef0efa7a1f43989320bb28231702f8840897eb7acacf174f6e71f

            # Potential optimization not implemented:
            # TODO: currently implementation does not guarantee single traversal of model dict and submits to mutliple targets in order.
            # If there are more targets than source, we are registering/all-gather/resharding/deregistering multiple times for same weights.
            # TODO: maybe pin the memory for GPUs instead of register + deregester each time after resharding.
            # TODO: increase concurrency with non-blocking transfers somehow. Note the reshaped tensors are temporary.
            # TODO: skip the forced all-gather for same shard tensors and instead convert directly.
            # TODO: finer granularity weight transfer where a multiple source instance can update a singular target instance.

            _ = register_memory_transfer_engine(converted_named_tensors, self.transfer_engine)
            logger.info(
                f"[RDMA] Registered {len(converted_named_tensors)} tensors with transfer engine for session {session_id}."
            )
            logger.info(f"[RDMA] Transfering {list(name for name, _ in converted_named_tensors)}")
            # Verify the 1-to-1 mapping between registered weights and remote weights expected.
            source_ptrs, target_ptrs, source_lens = [], [], []
            for name, tensor in converted_named_tensors:
                if name not in self.remote_weight_infos_by_session_id[session_id]:
                    raise RuntimeError(
                        f"Registered weight {name} not found in remote weight info for session {session_id}."
                    )
                remote_ptr, remote_numel, remote_element_size = self.remote_weight_infos_by_session_id[session_id][
                    name
                ]
                if tensor.numel() != remote_numel or tensor.element_size() != remote_element_size:
                    raise RuntimeError(
                        f"Registered weight {name} numel {tensor.numel()} size {tensor.element_size()} does not match remote numel {remote_numel} size {remote_element_size}."
                    )
                source_ptrs.append(tensor.data_ptr())
                target_ptrs.append(remote_ptr)
                source_lens.append(tensor.numel() * tensor.element_size())

            # Batch transfer weights through RDMA
            ret = self.transfer_engine.batch_transfer_sync(session_id, source_ptrs, target_ptrs, source_lens)
            logger.info(f"[RDMA] Batch transferred {len(converted_named_tensors)} tensors to session {session_id}.")
            if ret < 0:
                raise RuntimeError(f"Batch transfer weights via RDMA failed with error code {ret}.")
            self.transfer_engine.batch_deregister(source_ptrs)
            logger.info(f"[RDMA] Batch deregistered {len(converted_named_tensors)} tensors.")
            converted_named_tensors.clear()

        finally:
            # Release the lock (same as parent)
            ray.get(self.rollout_engine_lock.release.remote())
            if pbar:
                pbar.update(1)
