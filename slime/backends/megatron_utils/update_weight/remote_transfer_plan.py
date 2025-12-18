"""
Remote Transfer Plan - Abstract transfer planning for NCCL and RDMA weight updates.

This module provides a unified interface for determining transfer sources and planning
weight transfer tasks across different communication backends (NCCL, RDMA).
"""

import logging
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from megatron.core import mpu

from .common import expert_named_params_and_buffers, non_expert_named_params_and_buffers

logger = logging.getLogger(__name__)


@dataclass
class TransferTask:
    """
    Attributes:
        session: Session identifier (e.g., NCCL group name or Transfer Engine Session Id)
        named_params_and_buffers: tensors to be transferred from this rank.
        tensor_type: "expert" or "non-expert" are two diverse types of tasks.
    """

    named_params_and_buffers: list[tuple[str, torch.Tensor]]
    session: str  # NCCL group name or target entity id.
    tensor_type: Literal["expert", "non-expert"]


@dataclass
class TransferTaskP2PMeta:
    """
    Specifies a engine rollout rank to connect to.
    """

    engine_ind: int  # The index of the target rollout engine.
    engine_rank: int  # The shard within the target rollout engine.
    group: Literal["expert", "non-expert"]


class RemoteTransferPlan:
    """
    Plans and manages remote weight transfers for both NCCL and RDMA backends, assuming static training and rollout placements.

    At the moment, the plan assumes an all-gather in the tp/ep dimension on a bucketed basis.

    NCCL Plan: Use a single broadcast from DP=TP=0 PP rank to all rollout engines in a new process group.
    RDMA P2P Plan:
    The current execution plan prioritizes simplicity and general applicability for all supported models. It reuses existing
    componenets of slime distributed update as well as sglang remote instance load mechanisms. The plan follows:
    1. Calculate total number of source full replica (up to pp dimension) after all-gather in tp/ep dimension, for both
        expert and non-expert parameters.
    2. For each rollout engine, assign source ranks in a round-robin manner for both expert and non-expert parameters.
    3. During initialization, query each target rollout engine ranks for remote parameter names and session identifiers.
    4. Generate transfer tasks for each source rank based on remote parameter and local parameter availability.

    """

    def __init__(
        self, args: Namespace, model: Sequence[torch.nn.Module], mode: Literal["nccl", "rdma"] = "nccl"
    ) -> None:
        """
        Initialize the transfer plan.

        Args:
            args: Configuration namespace containing parallelism settings
            mode: Transfer backend mode - either "nccl" or "rdma"
        """
        self.mode = mode
        self._get_parallel_info(args)
        self.targets: list[TransferTaskP2PMeta] = self._plan_p2p() if mode == "rdma" else []
        self.transfer_tasks: list[TransferTask] = []
        self.non_expert_params_buffers = list(non_expert_named_params_and_buffers(args, model))
        self.expert_params_buffers = list(expert_named_params_and_buffers(args, model))

    def _get_parallel_info(self, args: Namespace) -> None:
        # Gather the source (current trainer) information.
        self._pp_rank, self._pp_size = (
            mpu.get_pipeline_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_world_size(),
        )
        self._ep_rank, self._ep_size = mpu.get_expert_model_parallel_rank(), mpu.get_expert_model_parallel_world_size()
        self._tp_rank, self._tp_size = mpu.get_tensor_model_parallel_rank(), mpu.get_tensor_model_parallel_world_size()
        self._etp_rank, self._etp_size = (
            mpu.get_expert_tensor_parallel_rank(),
            mpu.get_expert_tensor_parallel_world_size(),
        )
        self._dp_rank, self._dp_size = mpu.get_data_parallel_rank(
            with_context_parallel=True
        ), mpu.get_data_parallel_world_size(with_context_parallel=True)

        # Gather the target (rollout engine count and parallelism) information.
        self._rollout_tp_size = args.sglang_tp_size
        self._rollout_dp_size = args.sglang_dp_size
        self._rollout_ep_size = args.sglang_ep_size
        # PP sizes are not supported currently.
        self._rollout_pp_size = args.sglang_pp_size
        if self._rollout_ep_size != 1 or self._rollout_pp_size != 1:
            raise NotImplementedError("Rollout expert and pipeline parallelisms are not supported yet.")
        num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
        self._rollout_engine_count = args.rollout_num_gpus // num_gpu_per_engine
        logger.info(
            f"RemoteTransferPlan initialized: mode={self.mode}, pp_rank={self._pp_rank}/{self._pp_size}, tp_rank={self._tp_rank}/{self._tp_size}, "
            f"ep_rank={self._ep_rank}/{self._ep_size}, etp_rank={self._etp_rank}/{self._etp_size}, dp_rank={self._dp_rank}/{self._dp_size}"
        )
        logger.info(
            f"Rollout engine count: {self._rollout_engine_count}, tp_size={self._rollout_tp_size}, ep_size={self._rollout_ep_size}, dp_size={self._rollout_dp_size}"
        )

        # Expert and non expert parameters can have different parallel groups after all-gather.
        self._gathered_dp_size = self._dp_size * self._tp_size
        self._gathered_dp_rank = self._dp_rank * self._tp_size + self._tp_rank
        expert_tp_size = self._ep_size * self._etp_size
        self._gathered_expert_dp_size = self._dp_size * expert_tp_size
        self._gathered_expert_dp_rank = (
            self._dp_rank * expert_tp_size + self._ep_rank * self._etp_size + self._etp_rank
        )
        logger.info(
            f"Gathered dp_size={self._gathered_dp_size}, gathered expert dp_size={self._gathered_expert_dp_size}"
        )
        logger.info(
            f"Gathered dp_rank={self._gathered_dp_rank}, gathered expert dp_rank={self._gathered_expert_dp_rank}"
        )

    def _plan_p2p(self) -> list[TransferTaskP2PMeta]:
        def plan(
            source_size: int,
            source_rank: int,
            num_rank_in_target: int,
            num_targets: int,
            params: str,
            cur_active_rank: int = 0,
        ) -> list[TransferTaskP2PMeta]:
            transfer_tasks = []
            for target_ind in range(num_targets):
                for target_rank in range(num_rank_in_target):
                    if cur_active_rank % source_size == source_rank:
                        # TODO(jd): instead of doing round robin, we should prioritize reusing source ranks with same target_rank to
                        # avoid duplicating local copies.
                        transfer_tasks.append(
                            TransferTaskP2PMeta(engine_ind=target_ind, engine_rank=target_rank, group=params)
                        )
                        logger.info(
                            f"Planned P2P transfer task: source_rank={source_rank} -> target_engine_ind={target_ind}, target_engine_rank={target_rank}, group={params}"
                        )
                    cur_active_rank += 1
            return transfer_tasks

        # TODO(JD): Due to the local replica design, the plan should proritize reusing existing copies and merge sources.

        non_expert_plan = plan(
            source_size=self._gathered_dp_size,
            source_rank=self._gathered_dp_rank,
            num_rank_in_target=self._rollout_dp_size * self._rollout_tp_size,
            num_targets=self._rollout_engine_count,
            params="non-expert",
        )
        return non_expert_plan + plan(
            source_size=self._gathered_expert_dp_size,
            source_rank=self._gathered_expert_dp_rank,
            num_rank_in_target=self._rollout_dp_size * self._rollout_ep_size,
            num_targets=self._rollout_engine_count,
            params="expert",
        )

    def is_source(self) -> bool:
        """
        Determine if the current rank needs to initiate weight transfer.

        Returns:
            bool - True if the current rank is a source for weight transfer, False otherwise.
        """
        if self.mode == "nccl":
            # NCCL only load from DP=TP=0 PP ranks to all rollout engines.
            return (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            )
        return len(self.targets) > 0

    def add_transfer_task(self, session: str, param_group: Literal["expert", "non-expert"]) -> None:
        """
        Add a transfer task to the plan using remote instance session and tensor names.
        """
        params = self.non_expert_params_buffers if param_group == "non-expert" else self.expert_params_buffers
        if params:
            self.transfer_tasks.append(
                TransferTask(session=session, named_params_and_buffers=params, tensor_type=param_group)
            )
            logger.info(f"Added {param_group} parameter transfer task: session={session}, num_tensors={len(params)}")

    def clear_transfer_tasks(self) -> None:
        self.transfer_tasks = []

    def get_transfer_tasks(self) -> list[TransferTask]:
        # Generate session identifier based on mode
        if self.mode == "nccl":
            session = f"slime-pp_{self._pp_rank}"
            # In NCCL mode, the transfer is simply a broadcast from DP=TP=0 to all rollout engines.
            return [
                TransferTask(
                    session=session, named_params_and_buffers=self.non_expert_params_buffers, tensor_type="non-expert"
                ),
                TransferTask(
                    session=session, named_params_and_buffers=self.expert_params_buffers, tensor_type="expert"
                ),
            ]
        if self.targets and not self.transfer_tasks:
            raise RuntimeError("RDMA need to query target engine information for transfer task generations.")
        return self.transfer_tasks
