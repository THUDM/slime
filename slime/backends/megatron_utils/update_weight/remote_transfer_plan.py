"""
Remote Transfer Plan - Abstract transfer planning for NCCL and RDMA weight updates.

This module provides a unified interface for determining transfer sources and planning
weight transfer tasks across different communication backends (NCCL, RDMA).
"""

import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from megatron.core import mpu

logger = logging.getLogger(__name__)


@dataclass
class TransferTaskP2PMeta:
    """
    Specifies a engine rollout rank to connect to.
    """

    engine_ind: int  # The index of the target rollout engine.
    engine_rank: int  # The shard within the target rollout engine.
    source_shard: int = 0  # The source pp shard index.


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
        self._get_parallelism(args)

    def _get_parallelism(self, args: Namespace) -> None:
        """
        Collecting and printing out parallelism information for both source (trainer) and target (rollout engines).
        Also print out the parallelism information after the ep/tp all-gather for the 2 parameter groups.
        """

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
        # EP and PP sizes are not tested and likely miss functionalities.
        self._rollout_pp_size = args.sglang_pp_size
        if self._rollout_ep_size != 1 or self._rollout_pp_size != 1:
            raise NotImplementedError("Rollout expert and pipeline parallelisms are not supported yet.")
        self._num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
        self._rollout_engine_count = args.rollout_num_gpus // self._num_gpu_per_engine
        self._rollout_num_gpus = args.rollout_num_gpus
        logger.info(
            f"RemoteTransferPlan initialized: mode={self.mode}, pp_rank={self._pp_rank}/{self._pp_size}, tp_rank={self._tp_rank}/{self._tp_size}, "
            f"ep_rank={self._ep_rank}/{self._ep_size}, etp_rank={self._etp_rank}/{self._etp_size}, dp_rank={self._dp_rank}/{self._dp_size}"
        )
        logger.info(
            f"Rollout engine count: {self._rollout_engine_count}, tp_size={self._rollout_tp_size}, ep_size={self._rollout_ep_size}, dp_size={self._rollout_dp_size}"
        )

        self._gathered_dp_size = self._dp_size * self._tp_size
        self._gathered_dp_rank = self._dp_rank * self._tp_size + self._tp_rank
        # TODO: If I understand correctly the final size should be same as we now only have pp - dp dimensions for both param groups?
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

        self._rank = self._gathered_dp_rank

    def get_nccl_group(self) -> str:
        """
        Get the NCCL group name for weight transfer.

        Returns:
            str - NCCL group name
        """
        assert self.mode == "nccl", "NCCL group only applicable for NCCL mode."
        return f"slime-pp_{self._pp_rank}"

    def plan_p2p(self) -> list[TransferTaskP2PMeta]:
        """
        For each pp shard source rank, we plan the mapping relationship between n source dp ranks, m target rollout engines with k ranks each.
        The Transfer Plan Mapping Heuristics works as follows:
        1. for each target engine (idx, rank), assign source ranks in a round-robin manner until all source ranks are assigned at least once.
        2. for the reminder target (idx, rank), assign them to source ranks by priotizing the source with existing assignmeng of same rank.

        For example, 4 source ranks (0,1,2,3), 2 target engines with 3 ranks each (0,0),(0,1),(0,2),(1,0),(1,1),(1,2).
        The first round of assignment:
        source_rank=0 -> target (0,0)
        source_rank=1 -> target (0,1)
        source_rank=2 -> target (0,2)
        source_rank=3 -> target (1,0)
        The reminder assignment:
        source_rank=1 -> target (1,1)  # prioritize source_rank=1 as it had (0,1) assigned already.
        source_rank=2 -> target (1,2)

        Finally extract the transfer tasks matching the current dp_rank.
        """

        all_targets = [
            (m_idx, k_idx) for m_idx in range(self._rollout_engine_count) for k_idx in range(self._num_gpu_per_engine)
        ]
        # Assignments: source_rank -> {engin_rank: [engine_indices]}
        assignements = defaultdict(lambda: defaultdict(list))
        # First round robin assignment
        i = -1
        for source_rank, (idx, target) in zip(range(self._gathered_dp_size), enumerate(all_targets), strict=False):
            i = idx
            m_idx, k_idx = target
            assignements[source_rank][k_idx].append(m_idx)

        def count_engine_index_assignments(k_idx: int) -> int:
            return [len(assignements[source][k_idx]) for source in range(self._gathered_dp_size)]

        # Reminder assignment by least_assigned_source
        cur_source_index = 0
        if i < len(all_targets) - 1:
            for target in all_targets[i + 1 :]:
                m_idx, k_idx = target
                # count current assignments for source who has k_idx
                counted = count_engine_index_assignments(k_idx)
                # If any source has existing assignment for k_idx, assign it.
                if max(counted) > 0:
                    _, select_source = min((val, idx) for (idx, val) in enumerate(counted) if val > 0)
                # Else go back to round robin.
                else:
                    select_source = cur_source_index % self._gathered_dp_size
                    cur_source_index += 1
                assignements[select_source][k_idx].append(m_idx)

        # Extract transfer tasks for current rank.
        logger.info(f"[TransferPlanner] Full transfer assignments: {dict(assignements)}")
        transfer_tasks = []
        for engine_rank, engine_indices in assignements[self._rank].items():
            for engine_ind in engine_indices:
                logger.info(
                    f"[TransferPlanner] New task: source_rank={self._rank} pp_shard={self._pp_rank} -> target_engine_ind={engine_ind}, target_engine_rank={engine_rank}"
                )
                transfer_tasks.append(
                    TransferTaskP2PMeta(source_shard=self._pp_rank, engine_ind=engine_ind, engine_rank=engine_rank)
                )
        return transfer_tasks

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
        # Only case where RDMA P2P is not sending is when the current DP rank is >= total number of rollout GPUs.
        return False if (self._rank >= self._rollout_num_gpus) else True
