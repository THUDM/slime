import dataclasses
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import sglang.srt.layers.dp_attention as sglang_dp_attention
import sglang.srt.server_args as sglang_server_args
import torch
from mooncake.engine import TransferEngine
from ray.actor import ActorHandle
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig

# from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.model_loader import get_model
from sglang.srt.server_args import ServerArgs
from tqdm import tqdm

from slime.utils.memory_utils import print_memory

from .common import register_memory_transfer_engine
from .update_weight_from_remote import UpdateWeightFromRemote

logger = logging.getLogger(__name__)


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


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

        if self._is_source:
            # Query Engine session and weight info from rollout instances according to the transfer plan
            self.remote_weight_infos_by_session_id = {}
            targets_to_query = set((target.engine_ind, target.engine_rank) for target in self.transfer_plan.targets)
            targets_to_session_id, self.session_id_to_engine_rank = {}, {}
            self.session_id_to_server_args = {}
            for engine_ind, engine_rank in targets_to_query:
                session_id, weights_info = ray.get(
                    self.rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
                )
                self.session_id_to_engine_rank[session_id] = engine_rank
                self.session_id_to_server_args[session_id] = create_server_args_from_dict(
                    ray.get(self.rollout_engines[engine_ind].get_server_info.remote())
                )
                assert (
                    session_id is not None
                ), f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
                logger.info(
                    f"[RDMA] Obtained remote {session_id} info from rollout engine {engine_ind} rank {engine_rank}"
                )
                logger.info(f"[RDMA] Remote weight info has {len(weights_info)} tensors.")
                # logger.info(list(weights_info.keys()))
                self.remote_weight_infos_by_session_id[session_id] = weights_info
                targets_to_session_id[(engine_ind, engine_rank)] = session_id

            print_memory("[RDMA] After obtaining remote weight info")

            # Local model with identical shape to remote. Create at most one copy per target rank, and link
            # them by session id.
            self.engines = {}
            # Associate transfer tasks based on obtained session and weight info
            for target in self.transfer_plan.targets:
                session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                self.transfer_plan.add_transfer_task(
                    session=session_id,
                    param_group=target.group,
                )
                # Instantiate the local model replicas and a corresponding transfer engine with memory registry for each type of rollout shard.
                if target.engine_rank not in self.engines:
                    transfer_engine = self._create_transfer_engine()
                    model_replica = self._create_inference_replica(
                        self.args.hf_checkpoint,
                        target_rank=target.engine_rank,
                        target_tp=self.args.rollout_num_gpus_per_engine,
                        server_args=self.session_id_to_server_args[session_id],
                    )
                    print_memory("[RDMA] After model replica")
                    weight_memory_registry = self._register_replica_memory(
                        model_replica, self.remote_weight_infos_by_session_id[session_id], transfer_engine
                    )
                    self.engines[target.engine_rank] = (model_replica, transfer_engine, weight_memory_registry)

            print_memory("[RDMA] After Local Engine Replicas and engine Creation")

    def _register_replica_memory(self, model_replica, remote_weight_info, transfer_engine) -> dict:

        old_cuda_alloc_value = os.environ.get("PYTORCH_ALLOC_CONF", "")
        os.environ["PYTORCH_ALLOC_CONF"] = ""
        to_register_named_tensors = []
        named_tensors = dict(model_replica.named_parameters())
        # Verify the 1-to-1 mapping between registered weights and remote weights expected.
        for name, info in remote_weight_info.items():
            (_, remote_numel, remote_ele_size) = info
            if name not in named_tensors:
                raise RuntimeError(f"Remote replica parameter {name} not found in local replica.")
            tensor = named_tensors[name]
            if tensor.numel() != remote_numel or tensor.element_size() != remote_ele_size:
                raise RuntimeError(
                    f"Local replica parameter {name} numel {tensor.numel()} size {tensor.element_size()} does not match remote numel {remote_numel} size {remote_ele_size}."
                )
            if tensor.device.type != "cuda":
                raise RuntimeError(f"Local replica parameter {name} is not on CUDA device.")
            to_register_named_tensors.append((name, tensor))
        weight_memory_registry = register_memory_transfer_engine(to_register_named_tensors, transfer_engine)

        os.environ["PYTORCH_ALLOC_CONF"] = old_cuda_alloc_value
        logger.info(
            f"[RDMA] Registered {len(to_register_named_tensors)} tensors of total {len(named_tensors)} from replica with transfer engine."
        )
        return weight_memory_registry

    def _create_transfer_engine(self) -> TransferEngine:
        # local_ip = ray._private.services.get_node_ip_address()
        # transfer_engine = MooncakeTransferEngine(local_ip, None, None)
        transfer_engine = TransferEngine()
        local_ip = ray._private.services.get_node_ip_address()
        transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")

        logger.info(f"[RDMA] Local replica Transfer Engine initialized at port {transfer_engine.get_rpc_port()}")
        return transfer_engine

    def _create_inference_replica(self, model_path: str, target_rank: int, target_tp: int, server_args: ServerArgs):
        """
        Create model replica for target rank with correct tp settings.

        Uses MockSglangDistributedContext to avoid initializing actual distributed environment
        while ensuring the model weights have the correct shape for the target rank.
        """
        model_config = ModelConfig(model_path)
        load_config = LoadConfig(
            load_format="auto",
            tp_rank=target_rank,
            model_loader_extra_config=server_args.model_loader_extra_config,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        device_config = DeviceConfig()

        # Mock the distributed environment to get correct weight shapes
        # TODO: Reuse https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L845
        # For memory pinning and CPU offloading?
        with MockSglangDistributedContext(tp_size=target_tp, tp_rank=target_rank, server_args=server_args):
            model = get_model(
                model_config=model_config,
                load_config=load_config,
                device_config=device_config,
            )
        device = next(model.parameters()).device
        logger.info(f" Model {device}, params: {sum(p.numel() for p in model.parameters())} ")
        return model

    def _execute_transfer(self, session_id: str) -> None:
        """
        Execute weight transfer for a single transfer task using RDMA P2P transfer engine.
        """
        _, engine, weight_memory_registry = self.engines[self.session_id_to_engine_rank[session_id]]
        remote_weight_info = self.remote_weight_infos_by_session_id[session_id]
        source_ptrs, target_ptrs, source_lens = [], [], []
        for name, tensor_register in weight_memory_registry.items():
            data_ptr, numel, ele_size = tensor_register
            source_ptrs.append(data_ptr)
            target_ptrs.append(remote_weight_info[name][0])  # remote address
            source_lens.append(numel * ele_size)

        # Batch transfer weights through RDMA
        ret = engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        # logger.info(f"[RDMA] Batch transferred {len(weight_memory_registry)} tensors to session {session_id}.")
        if ret < 0:
            raise RuntimeError(f"Batch transfer weights via RDMA failed with error code {ret}.")

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        ray.get(
            [
                engine.update_weight_version.remote(weight_version=str(self.weight_version))
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

        # Refactoring needed:
        # TODO: refactor update_weight to still do a single traversal of the model dict; session_id should be per weight instead.
        # TODO: There is probably enough difference to the UpdateFromNCCL that we should just rebuild from scratch maybe?
        # Functionality missing:
        # TODO: Fix learner PP, right now we still send all weights from any source.
        # TODO: Support engine expert parallel, which has a bunch of details like dp_attnetion_tp etc.
        # TODO: Extensive tests on different pp/ep/tp settings --> tp/dp/ep settings.
        # TODO: Need a correctness test of the model weights similar to the test:https://github.com/sgl-project/sglang/pull/14997/changes#diff-6efab5fd819ef0efa7a1f43989320bb28231702f8840897eb7acacf174f6e71f
        # TODO: Memory profiling.
        # TODO: Design of experiments --- what other configurations do we need to enable.
        # Optimizations:
        # TODO: remote transfer plan optimizes for reduce local copy memory usage
        # TODO: pipeline the all-gather/reshard with transfer engine transfer calls for performance.
        # TODO: increase concurrency with non-blocking transfers to multiple targets.
        # TODO: memory offloading if the replica becomes a bottleneck.
        # Question:
        # 1. Do we really want to support sglang pipeline paralell?

        # Load weights into local replica matching the target session, this handles sharding and reshaping.
        self.engines[self.session_id_to_engine_rank[session_id]][0].load_weights(converted_named_tensors)
        converted_named_tensors.clear()

    def finish_transfer_task(self, session: str) -> None:
        self._execute_transfer(session)
        return


class MockSglangDistributedContext:
    def __init__(self, tp_size: int, tp_rank: int, server_args: ServerArgs):
        """
        TODO: Extend this to support ep, and dp attention?
        """
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = 1
        self.pp_rank = 0
        self.attn_tp_size = tp_size
        self.attn_tp_rank = tp_rank
        self.server_args = server_args
        # Store active patches for cleanup
        self._patches = []

    def __enter__(self):
        """Apply function-level mocks using unittest.mock.patch."""
        from unittest.mock import MagicMock, patch

        # Mock TP group
        mock_group = MagicMock()
        mock_group.world_size = self.tp_size
        mock_group.rank_in_group = self.tp_rank

        # Mock PP group with proper attributes
        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = self.pp_rank
        mock_pp_group.world_size = self.pp_size
        # Mock underlying global variables
        sglang_server_args._global_server_args = self.server_args
        sglang_dp_attention._ATTN_TP_RANK = self.attn_tp_rank
        sglang_dp_attention._ATTN_TP_SIZE = self.attn_tp_size
        sglang_dp_attention._ATTN_DP_RANK = None
        sglang_dp_attention._ATTN_DP_SIZE = 1
        # Mock parallelism getter
        self._patches = [
            patch("sglang.srt.distributed.parallel_state.get_tp_group", return_value=mock_group),
            patch("sglang.srt.distributed.get_pp_group", return_value=mock_pp_group),
            patch(
                "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_world_size", return_value=self.tp_size
            ),
            patch("sglang.srt.distributed.parallel_state.get_tensor_model_parallel_rank", return_value=self.tp_rank),
            patch("sglang.srt.layers.dp_attention.get_attention_tp_rank", return_value=self.attn_tp_rank),
            patch("sglang.srt.layers.dp_attention.get_attention_tp_size", return_value=self.attn_tp_size),
        ]

        for p in self._patches:
            p.start()

        logger.info(
            f"[MockDist] Activated: TP={self.tp_rank}/{self.tp_size}, "
            f"PP={self.pp_rank}/{self.pp_size}, AttnTP={self.attn_tp_rank}/{self.attn_tp_size}"
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all patches and restore original functions."""
        for p in self._patches:
            p.stop()
        sglang_server_args._global_server_args = None
        self._patches.clear()
        logger.info("[MockDist] Deactivated")
        return False  # Don't suppress exceptions
