import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
from ray.actor import ActorHandle
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig

# from mooncake.engine import TransferEngine
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.model_loader.loader import get_model_loader
from tqdm import tqdm

from slime.backends.megatron_utils.update_weight.remote_transfer_plan import TransferTask
from slime.utils.memory_utils import print_memory

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

        if self._is_source:
            local_ip = ray._private.services.get_node_ip_address()
            self.transfer_engine = MooncakeTransferEngine(local_ip, None, None)
            logger.info(f"[RDMA] Transfer Engine initialized at port {self.transfer_engine.session_id}")

            # Query Engine session and weight info from rollout instances according to the transfer plan
            self.remote_weight_infos_by_session_id = {}
            targets_to_query = set((target.engine_ind, target.engine_rank) for target in self.transfer_plan.targets)
            targets_to_session_id, self.session_id_to_engine_rank = {}, {}
            for engine_ind, engine_rank in targets_to_query:
                session_id, weights_info = ray.get(
                    self.rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
                )
                self.session_id_to_engine_rank[session_id] = engine_rank
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

            print_memory("[RDMA] After obtaining remote weight info")

            # Local model with identical shape to remote. Create at most one copy per target rank, and link
            # them by session id.
            self.engines, self.session_id_to_local_replicas = {}, {}
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
                # Instantiate the local model replicas and a corresponding transfer engine with memory registry for each type of rollout shard.
                if target.engine_rank not in self.engines:
                    model_replica = self._create_inference_replica(
                        self.args.hf_checkpoint,
                        target_tp=self.args.rollout_num_gpus_per_engine,
                        target_rank=target.engine_rank,
                    )
                    transfer_engine = self._create_transfer_engine()
                    weight_memory_registry = self._register_replica_memory(
                        model_replica, self.remote_weight_infos_by_session_id[session_id], transfer_engine
                    )
                    self.engines[target.engine_rank] = (model_replica, transfer_engine, weight_memory_registry)

            print_memory("[RDMA] After Local Engine Replicas and engine Creation")

    def _register_replica_memory(self, model_replica, remote_weight_info, transfer_engine) -> dict:
        to_register_named_tensors = []
        named_tensors = dict(model_replica.named_parameters())
        # Verify the 1-to-1 mapping between registered weights and remote weights expected.
        for name, (_, remote_numel, remote_ele_size) in remote_weight_info:
            if name not in named_tensors:
                raise RuntimeError(f"Remote replica parameter {name} not found in local replica.")
            tensor = named_tensors[name]
            if tensor.numel() != remote_numel or tensor.element_size() != remote_ele_size:
                raise RuntimeError(
                    f"Local replica parameter {name} numel {tensor.numel()} size {tensor.element_size()} does not match remote numel {remote_numel} size {remote_ele_size}."
                )
            to_register_named_tensors.append((name, tensor))
        weight_memory_registry = register_memory_transfer_engine(to_register_named_tensors, transfer_engine)
        logger.info(
            f"[RDMA] Registered {len(to_register_named_tensors)} tensors of total {len(named_tensors)} from replica with transfer engine."
        )
        return weight_memory_registry

    def _create_transfer_engine(self) -> MooncakeTransferEngine:
        local_ip = ray._private.services.get_node_ip_address()
        transfer_engine = MooncakeTransferEngine(local_ip, None, None)
        logger.info(f"[RDMA] Local replica Transfer Engine initialized at port {transfer_engine.session_id}")
        return transfer_engine

    def _create_inference_replica(self, model_path, target_tp, target_rank):
        # Create model replica for target rank with correct tp settings.
        # FIXME: how to validate this works?
        model_config = ModelConfig(model_path)
        loader = get_model_loader(
            load_config=LoadConfig(load_format="direct", tp_rank=target_rank),
            model_config=model_config,
        )
        return loader.load_model(
            model_config=model_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )

    def _execute_transfer(self, session_id: str) -> None:
        """
        Execute weight transfer for a single transfer task using RDMA P2P transfer engine.
        """
        _, engine, weight_memory_registry = self.engines[self.session_id_to_engine_rank[session_id]]
        remote_weight_info = self.remote_weight_infos_by_session_id[session_id]
        source_ptrs, target_ptrs, source_lens = [], [], []
        for name, tensor in weight_memory_registry.items():
            source_ptrs.append(tensor.data_ptr())
            target_ptrs.append(remote_weight_info[name][0])  # remote address
            source_lens.append(tensor.numel() * tensor.element_size())

        # Batch transfer weights through RDMA
        ret = engine.batch_transfer_sync(session_id, source_ptrs, target_ptrs, source_lens)
        logger.info(f"[RDMA] Batch transferred {len(weight_memory_registry)} tensors to session {session_id}.")
        if ret < 0:
            raise RuntimeError(f"Batch transfer weights via RDMA failed with error code {ret}.")

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

        # Refactoring needed:
        # TODO: refactor update_weight to still do a single traversal of the model dict; session_id should be per weight instead.
        # TODO: There is probably enough difference to the UpdateFromNCCL that we should just rebuild from scratch maybe?
        # Functionality missing:
        # TODO: Fix learner PP, right now we still send all weights from any source.
        # TODO: Support engine expert parallel
        # TODO: Extensive tests on different pp/ep/tp settings --> tp/dp/ep settings.
        # TODO: Need a correctness test of the model weights similar to the test:https://github.com/sgl-project/sglang/pull/14997/changes#diff-6efab5fd819ef0efa7a1f43989320bb28231702f8840897eb7acacf174f6e71f
        # TODO: Memory profiling.
        # TODO: Design of experiments --- what other configurations do we need to enable.
        # Optimizations:
        # TODO: remote transfer plan optimizes for reduce local copy memory usage
        # TODO: pipeline the all-gather/reshard with transfer engine transfer calls for performance.
        # TODO: increase concurrency with non-blocking transfers to multiple targets.
        # TODO: memory offloading if the replica becomes a bottleneck.

        # Load weights into local replica matching the target session, this handles sharding and reshaping.
        self.session_id_to_local_replicas[session_id].load_weights(converted_named_tensors)
        converted_named_tensors.clear()

    def finish_transfer_task(self, task: TransferTask) -> None:
        self._execute_transfer(task.session)
        return
