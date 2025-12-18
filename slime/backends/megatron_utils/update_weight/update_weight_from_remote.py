from abc import abstractmethod
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Literal
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group
from slime.utils.timer import timer

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param
from .remote_transfer_plan import RemoteTransferPlan


class UpdateWeightFromRemote:
    """
    Abstract base class for remote bucketed tensor weight update. Weights are all-gathered in TP EP dimension
    for each bucket of predefined size and processed into HF format.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        weight_update_mode: Literal["nccl", "rdma"] = "nccl",
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.transfer_plan = RemoteTransferPlan(args, model, weight_update_mode)
        self._is_source = self.transfer_plan.is_source()

    @abstractmethod
    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Establish connection to remote rollout engines.
        """

    @abstractmethod
    def _update_bucket_weights_from_remote(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], session_id: str, pbar: tqdm | None = None
    ) -> None:
        """
        Implementation of the bucketed parameter update from remote. session_id is used as the identifier
        for the operation, either NCCL group name or Transfer Engine session id.
        TODO(jd): to avoid traversing the model dict multiple times we need session_id to be a list.
        """

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            for transfer_task in self.transfer_plan.get_transfer_tasks():
                # Update non-expert or expert weights
                if transfer_task.tensor_type == "non-expert":
                    self._update_weights(transfer_task.named_params_and_buffers, transfer_task.session)
                elif transfer_task.tensor_type == "expert":
                    self._update_expert_weights(transfer_task.named_params_and_buffers, transfer_task.session)
                else:
                    raise ValueError(f"Unknown tensor type {transfer_task.tensor_type} in transfer task.")
                dist.barrier(group=get_gloo_group())
                self.finish_transfer_task(transfer_task.session)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            self.leader_post_update()
        dist.barrier(group=get_gloo_group())

    def leader_post_update(self) -> None:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        return

    def finish_transfer_task(self, session: str) -> None:
        return

    def _update_expert_weights(
        self, named_params_and_buffers: Sequence[tuple[str, torch.Tensor]], session_id: str
    ) -> None:
        pbar = tqdm(desc=f"[{session_id}] Update Expert Weights", total=0) if self._is_source else None
        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers:
            # transfer expert tensors
            assert ".experts." in name, "Function intended for expert params only."
            buffer_size = self._update_expert_weight_from_remote(
                name, param, named_tensors, buffer_size, session_id, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_remote(named_tensors, session_id, pbar=pbar)

    def _update_weights(self, named_params_and_buffers: Sequence[tuple[str, torch.Tensor]], session_id: str) -> None:
        pbar = tqdm(desc=f"[{session_id}] Update Weights", total=0) if self._is_source else None
        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        for name, param in named_params_and_buffers:
            # transfer tp tensors
            assert ".experts." not in name, "Function intended for non-expert params only."
            buffer_size = self._update_weight_from_remote(
                name, param, converted_named_tensors, buffer_size, session_id, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_remote(converted_named_tensors, session_id, pbar=pbar)

    def _update_weight_from_remote(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        session_id: str,
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(name, param)
        if not self._is_source:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_remote(converted_named_tensors, session_id, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_remote(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        session_id: str,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size and named_tensors:
            self._update_expert_bucket_weights_from_remote(named_tensors, session_id, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_remote(
        self, named_tensors: list[tuple[str, torch.Tensor]], session_id: str, pbar: tqdm | None = None
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_source:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_remote(converted_hf_tensors, session_id, pbar)
