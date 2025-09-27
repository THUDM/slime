import socket

import ray
import torch
import torch.distributed as dist
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm

from slime.utils.distributed_utils import init_process_group
from slime.utils.memory_utils import clear_memory

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

    use_flattened_tensor_bucket = True
except:
    use_flattened_tensor_bucket = False


class UpdateWeightFromTensor:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self):
        monkey_patch_torch_reductions()
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            named_tensors = [(name, param) for name, param in self.model.state_dict().items()]

        if use_flattened_tensor_bucket:
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()

            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            serialized_tensors = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
        else:
            serialized_tensors = MultiprocessingSerializer.serialize(named_tensors, output_str=True)

        serialized_named_tensors = (
            [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            kwargs = {
                "serialized_named_tensors": serialized_named_tensors,
            }
            if use_flattened_tensor_bucket:
                kwargs["load_format"] = "flattened_bucket"

            ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
            ray.get(ref)


## reference from xtuner_utils.update_weight_utils.UpdateWeightFromDistributed
class UpdateWeightFromDistributed:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            self._group_name = f"slime"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            ## TODO: why +1?
            world_size = self.args.rollout_num_gpus + 1

            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    i * self.args.rollout_num_gpus_per_engine + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    @torch.no_grad()
    def update_weights(self):
        model = self.model
        torch.cuda.empty_cache()
        clear_memory()

        # Use standard FSDP method to get full state dict
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()

        # Send weights one by one to minimize memory usage
        param_names = list(state_dict.keys())

        for i, name in enumerate(tqdm(param_names, desc="[broadcast weight]")):
            # Process one parameter at a time to minimize memory usage
            param = state_dict[name].to(torch.bfloat16)
            single_param_dict = {name: param}

            # Send this single parameter
            self.request_update_params(single_param_dict)

        dist.barrier()
        torch.cuda.empty_cache()
        return

    def request_update_params(self, state_dict):
        if not self._is_src_rank or not state_dict:
            return

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in state_dict.items()],
                dtypes=[param.dtype for _, param in state_dict.items()],
                shapes=[param.shape for _, param in state_dict.items()],
                group_name=self._group_name,
            )
            for engine in self.rollout_engines
        ]

        # Broadcast parameters one by one with memory management
        for name, param in state_dict.items():
            torch.cuda.empty_cache()
            # Ensure tensor is contiguous and on the right device
            param_data = param.data.contiguous()

            # Synchronous broadcast to avoid memory buildup
            dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=False)

            # Clean up immediately after broadcast
            del param_data

        ray.get(refs)
