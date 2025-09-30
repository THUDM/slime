import socket

import ray
import torch
import torch.distributed as dist
import logging
import gc
import os
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from tqdm import tqdm

from slime.utils.distributed_utils import init_process_group
from slime.utils.memory_utils import clear_memory
from slime.utils.types import ParamInfo

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
    use_flattened_tensor_bucket = True
except ImportError:
    use_flattened_tensor_bucket = False
# Note: FSDP v1 imports removed - we only support FSDP v2

# FSDP v2 imports
from torch.distributed.tensor import DTensor
from slime.utils.memory_utils import clear_memory

# Set up logger for FSDP weight updates
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket, LocalSerializedTensor

    use_flattened_tensor_bucket = True
except:
    use_flattened_tensor_bucket = False


# Use the preprocessing function from utils
# (keeping the same name for backward compatibility)


def get_fsdp_param_info_buckets(args, weights) -> list[list[ParamInfo]]:
    """Create parameter info buckets similar to Megatron's approach."""
    # Create ParamInfo objects for each parameter
    param_infos = []
    rank = dist.get_rank()
    
    for name, param in weights["actor"].items():
        param_infos.append(ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={},  # FSDP doesn't need complex tensor parallel attrs
            size=param.numel() * param.element_size(),
            src_rank=rank,  # All parameters available on all ranks for FSDP
        ))
    
    # Sort by name for consistency
    param_infos = sorted(param_infos, key=lambda info: info.name)
    
    # Create buckets based on buffer size (similar to Megatron)
    param_info_buckets = [[]]
    buffer_size = 0
    buffer_size_limit = args.update_weights_bucket_size
    
    for info in param_infos:
        param_size = info.size
        
        if buffer_size + param_size > buffer_size_limit and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0
        param_info_buckets[-1].append(info)
        buffer_size += param_size
    
    return param_info_buckets


class UpdateWeightFromTensor:
    def __init__(self, args, model, weights, full_params: bool = False):
        self.args = args
        self.model = model
        self.weights = weights  # CPU parameter storage
        self.full_params = full_params
        
        # Bucket-based loading is automatically enabled when full_params=False
        # This provides the Megatron-style optimization for sharded mode
        
        # Create parameter info buckets once during initialization (like Megatron)
        if not self.full_params and self.weights is not None:
            self.param_info_buckets = get_fsdp_param_info_buckets(self.args, self.weights)
            logger.info(f"Created {len(self.param_info_buckets)} parameter buckets for sharded mode")
        else:
            self.param_info_buckets = None
        
        # FSDP v2 model expected
        logger.info(f"Full params mode: {self.full_params}")
        logger.info(f"Bucket-based loading: {not self.full_params} (automatic when full_params=False)")
            
        # Set up tensor parallel configuration for SGLang
        self.tp_size = args.rollout_num_gpus_per_engine
        # tp_rank will be set during connect_rollout_engines based on the IPC group


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
                # Calculate TP rank within this SGLang engine group
                self.tp_rank = dist.get_rank() - start_rank

    @torch.no_grad()
    def update_weights(self):
        logger.info("Starting weight update")
        
        monkey_patch_torch_reductions()
        
        # Get state dict based on configuration
        if self.full_params:
            logger.info("Using FULL_STATE_DICT path")
            # FSDP v2 doesn't need context managers - get state dict directly
            state_dict = self.model.state_dict()
            
            # Preprocess tensors to handle DTensor -> full tensor conversion
            named_tensors = []
            for name, param in state_dict.items():
                # Convert DTensor to full tensor if needed
                if isinstance(param, DTensor):
                    param = param.full_tensor()
                named_tensors.append((name, param))
            del state_dict
            clear_memory()

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
            
            # Clear memory after serialization
            clear_memory()

            serialized_named_tensors = (
                [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
            )
            dist.gather_object(
                serialized_tensors,
                object_gather_list=serialized_named_tensors,
                dst=self._ipc_gather_src,
                group=self._ipc_gather_group,
            )
            clear_memory()

            if dist.get_rank() == self._ipc_gather_src:
                kwargs = {
                    "serialized_named_tensors": serialized_named_tensors,
                }
                if use_flattened_tensor_bucket:
                    kwargs["load_format"] = "flattened_bucket"

                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)
                clear_memory()
        else:
            # For sharded mode (full_params=False), automatically use bucket-based loading
            # This provides Megatron-style memory optimization
            logger.info("Using SHARDED_STATE_DICT path with bucket-based loading from CPU storage")
            
            # Use pre-computed buckets (like Megatron)
            if self.param_info_buckets is None:
                raise RuntimeError("Parameter info buckets not initialized for sharded mode")
            
            for bucket_idx, param_infos in enumerate(self.param_info_buckets):
                # Load only the parameters in this bucket from CPU to GPU (Megatron-style)
                named_tensors_batch = []
                total_params_size = 0
                for param_info in param_infos:
                    # Load parameter from CPU storage to GPU (similar to Megatron approach)
                    cpu_param = self.weights["actor"][param_info.name]
                    gpu_param = cpu_param.to(device=torch.cuda.current_device(), non_blocking=True)
                    named_tensors_batch.append((param_info.name, MultiprocessingSerializer.serialize(gpu_param)))
                    total_params_size += cpu_param.numel()
                    # Clear GPU memory immediately after serialization
                    del gpu_param
                
                torch.cuda.synchronize()
                clear_memory()

                if self._ipc_gather_src == dist.get_rank():
                    # On rank 0, prepare a list to hold the gathered batches from all ranks.
                    gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
                else:
                    gathered_serialized_batches = None

                # Gather the named_tensors_batch from all ranks to rank 0.
                dist.gather_object(
                    obj=named_tensors_batch,
                    object_gather_list=gathered_serialized_batches,
                    dst=self._ipc_gather_src,
                    group=self._ipc_gather_group,
                )
                del named_tensors_batch
                clear_memory()

                if dist.get_rank() == self._ipc_gather_src:
                    # Use zip(*) to "transpose" the data structure.
                    logical_tensors = zip(*gathered_serialized_batches, strict=True)
                    del gathered_serialized_batches
                    clear_memory()

                    # Create LocalSerializedTensor objects for each logical tensor
                    update_tensors = [
                        (
                            tensor_group[0][0],  # Get the name from the first rank's data.
                            LocalSerializedTensor(
                                values=[rank_part[1] for rank_part in tensor_group]
                            ),
                        )
                        for tensor_group in logical_tensors
                    ]

                    # Serialize once and reuse for all TP ranks to avoid memory explosion
                    serialized_update_tensors = MultiprocessingSerializer.serialize(update_tensors, output_str=True)
                    
                    # Clear intermediate data to free memory
                    del update_tensors
                    clear_memory()
                    
                    kwargs = {
                        "serialized_named_tensors": [serialized_update_tensors for _ in range(self.tp_size)],
                        "flush_cache": False,
                    }
                    
                    ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                    ray.get(ref)
                    clear_memory()
                    
                    # Clear serialized data
                    del serialized_update_tensors, kwargs
                    clear_memory()
            
            # Flush cache after all updates
            if dist.get_rank() == self._ipc_gather_src:
                ref = self._ipc_engine.flush_cache.remote()
                ray.get(ref)
                clear_memory()
        
        logger.info("Weight update completed")





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

        # FSDP v2 doesn't need context managers - get state dict directly
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
