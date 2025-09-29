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


class UpdateWeightFromTensor:
    def __init__(self, args, model, full_params: bool = False):
        self.args = args
        self.model = model
        self.full_params = full_params
        
        # FSDP v2 model expected
        logger.info(f"Full params mode: {self.full_params}")
            
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
            logger.info("Using SHARDED_STATE_DICT path")
            # Use SHARDED_STATE_DICT following veRL pattern
            params = self.model.state_dict()
            
            # Preprocess tensors to handle DTensor -> full tensor conversion
            named_tensors = []
            for k, v in params.items():
                # Convert DTensor to full tensor if needed
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                named_tensors.append((k, v))
            del params
            clear_memory()
            
            # Use veRL-style batched weight update approach
            self._update_weights_sharded(named_tensors)
        
        logger.info("Weight update completed")
    
    def _update_weights_sharded(self, named_tensors):
        """Update weights using sharded approach similar to veRL's implementation."""
        logger.info("Starting sharded weight update")
        
        load_format = None
        update_weights_bucket_megabytes = getattr(self.args, 'update_weights_bucket_megabytes', 100)
        update_weights_bucket_bytes = int(update_weights_bucket_megabytes) << 20
        
        # Use batched approach similar to fsdp_sglang.py
        for batch in self._get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            # On each rank, serialize a batch of (name, tensor) tuples.
            # named_tensors_batch will be a list like:
            # [(name0, serialized_tensor0_tp0), (name1, serialized_tensor1_tp0), ...]
            named_tensors_batch = []
            for name, tensor in batch:
                # Convert DTensor to full tensor if needed
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                named_tensors_batch.append((name, MultiprocessingSerializer.serialize(tensor)))
            del batch
            clear_memory()

            if self._ipc_gather_src == dist.get_rank():
                # On rank 0, prepare a list to hold the gathered batches from all ranks.
                gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
            else:
                gathered_serialized_batches = None

            # Gather the named_tensors_batch from all ranks to rank 0.
            # After this, on rank 0, gathered_serialized_batches will be a list of lists:
            # [ [ (name0, s_t0_tp0), (name1, s_t1_tp0), ... ],  # batch from TP rank 0
            #   [ (name0, s_t0_tp1), (name1, s_t1_tp1), ... ],  # batch from TP rank 1
            #   ... ]
            # On other ranks, gathered_serialized_batches will be None.
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
                # This groups the serialized parts for each individual tensor across all TP ranks.
                # Example: from [[(n0, t0_tp0), (n1, t1_tp0)], [(n0, t0_tp1), (n1, t1_tp1)]]
                # to [ ( (n0, t0_tp0), (n0, t0_tp1) ), ( (n1, t1_tp0), (n1, t1_tp1) ) ]
                logical_tensors = zip(*gathered_serialized_batches, strict=True)
                del gathered_serialized_batches
                clear_memory()

                # Create LocalSerializedTensor objects for each logical tensor
                update_tensors = [
                    (
                        tensor_group[0][0],  # Get the name from the first rank's data.
                        LocalSerializedTensor(
                            # 'rank_part' is the (name, serialized_tensor) tuple from one specific rank.
                            values=[rank_part[1] for rank_part in tensor_group]
                        ),
                    )
                    for tensor_group in logical_tensors
                    # each tensor_group is like ( (n0, t0_tp0), (n0, t0_tp1) )
                ]

                # Serialize once and reuse for all TP ranks to avoid memory explosion
                serialized_update_tensors = MultiprocessingSerializer.serialize(update_tensors, output_str=True)
                
                logger.info(f"Sending batch of {len(update_tensors)} parameters to SGLang")
                
                # Clear intermediate data to free memory
                del update_tensors
                clear_memory()
                
                kwargs = {
                    "serialized_named_tensors": [serialized_update_tensors for _ in range(self.tp_size)],
                    "load_format": load_format,
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
            
        logger.info("Sharded weight update completed")
        
    def _get_named_tensor_buckets(self, iterable, bucket_bytes):
        """
        Group tensors into buckets based on a specified size in bytes.
        Similar to the implementation in fsdp_sglang.py.
        
        Args:
            iterable: An iterator of tuples containing tensor names and tensors.
            bucket_bytes: The maximum size of each bucket in bytes.

        Yields:
            Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.
        """
        if bucket_bytes <= 0:
            raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

        current_bucket = []
        current_size = 0
        for name, tensor in iterable:
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > bucket_bytes:
                if current_bucket:
                    yield current_bucket
                current_bucket = [(name, tensor)]
                current_size = tensor_size
            else:
                current_bucket.append((name, tensor))
                current_size += tensor_size

        if current_bucket:
            yield current_bucket


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
    
    def _get_named_tensor_buckets(self, iterable, bucket_bytes):
        """
        Group tensors into buckets based on a specified size in bytes.
        Similar to the implementation in fsdp_sglang.py.
        
        Args:
            iterable: An iterator of tuples containing tensor names and tensors.
            bucket_bytes: The maximum size of each bucket in bytes.

        Yields:
            Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.
        """
        if bucket_bytes <= 0:
            raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

        current_bucket = []
        current_size = 0
        for name, tensor in iterable:
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > bucket_bytes:
                if current_bucket:
                    yield current_bucket
                current_bucket = [(name, tensor)]
                current_size = tensor_size
            else:
                current_bucket.append((name, tensor))
                current_size += tensor_size

        if current_bucket:
            yield current_bucket
