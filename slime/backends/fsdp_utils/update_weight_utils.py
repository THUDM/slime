import ray
import torch
import torch.distributed as dist
import logging
import gc
import os
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
    use_flattened_tensor_bucket = True
except ImportError:
    use_flattened_tensor_bucket = False
# Note: FSDP v1 imports removed - we only support FSDP v2

# Import FSDP version utilities
from .fsdp_version_utils import (
    preprocess_tensor_for_update_weights as _preprocess_tensor_for_update_weights,
    verify_model_is_fsdp_v2,
)
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
        
        # Verify FSDP v2 (will raise error if not FSDP v2)
        verify_model_is_fsdp_v2(model)
        logger.info("Detected FSDP version: 2")
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
            named_tensors = [(name, _preprocess_tensor_for_update_weights(param)) for name, param in state_dict.items()]
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
            
            # Preprocess tensors to handle DTensor/ShardedTensor -> full tensor conversion
            named_tensors = [(k, _preprocess_tensor_for_update_weights(v)) for k, v in params.items()]
            del params
            clear_memory()
            
            # Use veRL-style batched weight update approach
            self._update_weights_sharded(named_tensors)
        
        logger.info("Weight update completed")
    
    def _update_weights_sharded(self, named_tensors):
        """Update weights using sharded approach similar to veRL's implementation."""
        logger.info("Starting sharded weight update")
        
        load_format = None
        
        # Serialize tensors on each rank
        named_tensors_batch = [
            (name, MultiprocessingSerializer.serialize(tensor))
            for name, tensor in named_tensors
        ]
        # Clear original tensors to free memory
        del named_tensors
        clear_memory()

        # Use IPC group approach to gather tensors from all FSDP ranks
        gathered_serialized_batches = (
            [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            named_tensors_batch,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )
        # Clear batch data to free memory
        del named_tensors_batch
        clear_memory()

        if dist.get_rank() == self._ipc_gather_src:
            # Use zip(*) to "transpose" the data structure following veRL pattern
            logical_tensors = zip(*gathered_serialized_batches, strict=True)

            # Create LocalSerializedTensor objects for each logical tensor
            update_tensors = [
                (
                    tensor_group[0][0],  # Get the name from the first rank's data
                    LocalSerializedTensor(
                        values=[rank_part[1] for rank_part in tensor_group]
                    ),
                )
                for tensor_group in logical_tensors
            ]

            # Serialize once and reuse for all TP ranks to avoid memory explosion
            serialized_update_tensors = MultiprocessingSerializer.serialize(update_tensors, output_str=True)
            # Clear intermediate data to free memory
            del update_tensors, gathered_serialized_batches
            clear_memory()
            
            kwargs = {
                "serialized_named_tensors": [serialized_update_tensors for _ in range(self.tp_size)],
                "load_format": load_format,
                "flush_cache": False,
            }

            logger.info("Sending weights to SGLang")
            ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
            ray.get(ref)
            clear_memory()
            
            # Clear serialized data
            del serialized_update_tensors, kwargs
            clear_memory()
            
            # Flush cache after all updates
            ref = self._ipc_engine.flush_cache.remote()
            ray.get(ref)
            clear_memory()
            
        logger.info("Sharded weight update completed")
