import ray
import torch
import torch.distributed as dist
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from torch.distributed.device_mesh import init_device_mesh


class UpdateWeightFromTensor:
    """
    FSDP weight update implementation following VERL's exact pattern.
    
    This implementation:
    1. Uses SHARDED_STATE_DICT for memory efficiency (like VERL)
    2. Calls SGLang's sgl_update_weights directly (like VERL)
    3. Handles bucketing for large models (like VERL)
    
    The key insight: SGLang's sgl_update_weights is a utility function that should be
    called directly, not through Ray. Ray is only used for managing rollout engines,
    not for the weight synchronization logic itself.
    """
    
    def __init__(self, args, model):
        self.args = args
        self.model = model
        # Set FSDP to use SHARDED_STATE_DICT for memory efficiency (VERL approach)
        FSDP.set_state_dict_type(
            self.model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(),
        )

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines
        
        # Create device mesh for SGLang weight sync (following VERL pattern)
        world_size = dist.get_world_size()
        device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))
        self.device_mesh = {"infer_tp": device_mesh}

    @torch.no_grad()
    async def update_weights(self):
        """
        Weight update implementation following VERL's exact pattern.
        
        This calls sgl_update_weights directly like VERL does - no Ray involved
        in the weight sync logic itself. Ray engines are just the target destinations.
        """
        
        # Step 1: Get sharded state dict (VERL approach - returns DTensors)
        params = self.model.state_dict()
        
        # Step 2: Convert to named_tensors format (VERL pattern)
        named_tensors = [(k, v) for k, v in params.items()]
        
        # Step 3: Bucket parameters for memory efficiency (VERL approach)
        # Convert megabytes to bytes using bit shift : MB << 20 = MB * 1024 * 1024
        update_weights_bucket_bytes = int(
            getattr(self.args, 'update_weights_bucket_megabytes', 512)
        ) << 20
        
        param_buckets = self._get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes)
        
        # Step 4: Call sgl_update_weights for each bucket (exactly like VERL)
        # SGLang handles all the complex logic internally:
        # - DTensor.full_tensor() conversion
        # - Serialization with MultiprocessingSerializer  
        # - Gathering to rank 0
        # - LocalSerializedTensor creation
        # - Transmission to rollout engine
        for params_batch in param_buckets:
            for engine in self.rollout_engines:
                await sgl_update_weights(
                    engine=engine,
                    params_batch=params_batch,
                    device_mesh_key="infer_tp", 
                    device_mesh=self.device_mesh,
                )
        
        # Step 5: Flush cache (VERL pattern)
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            for engine in self.rollout_engines:
                await engine.flush_cache()

    def _get_named_tensor_buckets(self, named_tensors, fsdp_update_weights_bucket_megabytes):
        """
        Create parameter buckets for memory efficiency.
        Simplified version of VERL's get_named_tensor_buckets logic.
        """
        buckets = []
        current_bucket = []
        current_size = 0

        for name, tensor in named_tensors:
            # Estimate tensor size
            tensor_size = tensor.numel() * tensor.element_size() if hasattr(tensor, 'element_size') else tensor.numel() * 4
            
            # If adding this tensor would exceed bucket size, start new bucket
            if current_size + tensor_size > fsdp_update_weights_bucket_megabytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append((name, tensor))
            current_size += tensor_size

        # Add the last bucket if it has any parameters
        if current_bucket:
            buckets.append(current_bucket)

        return buckets
