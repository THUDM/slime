import socket

import ray
import torch
import torch.distributed as dist
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

from slime.utils.memory_utils import clear_memory

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

    use_flattened_tensor_bucket = True
except:
    use_flattened_tensor_bucket = False


def get_param_info_buckets(args, weights) -> list[list[ParamInfo]]:
    """Create parameter info buckets similar to Megatron's approach."""
    # Create ParamInfo objects for each parameter
    param_infos = []
    rank = dist.get_rank()

    for name, param in weights["actor"].items():
        param_infos.append(
            ParamInfo(
                name=name,
                dtype=param.dtype,
                shape=param.shape,
                attrs={},  # FSDP doesn't need complex tensor parallel attrs
                size=param.numel() * param.element_size(),
                src_rank=rank,  # All parameters available on all ranks for FSDP
            )
        )

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
            self.param_info_buckets = get_param_info_buckets(self.args, self.weights)
        else:
            self.param_info_buckets = None

        # FSDP v2 model expected

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

        monkey_patch_torch_reductions()

        if self.full_params:
            print("Using FULL_STATE_DICT path")
            state_dict = self.model.state_dict()

            # Preprocess tensors to handle DTensor -> full tensor conversion
            named_tensors = [(name, param) for name, param in self.model.state_dict().items()]
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

            clear_memory()

            serialized_named_tensors = (
                [None] * dist.get_world_size(self._ipc_gather_group)
                if self._ipc_gather_src == dist.get_rank()
                else None
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
            print("Using SHARDED_STATE_DICT path with bucket-based loading from CPU storage")
            if self.param_info_buckets is None:
                raise RuntimeError("Parameter info buckets not initialized for sharded mode")

            for param_infos in self.param_info_buckets:
                # Load only the parameters in this bucket from CPU to GPU
                named_tensors_batch = []
                for param_info in param_infos:
                    cpu_param = self.weights["actor"][param_info.name]
                    gpu_param = cpu_param.to(device=torch.cuda.current_device(), non_blocking=True)
                    named_tensors_batch.append((param_info.name, gpu_param))

                torch.cuda.synchronize()

                # Use flattened bucket approach similar to Megatron and full_params=True
                if use_flattened_tensor_bucket:
                    print("Using flattened tensor bucket")
                    # Group tensors by dtype (same as Megatron)
                    named_tensors_by_dtypes = {}
                    for name, tensor in named_tensors_batch:
                        dtype = tensor.dtype
                        if dtype not in named_tensors_by_dtypes:
                            named_tensors_by_dtypes[dtype] = []
                        named_tensors_by_dtypes[dtype].append((name, tensor))

                    # Create flattened bucket for each dtype group
                    serialized_tensors = []
                    for dtype, named_tensors in named_tensors_by_dtypes.items():
                        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
                        metadata = flattened_tensor_bucket.get_metadata()
                        flattened_tensor_data = {
                            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                            "metadata": metadata,
                        }
                        serialized_tensors.append(
                            MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                        )
                else:
                    # Fallback to non-flattened approach
                    serialized_tensors = MultiprocessingSerializer.serialize(named_tensors_batch, output_str=True)

                # Clean up GPU tensors after serialization
                del named_tensors_batch
                clear_memory()

                if self._ipc_gather_src == dist.get_rank():
                    # On rank 0, prepare a list to hold the gathered batches from all ranks.
                    gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
                else:
                    gathered_serialized_batches = None

                # Gather the serialized batches from all ranks to rank 0.
                dist.gather_object(
                    obj=serialized_tensors,
                    object_gather_list=gathered_serialized_batches,
                    dst=self._ipc_gather_src,
                    group=self._ipc_gather_group,
                )
                del serialized_tensors
                clear_memory()

                if dist.get_rank() == self._ipc_gather_src:
                    if use_flattened_tensor_bucket:
                        # Handle flattened bucket format (same as Megatron approach)
                        # Each rank may have multiple dtype buckets
                        # TODO: here we assume all ranks have the same number of dtypes
                        num_dtypes = len(gathered_serialized_batches[0])
                        for i in range(num_dtypes):
                            kwargs = {
                                "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                                "load_format": "flattened_bucket",
                                "flush_cache": False,
                            }
                            ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                            ray.get(ref)
                    else:
                        # Non-flattened approach
                        kwargs = {
                            "serialized_named_tensors": gathered_serialized_batches,
                            "flush_cache": False,
                        }
                        ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                        ray.get(ref)

                    del gathered_serialized_batches, kwargs
                    clear_memory()

            if dist.get_rank() == self._ipc_gather_src:
                ref = self._ipc_engine.flush_cache.remote()
                ray.get(ref)
                clear_memory()


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
