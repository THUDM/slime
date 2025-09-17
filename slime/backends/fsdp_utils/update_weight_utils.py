import ray
import torch
import torch.distributed as dist
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

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

        if self._is_src_rank:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
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

        if (model.config.float8_cfg is not None) and (model.config.float8_cfg.enable_float8):
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.bfloat16

        def get_params(tensor_list, name_list, save_dtype):
            _tensor_list, _spec_list = list(zip(*tensor_list))
            fsdp_unshard_tensor_list = model._fsdp_foreach_allgather(_tensor_list, _spec_list)
            if save_dtype == torch.float8_e4m3fn:
                fsdp_unshard_tensor_list, name_list = model._to_float8(
                    fsdp_unshard_tensor_list, name_list, _tensor_list, save_dtype
                )
            return fsdp_unshard_tensor_list, name_list

        saved_list = []
        for i, layer in tqdm(model.layers.items(), desc="[gather weight]"):
            tensor_list = []
            name_list = []
            for sub_name, param in layer.state_dict().items():
                saved_list.append(f"layers.{i}.{sub_name}")
                local_tensor = param._local_tensor if isinstance(param, DTensor) else param
                local_tensor = local_tensor.bfloat16()
                load_spec = model.load_spec_mapping.get(f"layers.{i}.{sub_name}")
                name = f"model.layers.{i}.{sub_name}"
                if ".experts." in name and ".mlp.experts." not in name:
                    name = name.replace(".experts.", ".mlp.experts.")
                if ".gate." in name and ".mlp.gate." not in name:
                    name = name.replace(".gate.", ".mlp.gate.")
                name_list.append(name)
                tensor_list.append((local_tensor, load_spec))
            fsdp_unshard_tensor_list, name_list = get_params(tensor_list, name_list, dtype)
            state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
            self.request_update_params(state_dict)

        tensor_list = []
        name_list = []
        for name, param in model.state_dict().items():
            if name in saved_list:
                continue
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            load_spec = model.load_spec_mapping.get(name)
            if name == "norm.weight":
                name = "model.norm.weight"
            elif name == "embed_tokens.weight":
                name = "model.embed_tokens.weight"
            tensor_list = [(local_tensor, load_spec)]
            name_list = [name]
            fsdp_unshard_tensor_list, name_list = get_params(tensor_list, name_list, dtype)
            state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
            self.request_update_params(state_dict)

        self.request_update_params({}, finished=True)

        dist.barrier()
        torch.cuda.empty_cache()
        return

    def request_update_params(self, state_dict, finished=False):
        if not self._is_src_rank:
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

        handles = []
        for _, param in state_dict.items():
            handles.append(dist.broadcast(param.data, 0, group=self._model_update_groups, async_op=True))
        for handle in handles:
            handle.wait()

        ray.get(refs)
