import socket

import ray
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from slime.utils.distributed_utils import init_process_group


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
