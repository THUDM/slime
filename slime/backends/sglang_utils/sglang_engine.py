import dataclasses

import os
from typing import TYPE_CHECKING

from sglang.srt.server_args import ServerArgs
from slime.utils.http_utils import get_host_info
from .http_server_engine import HttpServerEngineAdapter

if TYPE_CHECKING:
    pass


def get_base_gpu_id(args, rank):
    num_gpus = min(args.rollout_num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.rollout_num_gpus_per_node
    else:
        num_actor_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.rollout_num_gpus_per_node
    return start_index


class SglangEngine:

    def __init__(self, args, rank, dist_init_addr, port, nccl_port):
        self.args = args

        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        nnodes = max(1, args.rollout_num_gpus_per_engine // args.rollout_num_gpus_per_node)
        node_rank = rank % nnodes
        kwargs = {
            "model_path": args.hf_checkpoint,
            "trust_remote_code": True,
            "random_seed": args.seed + rank,
            # memory
            "enable_memory_saver": args.offload,
            # distributed
            "host": get_host_info()[1],
            "port": port,
            "nccl_port": nccl_port,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "dist_init_addr": dist_init_addr,
            "gpu_id_step": 1,
            "base_gpu_id": get_base_gpu_id(args, rank),
            # parallel
            "tp_size": args.rollout_num_gpus_per_engine,
            "dp_size": args.sglang_dp_size,
            "pp_size": args.sglang_pp_size,
            "ep_size": args.sglang_ep_size,
            # always skip warmup to prevent warmup timeout.
            "skip_server_warmup": True,
        }

        unused_keys = set(kwargs.keys())
        for attr in dataclasses.fields(ServerArgs):
            if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
                kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
            unused_keys.discard(attr.name)

        # for compatibility with old args
        if len(unused_keys) > 0:
            print(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
            for key in unused_keys:
                kwargs.pop(key)

        self.llm = HttpServerEngineAdapter(
            router_ip=args.sglang_router_ip, router_port=args.sglang_router_port, **kwargs
        )

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.llm.init_weights_update_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        self.llm.update_weights_from_distributed(names, dtypes, shapes, group_name)
        return

    def update_weights_from_tensor(self, ipc_handles):
        self.llm.update_weights_from_tensor(ipc_handles)
        return

    def reset_prefix_cache(self):
        self.llm.flush_cache()

    def sleep(self, level=1):
        # Adhoc solution to ensure no running requests
        self.llm.flush_cache()
        self.llm.release_memory_occupation()

    def wake_up(self):
        self.llm.resume_memory_occupation()

    def pause_generation(self):
        self.llm.pause_generation()

    def continue_generation(self):
        self.llm.continue_generation()
