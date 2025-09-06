import dataclasses
import multiprocessing
import time
from typing import List, Optional

import requests
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info


def get_base_gpu_id(args, rank):
    num_gpus = min(args.rollout_num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.rollout_num_gpus_per_node
    else:
        num_actor_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.rollout_num_gpus_per_node
    return start_index


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:

    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return

    base_url = server_args.url()

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {server_args.api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                if response.status_code == 200:
                    break

            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

    return p


class SGLangEngine(RayActor):
    def __init__(self, args, rank: int):
        self.args = args
        self.rank = rank

    def init(self, dist_init_addr, port, nccl_port):
        args = self.args
        rank = self.rank

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

        self.router_ip = args.sglang_router_ip
        self.router_port = args.sglang_router_port
        self.server_args = ServerArgs(**kwargs)
        self.node_rank = self.server_args.node_rank
        print(f"Launch HttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}")
        self.process = launch_server_process(self.server_args)
        if self.node_rank == 0 and self.router_ip and self.router_port:
            requests.post(
                f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_args.host}:{self.server_args.port}"
            )

    def _make_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        while True:
            try:
                response = requests.get(f"http://{self.server_args.host}:{self.server_args.port}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                print(f"Error flushing cache: {e}")
                continue

    def shutdown(self):
        requests.post(
            f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_args.host}:{self.server_args.port}"
        )
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_args.host}:{self.server_args.port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: List[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: Optional[str] = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        return requests.post(f"http://{self.server_args.host}:{self.server_args.port}/pause_generation", json={})

    def continue_generation(self):
        return requests.post(f"http://{self.server_args.host}:{self.server_args.port}/continue_generation", json={})

    def start_profile(
        self,
        # The output directory
        output_dir: Optional[str] = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        activities: Optional[List[str]] = None,
        profile_by_stage: bool = False,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
    ):
        return requests.post(
            f"http://{self.server_args.host}:{self.server_args.port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )

    def stop_profile(self):
        return requests.post(f"http://{self.server_args.host}:{self.server_args.port}/stop_profile", json={})
