import multiprocessing
import time
from typing import List, Optional

import aiohttp
import requests
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree


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


class HttpServerEngineAdapter(EngineBase):
    """
    You can use this class to launch a server from a VerlEngine instance.
    We recommend using this class only you need to use http server.
    Otherwise, you can use Engine directly.
    """

    def __init__(self, router_ip=None, router_port=None, **kwargs):
        self.router_ip = router_ip
        self.router_port = router_port
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

        if ("generate" in endpoint or "worker" in endpoint) and self.router_ip and self.router_port:
            url = f"http://{self.router_ip}:{self.router_port}/{endpoint}"
        else:
            url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()

    async def _make_request_async(self, endpoint: str, payload: Optional[dict] = None):
        """Make an asynchronous POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        if ("generate" in endpoint or "worker" in endpoint) and self.router_ip and self.router_port:
            url = f"http://{self.router_ip}:{self.router_port}/{endpoint}"
        else:
            url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload or {}) as response:
                response.raise_for_status()
                return await response.json(content_type=None)

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        while True:
            response = requests.get(f"http://{self.server_args.host}:{self.server_args.port}/flush_cache")
            if response.status_code == 200:
                break

    def shutdown(self):
        requests.post(
            f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_args.host}:{self.server_args.port}"
        )
        kill_process_tree(self.process.pid)

    def generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        token_ids_logprob=None,
        lora_path=None,
        custom_logit_processor=None,
    ):
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("generate", payload)

    async def generate_async(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        token_ids_logprob=None,
        lora_path=None,
        custom_logit_processor=None,
    ):
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self._make_request_async("generate", payload)

    def release_memory_occupation(self):
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self):
        return self._make_request("resume_memory_occupation")

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

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        return self._make_request(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
            },
        )

    def pause_generation(self):
        return requests.post(f"http://{self.server_args.host}:{self.server_args.port}/pause_generation", json={})

    def continue_generation(self):
        return requests.post(f"http://{self.server_args.host}:{self.server_args.port}/continue_generation", json={})
