import datetime
import os
import random
import time

import ray
import requests
from kimina_client import AsyncKiminaClient, CheckResponse
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.utils.misc import exec_command, get_free_port

# TODO handle docker stop more gracefully later
_KILL_PREVIOUS_KIMINA_DOCKER = bool(int(os.environ.get("SLIME_KILL_PREVIOUS_KIMINA_DOCKER", "1")))


class KiminaServerAndClientCluster:
    def __init__(self):
        self._servers = _create_actor_per_node(actor_cls=_KiminaServerActor)
        self._client_cluster = _KiminaClientCluster(self._servers)

    async def check(self, *args, **kwargs) -> CheckResponse:
        return await self._client_cluster.check(*args, **kwargs)


class _KiminaClientCluster:
    def __init__(self, servers: list["_KiminaServerActor"]):
        self._clients = [AsyncKiminaClient(api_url=ray.get(server.get_api_url.remote())) for server in servers]
        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


def _create_actor_per_node(actor_cls) -> list:
    # for simplicity, we use all available nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    assert len(nodes) > 0

    actors = []
    for node in nodes:
        actors.append(
            actor_cls.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
                num_cpus=0.001,
            ).remote()
        )

    return actors


@ray.remote
class _KiminaServerActor:
    def __init__(self):
        self.port = get_free_port()

        if _KILL_PREVIOUS_KIMINA_DOCKER:
            _docker_stop_all()

        self.docker_name = _docker_start(port=self.port)
        _wait_server_ready(base_url=self.get_api_url())

    def get_api_url(self):
        return f"http://{self.docker_name}:8000"


def _docker_start(port: int):
    docker_name = (
        f"kimina_lean_server_auto_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    )
    exec_command(
        "docker run "
        "-d "
        f"--name {docker_name} "
        "--restart unless-stopped "
        "--network formal_math "
        # "--env-file .env "  # do not use env yet
        f"-p {port}:8000 "
        f"projectnumina/kimina-lean-server:2.0.0"
    )
    return docker_name


def _wait_server_ready(base_url: str):
    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health")
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            print(f"Wait kimina server ready ({base_url})...")
            time.sleep(2)


def _docker_stop_all():
    exec_command(
        'ids=$(docker ps -a --filter "name=kimina_lean_server_auto" -q); '
        '[ -n "$ids" ] && docker stop $ids && docker rm $ids; '
        "true"
    )
