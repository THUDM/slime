"""CPU lifecycle tests for workers-first Omni deployment startup."""

from argparse import Namespace
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import requests

from slime.backends.sglang_utils import deployment
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@dataclass
class _Handle:
    operation: str
    engine: object
    error: Exception | None = None


class _Engine:
    def __init__(self, harness, number):
        self.harness = harness
        self.number = number
        self.url = f"http://worker-{number}:15000"
        self.shutdown = SimpleNamespace(remote=self._shutdown)
        self.attach_router = SimpleNamespace(remote=self._attach_router)

    def _shutdown(self):
        self.harness.events.append(("shutdown", self.number))
        if self.number in self.harness.shutdown_errors:
            raise RuntimeError("shutdown RPC unavailable")
        return _Handle("shutdown", self)

    def _attach_router(self, host, port):
        self.harness.events.append(("attach", self.number, host, port))
        return _Handle("attach", self, self.harness.attach_error)


class _Group:
    def __init__(self, harness, configuration):
        self.harness = harness
        self.configuration = configuration
        self.router_ip = self.router_port = None
        self.nodes_per_engine = 1
        self.all_engines = []
        self.engine_urls = {}

    @property
    def engines(self):
        return self.all_engines

    def start_engines(self, port_cursors):
        self.harness.events.append(("start_engines",))
        count = self.configuration.num_gpus // self.configuration.num_gpus_per_engine
        for _ in range(count):
            engine = _Engine(self.harness, len(self.harness.engines))
            self.harness.engines.append(engine)
            self.all_engines.append(engine)
            self.engine_urls[len(self.all_engines) - 1] = engine.url
            if self.harness.start_error is not None:
                self.all_engines.append(None)
                raise self.harness.start_error
        return [_Handle("init", engine, self.harness.init_error) for engine in self.engines], {0: 16000}


class _Process:
    def __init__(self, harness, *, target, args):
        self.harness = harness
        self.number = len(harness.processes)
        self.target, self.args = target, args
        self.alive = False
        self.exitcode = 23
        harness.processes.append(self)

    def start(self):
        self.harness.events.append(("router_start", self.number))
        self.alive = self.number not in self.harness.exited_routers

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.harness.events.append(("router_terminate", self.number))
        if not self.harness.stubborn_router:
            self.alive = False

    def join(self, *, timeout):
        self.harness.events.append(("router_join", self.number, timeout))

    def kill(self):
        self.harness.events.append(("router_kill", self.number))
        self.alive = False


class _Session:
    def __init__(self, harness):
        self.harness = harness
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, url, *, timeout):
        self.harness.events.append(("router_live", url, timeout, self.trust_env))
        if self.harness.http_error is not None:
            raise self.harness.http_error
        return SimpleNamespace(raise_for_status=lambda: None)


class _Harness:
    def __init__(self, monkeypatch):
        self.args = Namespace(
            rollout_backend="sglang-omni",
            rollout_external=False,
            colocate=False,
            offload_rollout=False,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            num_gpus_per_node=8,
            rollout_num_gpus_per_engine=1,
            hf_checkpoint="policy",
            sglang_router_ip=None,
            sglang_router_port=3001,
            sglang_router_request_timeout_secs=30,
            omni_router_policy="round_robin",
            omni_python="/python/omni",
            omni_admin_key="test-admin-key",
            rollout_health_check_timeout=0.3,
            rollout_cleanup_timeout=5,
        )
        self.models = [self.model("actor", replicas=2)]
        self.events, self.engines, self.groups, self.processes = [], [], [], []
        self.init_error = self.attach_error = self.start_error = self.http_error = None
        self.wait_error = None
        self.shutdown_errors, self.exited_routers = set(), set()
        self.stubborn_router = False
        self.now = 0
        monkeypatch.setattr(deployment, "resolve_sglang_config", lambda args: SglangConfig(self.models))
        monkeypatch.setattr(deployment, "ServerGroupPlacement", lambda **kwargs: self)
        monkeypatch.setattr(deployment, "get_host_info", lambda: ("hostname", "127.0.0.1"))
        monkeypatch.setattr(deployment, "find_available_port", lambda start: 3002)
        monkeypatch.setattr(deployment, "multiprocessing", SimpleNamespace(Process=self.process))
        monkeypatch.setattr(
            deployment,
            "requests",
            SimpleNamespace(Session=lambda: _Session(self), RequestException=requests.RequestException),
        )
        monkeypatch.setattr(deployment, "ray", SimpleNamespace(get=self.get, wait=self.wait, kill=self.kill))
        monkeypatch.setattr(deployment, "time", SimpleNamespace(monotonic=lambda: self.now, sleep=self.sleep))

    @staticmethod
    def model(name, *, replicas=1, update_weights=True):
        return ModelConfig(
            name=name,
            update_weights=update_weights,
            server_groups=[ServerGroupConfig("regular", replicas, num_gpus_per_engine=1)],
        )

    def create(self, configuration, router_ip, router_port):
        assert router_ip is None and router_port is None
        group = _Group(self, configuration)
        self.groups.append(group)
        return group

    def process(self, **kwargs):
        return _Process(self, **kwargs)

    def sleep(self, duration):
        self.now += duration

    def get(self, handles):
        for handle in handles:
            self.events.append(("get", handle.operation, handle.engine.number))
            if handle.error is not None:
                raise handle.error
        return [None] * len(handles)

    def wait(self, handles, *, num_returns, timeout):
        self.events.append(("wait_shutdown", tuple(handle.engine.number for handle in handles), num_returns, timeout))
        if self.wait_error is not None:
            raise self.wait_error
        return handles, []

    def kill(self, engine, *, no_restart):
        self.events.append(("kill", engine.number, no_restart))

    def assert_cleaned(self):
        assert [event[1] for event in self.events if event[0] == "shutdown"] == list(range(len(self.engines)))
        assert [event[1:] for event in self.events if event[0] == "kill"] == [
            (number, True) for number in range(len(self.engines))
        ]
        assert all(not process.is_alive() for process in self.processes)
        assert not hasattr(self.args, "sglang_model_routers")


@pytest.fixture
def harness(monkeypatch):
    return _Harness(monkeypatch)


def test_workers_are_ready_before_named_routers_are_started(harness):
    from slime.backends.sglang_utils.omni_engine import OMNI_WORKER_CAPABILITIES

    harness.models.append(harness.model("teacher", update_weights=False))
    servers, handles = deployment.start_rollout_servers(harness.args, pg=object())

    assert handles == []
    assert list(servers) == ["actor", "teacher"]
    assert harness.args.sglang_model_routers == {"actor": ("127.0.0.1", 3001), "teacher": ("127.0.0.1", 3002)}
    for process, numbers, disabled in [(harness.processes[0], [0, 1], True), (harness.processes[1], [2], False)]:
        _, configuration, initially_disabled, admin_key = process.args
        assert initially_disabled is disabled
        assert admin_key == "test-admin-key"
        assert configuration["workers"] == [
            {"url": harness.engines[number].url, "capabilities": OMNI_WORKER_CAPABILITIES} for number in numbers
        ]
        router_started = harness.events.index(("router_start", process.number))
        for number in numbers:
            assert harness.events.index(("get", "init", number)) < router_started
            assert (
                harness.events.index(("attach", number, configuration["host"], configuration["port"])) > router_started
            )
    assert all(event[-1] is False for event in harness.events if event[0] == "router_live")
    assert all(group.router_ip == "127.0.0.1" for group in harness.groups)
    assert not any(event[0] in {"shutdown", "kill"} for event in harness.events)


@pytest.mark.parametrize("failure", ["partial_start", "init", "router_timeout"])
def test_startup_failure_shuts_down_and_kills_all_created_actors(harness, failure):
    error = RuntimeError(f"original {failure} failure")
    if failure == "partial_start":
        harness.start_error = error
    elif failure == "init":
        harness.init_error = error
    else:
        harness.http_error = requests.ConnectionError("router not ready")

    expected_type = TimeoutError if failure == "router_timeout" else RuntimeError
    with pytest.raises(expected_type) as caught:
        deployment.start_rollout_servers(harness.args, pg=object())

    if failure in {"partial_start", "init"}:
        assert caught.value is error
    else:
        assert "startup timed out" in str(caught.value)
    harness.assert_cleaned()
    waits = [event for event in harness.events if event[0] == "wait_shutdown"]
    assert len(waits) == 1
    assert waits[0][2] == len(harness.engines)
    assert 0 < waits[0][3] <= harness.args.rollout_cleanup_timeout / 2
    if failure in {"partial_start", "init"}:
        assert harness.processes == []


def test_later_named_router_failure_cleans_up_earlier_model(harness):
    harness.models.append(harness.model("teacher", update_weights=False))
    harness.exited_routers.add(1)

    with pytest.raises(RuntimeError, match="exited with code 23"):
        deployment.start_rollout_servers(harness.args, pg=object())

    assert len(harness.engines) == 3
    assert len(harness.processes) == 2
    harness.assert_cleaned()
    assert ("router_terminate", 0) in harness.events
    assert ("router_terminate", 1) not in harness.events


def test_cleanup_errors_preserve_startup_failure_and_force_router_kill(harness):
    error = RuntimeError("attach failed")
    harness.attach_error = error
    harness.shutdown_errors.add(0)
    harness.wait_error = RuntimeError("shutdown wait failed")
    harness.stubborn_router = True

    with pytest.raises(RuntimeError) as caught:
        deployment.start_rollout_servers(harness.args, pg=object())

    assert caught.value is error
    harness.assert_cleaned()
    assert ("router_terminate", 0) in harness.events
    assert ("router_kill", 0) in harness.events
    joins = [event for event in harness.events if event[0] == "router_join"]
    assert len(joins) == 2
    assert all(0 < event[2] <= harness.args.rollout_cleanup_timeout for event in joins)


def test_pipeline_replicas_use_physical_gpus_and_actual_node_ports(monkeypatch):
    from slime.backends.sglang_utils import engine_group

    actor_locations = [("10.0.0.2", 18004), ("10.0.0.2", 18010), ("2001:db8::5", 18100)]
    actors, actor_requests, allocations, initializations = [], [], [], []

    def create_actor(args, **kwargs):
        host, port = actor_locations[len(actors)]
        handle = object()

        def allocate(*, start_port, consecutive):
            allocations.append((host, start_port, consecutive))
            return host, port

        def initialize(**address):
            initializations.append(address)
            return handle

        actor = SimpleNamespace(
            _get_current_node_ip_and_free_port=SimpleNamespace(remote=allocate),
            init=SimpleNamespace(remote=initialize),
            init_handle=handle,
        )
        actors.append(actor)
        actor_requests.append(kwargs)
        return actor

    scheduling_requests = []

    def schedule(**kwargs):
        scheduling_requests.append(kwargs)
        return kwargs

    remote_actor = SimpleNamespace(options=lambda **options: SimpleNamespace(remote=create_actor))
    monkeypatch.setattr(engine_group, "ray", SimpleNamespace(remote=lambda cls: remote_actor, get=lambda ref: ref))
    monkeypatch.setattr(engine_group, "PlacementGroupSchedulingStrategy", schedule)
    args = Namespace(
        rollout_backend="sglang-omni",
        debug_train_only=False,
        num_gpus_per_node=8,
        rollout_num_gpus=6,
        rollout_num_gpus_per_engine=2,
    )
    placement_group = object()
    group = engine_group.ServerGroup(
        args=args,
        pg=(placement_group, [7, 8, 13, 14, 2, 3], [2, 3, 4, 5, 0, 1]),
        all_engines=[None] * 3,
        num_gpus_per_engine=2,
        num_new_engines=0,
        rank_offset=3,
    )
    handles, cursors = group.start_engines({"10.0.0.9": 18000})

    assert handles == [actor.init_handle for actor in actors]
    assert [request["gpu_ids"] for request in actor_requests] == [[2, 3], [4, 5], [0, 1]]
    assert [request["base_gpu_id"] for request in actor_requests] == [2, 4, 0]
    assert [request["rank"] for request in actor_requests] == [3, 4, 5]
    assert [request["placement_group_bundle_index"] for request in scheduling_requests] == [7, 13, 2]
    assert all(request["placement_group"] is placement_group for request in scheduling_requests)
    assert allocations == [("10.0.0.2", 18000, 3), ("10.0.0.2", 18007, 3), ("2001:db8::5", 18013, 3)]
    assert cursors == {"10.0.0.9": 18000, "10.0.0.2": 18013, "2001:db8::5": 18103}
    assert initializations == [
        {
            "host": host,
            "port": port,
            "nccl_port": port + 1,
            "dist_init_addr": f"{host}:{port + 2}",
            "router_ip": None,
            "router_port": None,
        }
        for host, port in actor_locations
    ]
    assert group.engine_urls == {
        0: "http://10.0.0.2:18004",
        1: "http://10.0.0.2:18010",
        2: "http://[2001:db8::5]:18100",
    }
    assert group.num_new_engines == 3
    assert group.start_engines(cursors) == ([], cursors)
    assert group.num_new_engines == 0
    assert len(actors) == 3
