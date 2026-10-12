"""CPU contract tests for the complete-pipeline Omni serving adapter."""

import json
import sys
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests
import torch
import yaml

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def _stage_result(data=None, *, success=True):
    return {
        "success": success,
        "results": [
            {
                "stage": "tts_engine",
                "success": success,
                "data": {"success": success, **(data or {})},
            }
        ],
    }


class _AdminServer:
    def __init__(self):
        self.requests = []
        self.responses = {}
        self.weight_version = "0"
        self.healthy = True
        self.disabled = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._handle()

            def do_POST(self):
                self._handle()

            def do_PUT(self):
                self._handle()

            def do_DELETE(self):
                self._handle()

            def _handle(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else None
                path = urlsplit(self.path).path
                owner.requests.append((self.command, path, body, dict(self.headers)))
                response = owner.responses.get(path)
                if callable(response):
                    response = response(body)
                if response is None:
                    operation = path.rsplit("/", 1)[-1]
                    if path.startswith("/workers/"):
                        if self.command == "PUT":
                            owner.disabled = body["disabled"]
                        response = {"active_requests": 0, "disabled": owner.disabled}
                    elif operation == "model_info":
                        response = _stage_result(
                            {"weight_version": owner.weight_version, "stage_tp_size": 1, "engine_paused": True}
                        )
                        response["weight_version"] = owner.weight_version
                    elif operation == "health":
                        response = {"running": owner.healthy, "total_requests": 0, "pending_completions": 0}
                    elif operation == "pause_generation":
                        response = _stage_result({"engine_paused": True, "keep_pause": True})
                    elif operation == "continue_generation":
                        response = _stage_result({"engine_paused": False, "keep_pause": False})
                    elif operation in {"update_weights_from_distributed", "update_weights_from_tensor"}:
                        owner.weight_version = body["weight_version"]
                        response = _stage_result({"flush_success": True, "keep_pause": True, "engine_paused": True})
                    else:
                        response = _stage_result()
                status, response = response if isinstance(response, tuple) else (200, response)
                encoded = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def admin_server():
    server = _AdminServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def omni_args(tmp_path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "config_cls": "MossTTSPrismPipelineConfig",
                "enable_rl": True,
                "stages": [{"name": "tts_engine", "type": "tts", "gpu": 0, "args": {"model_path": "checkpoint"}}],
            }
        )
    )
    return Namespace(
        rollout_backend="sglang-omni",
        omni_config=str(config),
        omni_python=sys.executable,
        omni_admin_key=None,
        hf_checkpoint="checkpoint",
        colocate=False,
        offload_rollout=False,
        rollout_external=False,
        num_gpus_per_node=8,
        distributed_timeout_minutes=1,
        rollout_health_check_timeout=1,
        rollout_cleanup_timeout=1,
    )


def _engine(args, server=None, *, base_gpu_id=0, num_gpus_per_engine=1, sglang_overrides=None, **kwargs):
    from slime.backends.sglang_utils.omni_engine import OmniEngine

    engine = OmniEngine(
        args,
        rank=0,
        base_gpu_id=base_gpu_id,
        num_gpus_per_engine=num_gpus_per_engine,
        sglang_overrides=(
            {"model_path": "checkpoint", "update_weights": True} if sglang_overrides is None else sglang_overrides
        ),
        **kwargs,
    )
    if server is not None:
        engine.server_host = "127.0.0.1"
        engine.server_port = server.port
        engine.router_ip = engine.router_port = None
    return engine


def _update_bucket(engine, version="1", name="layer.weight"):
    return engine.update_weights_from_distributed(
        names=[name],
        dtypes=[torch.float32],
        shapes=[[2, 3]],
        group_name="policy_update",
        weight_version=version,
        load_format=None,
    )


def _publish(engine, version="0"):
    engine.pause_generation()
    _update_bucket(engine, version=version)
    engine.continue_generation()


def test_publication_commits_only_after_all_buckets_and_resume(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    assert engine.get_weight_version() is None
    _publish(engine)
    assert engine.get_weight_version() == "0"
    admin_server.requests.clear()

    engine.pause_generation(mode="abort")
    engine.init_weights_update_group("127.0.0.1", 23456, 2, 5, "policy_update", "nccl")
    _update_bucket(engine, name="layer0.weight")
    _update_bucket(engine, name="layer1.weight")
    assert engine.get_weight_version() == "0"
    engine.continue_generation()
    assert engine.get_weight_version() == "1"

    posts = [(path, body) for method, path, body, _ in admin_server.requests if method == "POST"]
    assert [path for path, _ in posts] == [
        "/pause_generation",
        "/init_weights_update_group",
        "/update_weights_from_distributed",
        "/update_weights_from_distributed",
        "/model_info",
        "/continue_generation",
    ]
    assert all(body["stages"] == ["tts_engine"] and body["timeout_s"] == 60 for _, body in posts)
    assert posts[0][1]["mode"] == "abort"
    assert posts[1][1] == {
        "master_address": "127.0.0.1",
        "master_port": 23456,
        "rank_offset": 2,
        "world_size": 5,
        "group_name": "policy_update",
        "backend": "nccl",
        "stages": ["tts_engine"],
        "timeout_s": 60,
    }
    for _, body in posts[2:4]:
        assert body["dtypes"] == ["float32"]
        assert body["shapes"] == [[2, 3]]
        assert body["weight_version"] == "1"
        assert body["flush_cache"] is True and body["keep_pause"] is True


@pytest.mark.parametrize(
    "failure", ["stage", "data", "rank", "rank_data", "flush", "rank_flush", "resume", "skipped", "unsupported"]
)
def test_failed_bucket_stays_gated_after_an_earlier_success(omni_args, admin_server, failure):
    engine = _engine(omni_args, admin_server)
    _publish(engine)
    engine.attach_router("127.0.0.1", admin_server.port, register=True)
    engine.pause_generation()
    _update_bucket(engine, name="layer0.weight")
    response = _stage_result({"flush_success": True, "keep_pause": True, "engine_paused": True})
    stage = response["results"][0]
    if failure == "stage":
        stage["success"] = False
    elif failure == "data":
        stage["data"]["success"] = False
    elif failure == "flush":
        stage["data"]["flush_success"] = False
    elif failure == "resume":
        stage["data"]["engine_paused"] = False
    elif failure in {"skipped", "unsupported"}:
        stage["data"][failure] = True
    else:
        stage["data"]["rank_results"] = [
            {"rank": 0, "success": True, "data": {"success": True, "flush_success": True, "engine_paused": True}}
        ]
        rank = stage["data"]["rank_results"][0]
        if failure == "rank":
            rank["success"] = False
        elif failure == "rank_data":
            rank["data"]["success"] = False
        else:
            rank["data"]["flush_success"] = False
    admin_server.responses["/update_weights_from_distributed"] = response
    with pytest.raises(RuntimeError):
        _update_bucket(engine, name="layer1.weight")
    assert engine.get_weight_version() == "0"
    assert admin_server.disabled is True
    admin_server.requests.clear()
    with pytest.raises(RuntimeError):
        engine.continue_generation()
    assert not any(path == "/continue_generation" for _, path, _, _ in admin_server.requests)
    assert engine.get_weight_version() == "0"
    assert admin_server.disabled is True


def test_http_bucket_failure_cannot_commit_the_earlier_bucket(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    _publish(engine)
    engine.pause_generation()
    _update_bucket(engine, name="layer0.weight")
    admin_server.responses["/update_weights_from_distributed"] = (503, {"error": "AR worker unavailable"})
    with pytest.raises(requests.HTTPError):
        _update_bucket(engine, name="layer1.weight")
    admin_server.responses.clear()
    with pytest.raises(RuntimeError):
        engine.continue_generation()
    assert engine.get_weight_version() == "0"


def test_failed_publication_requires_a_new_pause_before_retry(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    _publish(engine)
    engine.pause_generation()
    admin_server.responses["/update_weights_from_distributed"] = _stage_result(success=False)
    with pytest.raises(RuntimeError):
        _update_bucket(engine)
    admin_server.responses.clear()
    with pytest.raises(RuntimeError):
        _update_bucket(engine)
    engine.pause_generation()
    _update_bucket(engine)
    engine.continue_generation()
    assert engine.get_weight_version() == "1"


def test_replacement_registers_disabled_until_current_weights_commit(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    engine.attach_router("127.0.0.1", admin_server.port, register=True)
    assert admin_server.disabled is True
    registration = next(body for _, path, body, _ in admin_server.requests if path == "/workers")
    assert registration == {
        "url": f"http://127.0.0.1:{admin_server.port}",
        "capabilities": ["chat", "audio_input", "audio_output"],
    }
    engine.pause_generation()
    _update_bucket(engine)
    assert admin_server.disabled is True
    engine.continue_generation()
    assert engine.get_weight_version() == "1"
    assert admin_server.disabled is False


def test_pause_drains_router_and_entire_pipeline_before_pausing_ar(omni_args, admin_server, monkeypatch):
    from slime.backends.sglang_utils import omni_engine

    engine = _engine(omni_args, admin_server)
    engine.attach_router("127.0.0.1", admin_server.port, register=True)
    router_calls = health_calls = 0
    worker_path = next(path for method, path, _, _ in admin_server.requests if method == "PUT")

    def worker_response(body):
        nonlocal router_calls
        if body is not None:
            admin_server.disabled = body["disabled"]
            return {"disabled": admin_server.disabled}
        router_calls += 1
        return {"active_requests": int(router_calls == 1)}

    def health_response(body):
        nonlocal health_calls
        health_calls += 1
        return {"running": True, "total_requests": 0, "pending_completions": int(health_calls < 3)}

    admin_server.responses[worker_path] = worker_response
    admin_server.responses["/health"] = health_response
    monkeypatch.setattr(omni_engine.time, "sleep", lambda _: None)
    admin_server.requests.clear()
    engine.pause_generation()
    assert router_calls == health_calls == 3
    assert admin_server.disabled is True
    assert admin_server.requests[-1][1] == "/pause_generation"


def test_pause_timeout_leaves_router_disabled_and_does_not_pause_with_live_outputs(omni_args, admin_server):
    omni_args.distributed_timeout_minutes = 0.00001
    engine = _engine(omni_args, admin_server)
    engine.attach_router("127.0.0.1", admin_server.port, register=True)
    admin_server.responses["/health"] = {"running": True, "total_requests": 0, "pending_completions": 1}
    with pytest.raises(TimeoutError, match="drain"):
        engine.pause_generation()
    assert admin_server.disabled is True
    assert not any(path == "/pause_generation" for _, path, _, _ in admin_server.requests)


def test_mixed_version_bucket_requires_fresh_full_publication(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    _publish(engine)
    engine.pause_generation()
    _update_bucket(engine, version="1")
    with pytest.raises(RuntimeError, match="version"):
        _update_bucket(engine, version="2")
    admin_server.requests.clear()
    with pytest.raises(RuntimeError):
        engine.continue_generation()
    assert engine.get_weight_version() == "0"
    assert not any(path == "/continue_generation" for _, path, _, _ in admin_server.requests)


@pytest.mark.parametrize("mismatch", ["top_level", "stage"])
def test_resume_rejects_installed_version_mismatch(omni_args, admin_server, mismatch):
    engine = _engine(omni_args, admin_server)
    _publish(engine)
    engine.pause_generation()
    _update_bucket(engine)
    response = _stage_result({"weight_version": "1", "stage_tp_size": 1, "engine_paused": True})
    response["weight_version"] = "1"
    if mismatch == "top_level":
        response["weight_version"] = "0"
    else:
        response["results"][0]["data"]["weight_version"] = "0"
    admin_server.responses["/model_info"] = response
    admin_server.requests.clear()
    with pytest.raises(RuntimeError, match="version"):
        engine.continue_generation()
    assert engine.get_weight_version() == "0"
    assert not any(path == "/continue_generation" for _, path, _, _ in admin_server.requests)


def test_resume_stage_failure_does_not_commit(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    _publish(engine)
    engine.pause_generation()
    _update_bucket(engine)
    admin_server.responses["/continue_generation"] = _stage_result(success=False)
    with pytest.raises(RuntimeError):
        engine.continue_generation()
    assert engine.get_weight_version() == "0"


@pytest.mark.parametrize("failure", ["not_running", "stage", "http"])
def test_health_raises_when_pipeline_is_unhealthy(omni_args, admin_server, failure):
    engine = _engine(omni_args, admin_server)
    if failure == "not_running":
        admin_server.healthy = False
    elif failure == "stage":
        admin_server.responses["/model_info"] = _stage_result(success=False)
    else:
        admin_server.responses["/health"] = (503, {"running": False})
    with pytest.raises((RuntimeError, requests.HTTPError)):
        engine.health_generate(timeout=0.5)


def test_health_forwards_bounded_timeout_to_stage_probe(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    assert engine.health_generate(timeout=0.25) is True
    assert [(method, path) for method, path, _, _ in admin_server.requests] == [
        ("GET", "/health"),
        ("POST", "/model_info"),
    ]
    stage_request = admin_server.requests[1][2]
    assert stage_request["stages"] == ["tts_engine"]
    assert 0 < stage_request["timeout_s"] <= 0.25


def test_health_shares_one_deadline_across_pipeline_and_ar_probes(omni_args, admin_server, monkeypatch):
    from slime.backends.sglang_utils import omni_engine

    clock = {"now": 10.0, "health_cost": 0.125}
    monkeypatch.setattr(omni_engine, "time", SimpleNamespace(monotonic=lambda: clock["now"]))

    def health_response(body):
        clock["now"] += clock["health_cost"]
        return {"running": True}

    def model_response(body):
        clock["now"] += 0.05
        return _stage_result({"weight_version": "0", "stage_tp_size": 1})

    admin_server.responses["/health"] = health_response
    admin_server.responses["/model_info"] = model_response
    engine = _engine(omni_args, admin_server)
    assert engine.health_generate(timeout=0.2) is True
    assert admin_server.requests[1][2]["timeout_s"] == pytest.approx(0.075)

    clock["health_cost"] = 0.25
    admin_server.requests.clear()
    with pytest.raises(TimeoutError, match="health probe timed out"):
        engine.health_generate(timeout=0.2)
    assert [(method, path) for method, path, _, _ in admin_server.requests] == [("GET", "/health")]


def test_health_rejects_unexpected_ar_tensor_parallel_size(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    response = _stage_result({"weight_version": "0", "stage_tp_size": 2})
    admin_server.responses["/model_info"] = response
    with pytest.raises(RuntimeError, match="TP=1"):
        engine.health_generate()


def test_group_reset_cleans_attempted_init_after_failed_reply(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    admin_server.responses["/init_weights_update_group"] = _stage_result(success=False)
    with pytest.raises(RuntimeError):
        engine.init_weights_update_group("127.0.0.1", 23456, 1, 2, "policy_update", "nccl")
    engine.reset_weights_update_groups()
    engine.reset_weights_update_groups()
    deletes = [body for _, path, body, _ in admin_server.requests if path == "/destroy_weights_update_group"]
    assert deletes == [{"group_name": "policy_update", "stages": ["tts_engine"], "timeout_s": 1}]


@pytest.mark.parametrize("option", ["rollout_external"])
def test_constructor_rejects_unsupported_deployment_modes(omni_args, option):
    setattr(omni_args, option, True)
    with pytest.raises(ValueError):
        _engine(omni_args)


def test_constructor_rejects_multinode_replica(omni_args):
    with pytest.raises(ValueError, match="one node"):
        _engine(omni_args, num_gpus_per_engine=9)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("flush_cache", ()),
    ],
)
def test_unsupported_control_operations_are_explicit(omni_args, method, args):
    engine = _engine(omni_args)
    with pytest.raises(NotImplementedError):
        getattr(engine, method)(*args)


def test_launch_uses_reserved_gpu_layout_strict_port_and_clean_child_allocator(omni_args, admin_server, monkeypatch):
    from slime.backends.sglang_utils import omni_engine

    omni_args.omni_admin_key = "test-admin-key"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "TORCHELASTIC_USE_AGENT_STORE"):
        monkeypatch.setenv(name, "inherited-learner-value")
    captured = {}

    class Process:
        def __init__(self, *, target, args):
            captured["target"] = target
            captured["args"] = args

        def start(self):
            captured["started"] = True

        def is_alive(self):
            return True

    monkeypatch.setattr(omni_engine.multiprocessing, "get_context", lambda method: SimpleNamespace(Process=Process))
    engine = _engine(omni_args, base_gpu_id=3, num_gpus_per_engine=2)
    engine.init("127.0.0.1:23456", admin_server.port, 23457, host="127.0.0.1")
    command, environment, parent_pid = captured["args"]
    assert captured["started"] is True and captured["target"] is omni_engine.launch_omni_process
    assert command == [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        omni_args.omni_config,
        "--model-path",
        "checkpoint",
        "--host",
        "127.0.0.1",
        "--port",
        str(admin_server.port),
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "3,4"
    assert environment["SGLANG_OMNI_STRICT_PORT"] == "1"
    assert environment["SGLANG_OMNI_ADMIN_KEY"] == "test-admin-key"
    assert environment["SGLANG_OMNI_STARTUP_TIMEOUT"] == "60"
    assert environment["MASTER_PORT"] == "23457"
    assert not {"MASTER_ADDR", "RANK", "WORLD_SIZE", "LOCAL_RANK", "TORCHELASTIC_USE_AGENT_STORE"} & environment.keys()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in environment and "PYTORCH_ALLOC_CONF" not in environment
    assert isinstance(parent_pid, int) and parent_pid > 0
    assert all(headers["Authorization"] == "Bearer test-admin-key" for _, _, _, headers in admin_server.requests)
    assert engine.get_weight_version() is None


@pytest.mark.parametrize("invalid", ["disabled_rl", "tp", "gpu_outside_replica", "wrong_pipeline"])
def test_invalid_pipeline_config_is_rejected_before_launch(omni_args, monkeypatch, invalid):
    from slime.backends.sglang_utils import omni_engine

    with open(omni_args.omni_config) as stream:
        config = yaml.safe_load(stream)
    if invalid == "disabled_rl":
        config["enable_rl"] = False
    elif invalid == "tp":
        config["stages"][0]["tp_size"] = 2
    elif invalid == "gpu_outside_replica":
        config["stages"][0]["gpu"] = 1
    else:
        config["config_cls"] = "UnrelatedPipelineConfig"
    with open(omni_args.omni_config, "w") as stream:
        yaml.safe_dump(config, stream)
    monkeypatch.setattr(
        omni_engine.multiprocessing,
        "get_context",
        lambda _: pytest.fail("invalid pipeline attempted to start a process"),
    )
    with pytest.raises(ValueError):
        _engine(omni_args).init("127.0.0.1:23456", 23458, 23457, host="127.0.0.1")


@pytest.mark.parametrize(
    ("model_path", "configured_role", "resolved_role"),
    [("checkpoint", None, True), ("teacher-checkpoint", None, False), ("checkpoint", False, False)],
)
def test_named_model_resolves_and_injects_omni_weight_update_role(
    omni_args, model_path, configured_role, resolved_role
):
    from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig

    group = ServerGroupConfig(worker_type="regular", num_gpus=4)
    model = ModelConfig(
        name="policy" if resolved_role else "domain-teacher",
        model_path=model_path,
        num_gpus_per_engine=2,
        update_weights=configured_role,
        server_groups=[group],
    )
    model.resolve(omni_args)
    assert model.update_weights is resolved_role
    assert group.overrides["update_weights"] is resolved_role
    assert group.overrides["model_path"] == model_path
    assert group.num_gpus == 4 and group.num_gpus_per_engine == 2


@pytest.mark.parametrize("update_weights", [True, False], ids=["student", "frozen-score-teacher"])
def test_initialized_model_role_controls_admission_and_initial_version(
    omni_args, admin_server, monkeypatch, update_weights
):
    from slime.backends.sglang_utils import omni_engine

    if not update_weights:
        with open(omni_args.omni_config) as stream:
            config = yaml.safe_load(stream)
        config["config_cls"] = "MossTTSPrismScorePipelineConfig"
        config["enable_rl"] = False
        with open(omni_args.omni_config, "w") as stream:
            yaml.safe_dump(config, stream)
    admin_server.weight_version = "checkpoint-version-7"

    class Process:
        def __init__(self, *, target, args):
            pass

        def start(self):
            assert admin_server.disabled is update_weights
            assert [method for method, path, _, _ in admin_server.requests if path.startswith("/workers")] == [
                "POST",
                "PUT",
            ]

        def is_alive(self):
            return True

    monkeypatch.setattr(omni_engine.multiprocessing, "get_context", lambda method: SimpleNamespace(Process=Process))
    engine = _engine(
        omni_args,
        sglang_overrides={"model_path": "checkpoint", "update_weights": update_weights},
    )
    engine.init(
        "127.0.0.1:23456",
        admin_server.port,
        23457,
        host="127.0.0.1",
        router_ip="127.0.0.1",
        router_port=admin_server.port,
    )
    pause_requests = [request for request in admin_server.requests if request[1] == "/pause_generation"]
    assert bool(pause_requests) is update_weights
    assert admin_server.disabled is update_weights
    assert engine.get_weight_version() == (None if update_weights else "checkpoint-version-7")


@pytest.mark.parametrize(("backend", "receiver_gpus"), [("sglang-omni", 1), ("sglang", 2)])
def test_pipeline_placement_gpus_are_distinct_from_weight_receiver_gpus(omni_args, backend, receiver_gpus):
    from slime.backends.sglang_utils.engine_group import RolloutServer, ServerGroup

    omni_args.rollout_backend = backend
    replicas = [object(), object()]
    group = ServerGroup(
        args=omni_args,
        pg=None,
        all_engines=replicas,
        num_gpus_per_engine=2,
        num_new_engines=0,
        gpu_offset=4,
    )
    server = RolloutServer(server_groups=[group], model_name="policy")
    assert server.engine_gpu_counts == [receiver_gpus, receiver_gpus]
    assert server.engine_gpu_offsets == [4, 6]
    assert group.num_gpus_per_engine == 2
    assert server.engines == replicas
    assert (
        server.engine_parallel_configs
        == [{"tp_size": receiver_gpus, "pp_size": 1, "ep_size": 1, "moe_dp_size": 1}] * 2
    )


@pytest.mark.parametrize("delete_status", [200, 404])
def test_omni_unregister_uses_worker_id_and_admin_authorization(admin_server, delete_status):
    from slime.backends.sglang_utils.server_control import unregister_worker

    admin_server.responses["/workers"] = {
        "workers": [
            {"worker_id": "target-worker", "id": "legacy-target", "url": "http://target-pipeline"},
            {"worker_id": "healthy-worker", "id": "legacy-healthy", "url": "http://healthy-pipeline"},
        ]
    }
    admin_server.responses["/workers/target-worker"] = (delete_status, {})
    unregister_worker(
        f"http://127.0.0.1:{admin_server.port}",
        "http://target-pipeline",
        timeout=0.5,
        backend="sglang-omni",
        admin_key="router-admin-key",
    )
    assert [(method, path) for method, path, _, _ in admin_server.requests] == [
        ("GET", "/workers"),
        ("DELETE", "/workers/target-worker"),
    ]
    assert all(headers["Authorization"] == "Bearer router-admin-key" for _, _, _, headers in admin_server.requests)


def test_tensor_publication_keeps_admission_closed_until_kv_resumes(omni_args, admin_server):
    omni_args.colocate = omni_args.offload_rollout = True
    engine = _engine(omni_args, admin_server)
    engine.attach_router("127.0.0.1", admin_server.port)
    engine.offload_tags = {"kv_cache", "cuda_graph"}
    engine.pause_generation()
    engine.update_weights_from_tensor(["ipc-bucket"], load_format="flattened_bucket", weight_version="4")
    engine.continue_generation()
    assert engine.get_weight_version() == "4"
    assert admin_server.disabled is True
    admin_server.responses["/resume_memory_occupation"] = {
        "success": True,
        "results": [{"stage": stage, "success": True, "data": {"offload_tags": []}} for stage in engine.memory_stages],
    }
    engine.resume_memory_occupation(["kv_cache", "cuda_graph"])
    assert admin_server.disabled is False
    updates = [body for _, path, body, _ in admin_server.requests if path == "/update_weights_from_tensor"]
    assert updates[0]["keep_pause"] and updates[0]["flush_cache"]
    assert updates[0]["stages"] == ["tts_engine"]
    assert not any(path == "/flush_cache" for _, path, _, _ in admin_server.requests)


def test_tensor_publication_failure_requires_full_retry(omni_args, admin_server):
    engine = _engine(omni_args, admin_server)
    engine.pause_generation()
    engine.update_weights_from_tensor(["first"], weight_version="1")
    admin_server.responses["/update_weights_from_tensor"] = _stage_result(success=False)
    with pytest.raises(RuntimeError):
        engine.update_weights_from_tensor(["second"], weight_version="1")
    with pytest.raises(RuntimeError):
        engine.continue_generation()
    admin_server.responses.clear()
    with pytest.raises(RuntimeError):
        engine.update_weights_from_tensor(["retry"], weight_version="1")
    engine.pause_generation()
    engine.update_weights_from_tensor(["full-retry"], weight_version="1")
    engine.continue_generation()
    assert engine.get_weight_version() == "1"
