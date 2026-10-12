"""Ray control adapter for complete, separately placed Omni pipeline replicas."""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import signal
import sys
import time
from argparse import Namespace
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

from slime.backends.sglang_utils.server_control import unregister_worker
from slime.observability.logging_utils import configure_logger
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)
OMNI_AR_STAGE = "tts_engine"
OMNI_WORKER_CAPABILITIES = ["chat", "audio_input", "audio_output"]


def launch_omni_process(command: list[str], environment: dict[str, str], parent_pid: int) -> None:
    """Replace the supervised child with the pipeline launcher."""
    os.setsid()
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot install parent-death signal")
    if os.getppid() != parent_pid:
        return
    os.execvpe(command[0], command, environment)


def validate_admin_result(payload: object, action: str, *, require_flush: bool = False) -> dict:
    """Reject HTTP-success responses containing failed stage or rank operations."""
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError(f"Omni {action} failed: {payload}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"Omni {action} returned no stage results")
    if len(results) != 1 or not isinstance(results[0], dict) or results[0].get("stage") != OMNI_AR_STAGE:
        raise RuntimeError(f"Omni {action} must target exactly one Prism AR stage")
    for result in results:
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError(f"Omni {action} failed stage result: {result}")
        stage_payload = result.get("data")
        if not isinstance(stage_payload, dict) or stage_payload.get("skipped") or stage_payload.get("unsupported"):
            raise RuntimeError(f"Omni {action} did not execute on the requested stage: {result}")
        if stage_payload.get("success") is False:
            raise RuntimeError(f"Omni {action} failed inside its stage: {result}")
        rank_results = stage_payload.get("rank_results", [])
        if not isinstance(rank_results, list):
            raise RuntimeError(f"Omni {action} returned invalid rank results")
        if len(rank_results) > 1:
            raise RuntimeError("Prism Omni RL currently supports AR TP=1 only")
        checked_payloads = [stage_payload]
        for rank_result in rank_results:
            if not isinstance(rank_result, dict) or rank_result.get("success") is not True:
                raise RuntimeError(f"Omni {action} failed rank result: {rank_result}")
            rank_payload = rank_result.get("data")
            if not isinstance(rank_payload, dict) or rank_payload.get("success") is False:
                raise RuntimeError(f"Omni {action} returned invalid rank data: {rank_result}")
            checked_payloads.append(rank_payload)
        if require_flush:
            for checked in checked_payloads:
                if checked.get("flush_success") is not True or checked.get("engine_paused") is not True:
                    raise RuntimeError(f"Omni {action} did not successfully flush its cache: {checked}")
            if stage_payload.get("keep_pause") is not True or stage_payload.get("engine_paused") is not True:
                raise RuntimeError(f"Omni {action} resumed before the publication completed")
    return payload


class OmniEngine(RayActor):
    def __init__(
        self,
        args: Namespace,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
        gpu_ids: list[int] | None = None,
    ) -> None:
        if args.colocate or args.offload_rollout:
            raise ValueError("Omni rollout requires separate GPUs and does not support colocate/offload")
        if args.rollout_external:
            raise ValueError("The Omni adapter owns pipeline processes; external rollout is unsupported")
        if worker_type != "regular":
            raise ValueError("Omni replicas must use regular server groups")
        configure_logger()
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.gpu_ids = gpu_ids
        self.num_gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
        if self.num_gpus_per_engine > args.num_gpus_per_node:
            raise ValueError("An Omni pipeline replica must fit on one node")
        self.overrides = dict(sglang_overrides or {})
        self.update_weights = self.overrides.pop("update_weights", True)
        self.config_path = Path(self.overrides.pop("omni_config", args.omni_config))
        self.model_path = self.overrides.pop("model_path", args.hf_checkpoint)
        if self.overrides:
            raise ValueError(f"Unsupported Omni server overrides: {sorted(self.overrides)}")
        self.operation_timeout = getattr(args, "distributed_timeout_minutes", 10) * 60
        self.process: multiprocessing.Process | None = None
        self.server_host: str | None = None
        self.server_port: int | None = None
        self.router_ip: str | None = None
        self.router_port: int | None = None
        self.paused = False
        self.pending_version: str | None = None
        self.committed_version: str | None = None
        self.publication_failed = False
        self.weight_update_groups: set[str] = set()
        self.session = requests.Session()
        self.session.trust_env = False
        if args.omni_admin_key:
            self.session.headers["Authorization"] = f"Bearer {args.omni_admin_key}"

    def init(
        self,
        dist_init_addr: str,
        port: int,
        nccl_port: int,
        host: str | None = None,
        disaggregation_bootstrap_port: int | None = None,
        router_ip: str | None = None,
        router_port: int | None = None,
    ) -> None:
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        with self.config_path.open() as stream:
            pipeline = yaml.safe_load(stream)
        if not isinstance(pipeline, dict):
            raise ValueError("Omni pipeline config must be a mapping")
        if pipeline.get("config_cls") not in {"MossTTSPrismPipelineConfig", "MossTTSPrismScorePipelineConfig"}:
            raise ValueError("The Omni RL adapter currently supports Prism pipeline configurations only")
        if self.update_weights and pipeline.get("enable_rl") is not True:
            raise ValueError("Student Omni pipeline config must set enable_rl: true")
        for stage in pipeline.get("stages", []):
            if stage.get("tp_size", 1) != 1:
                raise ValueError("Prism Omni RL currently supports AR TP=1 only")
            stage_devices = stage.get("gpu", 0)
            stage_devices = stage_devices if isinstance(stage_devices, list) else [stage_devices]
            if any(device is not None and not 0 <= device < self.num_gpus_per_engine for device in stage_devices):
                raise ValueError("Omni stage GPU ids must be local to the reserved replica GPUs")
        self.server_host = (host or get_host_info()[1]).strip("[]")
        self.server_port = port
        environment = os.environ.copy()
        environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        environment.pop("PYTORCH_ALLOC_CONF", None)
        # The pipeline owns its serving rendezvous. Platform/torchrun/Ray
        # learner ranks must not leak into a separately placed AR process.
        for name in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_USE_AGENT_STORE",
        ):
            environment.pop(name, None)
        # Omni's model worker uses MASTER_PORT for its local NCCL bootstrap.
        # Reuse the distinct port already reserved for this pipeline replica.
        environment["MASTER_PORT"] = str(nccl_port)
        if self.base_gpu_id is None:
            raise ValueError("Omni replicas require an explicit placement GPU id")
        devices = self.gpu_ids or list(range(self.base_gpu_id, self.base_gpu_id + self.num_gpus_per_engine))
        if len(devices) != self.num_gpus_per_engine or len(set(devices)) != len(devices):
            raise ValueError("Omni GPU placement must name each reserved GPU exactly once")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in devices)
        environment["SGLANG_OMNI_STRICT_PORT"] = "1"
        environment["SGLANG_OMNI_STARTUP_TIMEOUT"] = str(self.operation_timeout)
        environment.pop("SGLANG_OMNI_ADMIN_KEY", None)
        if self.args.omni_admin_key:
            environment["SGLANG_OMNI_ADMIN_KEY"] = self.args.omni_admin_key
        command = [
            self.args.omni_python,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(self.config_path),
            "--model-path",
            self.model_path,
            "--host",
            self.server_host,
            "--port",
            str(port),
        ]
        self.process = multiprocessing.get_context("spawn").Process(
            target=launch_omni_process, args=(command, environment, os.getpid())
        )
        deadline = time.monotonic() + self.operation_timeout
        try:
            if router_ip is not None and router_port is not None:
                self.attach_router(router_ip, router_port, register=True)
            self.process.start()
            while time.monotonic() < deadline:
                if not self.process.is_alive():
                    raise RuntimeError(f"Omni pipeline exited with code {self.process.exitcode}")
                try:
                    self.health_generate(timeout=min(5, max(0.1, deadline - time.monotonic())))
                    break
                except requests.RequestException as error:
                    if error.response is not None and error.response.status_code < 500:
                        raise
                    time.sleep(0.5)
            else:
                raise TimeoutError("Omni pipeline startup timed out")
            if self.update_weights:
                self.pause_generation()
            else:
                initial = self.request_admin("model_info")
                self.committed_version = initial.get("weight_version")
        except BaseException:
            self.shutdown()
            raise

    def get_url(self) -> str:
        if self.server_host is None or self.server_port is None:
            raise RuntimeError("Omni pipeline has not been initialized")
        host = f"[{self.server_host}]" if ":" in self.server_host else self.server_host
        return f"http://{host}:{self.server_port}"

    def request_admin(self, action: str, payload: dict | None = None, *, timeout: float | None = None) -> dict:
        timeout = self.operation_timeout if timeout is None else timeout
        response = self.session.post(
            f"{self.get_url()}/{action}",
            json={**(payload or {}), "stages": [OMNI_AR_STAGE], "timeout_s": timeout},
            timeout=timeout,
        )
        response.raise_for_status()
        return validate_admin_result(response.json(), action)

    def attach_router(self, router_ip: str, router_port: int, *, register: bool = False) -> None:
        self.router_ip, self.router_port = router_ip, router_port
        if register:
            response = self.session.post(
                f"{self.router_url()}/workers",
                json={"url": self.get_url(), "capabilities": OMNI_WORKER_CAPABILITIES},
                timeout=self.args.rollout_health_check_timeout,
            )
            response.raise_for_status()
        self.set_admission_disabled(self.update_weights and self.committed_version is None)

    def router_url(self) -> str:
        if self.router_ip is None or self.router_port is None:
            raise RuntimeError("Omni router has not been attached")
        return f"http://{self.router_ip}:{self.router_port}"

    def set_admission_disabled(self, disabled: bool) -> None:
        if self.router_ip is None:
            return
        response = self.session.put(
            f"{self.router_url()}/workers/{quote(self.get_url(), safe='')}",
            json={"disabled": disabled},
            timeout=self.args.rollout_health_check_timeout,
        )
        response.raise_for_status()

    def health_generate(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        response = self.session.get(f"{self.get_url()}/health", timeout=timeout)
        response.raise_for_status()
        if response.json().get("running") is not True:
            raise RuntimeError("Omni pipeline is not running")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Omni pipeline health probe timed out")
        result = self.request_admin("model_info", timeout=remaining)
        for stage in result["results"]:
            if stage["data"].get("stage_tp_size", 1) != 1:
                raise RuntimeError("The Omni RL adapter requires AR TP=1")
        return True

    def pause_generation(self, mode: str = "abort") -> dict:
        self.set_admission_disabled(True)
        deadline = time.monotonic() + self.operation_timeout
        while True:
            router_idle = True
            if self.router_ip is not None:
                response = self.session.get(
                    f"{self.router_url()}/workers/{quote(self.get_url(), safe='')}",
                    timeout=self.args.rollout_health_check_timeout,
                )
                response.raise_for_status()
                router_idle = response.json().get("active_requests") == 0
            response = self.session.get(f"{self.get_url()}/health", timeout=self.args.rollout_health_check_timeout)
            response.raise_for_status()
            health = response.json()
            if health.get("running") is not True:
                raise RuntimeError("Omni pipeline stopped while draining")
            if router_idle and health.get("total_requests") == 0 and health.get("pending_completions") == 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Omni pipeline did not drain after admission was disabled")
            time.sleep(0.1)
        result = self.request_admin("pause_generation", {"mode": mode})
        self.paused = True
        self.pending_version = None
        self.publication_failed = False
        return result

    def continue_generation(self) -> dict:
        if not self.paused or self.pending_version is None or self.publication_failed:
            raise RuntimeError("Omni cannot resume without a completed pending publication")
        installed = self.request_admin("model_info")
        if str(installed.get("weight_version")) != self.pending_version:
            raise RuntimeError("Omni AR worker did not install the pending policy version")
        for stage in installed["results"]:
            if str(stage["data"].get("weight_version")) != self.pending_version:
                raise RuntimeError("Omni returned inconsistent installed stage versions")
        result = self.request_admin("continue_generation")
        self.committed_version = self.pending_version
        self.pending_version = None
        self.paused = False
        self.set_admission_disabled(False)
        logger.info(
            "Omni replica rank=%s committed_version=%s url=%s", self.rank, self.committed_version, self.get_url()
        )
        return result

    def get_weight_version(self) -> str | None:
        return self.committed_version

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> dict:
        self.weight_update_groups.add(group_name)
        return self.request_admin(
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

    def destroy_weights_update_group(self, group_name: str) -> dict | None:
        if group_name not in self.weight_update_groups:
            return None
        result = self.request_admin(
            "destroy_weights_update_group",
            {"group_name": group_name},
            timeout=self.args.rollout_health_check_timeout,
        )
        self.weight_update_groups.discard(group_name)
        return result

    def reset_weights_update_groups(self) -> None:
        self.set_admission_disabled(True)
        for group_name in tuple(self.weight_update_groups):
            self.destroy_weights_update_group(group_name)

    def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list,
        shapes: list,
        group_name: str,
        flush_cache: bool = True,
        weight_version: str | None = None,
        load_format: str | None = None,
    ) -> dict:
        if not self.paused or weight_version is None or not names:
            raise RuntimeError("Omni weight buckets require a paused replica and an explicit version")
        version = str(weight_version)
        if self.pending_version is not None and self.pending_version != version:
            self.publication_failed = True
            raise RuntimeError("Omni cannot mix versions within one publication")
        if self.publication_failed:
            raise RuntimeError("A failed Omni publication requires a fresh pause and full synchronization")
        self.publication_failed = True
        result = self.request_admin(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).removeprefix("torch.") for dtype in dtypes],
                "shapes": [list(shape) for shape in shapes],
                "group_name": group_name,
                "load_format": load_format,
                "weight_version": version,
                "flush_cache": True,
                "keep_pause": True,
            },
        )
        validate_admin_result(result, "update_weights_from_distributed", require_flush=True)
        self.pending_version = version
        self.publication_failed = False
        return result

    def flush_cache(self) -> None:
        raise NotImplementedError("Omni flushes caches inside paused weight updates; no standalone endpoint exists")

    def update_weights_from_tensor(self, *args, **kwargs) -> None:
        raise NotImplementedError("Omni CUDA IPC weight updates are not supported")

    def release_memory_occupation(self) -> None:
        raise NotImplementedError("Omni rollout offload is not supported")

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError("Omni rollout offload is not supported")

    def shutdown(self) -> None:
        try:
            if self.router_ip is not None:
                unregister_worker(
                    self.router_url(),
                    self.get_url(),
                    timeout=self.args.rollout_health_check_timeout,
                    backend="sglang-omni",
                    admin_key=self.args.omni_admin_key,
                )
        finally:
            if self.process is not None:
                if self.process.is_alive():
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.join(timeout=self.args.rollout_cleanup_timeout)
                    if self.process.is_alive():
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.join(timeout=self.args.rollout_cleanup_timeout)
            self.session.close()
