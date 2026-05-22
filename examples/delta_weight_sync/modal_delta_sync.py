"""Modal Qwen3-4B rollout + slime trainer with disk delta weight sync.

Deploy:
    MIN_CONTAINERS=1 uv run modal deploy examples/delta_weight_sync/modal_delta_sync.py

Launch a deployed end-to-end run:
    uv run --with requests modal run examples/delta_weight_sync/modal_delta_sync.py::launch_run --num-rollout 2
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
import asyncio
import glob
from collections import deque
from pathlib import Path
from typing import Any

import modal
import modal.experimental


def _local_repo_root() -> Path:
    file_path = Path(__file__).resolve()
    candidates = [file_path.parent, *file_path.parents]
    for candidate in candidates:
        if (candidate / "docker/patch/latest/sglang.patch").exists():
            return candidate
    raise RuntimeError(f"Could not locate slime repo root from {file_path}")


IS_LOCAL = modal.is_local()
REPO_ROOT = _local_repo_root() if IS_LOCAL else Path("/root/slime")

MINUTES = 60
HOURS = 60 * MINUTES

APP_NAME = os.environ.get("SLIME_MODAL_APP_NAME", "slime-qwen3-4b-delta-sync")
MODEL_NAME = os.environ.get("SLIME_MODAL_MODEL_NAME", "Qwen/Qwen3-4B")
MODEL_REVISION = os.environ.get("SLIME_MODAL_MODEL_REVISION", "main")

SLIME_COMMIT = "0a664bc5eb776a785b4e035ddd57866f921d0cdc"
SGLANG_IMAGE_TAG = "v0.5.10.post1"
MEGATRON_COMMIT = "1dcf0dafa884ad52ffb243625717a3471643e087"

ROLLOUT_BASE_IMAGE = os.environ.get("SLIME_MODAL_ROLLOUT_BASE_IMAGE", f"slimerl/sglang:{SGLANG_IMAGE_TAG}")
TRAINER_BASE_IMAGE = os.environ.get("SLIME_MODAL_TRAINER_BASE_IMAGE", "slimerl/slime-test:nightly-dev-20260429b")
AUTOINFERENCE_UTILS_VERSION = os.environ.get("AUTOINFERENCE_UTILS_VERSION", "0.2.0")

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOLUME_NAME = os.environ.get("HF_CACHE_VOLUME_NAME", "huggingface-cache")
DELTA_MOUNT_PATH = os.environ.get("SLIME_DELTA_MOUNT_PATH", "/delta")
DELTA_VOLUME_NAME = os.environ.get("SLIME_DELTA_VOLUME_NAME", "slime-qwen3-4b-deltas")

SGLANG_INTERNAL_PORT = int(os.environ.get("SLIME_SGLANG_INTERNAL_PORT", "8001"))
SIDECAR_PORT = int(os.environ.get("SLIME_MODAL_SIDECAR_PORT", "8000"))
PROXY_REGIONS = os.environ.get("PROXY_REGIONS", "us-west").split(",")
REGION = os.environ.get("SLIME_MODAL_REGION", "us")
STARTUP_TIMEOUT = int(os.environ.get("SLIME_MODAL_STARTUP_TIMEOUT", str(45 * MINUTES)))
SCALEDOWN_WINDOW = int(os.environ.get("SLIME_MODAL_SCALEDOWN_WINDOW", str(15 * MINUTES)))
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", os.environ.get("SLIME_MODAL_ROLLOUT_MIN_CONTAINERS", "1")))
MAX_CONTAINERS = 1
TARGET_INPUTS = int(os.environ.get("SLIME_MODAL_TARGET_INPUTS", "16"))

GPU_TYPE = os.environ.get("SLIME_MODAL_GPU_TYPE", "H100")
GPU = f"{GPU_TYPE}:1"
MEMORY_MB = int(os.environ.get("SLIME_MODAL_MEMORY_MB", "131072"))

SGLANG_CONTEXT_LENGTH = int(os.environ.get("SLIME_SGLANG_CONTEXT_LENGTH", "2048"))
SGLANG_MEM_FRACTION_STATIC = os.environ.get("SLIME_SGLANG_MEM_FRACTION_STATIC", "0.50")

HF_IMAGE_ENV = {
    "HF_HOME": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}
RUNTIME_ENV = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": "/root:/root/Megatron-LM:/root/slime",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SLIME_DELTA_VOLUME_NAME": DELTA_VOLUME_NAME,
}

APPLY_SGLANG_PATCH = (
    "set -eux; "
    'SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"; '
    '[ -d "$SGLANG_DIR" ] || SGLANG_DIR="/root/src/sglang"; '
    'test -d "$SGLANG_DIR"; '
    'cd "$SGLANG_DIR"; '
    "git update-index --refresh || true; "
    "if git apply --check /tmp/sglang.patch; then "
    "git apply --3way /tmp/sglang.patch; "
    "elif git apply --reverse --check /tmp/sglang.patch; then "
    'echo "sglang.patch is already applied"; '
    "else "
    "git apply --3way /tmp/sglang.patch; "
    "fi; "
    "if grep -R -n '^<<<<<<< ' python/sglang; then "
    'echo "SGLang patch conflict markers remain after sglang.patch"; '
    "exit 1; "
    "fi"
)

APPLY_SGLANG_DELTA_PATCH = (
    "set -eux; "
    'SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"; '
    '[ -d "$SGLANG_DIR" ] || SGLANG_DIR="/root/src/sglang"; '
    'test -d "$SGLANG_DIR"; '
    'cd "$SGLANG_DIR"; '
    'if ! grep -R "class DeltaEncoding" python/sglang/srt/managers/io_struct.py >/dev/null 2>&1; then '
    "git update-index --refresh || true; "
    "if git apply --check /tmp/sglang_delta_sync.patch; then "
    "git apply --3way /tmp/sglang_delta_sync.patch; "
    "elif git apply --reverse --check /tmp/sglang_delta_sync.patch; then "
    'echo "sglang_delta_sync.patch is already applied"; '
    "else "
    "git apply --3way /tmp/sglang_delta_sync.patch; "
    "fi; "
    "fi; "
    "if grep -R -n '^<<<<<<< ' python/sglang; then "
    'echo "SGLang patch conflict markers remain"; '
    "exit 1; "
    "fi; "
    "grep -R 'load_format == \"delta\"' python/sglang/srt/model_executor/model_runner.py"
)

app = modal.App(name=APP_NAME)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
delta_volume = modal.Volume.from_name(DELTA_VOLUME_NAME, create_if_missing=True)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _normalize_base_url(base_url: str) -> str:
    if base_url.startswith(("http://", "https://")):
        return base_url.rstrip("/")
    return f"http://{base_url.rstrip('/')}"


_DELTA_VERSION_RE = re.compile(r"^weight_v\d{6}$")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def validate_delta_update_payload(payload: dict[str, Any], *, delta_mount_path: str = "/delta") -> str:
    if payload.get("load_format") != "delta":
        raise ValueError("update_weights_from_disk sidecar only accepts load_format='delta'")

    model_path = payload.get("model_path")
    if not isinstance(model_path, str):
        raise ValueError("update_weights_from_disk payload requires string model_path")

    mount = os.path.realpath(delta_mount_path)
    real_model_path = os.path.realpath(model_path)
    if os.path.commonpath([mount, real_model_path]) != mount:
        raise ValueError(f"model_path must be under {delta_mount_path}")

    if os.path.dirname(real_model_path) != mount or not _DELTA_VERSION_RE.fullmatch(os.path.basename(real_model_path)):
        raise ValueError(f"model_path must match {delta_mount_path}/weight_vNNNNNN")

    return real_model_path


def verify_delta_dir_ready(model_path: str) -> None:
    done_path = os.path.join(model_path, "DONE")
    if not os.path.isfile(done_path):
        raise FileNotFoundError(f"missing delta DONE marker: {done_path}")
    if not glob.glob(os.path.join(model_path, "*.safetensors")):
        raise FileNotFoundError(f"missing delta safetensors files under: {model_path}")


async def _reload_volume(delta_volume_obj: Any) -> None:
    if delta_volume_obj is None:
        return
    reload_fn = getattr(delta_volume_obj, "reload", None)
    if reload_fn is None:
        raise TypeError("delta_volume must expose a reload() method")
    if asyncio.iscoroutinefunction(reload_fn):
        await reload_fn()
    else:
        await asyncio.to_thread(reload_fn)


def _forward_response_headers(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}


async def _proxy_request(
    request: Any,
    *,
    target_base_url: str,
    delta_volume_obj: Any,
    delta_mount_path: str,
    update_lock: Any,
) -> Any:
    import aiohttp
    from aiohttp import web

    endpoint = request.match_info["tail"]
    endpoint = f"/{endpoint}" if endpoint else "/"
    target_url = _join_url(target_base_url, endpoint)
    if request.query_string:
        target_url = f"{target_url}?{request.query_string}"

    excluded_headers = _HOP_BY_HOP_HEADERS | {"host"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded_headers}

    if endpoint == "/update_weights_from_disk":
        if request.method != "POST":
            return web.json_response({"error": "method not allowed"}, status=405)
        async with update_lock:
            try:
                payload = await request.json()
                model_path = validate_delta_update_payload(payload, delta_mount_path=delta_mount_path)
                await _reload_volume(delta_volume_obj)
                verify_delta_dir_ready(model_path)
            except json.JSONDecodeError as exc:
                return web.json_response({"error": f"invalid JSON payload: {exc}"}, status=400)
            except Exception as exc:  # noqa: BLE001 - sidecar validation errors should become HTTP errors
                return web.json_response({"error": str(exc)}, status=400)
            body = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
    else:
        body = await request.read()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(request.method, target_url, data=body, headers=headers) as response:
                content = await response.read()
                return web.Response(
                    body=content,
                    status=response.status,
                    headers=_forward_response_headers(response.headers),
                )
    except aiohttp.ClientError as exc:
        return web.json_response({"error": f"SGLang upstream unavailable: {exc}"}, status=503)


def create_delta_proxy_app(
    *,
    target_base_url: str = "http://127.0.0.1:8001",
    delta_volume_obj: Any = None,
    delta_mount_path: str = "/delta",
) -> Any:
    from aiohttp import web

    proxy_app = web.Application()
    proxy_app["target_base_url"] = _normalize_base_url(target_base_url)
    proxy_app["delta_volume"] = delta_volume_obj
    proxy_app["delta_mount_path"] = delta_mount_path
    proxy_app["update_lock"] = asyncio.Lock()

    async def handler(request: Any) -> Any:
        return await _proxy_request(
            request,
            target_base_url=proxy_app["target_base_url"],
            delta_volume_obj=proxy_app["delta_volume"],
            delta_mount_path=proxy_app["delta_mount_path"],
            update_lock=proxy_app["update_lock"],
        )

    proxy_app.router.add_route("*", "/{tail:.*}", handler)
    return proxy_app


def run_delta_proxy(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    target_base_url: str = "http://127.0.0.1:8001",
    delta_volume_obj: Any = None,
    delta_mount_path: str = "/delta",
) -> None:
    from aiohttp import web

    proxy_app = create_delta_proxy_app(
        target_base_url=target_base_url,
        delta_volume_obj=delta_volume_obj,
        delta_mount_path=delta_mount_path,
    )
    web.run_app(proxy_app, host=host, port=port, handle_signals=False)


def _build_rollout_image() -> modal.Image:
    image = modal.Image.from_registry(ROLLOUT_BASE_IMAGE)
    if IS_LOCAL:
        image = (
            image.add_local_file(
                REPO_ROOT / "docker/patch/latest/sglang.patch",
                "/tmp/sglang.patch",
                copy=True,
            )
            .add_local_file(
                REPO_ROOT / "docker/patch/latest/sglang_delta_sync.patch",
                "/tmp/sglang_delta_sync.patch",
                copy=True,
            )
            .run_commands(APPLY_SGLANG_PATCH)
            .run_commands(APPLY_SGLANG_DELTA_PATCH)
            .run_commands(
                "sed -i 's/timeout_keep_alive=5/timeout_keep_alive=300/g' "
                "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py || true",
                "apt-get update -qq && apt-get install -y -qq libcudart12 2>/dev/null "
                "|| pip install nvidia-cuda-runtime-cu12 --quiet",
                r"sed -i 's/self_named_buffers\[name\]\[\.\.\.] = tensor/self_named_buffers[name].data.copy_(tensor)/g' "
                "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py || true",
            )
        )
    image = image.uv_pip_install(
        f"autoinference-utils=={AUTOINFERENCE_UTILS_VERSION}",
        "aiohttp",
        "hf_transfer",
        "requests",
    )
    if IS_LOCAL:
        image = (
            image.add_local_dir(REPO_ROOT / "slime", "/root/slime/slime", copy=True)
            .add_local_file(REPO_ROOT / "examples/delta_weight_sync/modal_delta_sync.py", "/root/modal_delta_sync.py", copy=True)
        )
    return image.env(HF_IMAGE_ENV | RUNTIME_ENV)


def _build_trainer_image() -> modal.Image:
    image = modal.Image.from_registry(TRAINER_BASE_IMAGE).uv_pip_install("hf_transfer", "modal", "requests")
    if IS_LOCAL:
        image = (
            image.add_local_dir(REPO_ROOT / "slime", "/root/slime/slime", copy=True)
            .add_local_file(REPO_ROOT / "train.py", "/root/slime/train.py", copy=True)
            .add_local_file(REPO_ROOT / "examples/delta_weight_sync/modal_delta_sync.py", "/root/modal_delta_sync.py", copy=True)
        )
    return image.env(HF_IMAGE_ENV | RUNTIME_ENV)


rollout_image = _build_rollout_image()
trainer_image = _build_trainer_image()

with rollout_image.imports():
    from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions

SERVER_ARGS = {
    "--revision": MODEL_REVISION,
    "--served-model-name": MODEL_NAME,
    "--context-length": str(SGLANG_CONTEXT_LENGTH),
    "--mem-fraction-static": SGLANG_MEM_FRACTION_STATIC,
    "--reasoning-parser": "qwen3",
    "--trust-remote-code": "",
    "--cuda-graph-bs": "1 2 4 8 16",
    "--cuda-graph-max-bs": str(TARGET_INPUTS),
    "--max-running-requests": str(TARGET_INPUTS),
    "--disable-piecewise-cuda-graph": "",
    "--skip-server-warmup": "",
}

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}

MODEL_ARGS = [
    "--swiglu",
    "--num-layers",
    "36",
    "--hidden-size",
    "2560",
    "--ffn-hidden-size",
    "9728",
    "--num-attention-heads",
    "32",
    "--group-query-attention",
    "--num-query-groups",
    "8",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--rotary-base",
    "1000000",
    "--vocab-size",
    "151936",
    "--kv-channels",
    "128",
    "--qk-layernorm",
]


def _run(cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None, check: bool = True) -> int:
    print("+ " + shlex.join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_output: deque[str] = deque(maxlen=240)
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            last_output.append(line)
            print(line, flush=True)
    return_code = proc.wait()
    if check and return_code != 0:
        tail = "\n".join(last_output)
        raise RuntimeError(f"Command failed with exit code {return_code}: {shlex.join(cmd)}\n\nLast output:\n{tail}")
    return return_code


def _detect_nvlink() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return "0"
    return "1" if "NV" in output else "0"


def _write_smoke_dataset(path: str, rows: int = 8) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prompts = [
        ("What is 1 + 1? Answer with a single integer.", "2"),
        ("What is 3 + 4? Answer with a single integer.", "7"),
        ("What is 9 - 5? Answer with a single integer.", "4"),
        ("What is 6 / 2? Answer with a single integer.", "3"),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(rows):
            prompt, label = prompts[i % len(prompts)]
            f.write(json.dumps({"prompt": prompt, "label": label}) + "\n")


def _ensure_model_cached() -> str:
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=MODEL_NAME, revision=MODEL_REVISION)
    hf_cache_volume.commit()
    return model_path


def _build_train_args(
    rollout_url: str,
    *,
    model_path: str,
    num_rollout: int,
    response_len: int,
) -> list[str]:
    rollout_batch_size = 1
    n_samples_per_prompt = 1
    global_batch_size = rollout_batch_size * n_samples_per_prompt
    dataset_path = "/root/slime-data/qwen3_4b_smoke.jsonl"

    return [
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "1",
        "--megatron-to-hf-mode",
        "bridge",
        "--hf-checkpoint",
        model_path,
        "--load",
        model_path,
        *MODEL_ARGS,
        "--prompt-data",
        dataset_path,
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--apply-chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
        "--custom-rm-path",
        "slime.backends.sglang_utils.delta_sidecar.constant_reward",
        "--disable-rewards-normalization",
        "--num-rollout",
        str(num_rollout),
        "--rollout-batch-size",
        str(rollout_batch_size),
        "--n-samples-per-prompt",
        str(n_samples_per_prompt),
        "--num-steps-per-rollout",
        "1",
        "--global-batch-size",
        str(global_batch_size),
        "--rollout-max-context-len",
        str(SGLANG_CONTEXT_LENGTH),
        "--rollout-max-prompt-len",
        "512",
        "--rollout-max-response-len",
        str(response_len),
        "--rollout-temperature",
        "0.7",
        "--skip-eval-before-train",
        "--advantage-estimator",
        "grpo",
        "--kl-coef",
        "0.0",
        "--kl-loss-coef",
        "0.0",
        "--entropy-coef",
        "0.0",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--micro-batch-size",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "2048",
        "--rollout-external",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-router-url",
        rollout_url,
        "--rollout-external-engine-addrs",
        rollout_url,
        "--sglang-mem-fraction-static",
        SGLANG_MEM_FRACTION_STATIC,
        "--update-weight-mode",
        "delta",
        "--update-weight-transport",
        "disk",
        "--update-weight-encoding",
        "deltas",
        "--update-weight-delta-dir",
        DELTA_MOUNT_PATH,
        "--update-weight-delta-keep-files",
        "--custom-delta-pre-push-path",
        "slime.backends.sglang_utils.delta_sidecar.commit_modal_delta_volume",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "flash",
    ]


@app.cls(
    include_source=False,
    image=rollout_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        DELTA_MOUNT_PATH: delta_volume,
    },
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=24 * HOURS,
    startup_timeout=STARTUP_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    memory=MEMORY_MB,
    region=REGION,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT,
    proxy_regions=PROXY_REGIONS,
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class RolloutServer:
    @modal.enter()
    def startup(self) -> None:
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            worker_port=SGLANG_INTERNAL_PORT,
            tp=1,
            extra_server_args=SERVER_ARGS,
            health_timeout=STARTUP_TIMEOUT,
            health_poll_interval=5.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=SGLANG_INTERNAL_PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=2,
            request_timeout=90.0,
        )
        hf_cache_volume.commit()

        self.sidecar_thread = threading.Thread(
            target=run_delta_proxy,
            kwargs={
                "host": "0.0.0.0",
                "port": SIDECAR_PORT,
                "target_base_url": f"http://127.0.0.1:{SGLANG_INTERNAL_PORT}",
                "delta_volume_obj": delta_volume,
                "delta_mount_path": DELTA_MOUNT_PATH,
            },
            daemon=True,
        )
        self.sidecar_thread.start()
        print(f"{MODEL_NAME} rollout is serving through the delta sidecar on port {SIDECAR_PORT}.", flush=True)

    @modal.exit()
    def stop(self) -> None:
        if hasattr(self, "endpoint"):
            self.endpoint.stop()


@app.function(
    name="train_qwen3_4b",
    include_source=False,
    image=trainer_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        DELTA_MOUNT_PATH: delta_volume,
    },
    max_containers=1,
    timeout=24 * HOURS,
    memory=MEMORY_MB,
    region=REGION,
)
def train_qwen3_4b(rollout_url: str, num_rollout: int = 2, response_len: int = 16) -> dict[str, Any]:
    rollout_url = rollout_url.rstrip("/")
    model_path = _ensure_model_cached()
    dataset_path = "/root/slime-data/qwen3_4b_smoke.jsonl"
    _write_smoke_dataset(dataset_path)

    train_env = os.environ.copy()
    train_env.update(RUNTIME_ENV)
    train_env["NCCL_NVLS_ENABLE"] = _detect_nvlink()
    train_env["MASTER_ADDR"] = "127.0.0.1"
    train_env["RAY_DEDUP_LOGS"] = "0"

    _run(["ray", "stop", "--force"], check=False)
    _run(
        [
            "ray",
            "start",
            "--head",
            "--node-ip-address",
            "127.0.0.1",
            "--num-gpus",
            "1",
            "--disable-usage-stats",
            "--dashboard-host",
            "0.0.0.0",
            "--dashboard-port",
            "8265",
        ],
        env=train_env,
    )

    runtime_env_json = json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": train_env["PYTHONPATH"],
                "CUDA_DEVICE_MAX_CONNECTIONS": train_env["CUDA_DEVICE_MAX_CONNECTIONS"],
                "NCCL_NVLS_ENABLE": train_env["NCCL_NVLS_ENABLE"],
                "RAY_DEDUP_LOGS": train_env["RAY_DEDUP_LOGS"],
            }
        }
    )
    train_args = _build_train_args(
        rollout_url,
        model_path=model_path,
        num_rollout=num_rollout,
        response_len=response_len,
    )
    job_cmd = [
        "ray",
        "job",
        "submit",
        "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={runtime_env_json}",
        "--",
        "python3",
        "/root/slime/train.py",
        *train_args,
    ]
    started = time.time()
    try:
        _run(job_cmd, cwd="/root/slime", env=train_env)
    finally:
        _run(["ray", "stop", "--force"], check=False)

    delta_volume.commit()
    delta_volume.reload()
    version_dirs = []
    if os.path.isdir(DELTA_MOUNT_PATH):
        version_dirs = sorted(name for name in os.listdir(DELTA_MOUNT_PATH) if name.startswith("weight_v"))
    return {
        "rollout_url": rollout_url,
        "num_rollout": num_rollout,
        "elapsed_s": round(time.time() - started, 1),
        "delta_versions": version_dirs,
    }


def _deployed_rollout_url() -> str:
    rollout_cls = modal.Cls.from_name(APP_NAME, "RolloutServer")
    urls = rollout_cls._experimental_get_flash_urls()
    if not urls:
        raise RuntimeError(f"No http_server URL found for deployed Modal class {APP_NAME}::RolloutServer")
    return urls[0].rstrip("/")


def _assert_rollout_ready(rollout_url: str) -> None:
    import requests

    deadline = time.time() + STARTUP_TIMEOUT
    last_error = "not checked"
    while time.time() < deadline:
        try:
            response = requests.get(_join_url(rollout_url, "/health_generate"), timeout=30)
            if response.status_code == 200:
                break
            last_error = f"{response.status_code} {response.text[:300]}"
        except requests.RequestException as exc:
            last_error = repr(exc)
        print(f"Waiting for deployed rollout /health_generate: {last_error}", flush=True)
        time.sleep(10)
    else:
        raise TimeoutError(f"Rollout did not become healthy within {STARTUP_TIMEOUT}s: {last_error}")

    bad_update = requests.post(
        _join_url(rollout_url, "/update_weights_from_disk"),
        json={"load_format": "delta", "model_path": "/tmp/not-a-delta"},
        timeout=30,
    )
    if bad_update.status_code != 400:
        raise RuntimeError(f"sidecar accepted bad delta path: {bad_update.status_code} {bad_update.text}")


@app.local_entrypoint()
def rollout_url() -> None:
    print(_deployed_rollout_url())


@app.local_entrypoint()
def launch_run(num_rollout: int = 2, response_len: int = 16) -> None:
    rollout_url_value = _deployed_rollout_url()
    print(f"Using deployed rollout URL: {rollout_url_value}", flush=True)
    _assert_rollout_ready(rollout_url_value)
    train_fn = modal.Function.from_name(APP_NAME, "train_qwen3_4b")
    result = train_fn.remote(rollout_url_value, num_rollout=num_rollout, response_len=response_len)
    print(json.dumps(result, indent=2, sort_keys=True))
