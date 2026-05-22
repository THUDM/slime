import asyncio
import glob
import json
import os
import re
from typing import Any

from slime.utils.url_utils import join_url, normalize_base_url

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


def commit_modal_delta_volume(args: Any, version_dir: str, rollout_engines: Any) -> None:
    volume_name = os.environ.get("SLIME_DELTA_VOLUME_NAME")
    if not volume_name:
        raise RuntimeError("SLIME_DELTA_VOLUME_NAME must be set to commit a Modal delta volume")

    import modal

    modal.Volume.from_name(volume_name, create_if_missing=True).commit()
    print(f"Committed Modal delta volume {volume_name} for {version_dir}", flush=True)


async def constant_reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    if isinstance(sample_or_samples, list):
        return [1.0 for _ in sample_or_samples]
    return 1.0


async def _reload_volume(delta_volume: Any) -> None:
    if delta_volume is None:
        return
    reload_fn = getattr(delta_volume, "reload", None)
    if reload_fn is None:
        raise TypeError("delta_volume must expose a reload() method")
    if asyncio.iscoroutinefunction(reload_fn):
        await reload_fn()
    else:
        await asyncio.to_thread(reload_fn)


def _forward_response_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}


async def _proxy_request(request, *, target_base_url: str, delta_volume: Any, delta_mount_path: str, update_lock):
    import aiohttp
    from aiohttp import web

    endpoint = request.match_info["tail"]
    endpoint = f"/{endpoint}" if endpoint else "/"
    if request.query_string:
        target_url = f"{join_url(target_base_url, endpoint)}?{request.query_string}"
    else:
        target_url = join_url(target_base_url, endpoint)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS | {"host"}}

    if endpoint == "/update_weights_from_disk":
        if request.method != "POST":
            return web.json_response({"error": "method not allowed"}, status=405)
        async with update_lock:
            try:
                payload = await request.json()
                model_path = validate_delta_update_payload(payload, delta_mount_path=delta_mount_path)
                await _reload_volume(delta_volume)
                verify_delta_dir_ready(model_path)
            except json.JSONDecodeError as exc:
                return web.json_response({"error": f"invalid JSON payload: {exc}"}, status=400)
            except Exception as exc:  # noqa: BLE001 - return sidecar validation/reload errors as HTTP
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
    delta_volume: Any = None,
    delta_mount_path: str = "/delta",
):
    from aiohttp import web

    app = web.Application()
    app["target_base_url"] = normalize_base_url(target_base_url)
    app["delta_volume"] = delta_volume
    app["delta_mount_path"] = delta_mount_path
    app["update_lock"] = asyncio.Lock()

    async def handler(request):
        return await _proxy_request(
            request,
            target_base_url=app["target_base_url"],
            delta_volume=app["delta_volume"],
            delta_mount_path=app["delta_mount_path"],
            update_lock=app["update_lock"],
        )

    app.router.add_route("*", "/{tail:.*}", handler)
    return app


def run_delta_proxy(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    target_base_url: str = "http://127.0.0.1:8001",
    delta_volume: Any = None,
    delta_mount_path: str = "/delta",
) -> None:
    from aiohttp import web

    app = create_delta_proxy_app(
        target_base_url=target_base_url,
        delta_volume=delta_volume,
        delta_mount_path=delta_mount_path,
    )
    web.run_app(app, host=host, port=port, handle_signals=False)
