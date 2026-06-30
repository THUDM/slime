import asyncio
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import ray
from aiohttp import web


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "slime").is_dir() and (parent / "setup.py").is_file():
            return parent
    return Path(__file__).resolve().parents[4]


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread  # noqa: E402
from slime.backends.megatron_utils.server.arguments import (  # noqa: E402
    add_megatron_server_arguments,
    configure_megatron_server_args,
    validate_megatron_server_args,
)
from slime.utils.misc import Box  # noqa: E402


def _parse_sample_n(sample_n: Any) -> int:
    if sample_n is None:
        return 0
    if isinstance(sample_n, bool):
        raise ValueError("sample_n must be an integer >= 0")
    try:
        sample_n = int(sample_n)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_n must be an integer >= 0") from exc
    if sample_n < 0:
        raise ValueError("sample_n must be >= 0")
    return sample_n


def _normalize_label_token_ids(label_token_ids: Any, expected_length: int) -> list[list[int]] | None:
    if label_token_ids is None:
        return None
    if not isinstance(label_token_ids, list):
        raise ValueError("label_token_ids must be a 2D list with shape [len(input_ids) - 1, num_label_tokens]")
    if len(label_token_ids) != expected_length:
        raise ValueError("label_token_ids length must match len(input_ids) - 1")

    normalized = []
    width = None
    for row in label_token_ids:
        if not isinstance(row, list):
            raise ValueError("label_token_ids must be a 2D list with shape [len(input_ids) - 1, num_label_tokens]")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("label_token_ids rows must have the same length")
        try:
            normalized.append([int(token_id) for token_id in row])
        except (TypeError, ValueError) as exc:
            raise ValueError("label_token_ids must contain integers") from exc

    return normalized


def _get_max_request_length(args) -> int:
    return args.megatron_server_max_length


def _get_update_timeout_s(payload: dict[str, Any], args) -> float:
    return float(payload.get("timeout_s") or args.megatron_server_update_timeout_s)


@ray.remote
class SampleManager:
    """Minimal rollout-manager surface plus request queue for teacher-server mode."""

    def __init__(self, args, pg):
        self.args = args
        self.pg = pg
        self.train_parallel_config = None

        self._pending_requests = deque()
        self._inflight = defaultdict(deque)  # worker_id -> deque of batch_info
        self._results = {}
        self._next_request_id = 0
        self._canceled = set()
        self.total_finished_reqs = 0
        self.total_finished_tokens = 0

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def submit(
        self,
        input_ids,
        request_id=None,
        response_length=None,
        loss_mask=None,
        metadata=None,
        sample_n=0,
        label_token_ids=None,
    ):
        if len(input_ids) == 0:
            raise ValueError("input_ids is empty")
        sample_n = _parse_sample_n(sample_n)

        if response_length is None:
            response_length = max(len(input_ids) - 1, 0)

        if loss_mask is None:
            loss_mask = [1] * response_length

        if response_length > len(input_ids) - 1:
            raise ValueError("response_length error")
        if len(loss_mask) != response_length:
            raise ValueError("loss_mask length error")

        if request_id is None:
            rid = f"req_{self._next_request_id}"
            self._next_request_id += 1
        else:
            rid = request_id

        self._pending_requests.append(
            {
                "request_id": rid,
                "tokens": input_ids,
                "response_length": response_length,
                "loss_mask": loss_mask,
                "metadata": metadata,
                "sample_n": sample_n,
                "label_token_ids": label_token_ids,
            }
        )
        return rid

    def _stats(self):
        total_inflight = sum(len(q) for q in self._inflight.values())
        return {
            "queue_size": len(self._pending_requests),
            "inflight_size": total_inflight,
        }

    def get_stats(self):
        return self._stats()

    def get_global_stats(self):
        return {
            **self._stats(),
            "total_finished_reqs": self.total_finished_reqs,
            "total_finished_tokens": self.total_finished_tokens,
        }

    def get_loads(self):
        pending_tokens = sum(len(req["tokens"]) for req in self._pending_requests)
        pending_response_tokens = sum(req["response_length"] for req in self._pending_requests)
        running_by_worker = {worker_id: len(queue) for worker_id, queue in self._inflight.items()}
        running_tokens = sum(sum(info["response_lengths"]) for queue in self._inflight.values() for info in queue)
        return {
            **self.get_global_stats(),
            "pending_tokens": pending_tokens,
            "pending_response_tokens": pending_response_tokens,
            "running_tokens": running_tokens,
            "running_by_worker": running_by_worker,
        }

    def cancel_request(self, request_id: str) -> None:
        self._canceled.add(request_id)
        self._results.pop(request_id, None)
        if self._pending_requests:
            self._pending_requests = deque(
                req for req in self._pending_requests if req.get("request_id") != request_id
            )

    def _pop_next_request(self):
        while self._pending_requests:
            req = self._pending_requests.popleft()
            rid = req.get("request_id")
            if rid in self._canceled:
                continue
            return req
        return None

    def get_input_data(self, worker_id):
        if self.train_parallel_config is None or not self._pending_requests:
            return None

        req = self._pop_next_request()
        if req is None:
            return None

        tokens = req["tokens"]
        response_length = req["response_length"]
        loss_mask = req["loss_mask"]
        sample_n = req.get("sample_n", 0)
        label_token_ids = req.get("label_token_ids")

        request_ids = [req["request_id"]]
        is_dummy = [False]
        token_lens = [len(tokens)]
        response_lengths = [response_length]

        rollout_data = {
            "tokens": [tokens],
            "response_lengths": [response_length],
            "loss_masks": [loss_mask],
            "sample_indices": [0],
            "total_lengths": [len(tokens)],
            "sample_ns": [sample_n],
        }
        if label_token_ids is not None:
            rollout_data["label_token_ids"] = [label_token_ids]
        data_refs = [Box(ray.put(rollout_data))]

        self._inflight[worker_id].append(
            {
                "request_ids": request_ids,
                "is_dummy": is_dummy,
                "token_lens": token_lens,
                "response_lengths": response_lengths,
                "sample_ns": [sample_n],
            }
        )

        return data_refs

    def load(self, rollout_id=None):
        pass

    def save_log_probs(self, worker_id, outputs):
        if not self._inflight[worker_id]:
            raise KeyError(f"No inflight info for worker {worker_id}")

        info = self._inflight[worker_id].popleft()

        request_ids = info["request_ids"]
        is_dummy = info["is_dummy"]
        response_lengths = info["response_lengths"]

        assert len(outputs) == len(request_ids)

        for i, rid in enumerate(request_ids):
            if is_dummy[i]:
                continue

            self.total_finished_reqs += 1
            self.total_finished_tokens += response_lengths[i]

            if rid in self._canceled:
                continue

            output_item = outputs[i]
            if isinstance(output_item, dict):
                log_probs = output_item.get("log_probs")
                sampled_token_ids = output_item.get("sampled_token_ids")
                sampled_log_probs = output_item.get("sampled_log_probs")
                label_token_log_probs = output_item.get("label_token_log_probs")
            else:
                # 兼容旧格式
                log_probs = output_item
                sampled_token_ids = None
                sampled_log_probs = None
                label_token_log_probs = None

            self._results[rid] = {
                "log_probs": log_probs,
                "sampled_token_ids": sampled_token_ids,
                "sampled_log_probs": sampled_log_probs,
                "label_token_log_probs": label_token_log_probs,
            }
        return [rid for rid in request_ids if rid is not None]

    def get_result(self, request_id):
        return self._results.pop(request_id, None)


def _merge_log_probs(logp_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logp_parts.sort(key=lambda x: x.get("dp_rank"))
    merged = []
    for part in logp_parts:
        has_log_prob = part.get("log_prob") is not None
        has_sampled = part.get("sampled_log_probs") is not None or part.get("sampled_token_ids") is not None
        has_label = part.get("label_token_log_probs") is not None
        if not has_log_prob and not has_sampled and not has_label:
            continue
        merged.append(
            {
                "log_probs": part.get("log_prob"),
                "sampled_log_probs": part.get("sampled_log_probs"),
                "sampled_token_ids": part.get("sampled_token_ids"),
                "label_token_log_probs": part.get("label_token_log_probs"),
            }
        )
    return merged


def _run_stats_printer(sample_manager, interval=1.0):
    """像 SGLang 一样每隔一段时间打印系统状态"""
    last_time = time.time()
    last_reqs = 0
    last_tokens = 0

    print(f"Stats printer started (interval={interval}s)...", flush=True)

    while True:
        time.sleep(interval)
        try:
            # 获取全局统计
            stats = ray.get(sample_manager.get_global_stats.remote())

            now = time.time()
            delta_t = now - last_time
            if delta_t <= 0:
                continue

            # 计算增量
            curr_reqs = stats["total_finished_reqs"]
            curr_tokens = stats["total_finished_tokens"]

            delta_reqs = curr_reqs - last_reqs
            delta_tokens = curr_tokens - last_tokens

            rps = delta_reqs / delta_t
            tps = delta_tokens / delta_t

            # 格式化输出
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if stats["queue_size"] > 0 or stats["inflight_size"] > 0 or rps > 0 or tps > 0:
                print(
                    f"[{current_time_str}] "
                    f"Queue: {stats['queue_size']} | "
                    f"Running: {stats['inflight_size']} | "
                    f"RPS: {rps:.2f} | "
                    f"TPS: {tps:.2f} tokens/s",
                    flush=True,
                )

            last_time = now
            last_reqs = curr_reqs
            last_tokens = curr_tokens

        except Exception as e:
            print(f"Stats printer error: {e}", flush=True)


async def _ray_get(ref):
    return await asyncio.to_thread(ray.get, ref)


async def _read_json_payload(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    try:
        payload = await request.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _crop_sequence(value, expected_len: int):
    if value is not None and len(value) > expected_len:
        return value[:expected_len]
    return value


def _build_generate_response(request_id: str, result: dict[str, Any], expected_len: int) -> dict[str, Any]:
    response = {
        "request_id": request_id,
        "log_probs": _crop_sequence(result.get("log_probs"), expected_len),
    }
    for key in ("label_token_log_probs", "sampled_token_ids", "sampled_log_probs"):
        value = result.get(key)
        if value is not None:
            response[key] = _crop_sequence(value, expected_len)
    return response


async def _wait_until_idle(sample_manager, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while True:
        loads = await _ray_get(sample_manager.get_loads.remote())
        if loads["queue_size"] == 0 and loads["inflight_size"] == 0:
            return loads
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for queued/inflight requests to finish: {loads}")
        await asyncio.sleep(0.1)


def _get_update_model_path(payload: dict[str, Any]) -> str | None:
    model_path = payload.get("model_path") or payload.get("path") or payload.get("load")
    return str(model_path) if model_path else None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of an arbitrary value to something JSON-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _args_to_dict(args) -> dict[str, Any]:
    return {key: _jsonable(val) for key, val in sorted(vars(args).items())}


def _build_http_app(sample_manager, args, update_from_disk_fn=None):
    app = web.Application(client_max_size=64 * 1024 * 1024)
    update_state = {
        "in_progress": False,
        "updating_model_path": None,
        "update_future": None,
    }
    update_lock = asyncio.Lock()

    async def detect(_request: web.Request) -> web.Response:
        return web.json_response({"server_type": "megatron_server"})

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def info(_request: web.Request) -> web.Response:
        return web.json_response({"args": _args_to_dict(args)})

    async def get_loads(_request: web.Request) -> web.Response:
        return web.json_response(await _ray_get(sample_manager.get_loads.remote()))

    async def update_from_disk(request: web.Request) -> web.Response:
        if update_from_disk_fn is None:
            return _json_error("update_from_disk is not available during warmup", 503)

        payload = await _read_json_payload(request)
        model_path = _get_update_model_path(payload)
        if model_path is None:
            return _json_error("missing model_path", 400)

        async with update_lock:
            if getattr(args, "load", None) == model_path:
                return web.json_response({"ok": True, "model_path": model_path, "skipped": True})

            if update_state["in_progress"]:
                if update_state["updating_model_path"] == model_path and update_state["update_future"] is not None:
                    update_future = update_state["update_future"]
                    coalesced = True
                else:
                    updating_model_path = update_state["updating_model_path"]
                    return _json_error(f"update_from_disk is already in progress for {updating_model_path}", 409)
            else:
                update_future = asyncio.get_running_loop().create_future()
                update_state["in_progress"] = True
                update_state["updating_model_path"] = model_path
                update_state["update_future"] = update_future
                coalesced = False

        if coalesced:
            result = await asyncio.shield(update_future)
            if result.get("ok") is True:
                result = dict(result)
                result["coalesced"] = True
                return web.json_response(result)
            return _json_error(result.get("error", "update_from_disk failed"), int(result.get("status", 500)))

        timeout_s = _get_update_timeout_s(payload, args)
        result = None
        error = None
        try:
            before_loads = await _wait_until_idle(sample_manager, timeout_s)
            update_result = await asyncio.to_thread(update_from_disk_fn, model_path)
            after_loads = await _ray_get(sample_manager.get_loads.remote())
        except TimeoutError as e:
            error = {"ok": False, "status": 503, "error": str(e)}
        except Exception as e:
            error = {"ok": False, "status": 500, "error": f"update_from_disk failed: {e}"}
        finally:
            if error is None:
                result = {
                    "ok": True,
                    "model_path": model_path,
                    "before_loads": before_loads,
                    "after_loads": after_loads,
                    "update_result": update_result,
                }
                # Reflect the freshly loaded checkpoint in /info. The actors restore
                # their own args after loading, so only the server-side copy needs to be
                # kept in sync here.
                args.load = model_path
                args.ref_load = model_path

            async with update_lock:
                if result is not None:
                    update_future.set_result(result)
                else:
                    update_future.set_result(error)
                update_state["in_progress"] = False
                update_state["updating_model_path"] = None
                update_state["update_future"] = None

        if result is not None:
            return web.json_response(result)
        return _json_error(error["error"], error["status"])

    async def generate(request: web.Request) -> web.Response:
        if update_state["in_progress"]:
            return _json_error("server is updating from disk", 503)

        payload = await _read_json_payload(request)
        if "input_ids" not in payload:
            return _json_error("missing input_ids", 400)

        input_ids = payload["input_ids"]
        original_input_len = len(input_ids)
        max_request_length = _get_max_request_length(args)
        if max_request_length > 0 and original_input_len > max_request_length:
            return web.json_response(
                {
                    "error": (
                        f"input_ids length {original_input_len} exceeds configured maximum {max_request_length}"
                    ),
                    "input_length": original_input_len,
                    "max_length": max_request_length,
                },
                status=413,
            )

        try:
            sample_n = _parse_sample_n(payload.get("sample_n", 0))
            valid_response_length = max(original_input_len - 1, 0)
            label_token_ids = _normalize_label_token_ids(payload.get("label_token_ids"), valid_response_length)
        except ValueError as e:
            return _json_error(str(e), 400)

        try:
            stats = await _ray_get(sample_manager.get_stats.remote())
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{current_time_str}] Received Request | "
                f"Queue: {stats['queue_size']} | "
                f"Running: {stats['inflight_size']} | "
                f"Input Len: {original_input_len} | "
                f"Max Len: {max_request_length or 'disabled'} | "
                f"sample_n: {sample_n} | "
                f"label_width: {len(label_token_ids[0]) if label_token_ids else 0}",
                flush=True,
            )
        except Exception as e:
            print(f"stats failed: {e}", flush=True)

        request_id = None
        try:
            if update_state["in_progress"]:
                return _json_error("server is updating from disk", 503)

            response_length = valid_response_length
            loss_mask = [1] * response_length
            request_id = await _ray_get(
                sample_manager.submit.remote(
                    input_ids,
                    response_length=response_length,
                    loss_mask=loss_mask,
                    sample_n=sample_n,
                    label_token_ids=label_token_ids,
                )
            )

            while True:
                result = await _ray_get(sample_manager.get_result.remote(request_id))
                if result is not None:
                    break
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            if request_id is not None:
                print(f"cancel request {request_id} because client disconnected", flush=True)
                await _ray_get(sample_manager.cancel_request.remote(request_id))
            raise
        except Exception as e:
            prefix = "submit failed" if request_id is None else "result failed"
            return _json_error(f"{prefix}: {e}", 500)

        expected_log_probs_len = max(original_input_len - 1, 0)
        return web.json_response(_build_generate_response(request_id, result, expected_log_probs_len))

    app.router.add_get("/detect", detect)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/info", info)
    app.router.add_get("/get_loads", get_loads)
    app.router.add_post("/update_weights_from_disk", update_from_disk)
    app.router.add_post("/generate", generate)
    return app


def _start_http_server(sample_manager, args, update_from_disk_fn):
    app = _build_http_app(sample_manager, args, update_from_disk_fn=update_from_disk_fn)
    handle = run_app_in_thread(
        app,
        host="0.0.0.0",
        port=args.teacher_port,
        thread_name="teacher-http",
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )

    stats_thread = threading.Thread(
        target=_run_stats_printer,
        args=(sample_manager, 5.0),
        name="stats-printer",
        daemon=True,
    )
    stats_thread.start()
    return handle


def _run_warmup_via_private_http(sample_manager, args):
    warmup_timeout_s = args.teacher_warmup_timeout_s
    warmup_host = "127.0.0.1"
    warmup_port = args.teacher_warmup_port
    warmup_tokens = [101, 102, 103, 104]

    print(
        "start private warmup server: "
        f"http://{warmup_host}:{warmup_port}, timeout_s={warmup_timeout_s}, input_len={len(warmup_tokens)}",
        flush=True,
    )

    app = _build_http_app(sample_manager, args)
    warmup_handle = run_app_in_thread(
        app,
        host=warmup_host,
        port=warmup_port,
        thread_name="teacher-warmup-http",
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )

    try:
        req_start = time.time()
        timeout = httpx.Timeout(warmup_timeout_s + 60, connect=10.0)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(
                f"http://{warmup_host}:{warmup_handle.port}/generate",
                json={
                    "input_ids": warmup_tokens,
                    "sample_n": 0,
                    "timeout_s": warmup_timeout_s,
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f"warmup request failed with http {response.status_code}, body={response.text[:500]}")

        print(
            f"private warmup request finished in {time.time() - req_start:.2f}s",
            flush=True,
        )
    finally:
        warmup_handle.stop()
        print("private warmup server stopped", flush=True)


@ray.remote(num_cpus=0)
def run_megatron_dp_models_loop_worker(dp_rank, pp_size, sample_manager, actor_models):
    worker_id = f"rank_{dp_rank}"
    print(f"Start Async Pipeline Loop for {worker_id}", flush=True)

    MAX_INFLIGHT_BATCHES = pp_size + 1
    futures_queue = deque()

    while True:
        # 1. Submission Stage
        if len(futures_queue) < MAX_INFLIGHT_BATCHES:
            rollout_data_ref = ray.get(sample_manager.get_input_data.remote(worker_id))

            if rollout_data_ref is not None:
                logp_parts_refs = [actor.compute_logp.remote(rollout_data_ref) for actor in actor_models]
                futures_queue.append(logp_parts_refs)
            else:
                if not futures_queue:
                    time.sleep(0.02)

        # 2. Collection Stage
        if futures_queue:
            oldest_refs = futures_queue[0]
            should_block = len(futures_queue) >= MAX_INFLIGHT_BATCHES

            _, remaining_refs = ray.wait(
                oldest_refs, num_returns=len(oldest_refs), timeout=None if should_block else 0
            )

            if len(remaining_refs) == 0:
                logp_parts = ray.get(oldest_refs)
                futures_queue.popleft()

                merged_log_probs = _merge_log_probs(logp_parts)
                sample_manager.save_log_probs.remote(worker_id, merged_log_probs)


def _build_update_from_disk_fn(actor_model):
    def update_from_disk(model_path: str):
        refs = [actor.update_from_disk.remote(model_path) for actor in actor_model._actor_handlers]
        results = ray.get(refs)
        return {
            "num_ranks": len(results),
            "model_path": model_path,
            "results": results,
        }

    return update_from_disk


def launch(args):
    from slime.backends.megatron_utils.server.logprob_utils import TeacherLogpRayActor
    from slime.ray.placement_group import create_placement_groups, create_training_models

    configure_megatron_server_args(args)
    validate_megatron_server_args(args)
    pgs = create_placement_groups(args)

    sample_manager = SampleManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pgs["rollout"])

    print("initializing training models...", flush=True)
    actor_model, _ = create_training_models(args, pgs, sample_manager, actor_cls=TeacherLogpRayActor)
    parallel_infos = ray.get([actor.get_parallel_infos.remote() for actor in actor_model._actor_handlers])
    all_dp_ranks = set(info["dp_rank"] for info in parallel_infos)
    futures = []
    pp_size = parallel_infos[0]["pp_size"]

    for dp_rank in all_dp_ranks:
        models_in_same_dp_rank = [
            actor
            for info, actor in zip(parallel_infos, actor_model._actor_handlers, strict=True)
            if info["dp_rank"] == dp_rank
        ]

        # 注意：这里已经应用了之前讨论的异步 Worker
        fut = run_megatron_dp_models_loop_worker.remote(dp_rank, pp_size, sample_manager, models_in_same_dp_rank)
        futures.append(fut)

    if args.megatron_server_warmup:
        _run_warmup_via_private_http(sample_manager, args)
    else:
        print("warmup disabled", flush=True)
    print("training models ready and warmup done", flush=True)
    _start_http_server(sample_manager, args, update_from_disk_fn=_build_update_from_disk_fn(actor_model))

    ray.get(futures)


def main():
    from slime.utils.arguments import parse_args

    args = parse_args(add_custom_arguments=add_megatron_server_arguments)
    launch(args)


if __name__ == "__main__":
    main()
