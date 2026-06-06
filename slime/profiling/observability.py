import datetime
import json
import logging
import os
import platform
import socket
import sys
import time
import uuid
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_PROMETHEUS_SCRAPE_INTERVAL = "5s"
_PROMETHEUS_EVALUATION_INTERVAL = "15s"
_PROMETHEUS_ROUTER_METRICS_FILE = "sglang_router_metrics.json"
_PROMETHEUS_ROUTER_ENGINE_METRICS_FILE = "sglang_router_engine_metrics.json"

_SECRET_FRAGMENTS = ("key", "token", "secret", "password", "authorization", "credential")
_MANIFEST_CONFIG_ALLOWLIST = {
    "advantage_estimator",
    "actor_num_gpus_per_node",
    "actor_num_nodes",
    "colocate",
    "debug_rollout_only",
    "debug_train_only",
    "enable_observability",
    "eval_interval",
    "global_batch_size",
    "hf_checkpoint",
    "load",
    "n_samples_per_prompt",
    "num_critic_only_steps",
    "num_gpus_per_node",
    "num_rollout",
    "observability_enabled",
    "offload_rollout",
    "offload_train",
    "rollout_batch_size",
    "rollout_external",
    "rollout_max_response_len",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "run_dir",
    "run_id",
    "save",
    "sglang_dp_size",
    "sglang_ep_size",
    "sglang_pp_size",
    "start_rollout_id",
    "train_backend",
    "update_weights_interval",
    "use_critic",
}
_MANIFEST_CONFIG_PREFIX_ALLOWLIST = (
    "observability_",
    "sglang_enable_",
    "sglang_export_metrics_to_file",
    "sglang_log_requests",
)
_REQUEST_METRICS_FORBIDDEN_FIELDS = (
    "prompt",
    "input_ids",
    "output_ids",
    "output_text",
    "sampling_params",
    "text",
    "image_data",
    "audio_data",
    "video_data",
    "multi_modal_data",
)


def prepare_observability_args(args) -> None:
    profile = getattr(args, "observability_profile", None)
    if profile is not None:
        args.enable_observability = profile != "off"
        logger.warning(
            "--observability-profile is deprecated. Use --enable-observability for the single production profiling mode."
        )

    args.observability_enabled = bool(getattr(args, "enable_observability", False))

    if not args.observability_enabled:
        return

    if not getattr(args, "run_id", None):
        args.run_id = _generate_run_id()

    if not getattr(args, "run_dir", None):
        run_root = os.environ.get("SLIME_RUN_ROOT", "/tmp/slime-runs")
        args.run_dir = os.path.join(run_root, args.run_id)
    args.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))

    _prepare_observability_paths(args)
    args.observability_request_metrics_privacy_mode = "metadata_only_allowlist"

    _apply_sglang_request_profiling_defaults(args)


def _prepare_observability_paths(args) -> None:
    scratch_dir = getattr(args, "observability_scratch_dir", None) or os.environ.get("SLIME_OBS_SCRATCH_DIR")
    args.observability_scratch_dir = (
        _abspath_template(scratch_dir) if scratch_dir else os.path.join(args.run_dir, "nodes", "node={hostname}")
    )

    prometheus_tsdb_dir = getattr(args, "observability_prometheus_tsdb_dir", None) or os.environ.get(
        "SLIME_PROMETHEUS_TSDB_DIR"
    )
    args.observability_prometheus_tsdb_dir = (
        _abspath_template(prometheus_tsdb_dir)
        if prometheus_tsdb_dir
        else os.path.join(
            "/tmp",
            "slime-observability",
            args.run_id,
            "prometheus-tsdb",
        )
    )

    export_dir = getattr(args, "observability_export_dir", None) or os.environ.get("SLIME_OBS_EXPORT_DIR")
    args.observability_export_dir = (
        _abspath_template(export_dir) if export_dir else os.path.join(args.run_dir, "export")
    )

    status_dir = os.path.join(args.run_dir, "observability")
    prometheus_dir = os.path.join(args.run_dir, "prometheus")
    prometheus_file_sd_dir = os.path.join(prometheus_dir, "file_sd")
    path_attrs = {
        "observability_manifest_path": os.path.join(args.run_dir, "manifest.json"),
        "observability_status_dir": status_dir,
        "observability_status_path": os.path.join(status_dir, "status.json"),
        "observability_errors_path": os.path.join(status_dir, "errors.jsonl"),
        "observability_component_state_path": os.path.join(status_dir, "component_state.json"),
        "observability_prometheus_dir": prometheus_dir,
        "observability_prometheus_config": os.path.join(prometheus_dir, "prometheus.yml"),
        "observability_prometheus_file_sd_dir": prometheus_file_sd_dir,
        "observability_sglang_router_metrics_target_file": os.path.join(
            prometheus_file_sd_dir,
            _PROMETHEUS_ROUTER_METRICS_FILE,
        ),
        "observability_sglang_router_engine_metrics_target_file": os.path.join(
            prometheus_file_sd_dir,
            _PROMETHEUS_ROUTER_ENGINE_METRICS_FILE,
        ),
        "observability_sglang_request_metrics_dir": os.path.join(
            args.observability_scratch_dir,
            "request_metrics",
            "sglang",
        ),
        "observability_sglang_request_log_dir": os.path.join(
            args.observability_scratch_dir,
            "logs",
            "sglang",
        ),
        "observability_sglang_request_time_stats_log_dir": os.path.join(
            args.observability_scratch_dir,
            "request_time_stats",
            "sglang",
        ),
        "observability_sglang_logging_config_dir": os.path.join(
            args.run_dir,
            "logging_configs",
            "sglang",
        ),
    }
    for name, path in path_attrs.items():
        setattr(args, name, path)

    # Backward-compatible alias for existing callers and docs generated from
    # the first implementation.
    args.observability_sglang_router_target_file = args.observability_sglang_router_engine_metrics_target_file


def initialize_observability(args) -> None:
    if not getattr(args, "observability_enabled", False):
        return

    try:
        os.makedirs(args.run_dir, exist_ok=True)
        os.makedirs(args.observability_status_dir, exist_ok=True)
        os.makedirs(args.observability_prometheus_file_sd_dir, exist_ok=True)
        os.makedirs(args.observability_export_dir, exist_ok=True)
        os.makedirs(args.observability_sglang_logging_config_dir, exist_ok=True)

        _write_text_atomic(args.observability_prometheus_config, _render_prometheus_config(args))
        _write_json_atomic(args.observability_manifest_path, _build_manifest(args))
        _write_json_atomic(args.observability_component_state_path, _initial_component_state(args))
        _write_json_atomic(
            args.observability_status_path,
            {
                "schema_version": "slime.observability.status.v1",
                "state": "ok",
                "run_id": args.run_id,
                "updated_at": _utc_now(),
            },
        )
        logger.info(
            "Initialized observability bundle at %s.",
            args.run_dir,
        )
    except Exception:
        logger.exception("Failed to initialize observability bundle; continuing without failing training.")
        args.observability_enabled = False


def register_sglang_router(args, router_addr: str | None) -> None:
    if not getattr(args, "observability_enabled", False) or router_addr is None:
        return

    try:
        parsed = urlparse(router_addr)
        if not parsed.netloc:
            logger.warning(f"Skip writing Prometheus target file for invalid router address: {router_addr}")
            return

        router_payload = _build_sglang_router_target_payload(
            args,
            target=parsed.netloc,
            metrics_endpoint="router",
        )
        engine_payload = _build_sglang_router_target_payload(
            args,
            target=parsed.netloc,
            metrics_endpoint="engine_aggregated",
        )
        _write_json_atomic(args.observability_sglang_router_metrics_target_file, router_payload)
        _write_json_atomic(args.observability_sglang_router_engine_metrics_target_file, engine_payload)
        update_observability_component_state(
            args,
            "prometheus_file_sd",
            {
                "state": "ok",
                "router_metrics_target_file": args.observability_sglang_router_metrics_target_file,
                "engine_metrics_target_file": args.observability_sglang_router_engine_metrics_target_file,
                "target": parsed.netloc,
                "updated_at": _utc_now(),
            },
        )
        logger.info(
            "Registered SGLang router metrics target %s in %s.",
            parsed.netloc,
            args.observability_prometheus_file_sd_dir,
        )
    except Exception as exc:
        logger.exception("Failed to register SGLang Prometheus target; continuing without failing training.")
        record_observability_error(args, "prometheus_file_sd", "register_sglang_router_failed", exc)


def _render_prometheus_config(args) -> str:
    router_metrics_target_file = args.observability_sglang_router_metrics_target_file
    engine_metrics_target_file = args.observability_sglang_router_engine_metrics_target_file
    return (
        "global:\n"
        f"  scrape_interval: {_PROMETHEUS_SCRAPE_INTERVAL}\n"
        f"  evaluation_interval: {_PROMETHEUS_EVALUATION_INTERVAL}\n"
        "\n"
        "scrape_configs:\n"
        "  - job_name: slime_sglang_router\n"
        "    metrics_path: /metrics\n"
        "    file_sd_configs:\n"
        "      - files:\n"
        f"          - {router_metrics_target_file}\n"
        "\n"
        "  - job_name: slime_sglang_engine_aggregated\n"
        "    metrics_path: /engine_metrics\n"
        "    file_sd_configs:\n"
        "      - files:\n"
        f"          - {engine_metrics_target_file}\n"
    )


def _build_manifest(args) -> dict:
    return {
        "schema_version": "slime.observability.manifest.v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "run_dir": args.run_dir,
        "observability_mode": "production_profiling",
        "paths": {
            "scratch_dir": args.observability_scratch_dir,
            "export_dir": args.observability_export_dir,
            "manifest": args.observability_manifest_path,
            "prometheus_config": args.observability_prometheus_config,
            "prometheus_tsdb_dir": args.observability_prometheus_tsdb_dir,
            "sglang_router_metrics_file_sd": args.observability_sglang_router_metrics_target_file,
            "sglang_router_engine_metrics_file_sd": args.observability_sglang_router_engine_metrics_target_file,
            "sglang_request_metrics_dir": args.observability_sglang_request_metrics_dir,
            "sglang_request_log_dir": args.observability_sglang_request_log_dir,
            "sglang_request_time_stats_log_dir": args.observability_sglang_request_time_stats_log_dir,
            "sglang_logging_config_dir": args.observability_sglang_logging_config_dir,
            "status": args.observability_status_path,
            "errors": args.observability_errors_path,
            "component_state": args.observability_component_state_path,
        },
        "privacy": {
            "manifest_config_mode": "allowlist",
            "request_metrics_privacy_mode": args.observability_request_metrics_privacy_mode,
            "request_metrics_forbidden_fields": list(_REQUEST_METRICS_FORBIDDEN_FIELDS),
        },
        "storage_contract": {
            "run_dir": "durable low-frequency metadata, configs, status, and summaries",
            "scratch_dir": "node-local or node-sharded high-frequency request artifacts",
            "prometheus_tsdb_dir": "local Prometheus TSDB path; prefer local POSIX storage",
            "export_dir": "optional durable compacted or post-run outputs",
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "timezone": "UTC",
        },
        "versions": _collect_versions(),
        "config": _redact_config(vars(args)),
    }


def _apply_sglang_request_profiling_defaults(args) -> None:
    # SGLang's exporter currently writes one JSONL record per completed
    # request. Keep the logger metadata-only while slime records the effective
    # privacy mode and checks generated artifacts out of band.
    args.sglang_export_metrics_to_file = True
    if not getattr(args, "sglang_export_metrics_to_file_dir", None):
        args.sglang_export_metrics_to_file_dir = args.observability_sglang_request_metrics_dir

    args.sglang_log_requests = True
    args.sglang_log_requests_level = 0
    args.sglang_log_requests_format = "json"
    if not getattr(args, "sglang_log_requests_target", None):
        args.sglang_log_requests_target = [args.observability_sglang_request_log_dir]

    args.sglang_enable_request_time_stats_logging = True


def _collect_versions() -> dict:
    versions = {}
    for module_name in ("sglang", "sglang_router"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


def _redact_config(config: dict) -> dict:
    redacted = {}
    omitted_keys = []
    for key, value in sorted(config.items()):
        if not _is_manifest_config_allowed(key):
            omitted_keys.append(key)
            continue
        if _looks_secret(key):
            redacted[key] = "<redacted>" if value is not None else None
        else:
            redacted[key] = _json_safe(value)
    redacted["_omitted_keys_count"] = len(omitted_keys)
    return redacted


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _generate_run_id() -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{socket.gethostname()}_{uuid.uuid4().hex[:8]}"


def resolve_observability_path(args, path: str | None) -> str | None:
    if path is None:
        return None
    values = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "run_id": getattr(args, "run_id", "unknown"),
    }
    try:
        path = path.format(**values)
    except KeyError:
        pass
    return os.path.abspath(os.path.expanduser(path))


def record_observability_error(args, component: str, reason: str, error: Exception | str | None = None) -> None:
    if not getattr(args, "observability_enabled", False):
        return

    payload = {
        "schema_version": "slime.observability.error.v1",
        "created_at": _utc_now(),
        "ts_ns": time.time_ns(),
        "run_id": getattr(args, "run_id", None),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "component": component,
        "reason": reason,
        "error": repr(error) if error is not None else None,
    }
    try:
        _append_jsonl(args.observability_errors_path, payload)
        update_observability_component_state(
            args,
            component,
            {
                "state": "degraded",
                "reason": reason,
                "last_error": payload["error"],
                "updated_at": payload["created_at"],
            },
        )
    except Exception:
        logger.exception("Failed to record observability error for component %s.", component)


def update_observability_component_state(args, component: str, update: dict) -> None:
    if not getattr(args, "observability_enabled", False):
        return

    path = getattr(args, "observability_component_state_path", None)
    if path is None:
        return

    try:
        state = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                state = json.load(handle)
        components = state.setdefault("components", {})
        current = components.setdefault(component, {})
        current.update(update)
        state["updated_at"] = _utc_now()
        _write_json_atomic(path, state)
    except Exception:
        logger.exception("Failed to update observability component state for %s.", component)


def _build_sglang_router_target_payload(args, target: str, metrics_endpoint: str) -> list[dict]:
    return [
        {
            "targets": [target],
            "labels": {
                "component": "sglang",
                "metrics_endpoint": metrics_endpoint,
                "role": "router",
                "run_id": getattr(args, "run_id", ""),
            },
        }
    ]


def _initial_component_state(args) -> dict:
    now = _utc_now()
    return {
        "schema_version": "slime.observability.component_state.v1",
        "run_id": args.run_id,
        "updated_at": now,
        "components": {
            "manifest": {"state": "ok", "path": args.observability_manifest_path, "updated_at": now},
            "prometheus_config": {"state": "ok", "path": args.observability_prometheus_config, "updated_at": now},
            "prometheus_file_sd": {
                "state": "pending_target",
                "router_metrics_target_file": args.observability_sglang_router_metrics_target_file,
                "engine_metrics_target_file": args.observability_sglang_router_engine_metrics_target_file,
                "updated_at": now,
            },
            "request_metrics": {
                "state": "configured",
                "path_template": args.observability_sglang_request_metrics_dir,
                "privacy_mode": args.observability_request_metrics_privacy_mode,
                "updated_at": now,
            },
            "request_logs": {
                "state": "configured",
                "path_template": args.observability_sglang_request_log_dir,
                "updated_at": now,
            },
            "request_time_stats": {
                "state": "configured",
                "path_template": args.observability_sglang_request_time_stats_log_dir,
                "updated_at": now,
            },
        },
    }


def _append_jsonl(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
        handle.write("\n")


def _is_manifest_config_allowed(key: str) -> bool:
    return key in _MANIFEST_CONFIG_ALLOWLIST or any(
        key.startswith(prefix) for prefix in _MANIFEST_CONFIG_PREFIX_ALLOWLIST
    )


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SECRET_FRAGMENTS)


def _abspath_template(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_json_atomic(path: str, payload) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    _write_text_atomic(path, f"{text}\n")


def _write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)
