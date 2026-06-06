import json
import sys
from argparse import Namespace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.profiling import observability


def _make_args(tmp_path):
    return Namespace(
        enable_observability=True,
        observability_profile=None,
        observability_scratch_dir=None,
        observability_prometheus_tsdb_dir=None,
        observability_export_dir=None,
        run_id="test-run",
        run_dir=str(tmp_path / "run"),
        sglang_export_metrics_to_file_dir=None,
        sglang_log_requests_target=None,
        wandb_key="should-not-enter-manifest",
        prompt_data="/private/dataset.jsonl",
        rollout_batch_size=8,
        n_samples_per_prompt=2,
        global_batch_size=4,
    )


def test_observability_bundle_uses_split_paths_and_two_prometheus_jobs(tmp_path, monkeypatch):
    args = _make_args(tmp_path)
    monkeypatch.setattr(observability, "_collect_versions", lambda: {})

    observability.prepare_observability_args(args)
    observability.initialize_observability(args)
    observability.register_sglang_router(args, "http://10.0.0.1:3456")

    assert args.observability_sglang_request_metrics_dir.endswith("nodes/node={hostname}/request_metrics/sglang")
    assert args.sglang_export_metrics_to_file_dir == args.observability_sglang_request_metrics_dir
    assert args.sglang_log_requests_target == [args.observability_sglang_request_log_dir]

    prometheus_config = (tmp_path / "run" / "prometheus" / "prometheus.yml").read_text()
    assert "job_name: slime_sglang_router" in prometheus_config
    assert "metrics_path: /metrics" in prometheus_config
    assert "job_name: slime_sglang_engine_aggregated" in prometheus_config
    assert "metrics_path: /engine_metrics" in prometheus_config

    router_target = json.loads(
        (tmp_path / "run" / "prometheus" / "file_sd" / "sglang_router_metrics.json").read_text()
    )
    engine_target = json.loads(
        (tmp_path / "run" / "prometheus" / "file_sd" / "sglang_router_engine_metrics.json").read_text()
    )
    assert router_target[0]["targets"] == ["10.0.0.1:3456"]
    assert router_target[0]["labels"]["metrics_endpoint"] == "router"
    assert engine_target[0]["labels"]["metrics_endpoint"] == "engine_aggregated"

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["privacy"]["manifest_config_mode"] == "allowlist"
    assert "prompt_data" not in manifest["config"]
    assert "wandb_key" not in manifest["config"]
    assert manifest["paths"]["prometheus_tsdb_dir"].startswith("/tmp/slime-observability/test-run/")

    status = json.loads((tmp_path / "run" / "observability" / "status.json").read_text())
    assert status["state"] == "ok"
