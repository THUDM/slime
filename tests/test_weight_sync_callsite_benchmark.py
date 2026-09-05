import copy
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_weight_sync_callsite import (
    EVIDENCE_ROLES,
    POLICIES,
    SCHEMA,
    _load_production_callsite_from_source,
    artifact_digest,
    ordered_engine_indices,
    parse_size,
    resolve_policy,
    summarize_artifacts,
    verify_artifact,
)


NUM_GPUS = 0


def test_policy_matrix_uses_production_native_limits_for_logical_four_rank_layout():
    assert resolve_policy("isolated_a", 3, 2).active_engine_indices == (0,)
    assert resolve_policy("isolated_b", 3, 2).active_engine_indices == (1,)
    assert resolve_policy("baseline_overlap", 3, 2).max_inflight_engine_groups == 0
    assert resolve_policy("current_legal", 3, 2).max_inflight_engine_groups == 3
    assert resolve_policy("candidate_serialized", 3, 2).max_inflight_engine_groups == 1
    assert resolve_policy("candidate_windowed", 3, 2).max_inflight_engine_groups == 2


def test_policy_resolution_is_generic_and_has_no_special_world_size_branch():
    resolved = resolve_policy("candidate_windowed", engine_count=7, window_size=4)

    assert resolved.active_engine_indices == tuple(range(7))
    assert resolved.max_inflight_engine_groups == 4


def test_ba_order_only_swaps_the_named_pair():
    assert ordered_engine_indices((0, 1, 2), "ab") == (0, 1, 2)
    assert ordered_engine_indices((0, 1, 2), "ba") == (1, 0, 2)


def test_parse_size_checks_exact_positive_elements():
    assert parse_size("1MiB") == 1 << 20
    assert parse_size("2.5MB") == 2_500_000
    with pytest.raises(Exception, match="positive"):
        parse_size("0")


def test_minimal_container_source_loader_uses_the_production_function():
    module = _load_production_callsite_from_source()

    assert module.update_weights_in_engine_group_waves.__module__.endswith("update_weight_from_distributed")
    assert module.update_weights_in_engine_group_waves.__code__.co_filename.endswith(
        "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
    )


def _artifact(policy, role, run_index, *, world_size=4):
    metrics = {
        name: {"p50": float(run_index + 1), "p95": float(run_index + 2), "min": 1.0, "max": 3.0}
        for name in (
            "comm_a_ms",
            "comm_b_ms",
            "rank_local_pair_makespan_ms",
            "realized_b_minus_a_launch_offset_us",
            "consumer_a_wait_ms",
            "consumer_b_wait_ms",
            "consumer_ready_ms",
            "callsite_return_ms",
            "controller_device_ready_ms",
            "step_sync_ready_ms",
        )
    }
    payload = {
        "schema": SCHEMA,
        "run_id": f"{policy}-{role}-{run_index}",
        "process_launch_id": f"launch-{policy}-{role}-{run_index}",
        "evidence_role": role,
        "policy": policy,
        "order": "ab" if run_index % 2 == 0 else "ba",
        "resolved_policy": {},
        "timing_scope": {},
        "compatibility": {
            "backend": "gloo",
            "python": "3.12.0",
            "pytorch": "2.11.0",
            "cuda_runtime": None,
            "nccl": None,
            "nccl_launch_order_implicit": None,
            "dtype": "float32",
            "message_bytes": 1024,
            "tensor_elements": 256,
            "launched_world_size": world_size,
            "engine_group_count": world_size - 1,
            "process_group_membership": [[0, rank] for rank in range(1, world_size)],
            "graph_capture": False,
            "hostname": "test-host",
            "device": None,
            "container_image_digest": None,
        },
        "iterations": [],
        "summary": metrics,
        "observed_ranks": list(range(world_size)),
        "payload_validated": True,
        "claim_limits": ["fixture"],
    }
    payload["artifact_sha256"] = artifact_digest(payload)
    return payload


def _write_campaign(tmp_path, runs_per_role=2):
    paths = []
    for policy in POLICIES:
        for role in EVIDENCE_ROLES:
            for run_index in range(runs_per_role):
                path = tmp_path / f"{policy}-{role}-{run_index}.json"
                path.write_text(json.dumps(_artifact(policy, role, run_index)))
                paths.append(path)
    return paths


def test_campaign_counts_independent_processes_not_iterations(tmp_path):
    paths = _write_campaign(tmp_path)

    result = summarize_artifacts(paths, min_runs_per_role=2)

    assert result["artifact_count"] == len(POLICIES) * len(EVIDENCE_ROLES) * 2
    assert result["selection_and_confirmation_disjoint"] is True
    assert result["automatic_policy_selection"] is False
    for policy in POLICIES:
        for role in EVIDENCE_ROLES:
            summary = result["policy_summaries"][policy][role]
            assert summary["independent_process_runs"] == 2
            assert summary["orders"] == ["ab", "ba"]


def test_campaign_fails_closed_on_too_few_independent_runs(tmp_path):
    paths = _write_campaign(tmp_path, runs_per_role=1)

    with pytest.raises(ValueError, match="requires 2 independent process runs"):
        summarize_artifacts(paths, min_runs_per_role=2)


def test_campaign_rejects_cross_cell_evidence(tmp_path):
    paths = _write_campaign(tmp_path)
    changed = json.loads(paths[-1].read_text())
    changed["compatibility"]["message_bytes"] *= 2
    changed["artifact_sha256"] = artifact_digest(changed)
    paths[-1].write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="incompatible"):
        summarize_artifacts(paths, min_runs_per_role=2)


def test_campaign_rejects_reused_process_launch_identity(tmp_path):
    paths = _write_campaign(tmp_path)
    first = json.loads(paths[0].read_text())
    second = json.loads(paths[1].read_text())
    second["process_launch_id"] = first["process_launch_id"]
    second["artifact_sha256"] = artifact_digest(second)
    paths[1].write_text(json.dumps(second))

    with pytest.raises(ValueError, match="process_launch_id values must be unique"):
        summarize_artifacts(paths, min_runs_per_role=2)


def test_artifact_digest_detects_timing_mutation():
    payload = _artifact("baseline_overlap", "selection", 0)
    verify_artifact(payload)
    mutated = copy.deepcopy(payload)
    mutated["summary"]["step_sync_ready_ms"]["p50"] += 1

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_artifact(mutated)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
