from pathlib import Path

from slime.profiling.request_time_stats import load_request_time_stats, parse_request_time_stats_record


def test_parse_request_time_stats_record_accepts_flat_and_nested_attrs():
    parsed = parse_request_time_stats_record(
        {
            "request_id": "req-1",
            "schema_version": "custom.request_time_stats.v1",
            "attrs": {"pd_prefill_forward_duration": 0.25},
            "pd_decode_forward_duration": 0.5,
        },
        source="/tmp/stats.jsonl",
    )

    assert parsed == (
        "req-1",
        {
            "schema_version": "custom.request_time_stats.v1",
            "pd_prefill_forward_duration": 0.25,
            "pd_decode_forward_duration": 0.5,
            "request_time_stats_source": "/tmp/stats.jsonl",
        },
    )


def test_load_request_time_stats_reads_structured_jsonl_and_sglang_text(tmp_path: Path):
    stats_path = tmp_path / "stats.jsonl"
    stats_path.write_text(
        "\n".join(
            [
                '{"rid":"rid-1","pd_decode_forward_duration":0.75}',
                (
                    "ReqTimeStats(rid=rid-1, type=prefill, input_len=8, output_len=2): "
                    "bootstrap_duration=10.00ms, alloc_wait_duration=5.00ms, "
                    "forward_duration=20.00ms, transfer_speed=3.50GB/s"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, summary = load_request_time_stats(stats_path)

    assert summary["record_count"] == 2
    assert records["rid-1"]["pd_decode_forward_duration"] == 0.75
    assert records["rid-1"]["pd_prefill_forward_duration"] == 0.02
    assert records["rid-1"]["pd_bootstrap_duration"] == 0.01
    assert records["rid-1"]["pd_alloc_waiting_duration"] == 0.005
    assert records["rid-1"]["pd_transfer_speed_gb_s"] == 3.5
