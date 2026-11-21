import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.observability.trace_utils import TRACE_CHILDREN_KEY, build_sglang_meta_trace_attrs, trace_span  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def _load_trace_timeline_viewer_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "trace_timeline_viewer.py"
    module_name = "test_trace_timeline_viewer_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_build_sglang_meta_trace_attrs_keeps_standard_and_pd_fields():
    meta = {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "pd_prefill_forward_duration": 0.125,
        "pd_decode_transfer_duration": 0.05,
        "finish_reason": {"type": "stop"},
        "unused_field": "ignored",
    }

    attrs = build_sglang_meta_trace_attrs(meta)
    trace_children = attrs.pop(TRACE_CHILDREN_KEY)

    assert attrs == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "finish_reason": "stop",
    }
    assert trace_children[0]["name"] == "sglang_pd_prefill"
    assert trace_children[0]["children"][0]["attrs"] == {
        "pd_prefill_forward_duration": 0.125,
    }
    assert trace_children[1]["name"] == "sglang_pd_decode"
    assert trace_children[1]["children"][0]["attrs"] == {
        "pd_decode_transfer_duration": 0.05,
    }


@pytest.mark.unit
def test_trace_timeline_viewer_omits_virtual_pd_lanes_without_pd_attrs(tmp_path: Path):
    viewer = _load_trace_timeline_viewer_module()
    sample = Sample(index=0, prompt="hello")

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": 8}) as span:
        span.update(
            build_sglang_meta_trace_attrs(
                {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "cached_tokens": 1,
                    "finish_reason": {"type": "stop"},
                }
            )
        )

    pt_path = tmp_path / "rollout.pt"
    torch.save({"samples": [sample]}, pt_path)

    cache = viewer._build_cache_data(pt_path)

    assert cache["sample_count"] == 1
    row = cache["rows"][0]
    assert row["lane_count"] == 1
    assert row["item_count"] == 1
    assert row["closed_span_count"] == 1

    item = row["items"][0]
    assert item["name"] == "sglang_generate"
    assert item["attrs"]["end_attrs"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "cached_tokens": 1,
        "finish_reason": "stop",
    }
    assert "[P]" not in item["name"]
    assert "[D]" not in item["name"]


def test_trace_closes_cancelled_spans_and_jsonl_roundtrips(tmp_path):
    sample = Sample(index=3, group_index=1)
    with pytest.raises(asyncio.CancelledError):
        with trace_span(sample, "agent_run", record_error=False):
            raise asyncio.CancelledError("private endpoint must not be stored")
    events = sample.trace["events"]
    assert [event["type"] for event in events] == ["span_start", "span_end"]
    assert events[0]["span_id"] == events[1]["span_id"]
    assert "private endpoint" not in json.dumps(sample.trace)
    path = tmp_path / "coding.jsonl"
    path.write_text(json.dumps({"index": 3, "group_index": 1, "trace": sample.trace}) + "\n")
    cache = _load_trace_timeline_viewer_module()._build_cache_data(path)
    assert cache["sample_count"] == 1
    assert cache["rows"][0]["closed_span_count"] == 1
    assert cache["rows"][0]["items"][0]["name"] == "agent_run"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
