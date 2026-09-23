import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from slime.utils.score_centering import decode_score_centering_response, extract_sampler_topk, validate_sampler_topk
from slime.utils.types import Sample

NUM_GPUS = 0


def binary_meta(ids, logps):
    ids = np.asarray(ids, dtype="<i4")
    logps = np.asarray(logps, dtype="<f4")
    return {
        "output_topk_token_ids": pybase64.b64encode(ids.tobytes()).decode(),
        "output_topk_log_probs": pybase64.b64encode(logps.tobytes()).decode(),
        "output_topk_shape": list(ids.shape),
        "output_token_logprobs": [[float(row[0]), int(idx[0]), None] for row, idx in zip(logps, ids, strict=True)],
    }


def test_binary_roundtrip_and_sample_resume():
    ids = [[3, 1, 4], [2, 1, 0]]
    logps = [[-0.5, -2.0, -3.0], [-0.7, -1.5, -2.0]]
    expected = (np.asarray(ids, dtype=np.int32), np.asarray(logps, dtype=np.float32))
    info = binary_meta(ids, logps)
    info["routed_experts"] = "unchanged"
    result = decode_score_centering_response(json.dumps({"text": "x", "meta_info": info}), 3)
    actual = extract_sampler_topk(result["meta_info"], 2, 3)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert actual[0].dtype == np.int32 and actual[1].dtype == np.float32
    assert sum(x.nbytes for x in actual) == 2 * 3 * 8
    assert "output_topk_token_ids" not in result["meta_info"]
    assert result["meta_info"].pop("routed_experts") == "unchanged"
    args = SimpleNamespace(use_score_centering=True, score_centering_top_k=3, use_rollout_routing_replay=False)
    sample = Sample(tokens=[9])
    sample.append_response_tokens(args, tokens=[3, 2], log_probs=[-0.5, -0.7], meta_info=result["meta_info"])
    # Transport may hand out read-only NumPy views; continuation must not mutate them.
    for value in (sample.rollout_topk_token_ids, sample.rollout_topk_log_probs):
        value.flags.writeable = False
    sample = Sample.from_dict(sample.to_dict())
    sample.append_response_tokens(args, tokens=[8], trainable=False)
    validate_sampler_topk(sample, 3)
    assert sample.loss_mask == [1, 1, 0]
    np.testing.assert_array_equal(sample.rollout_topk_token_ids[:2], ids)


@pytest.mark.parametrize(
    "ids,logps",
    [
        ([[1, 1]], [[-1.0, -1.0]]),
        ([[-1, 2]], [[-1.0, -1.0]]),
        ([[1, 2]], [[float("nan"), -1.0]]),
        ([[1, 2]], [[0.1, -1.0]]),
        ([[1, 2]], [[-0.1, -0.1]]),
    ],
)
def test_binary_rejects_invalid_distribution(ids, logps):
    with pytest.raises(ValueError):
        extract_sampler_topk(binary_meta(ids, logps), 1, 2)


def test_binary_rejects_missing_or_misaligned_rows():
    info = binary_meta([[1, 2]], [[-1.0, -1.0]])
    with pytest.raises(ValueError, match="shape"):
        extract_sampler_topk(info, 2, 2)
    info["output_topk_log_probs"] = ""
    with pytest.raises(ValueError):
        extract_sampler_topk(info, 1, 2)


@pytest.mark.parametrize("missing", ["output_topk_token_ids", "output_topk_log_probs", "output_topk_shape"])
def test_binary_required_even_when_legacy_rows_are_present(missing):
    info = binary_meta([[1, 2]], [[-1.0, -1.0]])
    del info[missing]
    info["output_top_logprobs"] = [[[-1.0, 1, None], [-1.0, 2, None]]]
    with pytest.raises(ValueError, match="requires binary"):
        decode_score_centering_response(json.dumps({"meta_info": info}), 2)


def test_http_worker_decodes_off_event_loop(monkeypatch):
    from slime.utils import http_utils, score_centering

    main_thread = threading.get_ident()
    calls = []
    original = score_centering.decode_score_centering_response

    def decode(*args):
        calls.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(score_centering, "decode_score_centering_response", decode)
    content = json.dumps({"meta_info": binary_meta([[1, 2]], [[-1.0, -1.0]])}).encode()

    class Response:
        def raise_for_status(self):
            pass

        async def aread(self):
            return content

        async def aclose(self):
            pass

    class Client:
        async def post(self, *args, **kwargs):
            return Response()

    result = asyncio.run(
        http_utils._post(Client(), "http://sampler/generate", {}, max_retries=1, score_centering_top_k=2)
    )
    assert calls and calls[0] != main_thread
    assert result["meta_info"]["score_centering_topk"][0].tolist() == [[1, 2]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
