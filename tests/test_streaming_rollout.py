import asyncio
import base64
import json
import struct
import sys
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sglang_router  # noqa: F401
except ImportError:
    sys.modules["sglang_router"] = SimpleNamespace(__version__="0.3.0")

try:
    import transformers  # noqa: F401
except ImportError:
    sys.modules["transformers"] = SimpleNamespace(
        AutoProcessor=object,
        AutoTokenizer=object,
        PreTrainedTokenizerBase=object,
        ProcessorMixin=object,
    )

from slime.rollout import sglang_rollout
from slime.rollout import sglang_streaming_rollout as streaming
from slime.rollout.streaming_utils import SGLangStreamAccumulator
from slime.utils.types import Sample

NUM_GPUS = 0


def _chunk(text, pairs, output_length, **meta_info):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": pairs,
            "output_token_logprobs_length": output_length,
            **meta_info,
        },
    }


def _b64_int32(values):
    return base64.b64encode(struct.pack(f"<{len(values)}i", *values)).decode("ascii")


@pytest.mark.parametrize(
    ("output_mode", "chunks", "expected_updates", "expected"),
    [
        (
            "cumulative",
            [
                _chunk("a", [[-0.1, 11, None]], 1),
                _chunk("ab", [[-0.1, 11, None], [-0.2, 12, None]], 2),
            ],
            [[11], [12]],
            ([11, 12], [-0.1, -0.2], "ab"),
        ),
        (
            "incremental",
            [
                _chunk("a", [[-0.1, 11, None]], 1),
                _chunk("b", [[-0.2, 12, None]], 2),
                _chunk("", [], 2),
            ],
            [[11], [12], []],
            ([11, 12], [-0.1, -0.2], "ab"),
        ),
        (
            "incremental",
            [
                _chunk(None, [[-0.1, 11, None]], 1),
                _chunk(None, [[-0.2, 12, None]], 2),
            ],
            [[11], [12]],
            ([11, 12], [-0.1, -0.2], "<11><12>"),
        ),
    ],
    ids=["cumulative", "incremental", "null-text"],
)
def test_merge_stream_chunks(output_mode, chunks, expected_updates, expected):
    accumulator = SGLangStreamAccumulator(output_mode=output_mode)
    updates = []
    for chunk in chunks:
        updates.append(accumulator.add(chunk).tokens)

    decode_calls = 0

    def decode(token_ids):
        nonlocal decode_calls
        decode_calls += 1
        return "".join(f"<{token_id}>" for token_id in token_ids)

    result = (accumulator.tokens, accumulator.log_probs, accumulator.response_text(decode))

    assert updates == expected_updates
    assert result == expected
    assert decode_calls == int(chunks[-1]["text"] is None)


def test_merge_stream_rejects_inconsistent_length():
    accumulator = SGLangStreamAccumulator(
        output_mode="incremental",
        output_length=1,
        tokens=[11],
    )
    with pytest.raises(ValueError, match="output_token_logprobs_length"):
        accumulator.add(_chunk("bc", [[-0.2, 12, None], [-0.3, 13, None]], 4))


def _generation_state():
    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=False: "".join(f"<{token_id}>" for token_id in token_ids)
        ),
        processor=None,
        aborted=False,
        cancellable_tasks=set(),
        active_server_generations=0,
        pendings=set(),
        semaphore=asyncio.Semaphore(8),
        dp_rank_context=lambda: nullcontext(None),
    )


def _streaming_args():
    return SimpleNamespace(
        ci_test=False,
        sglang_router_ip="frontend",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        sglang_incremental_streaming_output=True,
        router_policy=None,
        sglang_speculative_algorithm=False,
        num_layers=1,
        moe_router_topk=1,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        custom_generate_function_path="slime.rollout.sglang_streaming_rollout.generate_streaming",
        group_rm=True,
        rollout_sample_hook_path=None,
    )


def _patch_streaming(monkeypatch, state, stream):
    monkeypatch.setattr(streaming, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: streaming.generate_streaming)
    monkeypatch.setattr(streaming, "_prepare_prompt_ids", lambda *_args: [1, 2])
    monkeypatch.setattr(streaming.http_utils, "_http_client", SimpleNamespace(stream=stream))
    monkeypatch.setattr(
        streaming,
        "trace_span",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(update=lambda *_args, **_kwargs: None)),
    )


def _http_stream(*chunks, lines=None, expected_payload=None, on_close=None):
    async def chunk_lines():
        for chunk in chunks:
            yield "data: " + json.dumps(chunk)

    response = SimpleNamespace(raise_for_status=lambda: None, aiter_lines=lines or chunk_lines)

    @asynccontextmanager
    async def stream(method, url, json, headers):
        assert (method, url, json["stream"]) == ("POST", "http://frontend:8000/generate", True)
        for key, value in (expected_payload or {}).items():
            assert json[key] == value
        try:
            yield response
        finally:
            if on_close is not None:
                on_close()

    return stream


def test_streaming_generate_selects_request_abort():
    assert streaming.generate_streaming.abort_mode == "request"


def test_stream_accumulator_requires_reported_length():
    with pytest.raises(ValueError, match="must include output_token_logprobs_length"):
        SGLangStreamAccumulator(output_mode="incremental").add(
            {"text": "x", "meta_info": {"output_token_logprobs": [[-0.1, 11, None]]}},
        )


def test_stream_accumulator_rejects_output_incompatible_with_configured_mode():
    accumulator = SGLangStreamAccumulator(output_mode="incremental")
    accumulator.add(_chunk("a", [[-0.1, 11, None]], 1))

    with pytest.raises(ValueError, match="incremental streaming output has inconsistent"):
        accumulator.add(
            _chunk(
                "ab",
                [[-0.1, 11, None], [-0.2, 12, None]],
                2,
            ),
        )


@pytest.mark.parametrize("stream_interval", [1, 20, 64])
@pytest.mark.parametrize("incremental", [True, False], ids=["incremental", "cumulative"])
@pytest.mark.parametrize("top_p_layout", ["final_full", "per_chunk"])
def test_generate_streaming_preserves_metadata_across_stream_intervals(
    monkeypatch, stream_interval, incremental, top_p_layout
):
    # Keep the response longer than the largest interval so the second chunk
    # makes cumulative versus incremental output observable on the wire.
    response_tokens = list(range(11, 76))
    chunks = []
    previous = 0
    for end in range(stream_interval, len(response_tokens) + stream_interval, stream_interval):
        end = min(end, len(response_tokens))
        start = previous if incremental else 0
        token_slice = response_tokens[start:end]
        is_final = end == len(response_tokens)
        top_p_tokens = token_slice if top_p_layout == "per_chunk" else response_tokens
        top_p_metadata = (
            {
                "top_p_token_ids": [token_id + 1000 for token_id in top_p_tokens],
                "top_p_token_offsets": list(range(len(top_p_tokens) + 1)),
            }
            if top_p_layout == "per_chunk" or is_final
            else {}
        )
        chunks.append(
            _chunk(
                "".join(f"<{token_id}>" for token_id in token_slice),
                [[-float(token_id), token_id, None] for token_id in token_slice],
                end,
                finish_reason={"type": "stop"} if is_final else None,
                **top_p_metadata,
            )
        )
        previous = end
        if end == len(response_tokens):
            break

    state = _generation_state()
    args = _streaming_args()
    args.sglang_incremental_streaming_output = incremental
    _patch_streaming(monkeypatch, state, _http_stream(*chunks))

    result = asyncio.run(
        streaming.generate_streaming(
            args,
            Sample(prompt="hello"),
            {"max_new_tokens": len(response_tokens), "skip_special_tokens": False},
        )
    )

    assert result.status == Sample.Status.COMPLETED
    assert result.tokens == [1, 2, *response_tokens]
    assert result.response_length == len(response_tokens)
    assert result.rollout_log_probs == [-float(token_id) for token_id in response_tokens]
    assert result.rollout_top_p_token_ids.tolist() == [token_id + 1000 for token_id in response_tokens]
    assert result.rollout_top_p_token_offsets.tolist() == list(range(len(response_tokens) + 1))


@pytest.mark.parametrize("incremental", [True, False], ids=["incremental", "cumulative"])
def test_generate_streaming_handles_encoded_terminal_metadata(monkeypatch, incremental):
    first = _chunk("λ", [[-0.1, 11, None]], 1)
    second = _chunk(
        "<|eot_id|>" if incremental else "λ<|eot_id|>",
        [[-0.2, 12, None]] if incremental else [[-0.1, 11, None], [-0.2, 12, None]],
        2,
    )
    terminal = _chunk(
        "" if incremental else "λ<|eot_id|>",
        [] if incremental else [[-0.1, 11, None], [-0.2, 12, None]],
        2,
        finish_reason={"type": "length"},
        top_p_token_ids=_b64_int32([1011, 1012]),
        top_p_token_offsets=_b64_int32([0, 1, 2]),
        routed_experts=_b64_int32([101, 102, 103]),
    )

    state = _generation_state()
    args = _streaming_args()
    args.use_rollout_routing_replay = True
    args.sglang_incremental_streaming_output = incremental
    stream = _http_stream(first, second, terminal, expected_payload={"return_routed_experts": True})
    _patch_streaming(monkeypatch, state, stream)

    result = asyncio.run(
        streaming.generate_streaming(
            args,
            Sample(prompt="hello"),
            {"max_new_tokens": 2, "skip_special_tokens": False},
        )
    )

    assert result.status == Sample.Status.TRUNCATED
    assert result.response == "λ<|eot_id|>"
    assert result.tokens == [1, 2, 11, 12]
    assert result.rollout_top_p_token_ids.tolist() == [1011, 1012]
    assert result.rollout_top_p_token_offsets.tolist() == [0, 1, 2]
    assert result.rollout_routed_experts.tolist() == [[[101]], [[102]], [[103]]]


def test_streaming_generator_fails_closed_on_unexpected_eof(monkeypatch):
    stream = _http_stream(_chunk("a", [[-0.1, 11, None]], 1))
    _patch_streaming(monkeypatch, _generation_state(), stream)

    with pytest.raises(RuntimeError, match="without a terminal finish_reason"):
        asyncio.run(
            streaming.generate_streaming(
                _streaming_args(),
                Sample(prompt="hello"),
                {"max_new_tokens": 2},
            )
        )


def test_server_abort_keeps_last_observed_prefix(monkeypatch):
    state = _generation_state()

    async def lines():
        state.aborted = True
        yield "data: " + json.dumps(_chunk("a", [[-0.1, 11, None]], 1))
        raise AssertionError("server-aborted stream should stop after the next observed chunk")

    _patch_streaming(monkeypatch, state, _http_stream(lines=lines))
    result = asyncio.run(
        streaming.generate_streaming(
            _streaming_args(),
            Sample(prompt="hello"),
            {"max_new_tokens": 2},
        )
    )

    assert result.status == Sample.Status.ABORTED
    assert (result.tokens, result.rollout_log_probs, result.response) == ([1, 2, 11], [-0.1], "a")
    assert not state.cancellable_tasks


def test_stream_cancellation_closes_request_and_keeps_prefix(monkeypatch):
    first_chunk_seen = asyncio.Event()
    request_closed = asyncio.Event()
    server_started = asyncio.Event()
    server_released = asyncio.Event()
    state = _generation_state()

    async def lines():
        yield "data: " + json.dumps(
            {
                "text": None,
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 11, None]],
                    "output_token_logprobs_length": 1,
                },
            }
        )
        first_chunk_seen.set()
        await asyncio.Event().wait()

    _patch_streaming(monkeypatch, state, _http_stream(lines=lines, on_close=request_closed.set))
    args = _streaming_args()
    args.custom_generate_function_path = None

    async def server_generate(args, sample, sampling_params):
        server_started.set()
        await server_released.wait()
        sample.status = Sample.Status.ABORTED
        return sample

    async def get_workers(_url):
        return {"workers": [{"url": "http://worker:30000"}]}

    async def abort_servers(urls):
        assert urls == ["http://worker:30000"]
        server_released.set()

    monkeypatch.setattr(sglang_rollout, "generate", server_generate)
    monkeypatch.setattr(sglang_rollout, "get", get_workers)
    monkeypatch.setattr(sglang_rollout, "abort_servers_until_idle", abort_servers)

    async def exercise():
        request_task = asyncio.create_task(
            sglang_rollout.generate_and_rm(
                args,
                Sample(prompt="hello", generate_function_path="streaming"),
                {"max_new_tokens": 8},
            )
        )
        server_task = asyncio.create_task(sglang_rollout.generate_and_rm(args, Sample(), {}))
        await asyncio.gather(first_chunk_seen.wait(), server_started.wait())
        await sglang_rollout.abort(args, rollout_id=0)
        return await asyncio.gather(request_task, server_task)

    result, server_result = asyncio.run(exercise())

    assert result.status == Sample.Status.ABORTED
    assert server_result.status == Sample.Status.ABORTED
    assert (result.tokens, result.rollout_log_probs, result.response) == ([1, 2, 11], [-0.1], "<11>")
    assert request_closed.is_set()
    assert not state.cancellable_tasks
    assert state.active_server_generations == 0


def test_unrelated_stream_cancellation_propagates(monkeypatch):
    request_started = asyncio.Event()
    state = _generation_state()

    async def lines():
        request_started.set()
        await asyncio.Event().wait()
        yield "unreachable"

    _patch_streaming(monkeypatch, state, _http_stream(lines=lines))
    args = _streaming_args()

    async def exercise():
        task = asyncio.create_task(
            sglang_rollout.generate_and_rm(
                args,
                Sample(prompt="hello"),
                {"max_new_tokens": 8},
            )
        )
        await request_started.wait()
        state.aborted = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert not state.cancellable_tasks


def test_partial_abort_buffers_and_resumes_only_aborted_siblings(monkeypatch):
    partial = Sample(
        tokens=[1, 11],
        response="x",
        response_length=1,
        rollout_log_probs=[-0.1],
        loss_mask=[1],
        status=Sample.Status.ABORTED,
    )
    terminal = Sample(
        tokens=[1, 21, 22],
        response="done",
        response_length=2,
        rollout_log_probs=[-0.3, -0.4],
        loss_mask=[1, 1],
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )
    empty = Sample(response="", response_length=0, status=Sample.Status.ABORTED)
    mixed_group = [terminal, partial]

    async def exercise():
        state = SimpleNamespace(
            aborted=False,
            cancellable_tasks=set(),
            active_server_generations=0,
            pendings={
                asyncio.create_task(asyncio.sleep(0, result=group)) for group in (mixed_group, [terminal], [empty])
            },
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        args = SimpleNamespace(
            partial_rollout=True,
            mask_offpolicy_in_partial_rollout=True,
            group_rm=False,
            custom_generate_function_path="test.resume_generate",
            rollout_sample_hook_path=None,
            sglang_enable_deterministic_inference=False,
            sglang_speculative_algorithm=False,
        )

        buffered_groups = await sglang_rollout.abort(args, rollout_id=7)
        assert buffered_groups == [mixed_group]

        async def resume_generate(args, sample, sampling_params):
            assert sample is partial
            sample.append_response_tokens(
                args,
                tokens=[12],
                log_probs=[-0.2],
                meta_info={"finish_reason": {"type": "stop"}},
                text="y",
            )
            return sample

        async def reward(_args, sample):
            assert sample is partial
            return 2.0

        monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: resume_generate)
        monkeypatch.setattr(sglang_rollout, "async_rm", reward)
        state.aborted = False
        state.semaphore = asyncio.Semaphore(2)
        state.dp_rank_context = lambda: nullcontext(None)

        resumed_group = await sglang_rollout.generate_and_rm_group(args, mixed_group, {"max_new_tokens": 1})
        return buffered_groups, resumed_group

    buffered_groups, resumed_group = asyncio.run(exercise())

    assert buffered_groups == [mixed_group]
    assert resumed_group == [terminal, partial]
    assert partial.metadata["start_rollout_id"] == 7
    assert terminal.metadata["start_rollout_id"] == 7
    assert terminal.status == Sample.Status.COMPLETED
    assert terminal.reward == 1.0
    assert partial.status == Sample.Status.COMPLETED
    assert partial.tokens == [1, 11, 12]
    assert partial.response == "xy"
    assert partial.response_length == 2
    assert partial.rollout_log_probs == [-0.1, -0.2]
    assert partial.loss_mask == [0, 1]
    assert partial.reward == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
