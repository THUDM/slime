import asyncio
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import ray  # noqa: F401
except ImportError:
    pass

try:
    from tests.plugin_contracts._shared import install_stubs
except ImportError:
    from plugin_contracts._shared import install_stubs

install_stubs(with_sglang_router=True, with_transformers=True)

from slime.backends.sglang_utils.http_endpoint import (  # noqa: E402
    normalize_rollout_http_endpoint_url,
    start_http_endpoint_rollout_servers,
)
from slime.rollout import sglang_rollout  # noqa: E402
from slime.rollout.sglang_rollout import abort, generate, get_model_url  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "ci_test": False,
        "rollout_http_endpoint_url": None,
        "rollout_http_endpoint_abort_strategy": "router-workers",
        "sglang_router_ip": "10.0.0.1",
        "sglang_router_port": 30000,
        "sglang_model_routers": None,
        "router_policy": None,
        "use_rollout_routing_replay": False,
        "partial_rollout": False,
        "mask_offpolicy_in_partial_rollout": False,
        "sglang_speculative_algorithm": None,
        "custom_rollout_request_hook_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


class _Tokenizer:
    def encode(self, prompt, add_special_tokens=False):
        assert add_special_tokens is False
        return [101, len(prompt)]


class _GenerateState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = _Tokenizer()
        self.processor = None
        self.pendings = set()
        self.aborted = False


def test_normalize_rollout_http_endpoint_url_requires_absolute_http_url():
    assert normalize_rollout_http_endpoint_url("https://rollout.example/") == "https://rollout.example"
    with pytest.raises(ValueError, match="absolute http"):
        normalize_rollout_http_endpoint_url("rollout.example")


def test_get_model_url_prefers_http_endpoint():
    args = _args(
        rollout_http_endpoint_url="https://rollout.example/base/",
        sglang_model_routers={"default": ("10.0.0.2", 30001)},
    )

    assert get_model_url(args, "default", "/generate") == "https://rollout.example/base/generate"
    assert get_model_url(args, "reward", "score") == "https://rollout.example/base/score"


def test_get_model_url_uses_model_router_without_http_endpoint():
    args = _args(sglang_model_routers={"reward": ("10.0.0.3", 30002)})

    assert get_model_url(args, "reward", "/generate") == "http://10.0.0.3:30002/generate"
    assert get_model_url(args, "missing", "/generate") == "http://10.0.0.1:30000/generate"


def test_generate_posts_to_http_endpoint(monkeypatch):
    captured = {}

    async def fake_post(url, payload, headers=None, **_kwargs):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {
            "text": " answer",
            "meta_info": {
                "output_token_logprobs": [[-0.25, 42]],
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "cached_tokens": 1,
            },
        }

    monkeypatch.setattr(sglang_rollout, "GenerateState", _GenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    args = _args(rollout_http_endpoint_url="https://rollout.example")
    sample = asyncio.run(generate(args, Sample(index=0, prompt="hi"), {"max_new_tokens": 8}))

    assert captured["url"] == "https://rollout.example/generate"
    assert captured["payload"]["input_ids"] == [101, 2]
    assert captured["payload"]["return_logprob"] is True
    assert sample.response == " answer"
    assert sample.tokens == [101, 2, 42]
    assert sample.status == Sample.Status.COMPLETED


def test_generate_request_hook_can_add_exact_weight_version(monkeypatch):
    captured = {}

    async def fake_post(url, payload, headers=None, max_retries=60, retry_sleep=1.0):
        captured["url"] = url
        captured["payload"] = payload
        captured["max_retries"] = max_retries
        captured["retry_sleep"] = retry_sleep
        return {
            "text": " answer",
            "meta_info": {
                "output_token_logprobs": [[-0.25, 42]],
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "cached_tokens": 1,
            },
        }

    monkeypatch.setattr(sglang_rollout, "GenerateState", _GenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    def hook(args, sample, request):
        assert args.rollout_http_endpoint_url == "https://rollout.example"
        assert sample.index == 0
        assert request["rollout_id"] == 9
        assert request["evaluation"] is False
        request["payload"]["weight_version"] = {"exact_version": request["rollout_id"]}
        request["max_retries"] = 123
        request["retry_sleep"] = 0.25

    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: hook)

    args = _args(
        rollout_http_endpoint_url="https://rollout.example",
        custom_rollout_request_hook_path="example.hook",
    )
    with sglang_rollout.rollout_request_context(args, rollout_id=9):
        sample = asyncio.run(generate(args, Sample(index=0, prompt="hi"), {"max_new_tokens": 8}))

    assert captured["url"] == "https://rollout.example/generate"
    assert captured["payload"]["weight_version"] == {"exact_version": 9}
    assert captured["max_retries"] == 123
    assert captured["retry_sleep"] == 0.25
    assert sample.status == Sample.Status.COMPLETED


def test_generate_request_hook_can_return_request_updates(monkeypatch):
    captured = {}

    async def fake_post(url, payload, headers=None, max_retries=60, retry_sleep=1.0):
        captured["url"] = url
        captured["payload"] = payload
        captured["max_retries"] = max_retries
        captured["retry_sleep"] = retry_sleep
        return {
            "text": " answer",
            "meta_info": {
                "output_token_logprobs": [[-0.25, 42]],
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "cached_tokens": 1,
            },
        }

    monkeypatch.setattr(sglang_rollout, "GenerateState", _GenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    async def hook(_args, _sample, request):
        payload = dict(request["payload"])
        payload["weight_version"] = {"min_required_version": request["rollout_id"]}
        return {
            "payload": payload,
            "max_retries": 123,
            "retry_sleep": 0.25,
        }

    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: hook)

    args = _args(
        rollout_http_endpoint_url="https://rollout.example",
        custom_rollout_request_hook_path="example.hook",
    )
    with sglang_rollout.rollout_request_context(args, rollout_id=9):
        sample = asyncio.run(generate(args, Sample(index=0, prompt="hi"), {"max_new_tokens": 8}))

    assert captured["url"] == "https://rollout.example/generate"
    assert captured["payload"]["weight_version"] == {"min_required_version": 9}
    assert captured["max_retries"] == 123
    assert captured["retry_sleep"] == 0.25
    assert sample.status == Sample.Status.COMPLETED


def test_generate_retries_until_exact_weight_version_is_available(monkeypatch):
    aiohttp_web = pytest.importorskip("aiohttp.web")
    httpx = pytest.importorskip("httpx")

    async def run():
        from slime.utils import http_utils

        attempts = []

        async def handle_generate(request):
            payload = await request.json()
            attempts.append(payload)
            assert payload["weight_version"] == {"exact_version": 11}
            if len(attempts) == 1:
                raise aiohttp_web.HTTPNotFound(text="weight version not loaded")
            if len(attempts) == 2:
                raise aiohttp_web.HTTPConflict(text="weight version still loading")
            return aiohttp_web.json_response(
                {
                    "text": " answer",
                    "meta_info": {
                        "output_token_logprobs": [[-0.25, 42]],
                        "finish_reason": {"type": "stop"},
                        "prompt_tokens": 2,
                        "cached_tokens": 0,
                    },
                }
            )

        app = aiohttp_web.Application()
        app.router.add_post("/generate", handle_generate)
        runner = aiohttp_web.AppRunner(app)
        await runner.setup()
        site = aiohttp_web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        old_client = http_utils._http_client
        old_distributed = http_utils._distributed_post_enabled
        old_post_actors = http_utils._post_actors
        client = httpx.AsyncClient(timeout=httpx.Timeout(None), trust_env=False)
        http_utils._http_client = client
        http_utils._distributed_post_enabled = False
        http_utils._post_actors = []
        try:
            monkeypatch.setattr(sglang_rollout, "GenerateState", _GenerateState)

            def hook(_args, _sample, request):
                request["payload"]["weight_version"] = {"exact_version": request["rollout_id"]}
                request["max_retries"] = 5
                request["retry_sleep"] = 0.01

            monkeypatch.setattr(sglang_rollout, "load_function", lambda path: hook)

            args = _args(
                rollout_http_endpoint_url=f"http://127.0.0.1:{port}",
                custom_rollout_request_hook_path="example.hook",
            )
            with sglang_rollout.rollout_request_context(args, rollout_id=11):
                sample = await generate(args, Sample(index=0, prompt="hi"), {"max_new_tokens": 8})
        finally:
            await client.aclose()
            http_utils._http_client = old_client
            http_utils._distributed_post_enabled = old_distributed
            http_utils._post_actors = old_post_actors
            await runner.cleanup()

        assert len(attempts) == 3
        assert sample.status == Sample.Status.COMPLETED
        assert sample.tokens == [101, 2, 42]

    asyncio.run(run())


def test_cancel_only_abort_does_not_query_router_workers(monkeypatch):
    async def run():
        async def never_finishes():
            await asyncio.sleep(60)

        task = asyncio.create_task(never_finishes())
        state = _GenerateState(_args())
        state.pendings.add(task)

        def fake_state(_args):
            return state

        async def fail_get(_url):
            raise AssertionError("cancel-only abort must not query router workers")

        monkeypatch.setattr(sglang_rollout, "GenerateState", fake_state)
        monkeypatch.setattr(sglang_rollout, "get", fail_get)

        result = await abort(_args(rollout_http_endpoint_abort_strategy="cancel-only"), rollout_id=7)

        assert result == []
        assert state.pendings == set()
        assert task.cancelled()

    asyncio.run(run())


def test_start_http_endpoint_rollout_servers_returns_no_engine_server():
    args = _args(rollout_http_endpoint_url="https://rollout.example/", rollout_num_engines=None)

    servers = start_http_endpoint_rollout_servers(args)

    server = servers["default"]
    assert args.rollout_http_endpoint_url == "https://rollout.example"
    assert args.rollout_num_engines == 1
    assert server.engines == []
    assert server.server_groups == []
    assert server.router_ip is None
