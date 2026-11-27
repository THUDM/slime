import asyncio

import httpx
import pytest

import slime.utils.http_utils as http_utils

NUM_GPUS = 0


def _post_with_status(monkeypatch, status_code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status_code, json={"error": "test"})

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(http_utils.asyncio, "sleep", no_sleep)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await http_utils._post(client, "http://engine/generate", {"text": "x"}, max_retries=3)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(run())
    assert exc_info.value.response.status_code == status_code
    return len(calls)


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_post_does_not_retry_client_errors(monkeypatch, status_code):
    assert _post_with_status(monkeypatch, status_code) == 1


@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
def test_post_retries_transient_errors(monkeypatch, status_code):
    assert _post_with_status(monkeypatch, status_code) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
