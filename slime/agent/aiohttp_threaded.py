"""Run an ``aiohttp.web.Application`` in a background daemon thread."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from aiohttp.web_log import AccessLogger


class FilteredAccessLogger(AccessLogger):
    SLOW_THRESHOLD_SEC = 120.0

    def log(self, request, response, time):
        if request.method == "HEAD":
            return
        if response.status == 200 and time <= self.SLOW_THRESHOLD_SEC:
            return
        super().log(request, response, time)


@dataclass
class AppHandle:
    host: str
    port: int
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    runner: web.AppRunner

    def stop(self) -> None:
        async def _shutdown() -> None:
            await self.runner.cleanup()

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)
            fut.result(timeout=10)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def run_app_in_thread(
    app: web.Application,
    *,
    host: str = "0.0.0.0",
    port: int = 0,
    thread_name: str = "aiohttp-app",
    start_timeout_sec: float = 15.0,
    runner_kwargs: dict[str, Any] | None = None,
) -> AppHandle:
    """Spin up ``app`` on a daemon thread; block until it is listening.

    ``runner_kwargs`` is forwarded to ``web.AppRunner`` (e.g. pass
    ``{"handler_cancellation": True}`` to make a client disconnect cancel
    the in-flight handler coroutine).
    """
    started = threading.Event()
    err_box: list[BaseException] = []
    box: dict[str, Any] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            runner = web.AppRunner(app, **(runner_kwargs or {}))
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, host, port)
            loop.run_until_complete(site.start())
            actual_port = port
            for sock in site._server.sockets:  # type: ignore[attr-defined]
                actual_port = sock.getsockname()[1]
                break
            box["loop"] = loop
            box["runner"] = runner
            box["port"] = actual_port
            started.set()
            loop.run_forever()
        except BaseException as e:  # pragma: no cover
            err_box.append(e)
            started.set()
            raise

    thread = threading.Thread(target=_run, name=thread_name, daemon=True)
    thread.start()
    started.wait(timeout=start_timeout_sec)
    if err_box:
        raise err_box[0]
    return AppHandle(
        host=host,
        port=int(box["port"]),
        thread=thread,
        loop=box["loop"],
        runner=box["runner"],
    )
