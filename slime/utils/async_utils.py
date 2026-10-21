import asyncio
import threading
from collections import deque

__all__ = ["AsyncPacer", "get_async_loop", "run"]


class AsyncPacer:
    """Reduce asyncio scheduling pressure by resuming waiting tasks in batches.

    At high concurrency, waking many tasks at once floods the loop's ready
    queue with callbacks, delaying socket I/O and timers. Release at most 64
    waiters per tick, with a 1 ms delay between batches, to spread that work
    across loop iterations. This paces task starts without limiting in-flight
    concurrency or waiting for previously released tasks to finish.

    Use each instance within a single event loop.
    """

    def __init__(self):
        self.pending = deque()
        self.scheduled = False

    async def wait(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending.append(future)
        if not self.scheduled:
            self.scheduled = True
            loop.call_soon(self._release)
        await future

    def _release(self):
        for _ in range(min(64, len(self.pending))):
            future = self.pending.popleft()
            if not future.done():
                future.set_result(None)
        if self.pending:
            # Let socket callbacks and timers run before waking more tasks.
            asyncio.get_running_loop().call_later(0.001, self._release)
        else:
            self.scheduled = False


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
