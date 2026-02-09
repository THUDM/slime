"""Fully-async colocate rollout: async generation on shared GPUs with offload/onload.

The key idea:
  1. Rollout engines run continuously in the background, generating samples.
  2. Once enough samples are collected for a training batch, we:
     a. Abort all in-flight generation requests (to release GPU quickly).
     b. Offload rollout engines (free VRAM for training).
     c. Run training on the same GPUs.
     d. Offload training (free VRAM for inference).
     e. Onload rollout engines and resume generation.

The main challenge is that training can take hundreds of seconds, during which
any in-flight HTTP requests to sglang would time out. We solve this by
**aborting all requests before offloading**, so there are no dangling requests.
After training completes and rollout engines are back online, we simply resume
generation from where we left off.
"""

import asyncio
import atexit
import logging
import queue
import threading
import time

from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer):
    """Get or create the global async worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.is_alive():
            logger.info("Creating new global async colocate worker...")
            _global_worker = AsyncColocateWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop the global worker (called at process exit)."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(stop_global_worker)


class AsyncColocateWorker:
    """Background worker that generates samples asynchronously.

    Unlike the vanilla fully-async worker, this worker can be **paused** and
    **resumed** so that we can offload the rollout engines for training.

    Lifecycle for each training step:
        1. ``resume()`` – start (or continue) generation.
        2. Poll ``output_queue`` until we have enough groups.
        3. ``pause()`` – abort in-flight requests and wait for quiescence.
        4. (caller offloads rollout, trains, offloads train, onloads rollout)
        5. Go to 1.
    """

    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.output_queue: queue.Queue = queue.Queue(maxsize=5000)

        # Control flags
        self._running = True  # overall lifecycle
        self._paused = threading.Event()  # set = paused
        self._paused.set()  # start in paused state; caller must resume()
        self._quiesced = threading.Event()  # set = all tasks drained after pause
        self._quiesced.set()

        self._thread: threading.Thread | None = None
        self._state: GenerateState | None = None
        self._current_rollout_id: int = -1  # Track current rollout step

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Started async colocate worker thread")

    def stop(self):
        self._running = False
        self._paused.clear()  # un-pause so the loop can exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        logger.info("Stopped async colocate worker thread")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def resume(self, rollout_id: int = -1):
        """Un-pause the worker so it starts submitting generation tasks.

        Args:
            rollout_id: Current training step id for tracking partial rollout generations.
        """
        self._current_rollout_id = rollout_id
        self._quiesced.clear()
        self._paused.clear()
        logger.info(f"Resumed async colocate worker (rollout_id={rollout_id})")

    def pause(self, timeout: float = 120):
        """Pause the worker: abort in-flight requests and wait until quiesced.

        Args:
            timeout: Maximum seconds to wait for all in-flight tasks to finish.
        """
        logger.info("Pausing async colocate worker (aborting in-flight requests)...")
        self._paused.set()
        # Wait for the background loop to finish draining
        if not self._quiesced.wait(timeout=timeout):
            logger.warning(f"Pause did not quiesce within {timeout}s – proceeding anyway")
        logger.info("Async colocate worker paused and quiesced")

    def drain_completed(self) -> list[list[Sample]]:
        """Drain all completed sample groups from the output queue."""
        results = []
        while True:
            try:
                results.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return results

    @property
    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def _run_loop(self):
        """Main loop running in a background thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_loop())
        finally:
            loop.close()

    async def _async_loop(self):
        """Core async loop: submit tasks when un-paused, drain when paused."""
        self._state = GenerateState(self.args)

        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.args.rollout_batch_size

        while self._running:
            # If paused, abort in-flight work and wait
            if self._paused.is_set():
                await self._drain_and_abort(active_tasks)
                active_tasks.clear()
                self._quiesced.set()
                # Busy-wait for resume (check every 100ms)
                while self._paused.is_set() and self._running:
                    await asyncio.sleep(0.1)
                if not self._running:
                    break
                # Reset state for new generation round
                self._state.reset()
                continue

            # Harvest completed tasks (non-blocking)
            if active_tasks:
                done = {t for t in active_tasks if t.done()}
                for t in done:
                    try:
                        group = t.result()
                        self.output_queue.put_nowait(group)
                    except Exception as e:
                        logger.error(f"Generation task failed: {e}")
                active_tasks -= done

            # Submit new tasks up to concurrency limit
            while len(active_tasks) < max_concurrent and not self._paused.is_set():
                try:
                    samples = self.data_buffer.get_samples(1)
                except Exception as e:
                    logger.error(f"Failed to get samples from data buffer: {e}")
                    await asyncio.sleep(1)
                    break

                for group in samples:
                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=self._state.sampling_params.copy(),
                            evaluation=False,
                        )
                    )
                    active_tasks.add(task)
                break  # one batch of samples per iteration

            await asyncio.sleep(0.1)

        # Cleanup on exit
        if active_tasks:
            await self._drain_and_abort(active_tasks)

    async def _drain_and_abort(self, active_tasks: set[asyncio.Task]):
        """Abort all in-flight sglang requests and wait for tasks to settle.

        After abort, the pending tasks will return quickly (with ABORTED status
        or partial results). We collect any already-completed results and push
        them to the output queue; aborted ones are returned to the data buffer.
        """
        if not active_tasks:
            return

        logger.info(f"Aborting {len(active_tasks)} in-flight generation tasks...")

        # Tell sglang to cancel all pending requests
        try:
            from slime.rollout.sglang_rollout import abort

            # Run abort to cancel in-flight sglang requests
            # abort() sets state.aborted = True and sends abort_request to all workers
            await abort(self.args, rollout_id=self._current_rollout_id)
        except Exception as e:
            logger.warning(f"Abort call failed (may be expected): {e}")

        # Now wait for all tasks to finish (they should return quickly after abort)
        if active_tasks:
            try:
                done, _ = await asyncio.wait(active_tasks, timeout=30)
                for t in done:
                    try:
                        group = t.result()
                        # Check if any sample was aborted
                        any_aborted = any(s.status == Sample.Status.ABORTED for s in group)
                        if not any_aborted:
                            # Completed successfully before abort - keep it
                            self.output_queue.put_nowait(group)
                        else:
                            # Return aborted samples to data buffer for retry
                            try:
                                self.data_buffer.add_samples([group])
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"Task error during drain: {e}")
            except Exception as e:
                logger.warning(f"Error waiting for tasks during drain: {e}")

            # Cancel any truly stuck tasks
            for t in active_tasks:
                if not t.done():
                    t.cancel()

        logger.info("All in-flight tasks drained")


def generate_rollout_fully_async_colocate(args, rollout_id, data_buffer, evaluation=False):
    """Rollout function for fully-async colocate mode.

    This function is called by the RolloutManager for each training step.
    It resumes the background worker, collects enough groups, then pauses
    the worker (aborting in-flight requests) so that the caller can safely
    offload the rollout engines.

    Args:
        args: Training arguments.
        rollout_id: Current training step index.
        data_buffer: The data source providing prompts.
        evaluation: Must be False (eval not supported in this mode).

    Returns:
        list[list[Sample]]: Collected sample groups for training.
    """
    if evaluation:
        raise ValueError("Evaluation is not supported in fully-async colocate mode")

    worker = get_global_worker(args, data_buffer)
    target_size = args.rollout_batch_size

    # Resume the worker – sglang engines should be online at this point
    worker.resume(rollout_id=rollout_id)

    data = []
    start_time = time.time()
    last_log_time = start_time
    log_interval = 60.0

    logger.info(f"[Step {rollout_id}] Collecting {target_size} groups (queue={worker.queue_size})...")

    while len(data) < target_size:
        completed = worker.drain_completed()
        for group in completed:
            # Skip aborted groups
            try:
                if any(s.status == Sample.Status.ABORTED for s in group):
                    try:
                        data_buffer.add_samples([group])
                    except Exception:
                        pass
                    continue
            except Exception:
                pass

            data.append(group)
            if len(data) >= target_size:
                break

        now = time.time()
        if now - last_log_time > log_interval:
            logger.info(
                f"[Step {rollout_id}] Progress: {len(data)}/{target_size}, "
                f"elapsed={now - start_time:.1f}s, queue={worker.queue_size}"
            )
            last_log_time = now

        if len(data) < target_size:
            time.sleep(0.05)

    elapsed = time.time() - start_time
    logger.info(f"[Step {rollout_id}] Collected {len(data)} groups in {elapsed:.1f}s")

    # Pause the worker – this aborts in-flight requests so we can safely offload
    worker.pause(timeout=120)

    # Sort by sample index for determinism
    data = sorted(data, key=lambda group: group[0].index)

    return data
