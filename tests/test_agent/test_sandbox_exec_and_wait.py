"""CPU tests for ``slime.agent.sandbox.exec_and_wait`` process lifetime.

The detached spawn is guarded by ``mkdir {lock_dir} || exit 0`` so that a
transport-level retry of the *same* spawn RPC (a severed response replayed by
``E2BSandbox._rpc_retry``) cannot double-execute the command. That guard must
not leak into the next *logical* invocation of the same tag: the lock dir used
to survive forever, so a second ``exec_and_wait`` with the same tag skipped the
spawn at the guard (before the stale-marker cleanup, which sat behind it) and
``_await_done_marker`` immediately read the previous run's exit-code marker.

Concrete victim: ``harness.common.install_npm_cli`` retries a failed npm
install three times with ``tag="harness-npm-install"`` — attempts 2 and 3 ran
nothing and returned attempt 1's exit code and log verbatim.

The fake sandbox here interprets the shell operations ``exec_and_wait`` issues
(mkdir guard short-circuit, marker cleanup, setsid launch, process-group
signals, and marker polls).  Tests assert the resulting process state rather
than merely looking for shell snippets in the command log.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import slime.agent.sandbox as sandbox_mod
from slime.agent.sandbox import exec_and_wait

NUM_GPUS = 0


_POLL_RE = re.compile(r"test -f (\S+) && cat \1")
_LOCK_RE = re.compile(r"mkdir (\S+) 2>/dev/null")
_DETACHED_LAUNCH_RE = re.compile(r"(setsid bash .*?< /dev/null > \S+ 2>&1 &)")
_PID_WAIT_RE = re.compile(r"test -s (\S+)")
_TAIL_RE = re.compile(r"tail -c \d+ (\S+)")
_TERMINATE_RE = re.compile(r"pid=\$\(cat (\S+) 2>/dev/null\)")


@dataclass
class FakeProcessGroup:
    leader_pid: int
    alive: bool = True
    ignore_term: bool = False
    signals: list[str] = field(default_factory=list)


class ShellFakeSandbox:
    """Interprets exec_and_wait's shell commands against an in-memory FS.

    ``run_script`` is called once per *actual* launch with the launcher path;
    it returns ``(exit_code, output)`` which land in the done/out marker files
    exactly like the real detached command would write them.
    """

    sandbox_id = "shell-fake"

    def __init__(
        self,
        run_script,
        *,
        ignore_term=False,
        terminate_error=None,
        delay_pid_write=False,
        block_spawn_response=False,
    ):
        self.run_script = run_script
        self.ignore_term = ignore_term
        self.terminate_error = terminate_error
        self.delay_pid_write = delay_pid_write
        self.block_spawn_response = block_spawn_response
        self.files: dict[str, str] = {}
        self.dirs: set[str] = set()
        self.launches = 0
        self.exec_log: list[str] = []
        self.process_groups: dict[int, FakeProcessGroup] = {}
        self.removed_files: list[str] = []
        self.spawned = asyncio.Event()
        self.allow_pid_write = asyncio.Event()
        self.allow_spawn_response = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def write_file(self, path, content, *, user="root"):
        self.files[path] = content

    async def read_file(self, path, *, user="root"):
        return self.files.get(path, "")

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        self.exec_log.append(cmd)

        poll = _POLL_RE.search(cmd)
        if poll:
            path = poll.group(1)
            if path in self.files:
                return 0, self.files[path], ""
            return 1, "", ""

        tail = _TAIL_RE.search(cmd)
        if tail:
            return 0, self.files.get(tail.group(1), "")[-512:], ""

        terminate = _TERMINATE_RE.search(cmd)
        if terminate:
            if self.terminate_error is not None:
                raise self.terminate_error
            pid_file = terminate.group(1)
            if _PID_WAIT_RE.search(cmd) and pid_file not in self.files:
                await self._wait_for_pid(pid_file)
            self._terminate_process_group(pid_file, cmd)
            return 0, "", ""

        if "setsid bash " in cmd:
            lock = _LOCK_RE.search(cmd)
            launch = _DETACHED_LAUNCH_RE.search(cmd)
            assert lock and launch
            lock_dir = lock.group(1)
            if lock_dir not in self.dirs:
                self.dirs.add(lock_dir)
                self._launch(launch.group(1))

            pid_wait = _PID_WAIT_RE.search(cmd)
            if pid_wait:
                await self._wait_for_pid(pid_wait.group(1))
            if self.block_spawn_response:
                await self.allow_spawn_response.wait()
            return 0, "", ""

        # Plain cleanup command(s): rm -rf / rm -f sequences.
        self._run_shell_fragment(cmd)
        return 0, "", ""

    def _run_shell_fragment(self, fragment):
        for part in fragment.split(";"):
            part = part.strip().rstrip("&").strip()
            if part.startswith(("rm -rf", "rm -f")):
                for token in shlex.split(part)[2:]:
                    if token in self.files:
                        self.removed_files.append(token)
                    self.files.pop(token, None)
                    self.dirs.discard(token)

    def _launch(self, command):
        tokens = shlex.split(command)
        pid_file = None
        if tokens[2] == "-c":
            assert tokens[3] == 'echo $$ > "$1" && exec bash "$2"'
            pid_file, launcher = tokens[5:7]
        else:
            launcher = tokens[2]

        assert launcher in self.files, "launcher must be written before the spawn"
        self.launches += 1
        pid = 1000 + self.launches
        process_group = FakeProcessGroup(leader_pid=pid, ignore_term=self.ignore_term)
        self.process_groups[pid] = process_group
        if pid_file is not None and self.delay_pid_write:
            asyncio.create_task(self._write_pid_when_allowed(pid_file, pid))
        elif pid_file is not None:
            self.files[pid_file] = f"{pid}\n"

        out_file = tokens[tokens.index(">") + 1]
        done_file = launcher.replace(".sh", ".done")
        outcome = self.run_script(self.launches)
        if outcome is None:
            self.files[out_file] = ""
        else:
            exit_code, output = outcome
            process_group.alive = False
            self.files[out_file] = output
            self.files[done_file] = f"{exit_code}\n"
        self.spawned.set()

    async def _write_pid_when_allowed(self, pid_file, pid):
        await self.allow_pid_write.wait()
        self.files[pid_file] = f"{pid}\n"

    async def _wait_for_pid(self, pid_file):
        while pid_file not in self.files:
            await self.allow_pid_write.wait()
            await asyncio.sleep(0)

    def _terminate_process_group(self, pid_file, command):
        pid_text = self.files.get(pid_file, "").strip()
        if not pid_text.isdigit():
            return
        process_group = self.process_groups.get(int(pid_text))
        if process_group is None or not process_group.alive:
            return

        term_at = command.find('kill -TERM -- -"$pid"')
        kill_at = command.find('kill -KILL -- -"$pid"')
        if term_at >= 0:
            process_group.signals.append("TERM")
            if not process_group.ignore_term:
                process_group.alive = False
        if process_group.alive and kill_at > term_at >= 0:
            process_group.signals.append("KILL")
            process_group.alive = False

    @property
    def last_process_group(self):
        return self.process_groups[max(self.process_groups)]


@pytest.fixture
def fast_marker_polls(monkeypatch):
    """_await_done_marker sleeps 5s between polls; make that instant."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(
        sandbox_mod,
        "asyncio",
        SimpleNamespace(sleep=_instant, CancelledError=asyncio.CancelledError),
    )


@pytest.mark.unit
def test_same_tag_reinvocation_actually_reruns(fast_marker_polls):
    """A retry loop (e.g. install_npm_cli) must re-run the command, not be fed
    the previous attempt's stale exit code and log."""
    outcomes = {1: (1, "attempt-1 failed"), 2: (0, "attempt-2 ok")}
    sb = ShellFakeSandbox(run_script=lambda n: outcomes[n])

    async def _two_attempts():
        first = await exec_and_wait(sb, cmd="npm install", time_budget_sec=60, tag="npm", want_output=True)
        second = await exec_and_wait(sb, cmd="npm install", time_budget_sec=60, tag="npm", want_output=True)
        return first, second

    (first_code, first_out), (second_code, second_out) = asyncio.run(_two_attempts())

    assert (first_code, first_out) == (1, "attempt-1 failed")
    assert sb.launches == 2, "second invocation must actually spawn the command"
    assert (second_code, second_out) == (0, "attempt-2 ok")
    assert "/tmp/.npm.pid" in sb.removed_files
    assert all(not process_group.signals for process_group in sb.process_groups.values())


@pytest.mark.unit
def test_transport_retry_of_the_spawn_stays_deduped(fast_marker_polls):
    """Replaying the spawn RPC itself (what the mkdir guard is *for*) must not
    double-execute the command."""
    sb = ShellFakeSandbox(run_script=lambda n: (0, "ok"))

    asyncio.run(exec_and_wait(sb, cmd="true", time_budget_sec=60, tag="job"))

    spawn_cmds = [c for c in sb.exec_log if "setsid" in c]
    assert len(spawn_cmds) == 1
    # Replay the identical spawn RPC, as _rpc_retry would after a severed
    # response: the guard must swallow it.
    asyncio.run(sb.exec(spawn_cmds[0]))
    assert sb.launches == 1

    # The per-invocation cleanup must NOT ride inside the guarded spawn —
    # behind the guard it never runs on a replayed tag.
    assert not any("rm -" in c and "setsid" in c for c in sb.exec_log)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ignore_term", "expected_signals"),
    [(False, ["TERM"]), (True, ["TERM", "KILL"])],
)
def test_time_budget_timeout_terminates_process_group(ignore_term, expected_signals):
    sb = ShellFakeSandbox(run_script=lambda _n: None, ignore_term=ignore_term)

    exit_code, output = asyncio.run(exec_and_wait(sb, cmd="sleep 600", time_budget_sec=0, tag="timed-out"))

    assert (exit_code, output) == (sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED, "")
    assert sb.last_process_group.signals == expected_signals
    assert not sb.last_process_group.alive


@pytest.mark.unit
def test_immediate_timeout_waits_for_delayed_pid_marker_before_cleanup():
    sb = ShellFakeSandbox(run_script=lambda _n: None, ignore_term=True, delay_pid_write=True)

    async def _run_with_delayed_pid():
        task = asyncio.create_task(exec_and_wait(sb, cmd="sleep 600", time_budget_sec=0, tag="delayed-pid"))
        await sb.spawned.wait()
        await asyncio.sleep(0)
        spawn_is_waiting_for_pid = not task.done()
        sb.allow_pid_write.set()
        result = await task
        await asyncio.sleep(0)
        return spawn_is_waiting_for_pid, result

    spawn_waited, result = asyncio.run(_run_with_delayed_pid())

    assert spawn_waited
    assert result == (sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED, "")
    assert sb.last_process_group.signals == ["TERM", "KILL"]
    assert not sb.last_process_group.alive


@pytest.mark.unit
def test_completed_command_is_never_signalled(fast_marker_polls):
    sb = ShellFakeSandbox(run_script=lambda _n: (7, "failed"))

    exit_code, output = asyncio.run(
        exec_and_wait(sb, cmd="exit 7", time_budget_sec=60, tag="completed", want_output=True)
    )

    assert (exit_code, output) == (7, "failed")
    assert sb.last_process_group.signals == []
    assert not sb.last_process_group.alive


@pytest.mark.unit
def test_cancellation_terminates_process_group_and_is_reraised():
    sb = ShellFakeSandbox(run_script=lambda _n: None, ignore_term=True)

    async def _cancel_while_waiting():
        task = asyncio.create_task(exec_and_wait(sb, cmd="sleep 600", time_budget_sec=60, tag="cancelled"))
        await sb.spawned.wait()
        await asyncio.sleep(0)
        task.cancel("caller cancelled")
        # Python 3.10 does not reliably propagate Task.cancel(msg) text to the
        # final waiter, so assert the cancellation type and cleanup effects.
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancel_while_waiting())

    assert sb.last_process_group.signals == ["TERM", "KILL"]
    assert not sb.last_process_group.alive


@pytest.mark.unit
def test_cancellation_during_spawn_rpc_waits_for_pid_and_terminates_group():
    sb = ShellFakeSandbox(
        run_script=lambda _n: None,
        ignore_term=True,
        delay_pid_write=True,
        block_spawn_response=True,
    )

    async def _cancel_during_spawn():
        task = asyncio.create_task(exec_and_wait(sb, cmd="sleep 600", time_budget_sec=60, tag="cancel-spawn"))
        await sb.spawned.wait()
        task.cancel("cancelled during spawn")
        await asyncio.sleep(0)
        sb.allow_pid_write.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

    asyncio.run(_cancel_during_spawn())

    assert sb.last_process_group.signals == ["TERM", "KILL"]
    assert not sb.last_process_group.alive


@pytest.mark.unit
def test_cancellation_preserves_original_error_when_cleanup_rpc_fails(caplog):
    sb = ShellFakeSandbox(run_script=lambda _n: None, terminate_error=RuntimeError("cleanup RPC failed"))

    async def _cancel_while_waiting():
        task = asyncio.create_task(exec_and_wait(sb, cmd="sleep 600", time_budget_sec=60, tag="cancel-failure"))
        await sb.spawned.wait()
        await asyncio.sleep(0)
        task.cancel("original cancellation")
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancel_while_waiting())

    assert "process-group cleanup after cancellation failed" in caplog.text


@pytest.mark.unit
def test_timeout_logs_cleanup_rpc_failure_and_returns_timeout(caplog):
    sb = ShellFakeSandbox(run_script=lambda _n: None, terminate_error=RuntimeError("cleanup RPC failed"))

    result = asyncio.run(exec_and_wait(sb, cmd="sleep 600", time_budget_sec=0, tag="timeout-failure"))

    assert result == (sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED, "")
    assert "process-group cleanup after timeout failed" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
