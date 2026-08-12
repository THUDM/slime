"""CPU tests for ``slime.agent.sandbox.exec_and_wait``'s spawn-lock contract.

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

The fake sandbox here interprets the actual command strings ``exec_and_wait``
issues (mkdir guard short-circuit, rm cleanup, setsid launch, marker polls), so
these tests hold for any command layout that keeps the semantics right and fail
for one that reuses stale state.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from types import SimpleNamespace

import pytest

import slime.agent.sandbox as sandbox_mod
from slime.agent.sandbox import exec_and_wait


_POLL_RE = re.compile(r"test -f (\S+) && cat \1")
_SPAWN_RE = re.compile(r"mkdir (\S+) 2>/dev/null \|\| exit 0; (.*)$")
_LAUNCH_RE = re.compile(r"setsid bash (\S+) ")
_TAIL_RE = re.compile(r"tail -c \d+ (\S+)")


class ShellFakeSandbox:
    """Interprets exec_and_wait's shell commands against an in-memory FS.

    ``run_script`` is called once per *actual* launch with the launcher path;
    it returns ``(exit_code, output)`` which land in the done/out marker files
    exactly like the real detached command would write them.
    """

    sandbox_id = "shell-fake"

    def __init__(self, run_script):
        self.run_script = run_script
        self.files: dict[str, str] = {}
        self.dirs: set[str] = set()
        self.launches = 0
        self.exec_log: list[str] = []

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

        spawn = _SPAWN_RE.search(cmd)
        if spawn:
            lock_dir, rest = spawn.groups()
            if lock_dir in self.dirs:
                return 0, "", ""  # guard hit: nothing after it runs
            self.dirs.add(lock_dir)
            self._run_shell_fragment(rest)
            return 0, "", ""

        # Plain cleanup command(s): rm -rf / rm -f sequences.
        self._run_shell_fragment(cmd)
        return 0, "", ""

    def _run_shell_fragment(self, fragment):
        for part in fragment.split(";"):
            part = part.strip().rstrip("&").strip()
            if part.startswith(("rm -rf", "rm -f")):
                for token in shlex.split(part)[2:]:
                    self.files.pop(token, None)
                    self.dirs.discard(token)
            launch = _LAUNCH_RE.search(part + " ")
            if launch:
                launcher = launch.group(1)
                assert launcher in self.files, "launcher must be written before the spawn"
                self.launches += 1
                exit_code, output = self.run_script(self.launches)
                out_file = re.search(r"> (\S+) 2>&1", part).group(1)
                done_file = launcher.replace(".sh", ".done")
                self.files[out_file] = output
                self.files[done_file] = f"{exit_code}\n"


@pytest.fixture
def fast_marker_polls(monkeypatch):
    """_await_done_marker sleeps 5s between polls; make that instant."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(sandbox_mod, "asyncio", SimpleNamespace(sleep=_instant))


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
