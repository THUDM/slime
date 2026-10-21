"""Sandbox-local CLI hooks for a bounded compaction budget.

This file is copied into the sandbox and uses only the Python standard library.
The launcher records its CLI child PID; a budget stop may signal only a verified
ancestor of the hook, never an unrelated process or the sandbox supervisor.
"""

import fcntl
import json
import os
import signal
import sys
from pathlib import Path


def _host_process_ids(pid):
    # PRoot translates numeric /proc paths, but status retains host PID values.
    status = Path(f"/proc/{pid}/status").read_text()
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    return int(fields["Pid"]), int(fields["PPid"])


def run_hook(event, *, directory, limit, stop_mode="signal"):
    if stop_mode not in {"signal", "json"}:
        raise ValueError(f"Unknown compaction stop mode: {stop_mode}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Open without append or truncation so the locked update replaces the JSON.
    fd = os.open(directory / "compaction.json", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        state = json.loads(stream.read() or "{}")
        state.setdefault("attempts", 0)
        state.setdefault("count", 0)
        state.setdefault("limit_reached", False)
        stop = event == "PreCompact" and limit >= 0 and state["attempts"] >= limit
        if stop:
            state["limit_reached"] = True
        elif event == "PreCompact":
            state["attempts"] += 1
        elif event == "PostCompact":
            state["count"] += 1
        else:
            raise ValueError(f"Unexpected compaction hook: {event}")
        stream.seek(0)
        json.dump(state, stream)
        stream.truncate()
        stream.flush()
    if stop:
        if stop_mode == "json":
            print(json.dumps({"continue": False, "stopReason": "Compaction budget reached"}))
            return 0
        cli_pid = int((directory / "cli.pid").read_text())
        if cli_pid <= 1:
            raise RuntimeError("Compaction budget PID is not an ancestor of this hook")
        cli_host_pid, _ = _host_process_ids(cli_pid)
        ancestor, parent = _host_process_ids(os.getppid())
        while ancestor != cli_host_pid:
            if parent <= 1:
                raise RuntimeError("Compaction budget PID is not an ancestor of this hook")
            expected_parent = parent
            ancestor, parent = _host_process_ids(expected_parent)
            if ancestor != expected_parent:
                raise RuntimeError("Compaction ancestor PID resolved to a different process")
        os.kill(cli_pid, signal.SIGTERM)
        return 2
    return 0


if __name__ == "__main__":
    request = json.load(sys.stdin)
    raise SystemExit(
        run_hook(
            request["hook_event_name"],
            directory=sys.argv[1],
            limit=int(sys.argv[2]),
            stop_mode=sys.argv[3] if len(sys.argv) > 3 else "signal",
        )
    )
