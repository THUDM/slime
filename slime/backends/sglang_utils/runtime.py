"""Start SGLang with its own Python using an ordinary subprocess."""

import json
import os
import re
import subprocess
import sys
import time

import psutil


def prepare_server_environment(environ):
    """Remove unsupported allocator settings and avoid Mooncake/HTTP port collisions."""
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        value = environ.get(name)
        if value is None:
            continue
        value = re.sub(r"(^|,)\s*expandable_segments\s*:\s*(?:True|False|true|false|1|0)(?=,|$)", "", value).strip(
            ", "
        )
        if value:
            environ[name] = value
        else:
            environ.pop(name, None)

    # Preserve explicit ranges, including Mooncake's legacy PRC spelling.
    if not any(f"MC_{bound}_{spelling}_PORT" in environ for bound in ("MIN", "MAX") for spelling in ("RPC", "PRC")):
        environ["MC_MIN_PRC_PORT"] = "27000"
        environ["MC_MAX_PRC_PORT"] = "28000"


def launch_server_subprocess(executable, server_args):
    # JSON keeps the child independent of the parent's imported Torch classes.
    payload = json.dumps({"server_args": server_args, "parent_pid": os.getpid()})
    environ = os.environ.copy()
    prepare_server_environment(environ)
    process = subprocess.Popen(
        [executable, "-m", "slime.backends.sglang_utils.runtime"],
        stdin=subprocess.PIPE,
        env=environ,
        text=True,
    )
    try:
        process.stdin.write(payload)
        process.stdin.close()
    except BaseException:
        process.kill()
        process.wait()
        raise
    return process


def terminate_server_process(process, timeout=30):
    """Wait for this server's process tree to exit before reusing its GPUs."""
    if isinstance(process, subprocess.Popen):
        if process.poll() is not None:
            return
    elif not process.is_alive():
        process.join(timeout=0)
        return
    try:
        parent = psutil.Process(process.pid)
        owned = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for child in owned:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + timeout
    while owned:
        alive = []
        for child in owned:
            try:
                # Orphaned zombies can await init's reaper; their CUDA resources
                # are already gone. psutil.wait_procs alone can wait on these.
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.append(child)
            except psutil.NoSuchProcess:
                pass
        owned = alive
        if not owned:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"SGLang processes still exiting: {[child.pid for child in owned]}")
        time.sleep(0.05)
    if isinstance(process, subprocess.Popen):
        process.wait(timeout=max(0, deadline - time.monotonic()))
    else:
        process.join(timeout=max(0, deadline - time.monotonic()))


def main():
    payload = json.load(sys.stdin)
    prepare_server_environment(os.environ)
    from sglang.srt.server_args import ServerArgs

    from slime.backends.sglang_utils.sglang_engine import _launch_server, _prepare_server_stdin, launch_server_process

    server_args = ServerArgs(**payload["server_args"])
    server_args.host = server_args.host.strip("[]")
    if getattr(server_args, "encoder_only", False):
        from sglang.srt.utils import kill_itself_when_parent_died

        kill_itself_when_parent_died()
        if os.getppid() != payload["parent_pid"]:
            return
        _prepare_server_stdin()
        process = launch_server_process(server_args)
        process.join()
        sys.exit(process.exitcode)
    else:
        _launch_server(server_args, payload["parent_pid"])


if __name__ == "__main__":
    main()
