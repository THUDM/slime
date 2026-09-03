"""Bootstrap one Ray cluster across the replicas of an Iris job.

Iris runs this module on every replica. Rank zero starts Ray's head and the
requested driver; all other ranks join as Ray workers and wait for the driver.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid


POLL_SECONDS = 5
RAY_PORT = 6379
RENDEZVOUS_FRESHNESS_SLACK = 60


def task_rank(task_id: str) -> int:
    """Return the replica index from an Iris task-attempt ID."""
    return int(task_id.rsplit("/", 1)[-1].split(":", 1)[0])


def _open(path: str, mode: str):
    if "://" not in path:
        local = Path(path)
        if "w" in mode:
            local.parent.mkdir(parents=True, exist_ok=True)
        return local.open(mode)
    import fsspec  # noqa: PLC0415

    return fsspec.open(path, mode).open()


def _exists(path: str) -> bool:
    if "://" not in path:
        return Path(path).exists()
    import fsspec  # noqa: PLC0415

    fs, remote_path = fsspec.core.url_to_fs(path)
    return fs.exists(remote_path)


def _remove(path: str) -> None:
    if "://" not in path:
        Path(path).unlink(missing_ok=True)
        return
    import fsspec  # noqa: PLC0415

    fs, remote_path = fsspec.core.url_to_fs(path)
    if fs.exists(remote_path):
        fs.rm(remote_path)


def _own_ip() -> str:
    advertised = os.environ.get("IRIS_ADVERTISE_HOST")
    if advertised and advertised != "127.0.0.1":
        return advertised
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]


def _ray_binary() -> str:
    candidate = Path(sys.executable).with_name("ray")
    return str(candidate) if candidate.exists() else "ray"


def _ray_start(*arguments: str) -> None:
    subprocess.run([_ray_binary(), "start", *arguments], check=True, timeout=300)


def _wait_for_nodes(address: str, expected: int, timeout: int, *, heartbeat=None) -> None:
    import ray  # noqa: PLC0415

    deadline = time.monotonic() + timeout
    ray.init(address=address, ignore_reinit_error=True)
    try:
        while time.monotonic() < deadline:
            if heartbeat is not None:
                heartbeat()
            if len([node for node in ray.nodes() if node.get("Alive")]) >= expected:
                return
            time.sleep(POLL_SECONDS)
    finally:
        ray.shutdown()
    raise TimeoutError(f"Ray cluster at {address} did not reach {expected} live nodes within {timeout}s")


def _paths(root: str) -> tuple[str, str]:
    base = root.rstrip("/")
    return f"{base}/ray-head.json", f"{base}/ray-head.done"


def _run_head(args: argparse.Namespace, command: list[str]) -> int:
    head_ip = _own_ip()
    address = f"{head_ip}:{args.ray_port}"
    rendezvous, done = _paths(args.rendezvous_dir)
    epoch = uuid.uuid4().hex
    _remove(rendezvous)
    _remove(done)
    _ray_start(
        "--head",
        f"--node-ip-address={head_ip}",
        f"--port={args.ray_port}",
        "--dashboard-host=0.0.0.0",
        "--disable-usage-stats",
    )

    def publish_rendezvous():
        with _open(rendezvous, "w") as destination:
            json.dump(
                {"epoch": epoch, "head_ip": head_ip, "port": args.ray_port, "written_at": time.time()},
                destination,
            )

    publish_rendezvous()
    _wait_for_nodes(address, args.num_tasks, args.cluster_join_timeout, heartbeat=publish_rendezvous)
    env = os.environ.copy()
    env["RAY_ADDRESS"] = address
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, env=env, check=False)
    if result.returncode == 0:
        with _open(done, "w") as destination:
            json.dump({"epoch": epoch, "outcome": "succeeded"}, destination)
    subprocess.run([_ray_binary(), "stop", "--force"], check=False, timeout=60)
    return result.returncode


def _poll_payload(path: str, timeout: int, *, started_at: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _exists(path):
                with _open(path, "r") as source:
                    payload = json.load(source)
                if float(payload.get("written_at", 0)) >= started_at - RENDEZVOUS_FRESHNESS_SLACK:
                    return payload
        except (OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"timed out waiting for Iris rank-zero rendezvous at {path}")


def _run_worker(args: argparse.Namespace) -> int:
    started_at = time.time()
    rendezvous, done = _paths(args.rendezvous_dir)
    payload = _poll_payload(rendezvous, args.rendezvous_timeout, started_at=started_at)
    address = f"{payload['head_ip']}:{payload['port']}"
    _ray_start(
        f"--address={address}",
        f"--node-ip-address={_own_ip()}",
        "--disable-usage-stats",
    )
    _wait_for_nodes(address, args.num_tasks, args.cluster_join_timeout)
    while True:
        if _exists(done):
            with _open(done, "r") as source:
                result = json.load(source)
            if result.get("epoch") == payload.get("epoch") and result.get("outcome") == "succeeded":
                subprocess.run([_ray_binary(), "stop", "--force"], check=False, timeout=60)
                return 0
        time.sleep(POLL_SECONDS)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendezvous-dir", required=True)
    parser.add_argument("--ray-port", type=int, default=RAY_PORT)
    parser.add_argument("--rendezvous-timeout", type=int, default=1800)
    parser.add_argument("--cluster-join-timeout", type=int, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a driver command is required after '--'")
    args.num_tasks = int(os.environ.get("IRIS_NUM_TASKS", "1"))
    rank = task_rank(os.environ.get("IRIS_TASK_ID", "0"))
    return _run_head(args, command) if rank == 0 else _run_worker(args)


def main() -> None:
    def stop(_signum, _frame):
        subprocess.run([_ray_binary(), "stop", "--force"], check=False, timeout=60)
        raise SystemExit(128 + _signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(run())


if __name__ == "__main__":
    main()
