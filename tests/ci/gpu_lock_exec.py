#!/usr/bin/env python3
import argparse
import fcntl
import os
import random
import sys
import time

SLEEP_BACKOFF = 5.0


def main():
    """
    Remark: Can use `lslocks` to debug
    """
    args = _parse_args()

    if args.print_only:
        _execute_print_only(args)
        return

    fd_locks = _try_acquire(args)

    dev_list = ",".join(str(x.gpu_id) for x in fd_locks)
    os.environ[args.target_env_name] = dev_list
    print(f"[gpu_lock_exec] Acquired GPUs: {dev_list}", flush=True)

    _os_execvp(args)


def _os_execvp(args):
    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]
    os.execvp(cmd[0], cmd)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=None, help="Acquire this many GPUs (any free ones)")
    p.add_argument("--devices", type=str, default=None, help="Comma separated explicit devices to acquire (e.g. 0,1)")
    p.add_argument("--total-gpus", type=int, default=8, help="Total GPUs on the machine")
    p.add_argument("--timeout", type=int, default=3600 * 24, help="Seconds to wait for locks before failing")
    p.add_argument(
        "--target-env-name", type=str, default="CUDA_VISIBLE_DEVICES", help="Which env var to set for devices"
    )
    p.add_argument(
        "--lock-path-pattern",
        type=str,
        default="/dev/shm/custom_gpu_lock_{gpu_id}.lock",
        help='Filename pattern with "{gpu_id}" placeholder',
    )
    p.add_argument("--print-only", action="store_true", help="Probe free devices and print them (does NOT hold locks)")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to exec after '--' (required unless --print-only)")
    args = p.parse_args()

    if "{gpu_id}" not in args.lock_path_pattern:
        raise Exception("ERROR: --lock-path-pattern must contain '{i}' placeholder.")

    if not args.cmd and not args.print_only:
        raise Exception("ERROR: missing command to run. Use -- before command.")

    return args


def _execute_print_only(args):
    free = []
    _ensure_lock_files(path_pattern=args.lock_path_pattern, total_gpus=args.total_gpus)
    for i in range(args.total_gpus):
        try:
            fd_lock = FdLock(args.lock_path_pattern, i)
            fd_lock.open()
            try:
                fd_lock.lock()
                fcntl.flock(fd_lock.fd, fcntl.LOCK_UN)
                free.append(i)
            except BlockingIOError:
                pass
            fd_lock.close()
        except Exception as e:
            print(f"Warning: Error while probing lock: {e}", file=sys.stderr, flush=True)

    print("Free GPUs:", ",".join(str(x) for x in free), flush=True)


def _try_acquire(args):
    if args.devices:
        devs = _parse_devices(args.devices)
        return _try_acquire_specific(devs, args.lock_path_pattern, args.timeout)
    else:
        return _try_acquire_count(args.count, args.total_gpus, args.lock_path_pattern, args.timeout)


def _try_acquire_specific(devs: list[int], path_pattern: str, timeout: int):
    fd_locks = []
    start = time.time()
    try:
        _ensure_lock_files(path_pattern, max(devs) + 1)
        for gpu_id in devs:
            fd_lock = FdLock(path_pattern, gpu_id=gpu_id)
            fd_lock.open()
            while True:
                try:
                    fd_lock.lock()
                    break
                except BlockingIOError as e:
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Timeout while waiting for GPU {gpu_id}") from e
                    time.sleep(SLEEP_BACKOFF * random.random())
            fd_locks.append(fd_lock)
        return fd_locks
    except Exception as e:
        print(f"Error during specific GPU acquisition: {e}", file=sys.stderr, flush=True)
        for fd_lock in fd_locks:
            fd_lock.close()
        raise


def _try_acquire_count(count: int, total_gpus: int, path_pattern: str, timeout: int):
    start = time.time()
    _ensure_lock_files(path_pattern, total_gpus)
    while True:
        fd_locks: list = []
        for gpu_id in range(total_gpus):
            fd_lock = FdLock(path_pattern, gpu_id=gpu_id)
            fd_lock.open()
            try:
                fd_lock.lock()
            except BlockingIOError:
                fd_lock.close()
                continue

            fd_locks.append(fd_lock)
            if len(fd_locks) == count:
                return fd_locks

        gotten_gpu_ids = [x.gpu_id for x in fd_locks]
        for fd_lock in fd_locks:
            fd_lock.close()
        del fd_lock

        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout acquiring {count} GPUs (out of {total_gpus})")

        print(f"[gpu_lock_exec] try_acquire_count failed, sleep and retry (only got: {gotten_gpu_ids})", flush=True)
        time.sleep(SLEEP_BACKOFF * random.random())


class FdLock:
    def __init__(self, path_pattern, gpu_id: int):
        self.gpu_id = gpu_id
        self.path = _get_lock_path(path_pattern, self.gpu_id)
        self.fd = None

    def open(self):
        assert self.fd is None
        self.fd = open(self.path, "a+")
        # try to avoid lock disappear when execvp
        os.set_inheritable(self.fd.fileno(), True)

    def lock(self):
        assert self.fd is not None
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        assert self.fd is not None
        try:
            self.fd.close()
        except Exception as e:
            print(f"Warning: Failed to close file descriptor: {e}", file=sys.stderr, flush=True)
        self.fd = None


def _ensure_lock_files(path_pattern: str, total_gpus: int):
    lock_dir = os.path.dirname(path_pattern)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    for gpu_id in range(total_gpus):
        p = _get_lock_path(path_pattern, gpu_id)
        try:
            open(p, "a").close()
        except Exception as e:
            print(f"Warning: Could not create lock file {p}: {e}", file=sys.stderr, flush=True)


def _get_lock_path(path_pattern: str, gpu_id: int) -> str:
    return path_pattern.format(gpu_id=gpu_id)


def _parse_devices(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    main()
