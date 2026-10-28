import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

NUM_GPUS = 0

HELPER = Path(__file__).with_name("gpu_lock_exec.py")
GRANDCHILD = """
import os
import signal
import sys
import time
from pathlib import Path

def terminate(signum, frame):
    Path(sys.argv[2]).touch()
    if sys.argv[3] != "stubborn":
        raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
record = Path(sys.argv[1])
record.with_suffix(".tmp").write_text(str(os.getpid()))
record.with_suffix(".tmp").replace(record)
while True:
    time.sleep(0.1)
"""
CHILD = """
import os
import subprocess
import sys
import time
from pathlib import Path

record = Path(sys.argv[1])
record.with_suffix(".tmp").write_text(str(os.getpgrp()))
record.with_suffix(".tmp").replace(record)
subprocess.Popen(
    [sys.executable, "-c", sys.argv[2], *sys.argv[3:6]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while True:
    if Path(sys.argv[6]).exists():
        raise SystemExit(0)
    time.sleep(0.1)
"""


@unittest.skipUnless(sys.platform == "linux", "process-state checks use Linux /proc")
class GpuLockCancellationTest(unittest.TestCase):
    def wait_for(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("timed out waiting for the command process state")
            time.sleep(0.01)

    def is_running(self, pid):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return False
        return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")

    def assert_lock_held(self, lock_path):
        with lock_path.open("a+") as lock:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def check_cancellation(self, stubborn, exit_first=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group_record = root / "group.pid"
            descendant_record = root / "descendant.pid"
            signal_record = root / "terminated"
            exit_request = root / "exit"
            lock_path = root / "gpu-0.lock"
            wrapper = subprocess.Popen(
                [
                    sys.executable,
                    str(HELPER),
                    "--count",
                    "1",
                    "--total-gpus",
                    "1",
                    "--lock-path-pattern",
                    str(root / "gpu-{gpu_id}.lock"),
                    "--",
                    sys.executable,
                    "-c",
                    CHILD,
                    str(group_record),
                    GRANDCHILD,
                    str(descendant_record),
                    str(signal_record),
                    "stubborn" if stubborn else "cooperative",
                    str(exit_request),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            group_id = None
            try:
                self.wait_for(group_record.exists)
                group_id = int(group_record.read_text())
                self.wait_for(descendant_record.exists)
                descendant_pid = int(descendant_record.read_text())
                self.assert_lock_held(lock_path)

                if exit_first:
                    wrapper.send_signal(signal.SIGSTOP)
                    self.wait_for(
                        lambda: Path(f"/proc/{wrapper.pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "T"
                    )
                    exit_request.touch()
                    self.wait_for(lambda: not self.is_running(group_id))
                wrapper.send_signal(signal.SIGTERM)
                if exit_first:
                    wrapper.send_signal(signal.SIGCONT)
                self.wait_for(lambda: signal_record.exists() or wrapper.poll() is not None)
                if stubborn:
                    self.assertIsNone(wrapper.poll(), "launcher exited while the command group was still running")
                    self.assertTrue(self.is_running(descendant_pid))
                    self.assert_lock_held(lock_path)

                stdout, stderr = wrapper.communicate(timeout=15)
                self.assertEqual(wrapper.returncode, 128 + signal.SIGTERM, (stdout, stderr))
                self.assertFalse(
                    self.is_running(descendant_pid), "launcher exited while a command descendant was running"
                )
                with lock_path.open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if group_id is not None:
                    try:
                        os.killpg(group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if wrapper.poll() is None:
                    wrapper.kill()
                wrapper.communicate(timeout=5)

    def test_cancellation_stops_the_command_group(self):
        self.check_cancellation(stubborn=False)

    def test_stubborn_descendant_keeps_the_lock_until_it_is_killed(self):
        self.check_cancellation(stubborn=True)

    def test_cancellation_cleans_up_after_the_direct_child_has_exited(self):
        self.check_cancellation(stubborn=False, exit_first=True)

    def test_normal_exit_status_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(HELPER),
                    "--count",
                    "1",
                    "--total-gpus",
                    "1",
                    "--lock-path-pattern",
                    str(Path(directory) / "gpu-{gpu_id}.lock"),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 7, (result.stdout, result.stderr))


if __name__ == "__main__":
    unittest.main()
