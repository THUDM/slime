import tempfile
import unittest
from pathlib import Path

import run_cpu_tests as runner

NUM_GPUS = 0


class RunCpuTestsTest(unittest.TestCase):
    def test_shards_include_each_test_once_and_keep_untagged_straw_tests(self):
        tests = [
            {"test_file": "normal_a.py", "num_gpus": 0},
            {"test_file": "normal_b.py", "num_gpus": 0, "test_args": "--mode kept"},
            {"test_file": "slow_a.py", "num_gpus": 0, "straw": True, "ci_group": 1},
            {"test_file": "slow_b.py", "num_gpus": 0, "straw": True, "ci_group": 2},
            {"test_file": "future_straw.py", "num_gpus": 0, "straw": True},
        ]

        shards = runner.build_cpu_test_shards(tests, "cpu", max_tests=2, split_straw=True)
        sharded_tests = [test for shard in shards for test in shard["tests"]]

        self.assertCountEqual(sharded_tests, tests)
        self.assertEqual(
            sum(test["test_file"] == "future_straw.py" for shard in shards for test in shard["tests"]),
            1,
        )
        future_shard = next(
            shard for shard in shards if any(test["test_file"] == "future_straw.py" for test in shard["tests"])
        )
        self.assertTrue(future_shard["straw"])

        default_shards = runner.build_cpu_test_shards(tests, "cpu", max_tests=2)
        default_tests = [test for shard in default_shards for test in shard["tests"]]
        self.assertCountEqual(default_tests, tests)
        self.assertEqual(len({shard["shard"] for shard in default_shards}), len(default_shards))

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            counts_dir = root / "counts"
            counts_dir.mkdir()
            for test in tests:
                marker = f"counts/{test['test_file']}.count"
                (tests_dir / test["test_file"]).write_text(
                    "from pathlib import Path\n"
                    f"marker = Path({marker!r})\n"
                    "count = int(marker.read_text()) if marker.exists() else 0\n"
                    "marker.write_text(str(count + 1))\n"
                )
            for shard in shards:
                self.assertEqual(runner.run_tests(shard["tests"], root=root), 0)
            for test in tests:
                marker = counts_dir / f"{test['test_file']}.count"
                self.assertEqual(marker.read_text(), "1")

    def test_runs_each_test_in_its_own_process_and_preserves_args(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "first.py").write_text(
                "import os, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:] == ['--mode', 'small', '--retry', '2']\n"
                "Path('first.pid').write_text(str(os.getpid()))\n"
            )
            (tests_dir / "second.py").write_text(
                "import os, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:] == ['--mode', 'large']\n"
                "assert int(Path('first.pid').read_text()) != os.getpid()\n"
                "Path('second.pid').write_text(str(os.getpid()))\n"
            )

            result = runner.run_tests(
                [
                    {"test_file": "first.py", "test_args": "--mode small --retry 2", "num_gpus": 0},
                    {"test_file": "second.py", "test_args": "--mode large", "num_gpus": 0},
                ],
                root=root,
            )

            self.assertEqual(result, 0)
            self.assertNotEqual((root / "first.pid").read_text(), (root / "second.pid").read_text())

    def test_continues_after_a_failed_test_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "fails.py").write_text("raise SystemExit(4)\n")
            (tests_dir / "still_runs.py").write_text("from pathlib import Path\nPath('continued').touch()\n")

            result = runner.run_tests(
                [
                    {"test_file": "fails.py", "num_gpus": 0},
                    {"test_file": "still_runs.py", "num_gpus": 0},
                ],
                root=root,
            )

            self.assertEqual(result, 1)
            self.assertTrue((root / "continued").exists())

    def test_rejects_gpu_tests_in_cpu_shards(self):
        with self.assertRaisesRegex(ValueError, "cannot run GPU test"):
            runner.run_tests([{"test_file": "gpu.py", "num_gpus": 1}])


if __name__ == "__main__":
    unittest.main()
