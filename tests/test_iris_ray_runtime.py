import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from infra.iris.ray_runtime import run, task_rank


NUM_GPUS = 0


@pytest.mark.parametrize(
    ("task_id", "expected"),
    [("/user/job/0:0", 0), ("/user/job/4:12", 4), ("7", 7)],
)
def test_task_rank_handles_iris_attempt_ids(task_id, expected):
    assert task_rank(task_id) == expected


def test_run_rejects_missing_driver_command(tmp_path):
    with pytest.raises(ValueError, match="driver command"):
        run(["--rendezvous-dir", str(tmp_path)])


def _fake_ray():
    return SimpleNamespace(
        init=lambda **_kwargs: None,
        nodes=lambda: [{"Alive": True}, {"Alive": True}],
        shutdown=lambda: None,
    )


def test_head_starts_cluster_runs_driver_and_publishes_success(tmp_path, monkeypatch):
    commands = []

    def run_process(command, **kwargs):
        commands.append((command, kwargs.get("env")))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run_process)
    monkeypatch.setitem(sys.modules, "ray", _fake_ray())
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/0:0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_ADVERTISE_HOST", "10.0.0.1")

    assert run(["--rendezvous-dir", str(tmp_path), "--", "train", "--steps", "2"]) == 0
    assert Path(commands[0][0][0]).name == "ray"
    assert commands[0][0][1:3] == ["start", "--head"]
    assert commands[1][0] == ["train", "--steps", "2"]
    assert commands[1][1]["RAY_ADDRESS"] == "10.0.0.1:6379"
    assert json.loads((tmp_path / "ray-head.done").read_text())["outcome"] == "succeeded"


def test_worker_joins_head_and_returns_after_matching_success(tmp_path, monkeypatch):
    epoch = "current-attempt"
    (tmp_path / "ray-head.json").write_text(
        json.dumps({"epoch": epoch, "head_ip": "10.0.0.1", "port": 6379, "written_at": time.time()})
    )
    (tmp_path / "ray-head.done").write_text(json.dumps({"epoch": epoch, "outcome": "succeeded"}))
    commands = []

    def run_process(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run_process)
    monkeypatch.setitem(sys.modules, "ray", _fake_ray())
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/1:0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_ADVERTISE_HOST", "10.0.0.2")

    assert run(["--rendezvous-dir", str(tmp_path), "--", "unused-driver"]) == 0
    assert Path(commands[0][0]).name == "ray"
    assert commands[0][1:3] == ["start", "--address=10.0.0.1:6379"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
