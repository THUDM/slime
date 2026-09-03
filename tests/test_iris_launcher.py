from pathlib import Path

import pytest

from infra.iris.launcher import IrisLaunchSpec, LaunchConfigError, run


NUM_GPUS = 0


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def submit(self, spec, workspace, *, wait):
        self.calls.append((spec, workspace, wait))
        return "/test/slime-job"


def base_spec(**overrides):
    values = {
        "job_name": "slime-test",
        "task_image": "registry.example/slime@sha256:abc",
        "command": ("python", "train.py"),
        "cluster": "cw-rno2a",
    }
    values.update(overrides)
    return IrisLaunchSpec(**values)


def test_spec_rejects_multiple_iris_tasks():
    with pytest.raises(LaunchConfigError, match="duplicate trainers"):
        base_spec(nodes=2).validate()


def test_spec_requires_exactly_one_controller_selection():
    with pytest.raises(LaunchConfigError, match="select exactly one"):
        base_spec(cluster=None).validate()

    with pytest.raises(LaunchConfigError, match="select exactly one"):
        base_spec(controller_url="https://iris.example").validate()


def test_spec_rejects_invalid_direct_api_values():
    with pytest.raises(LaunchConfigError, match="priority"):
        base_spec(priority="urgent").validate()

    with pytest.raises(LaunchConfigError, match="environment variable"):
        base_spec(env={"NOT-AN-ENV": "value"}).validate()


def test_secret_resolution_is_explicit_and_reports_missing_names():
    spec = base_spec(env={"PUBLIC": "visible"}, secret_env_names=("TOKEN",))
    assert spec.resolved_env({"TOKEN": "secret"}) == {"PUBLIC": "visible", "TOKEN": "secret"}

    with pytest.raises(LaunchConfigError, match="TOKEN"):
        spec.resolved_env({})


def test_redacted_request_contains_no_environment_values():
    spec = base_spec(env={"PUBLIC": "sensitive-ish"}, secret_env_names=("TOKEN",))
    rendered = spec.redacted_dict()

    assert rendered["env"] == ["PUBLIC"]
    assert rendered["secret_env_names"] == ["TOKEN"]
    assert "sensitive-ish" not in str(rendered)


def test_run_submits_validated_command_without_importing_iris(tmp_path):
    backend = RecordingBackend()
    exit_code = run(
        [
            "--cluster",
            "cw-rno2a",
            "--task-image",
            "registry.example/slime@sha256:abc",
            "--job-name",
            "slime-test",
            "--workspace",
            str(tmp_path),
            "--env",
            "RUN_NAME=test",
            "--no-wait",
            "--",
            "python",
            "train.py",
            "--actor-num-nodes",
            "1",
        ],
        backend=backend,
    )

    assert exit_code == 0
    assert len(backend.calls) == 1
    spec, workspace, wait = backend.calls[0]
    assert spec.command == ("python", "train.py", "--actor-num-nodes", "1")
    assert spec.env == {"RUN_NAME": "test"}
    assert workspace == Path(tmp_path)
    assert wait is False


def test_run_rejects_duplicate_public_environment_names(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        run(
            [
                "--cluster",
                "cw-rno2a",
                "--task-image",
                "image",
                "--workspace",
                str(tmp_path),
                "--env",
                "NAME=first",
                "--env",
                "NAME=second",
                "--",
                "true",
            ]
        )

    assert error.value.code == 2
    assert "provided more than once" in capsys.readouterr().err


def test_dry_run_checks_secret_presence_but_redacts_value(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("API_TOKEN", "do-not-print")
    exit_code = run(
        [
            "--controller-url",
            "https://iris.example",
            "--task-image",
            "image",
            "--workspace",
            str(tmp_path),
            "--secret-env",
            "API_TOKEN",
            "--dry-run",
            "--",
            "true",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "API_TOKEN" in output
    assert "do-not-print" not in output


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
