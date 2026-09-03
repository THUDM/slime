import json
import subprocess
from pathlib import Path

import pytest

from slime.agent.harbor_converter import (
    ConversionOverrides,
    UnsupportedHarborTaskError,
    convert_task,
    discover_task_dirs,
    main,
)


def _write_task(root: Path, *, name: str = "demo/fix-it", extra_toml: str = "") -> Path:
    task = root / name.replace("/", "-")
    workdir = task / "workspace"
    (task / "tests").mkdir(parents=True)
    (task / "setup_files").mkdir()
    (task / "solution").mkdir()
    workdir.mkdir()
    (task / "task.toml").write_text(
        f"""schema_version = "1.2"

[task]
name = "{name}"

[environment]
docker_image = "registry.example/harbor-task:latest"
workdir = "{workdir}"

[verifier]
timeout_sec = 5
{extra_toml}
"""
    )
    (task / "instruction.md").write_text(
        "<!-- harbor-canary GUID secret -->\n\nFix the file. Read /setup_files/context.txt first.\n"
    )
    (task / "setup_files" / "context.txt").write_text("available under /setup_files\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\npython3 /tests/check.py\n")
    (task / "tests" / "check.py").write_text(
        r"""import json
from pathlib import Path

passed = Path("fixed.txt").read_text() == "fixed\n"
reward = Path("/logs/verifier/reward.json")
reward.parent.mkdir(parents=True, exist_ok=True)
reward.write_text(json.dumps({"reward": int(passed), "detail": 0.25}))
"""
    )
    (task / "solution" / "secret.txt").write_text("NEVER_INCLUDE_ORACLE_SECRET")
    return task


def test_converted_task_materializes_setup_and_grades_hidden_tests(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    row = convert_task(task)
    metadata = row["metadata"]
    workdir = Path(metadata["workdir"])

    assert row["label"] == "demo/fix-it"
    assert row["prompt"].startswith("Fix the file.")
    assert "harbor-canary" not in row["prompt"]
    assert metadata["image"] == "registry.example/harbor-task:latest"
    assert "snapshot" not in metadata
    assert "NEVER_INCLUDE_ORACLE_SECRET" not in json.dumps(row)
    assert not (workdir / ".slime-harbor/tests").exists()

    subprocess.run(metadata["pre_commands"], cwd=workdir, shell=True, check=True)
    context = workdir / ".slime-harbor/setup_files/context.txt"
    assert context.read_text() == f"available under {context.parent}\n"

    (workdir / "fixed.txt").write_text("fixed\n")
    result = subprocess.run(metadata["eval_cmd"], cwd=workdir, shell=True, check=False)
    assert result.returncode == 0
    assert (workdir / ".slime-harbor/tests/check.py").is_file()

    (workdir / "fixed.txt").write_text("broken\n")
    result = subprocess.run(metadata["eval_cmd"], cwd=workdir, shell=True, check=False)
    assert result.returncode == 1


def test_snapshot_override_replaces_harbor_image(tmp_path: Path) -> None:
    task = _write_task(tmp_path)

    row = convert_task(task, ConversionOverrides(snapshot="snapshot-123"))

    assert row["metadata"]["snapshot"] == "snapshot-123"
    assert "image" not in row["metadata"]


def test_unsupported_features_are_reported_together(tmp_path: Path) -> None:
    task = _write_task(
        tmp_path,
        extra_toml="""
environment_mode = "separate"
env = { TOKEN = "${TOKEN}" }

[[verifier.collect]]
command = "prepare-results"
""",
    )
    with (task / "task.toml").open("a") as config:
        config.write(
            """
[agent]
user = "root"
timeout_sec = 20
"""
        )
    (task / "environment").mkdir()
    (task / "environment" / "docker-compose.yaml").write_text("services: {}\n")

    with pytest.raises(UnsupportedHarborTaskError) as caught:
        convert_task(task)

    assert set(caught.value.features) >= {
        "agent.user",
        "agent.timeout_sec",
        "verifier.env",
        "verifier.environment",
        "verifier.collect",
        "environment compose file",
    }


def test_resource_and_service_features_are_explicitly_rejected(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    config_path = task / "task.toml"
    config_path.write_text(
        config_path.read_text().replace(
            'schema_version = "1.2"',
            'schema_version = "1.2"\nartifacts = ["/logs/agent/result.json"]',
        )
    )
    with config_path.open("a") as config:
        config.write(
            """
[[environment.mcp_servers]]
name = "tools"
transport = "streamable-http"
url = "http://tools/mcp"

[environment.healthcheck]
command = "curl localhost"
"""
        )

    with pytest.raises(UnsupportedHarborTaskError) as caught:
        convert_task(task)

    assert set(caught.value.features) >= {
        "artifacts",
        "environment.mcp_servers",
        "environment.healthcheck",
    }


def test_discovery_is_deterministic_and_deduplicated(tmp_path: Path) -> None:
    second = _write_task(tmp_path, name="demo/z-task")
    first = _write_task(tmp_path, name="demo/a-task")

    discovered = discover_task_dirs([tmp_path, second / "task.toml", first])

    assert discovered == sorted({first.resolve(), second.resolve()}, key=lambda path: path.as_posix())


def test_cli_does_not_write_partial_output_on_error(tmp_path: Path) -> None:
    supported = _write_task(tmp_path, name="demo/supported")
    unsupported = _write_task(tmp_path, name="demo/unsupported")
    with (unsupported / "task.toml").open("a") as config:
        config.write("\n[agent]\nuser = 'root'\n")
    output = tmp_path / "tasks.jsonl"

    return_code = main([str(supported), str(unsupported), "--output", str(output)])

    assert return_code == 2
    assert not output.exists()


def test_cli_can_skip_unsupported_tasks(tmp_path: Path) -> None:
    supported = _write_task(tmp_path, name="demo/supported")
    unsupported = _write_task(tmp_path, name="demo/unsupported")
    with (unsupported / "task.toml").open("a") as config:
        config.write("\n[environment.tpu]\ntype = 'v6e'\ntopology = '2x4'\n")
    output = tmp_path / "tasks.jsonl"

    return_code = main([str(supported), str(unsupported), "--skip-unsupported", "--output", str(output)])

    assert return_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["label"] for row in rows] == ["demo/supported"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
