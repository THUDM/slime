"""Exercise launcher argument quoting through Bash without starting Ray."""

import json
import os
import sys

import pytest

from slime.utils.external_utils import command_utils

NUM_GPUS = 0


@pytest.mark.parametrize("source", ["mapping", "config_json"])
@pytest.mark.parametrize(
    "value",
    [
        "plain value with spaces",
        "owner's run",
        "a 'quoted' run",
        "'$(printf expanded)' and '`printf expanded`'",
        "quotes: '\"; newline:\nbackslash: \\; dollar: $HOME",
    ],
    ids=["spaces", "apostrophe", "paired_quotes", "literal_substitutions", "mixed_characters"],
)
def test_execute_train_preserves_runtime_env_values(monkeypatch, tmp_path, source, value):
    # Record the arguments that Bash actually passes to Ray, rather than only
    # checking the launcher's command string or using a second shell parser.
    argv_path = tmp_path / "argv.json"
    ray = tmp_path / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SLIME_TEST_RAY_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    ray.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SLIME_TEST_RAY_ARGV", str(argv_path))
    monkeypatch.setenv("SLIME_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)

    environment = {"RUN_NOTE": value}
    kwargs = (
        {"extra_env_vars": environment}
        if source == "mapping"
        else {"config": command_utils.ExecuteTrainConfig(extra_env_vars=json.dumps(environment))}
    )
    command_utils.execute_train(
        "--num-rollout 1 --save 'dir with spaces'",
        num_gpus_per_node=1,
        megatron_model_type=None,
        **kwargs,
    )

    argv = json.loads(argv_path.read_text())
    assert argv[:3] == ["job", "submit", "--address=http://127.0.0.1:8265"]
    assert argv[3].startswith("--runtime-env-json=")
    runtime_env = json.loads(argv[3].split("=", 1)[1])
    assert runtime_env["env_vars"]["RUN_NOTE"] == value
    assert runtime_env["env_vars"]["RAY_USE_UVLOOP"] == "0"
    assert argv[4:] == ["--", "python3", "train.py", "--num-rollout", "1", "--save", "dir with spaces"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
