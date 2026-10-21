"""Agent patches must exclude image-shipped changes and retain all agent edits."""

import asyncio
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl import swe  # noqa: E402
from examples.coding_agent_rl.sunabako_sandbox import SunabakoSandbox  # noqa: E402

NUM_GPUS = 0


class LocalShell:
    def __init__(self, login_banner=""):
        self.login_banner = login_banner

    async def exec(self, command, *, check=False, **kwargs):
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=check)
        return result.returncode, self.login_banner + result.stdout, result.stderr

    async def write_file(self, path, content, **kwargs):
        Path(path).write_text(content)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.mark.parametrize("edit_state", ["unstaged", "staged", "committed"])
@pytest.mark.parametrize("ignore_harness", [False, True])
@pytest.mark.parametrize("login_banner", [False, True])
def test_patch_uses_actual_image_baseline_without_changing_git_index(
    tmp_path, monkeypatch, edit_state, ignore_harness, login_banner
):
    repo = tmp_path / "agent repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    # Copy a stable fixture; modern Git may otherwise remove maintenance.lock
    # in the background while copytree is traversing .git.
    git(repo, "config", "maintenance.auto", "false")
    (repo / "source.py").write_text("result = 0\n")
    (repo / "deleted.py").write_text("obsolete = True\n")
    if ignore_harness:
        (repo / ".gitignore").write_text(".harness/\nPROBLEM_STATEMENT.md\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "original")
    # These are shipped by the image, before the agent starts.
    (repo / "source.py").write_text("# image setup\nresult = 0\n")
    (repo / "package-lock.json").write_text('{"image_dependency": true}\n')
    fresh = tmp_path / "grader repo"
    shutil.copytree(repo, fresh)
    if login_banner:
        shell = SunabakoSandbox("local-test")
        shell._sandbox = LocalShell("Image environment initialized\n\n")
    else:
        shell = LocalShell()
    monkeypatch.setattr(swe, "_PATCH", str(tmp_path / "agent.patch"))

    async def scenario():
        assert await shell.exec("printf 0") == (0, "0", "")
        assert await shell.exec("printf failure >&2; exit 7") == (7, "", "failure")
        original_index = (repo / ".git/index").read_bytes()
        baseline = await swe._repository_tree(shell, str(repo))
        assert (repo / ".git/index").read_bytes() == original_index
        assert await swe.git_diff(shell, str(repo), baseline) == ""

        (repo / "source.py").write_text("# image setup\nresult = 42\n")
        (repo / "new.bin").write_bytes(b"\0new binary content\xff")
        (repo / "deleted.py").unlink()
        if edit_state != "unstaged":
            git(repo, "add", "source.py", "new.bin", "deleted.py")
        if edit_state == "committed":
            git(repo, "commit", "-qm", "agent edits")
        (repo / "PROBLEM_STATEMENT.md").write_text("task text")
        (repo / ".harness").mkdir()
        (repo / ".harness/log").write_text("execution log")
        agent_index = (repo / ".git/index").read_bytes()
        agent_head = git(repo, "rev-parse", "HEAD")
        patch = await swe.git_diff(shell, str(repo), baseline)
        assert (repo / ".git/index").read_bytes() == agent_index
        assert git(repo, "rev-parse", "HEAD") == agent_head
        assert "package-lock.json" not in patch and "PROBLEM_STATEMENT" not in patch and ".harness" not in patch
        assert "GIT binary patch" in patch
        assert await swe._apply_diff(shell, str(fresh), patch)
        assert (fresh / "source.py").read_bytes() == (repo / "source.py").read_bytes()
        assert (fresh / "new.bin").read_bytes() == (repo / "new.bin").read_bytes()
        assert not (fresh / "deleted.py").exists()
        assert (fresh / "package-lock.json").read_text() == '{"image_dependency": true}\n'

    asyncio.run(scenario())


@pytest.mark.parametrize("wrapper", [None, "bash", "/bin/sh"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_large_grading_command_avoids_exec_argument_limit(tmp_path, monkeypatch, wrapper, exit_code):
    repo = tmp_path / "grader repo"
    repo.mkdir()
    monkeypatch.setattr(swe, "_EVAL_SCRIPT", str(tmp_path / "large eval.sh"))
    # Include quotes and literal expansion syntax inside a large here-document.
    payload = "quoted ' and \" text; $(false) $UNSET\n" * 24000
    command = f"cat <<'PAYLOAD' > result.txt\n{payload}PAYLOAD\nexit {exit_code}"
    if wrapper:
        command = shlex.join([wrapper, "-lc", command])
    reward = asyncio.run(swe._run_eval_cmd(LocalShell(), str(repo), command, 30))
    assert reward == (1.0 if exit_code == 0 else 0.0)
    assert (repo / "result.txt").read_text() == payload


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
