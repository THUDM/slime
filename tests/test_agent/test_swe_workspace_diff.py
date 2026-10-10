"""Agent patches must exclude image-shipped changes and retain all agent edits."""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl import swe  # noqa: E402

NUM_GPUS = 0


class LocalShell:
    async def exec(self, command, *, check=False, **kwargs):
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=check)
        return result.returncode, result.stdout, result.stderr

    async def write_file(self, path, content, **kwargs):
        Path(path).write_text(content)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.mark.parametrize("edit_state", ["unstaged", "staged", "committed"])
def test_patch_uses_actual_image_baseline_without_changing_git_index(tmp_path, monkeypatch, edit_state):
    repo = tmp_path / "agent repo"
    repo.mkdir()
    git(repo, "init", "-q")
    # Disable background auto-maintenance before copying the repository.
    git(repo, "config", "maintenance.auto", "false")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "source.py").write_text("result = 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "original")
    # These are shipped by the image, before the agent starts.
    (repo / "source.py").write_text("# image setup\nresult = 0\n")
    (repo / "package-lock.json").write_text('{"image_dependency": true}\n')
    fresh = tmp_path / "grader repo"
    shutil.copytree(repo, fresh)
    shell = LocalShell()
    monkeypatch.setattr(swe, "_PATCH", str(tmp_path / "agent.patch"))

    async def scenario():
        original_index = (repo / ".git/index").read_bytes()
        baseline = await swe._repository_tree(shell, str(repo))
        assert (repo / ".git/index").read_bytes() == original_index
        assert await swe.git_diff(shell, str(repo), baseline) == ""

        (repo / "source.py").write_text("# image setup\nresult = 42\n")
        (repo / "new.bin").write_bytes(b"\0new binary content\xff")
        if edit_state != "unstaged":
            git(repo, "add", "source.py", "new.bin")
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
        assert (fresh / "package-lock.json").read_text() == '{"image_dependency": true}\n'

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
