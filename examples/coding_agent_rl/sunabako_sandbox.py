"""Slime example adapter for the separately installed sunabako package.

SUNABAKO_CLUSTER selects allocated sandbox nodes; SUNABAKO_IMAGES maps dataset
image names to OCI rootfs bundles already imported on every selected node.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from slime.agent.sandbox import ExecResult, FileContent

logger = logging.getLogger(__name__)


class SunabakoSandbox:
    def __init__(self, image: str) -> None:
        self.image = image
        self.sandbox_id = ""
        self._sandbox = None
        self._commands: list[dict] = []
        self._workdir = "/testbed"

    async def __aenter__(self) -> SunabakoSandbox:
        from sunabako import AsyncSandbox, Cluster

        cluster = Cluster.from_file(os.environ["SUNABAKO_CLUSTER"])
        images = json.loads(Path(os.environ["SUNABAKO_IMAGES"]).read_text())
        info = images[self.image]
        self._workdir = info.get("workdir", "/testbed")
        proxy = {
            key: os.environ[key]
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
            if key in os.environ
        }
        self._sandbox = await AsyncSandbox.create(
            info["rootfs"],
            cluster=cluster,
            runtime=os.environ.get("SUNABAKO_RUNTIME", "proot"),
            network="host",
            memory_mb=int(os.environ.get("SUNABAKO_MEMORY_MB", "2048")),
            memory_mode="rss" if os.environ.get("SUNABAKO_ALLOW_TEST_MEMORY") == "1" else "cgroup",
            envs={**info.get("env", {}), **proxy},
        )
        self.sandbox_id = self._sandbox.sandbox_id
        logger.info("[sunabako] %s image=%s node=%s", self.sandbox_id, self.image, self._sandbox.node.name)
        if destination := os.environ.get("SUNABAKO_ARTIFACTS"):
            try:
                self._save_manifest(Path(destination) / self.sandbox_id)
            except Exception:
                # Record ownership before launching the CLI so an interrupted
                # trainer can clean up exactly its own remote sandboxes.
                await self._sandbox.kill()
                raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._sandbox is None:
            return
        try:
            if destination := os.environ.get("SUNABAKO_ARTIFACTS"):
                await self._save_artifacts(Path(destination) / self.sandbox_id)
        except Exception:
            logger.exception("[sunabako] could not save artifacts for %s", self.sandbox_id)
        finally:
            await self._sandbox.kill()

    def _save_manifest(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sandbox.json").write_text(
            json.dumps({"image": self.image, "sandbox_id": self.sandbox_id, "node": self._sandbox.node.name})
        )

    async def _save_artifacts(self, folder: Path) -> None:
        self._save_manifest(folder)
        (folder / "commands.json").write_text(json.dumps(self._commands, indent=2))
        try:
            info = await self._sandbox.get_info()
            (folder / "status.json").write_text(json.dumps(info.get("status", info), indent=2))
        except Exception:
            logger.exception("[sunabako] could not inspect %s before cleanup", self.sandbox_id)
        for path in (f"{self._workdir}/.harness/trajectory.jsonl", "/tmp/.harness-npm-install.out", "/tmp/.eval.out"):
            try:
                contents = await self._sandbox.read_file(path)
            except Exception:
                continue
            (folder / Path(path).name).write_text(contents)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        # sunabako never replays an exec after a transport failure.
        result = await self._sandbox.exec(cmd, user=user, env=env, timeout=timeout, check=False)
        self._commands.append({"command": cmd, "exit_code": result[0], "stdout": result[1], "stderr": result[2]})
        if check and result[0] != 0:
            raise RuntimeError(f"sunabako exec failed (exit={result[0]}): {cmd[:120]}\n{result[2][:400]}")
        return result

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        await self._sandbox.write_file(sandbox_path, content, user=user)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return await self._sandbox.read_file(sandbox_path, user=user)
