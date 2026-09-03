"""Submit a Slime command as an Iris GPU job.

Iris is imported only by :class:`IrisBackend`, so parsing, validation, and dry
runs work in a normal Slime development environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from collections.abc import Sequence


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SUPPORTED_PRIORITIES = ("production", "interactive", "batch")
DEFAULT_NODES = 1
DEFAULT_GPU_VARIANT = "H100"
DEFAULT_GPUS_PER_NODE = 8
DEFAULT_CPU = 64.0
DEFAULT_MEMORY = "512GB"
DEFAULT_DISK = "1TB"
DEFAULT_PRIORITY = "interactive"
DEFAULT_MAX_RETRIES_FAILURE = 0
DEFAULT_MAX_RETRIES_PREEMPTION = 1000
DEFAULT_MAX_TASK_FAILURES = 0
DEFAULT_TIMEOUT_SECONDS = 0


class LaunchConfigError(ValueError):
    """The requested launch cannot be represented safely by this launcher."""


@dataclass(frozen=True)
class IrisLaunchSpec:
    """Dependency-free description of one Slime task submitted to Iris."""

    job_name: str
    task_image: str
    command: tuple[str, ...]
    cluster: str | None = None
    cluster_config: Path | None = None
    controller_url: str | None = None
    nodes: int = DEFAULT_NODES
    gpu_variant: str = DEFAULT_GPU_VARIANT
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE
    cpu: float = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    disk: str = DEFAULT_DISK
    priority: str = DEFAULT_PRIORITY
    max_retries_failure: int = DEFAULT_MAX_RETRIES_FAILURE
    max_retries_preemption: int = DEFAULT_MAX_RETRIES_PREEMPTION
    max_task_failures: int = DEFAULT_MAX_TASK_FAILURES
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    env: dict[str, str] = field(default_factory=dict)
    secret_env_names: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    rendezvous_dir: str | None = None

    def validate(self) -> None:
        if not _JOB_NAME.fullmatch(self.job_name) or "/" in self.job_name:
            raise LaunchConfigError(
                "job name must start with an alphanumeric character and contain only letters, numbers, '.', '_', or '-'"
            )
        if not self.task_image.strip():
            raise LaunchConfigError("--task-image must not be empty")
        if not self.command or not self.command[0]:
            raise LaunchConfigError("a command is required after '--'")
        if self.nodes < 1:
            raise LaunchConfigError("--nodes must be at least 1")
        if self.nodes > 1 and not self.rendezvous_dir:
            raise LaunchConfigError("--rendezvous-dir is required when --nodes is greater than 1")
        selected_endpoints = sum(
            value is not None for value in (self.cluster, self.cluster_config, self.controller_url)
        )
        if selected_endpoints != 1:
            raise LaunchConfigError("select exactly one of --cluster, --cluster-config, or --controller-url")
        if self.cluster is not None and not self.cluster.strip():
            raise LaunchConfigError("--cluster must not be empty")
        if self.controller_url is not None and not self.controller_url.strip():
            raise LaunchConfigError("--controller-url must not be empty")
        if not self.gpu_variant.strip():
            raise LaunchConfigError("--gpu-variant must not be empty")
        if self.gpus_per_node < 1:
            raise LaunchConfigError("--gpus-per-node must be at least 1")
        if self.cpu <= 0:
            raise LaunchConfigError("--cpu must be greater than 0")
        if not self.memory.strip():
            raise LaunchConfigError("--memory must not be empty")
        if not self.disk.strip():
            raise LaunchConfigError("--disk must not be empty")
        if self.priority not in SUPPORTED_PRIORITIES:
            raise LaunchConfigError(f"--priority must be one of: {', '.join(SUPPORTED_PRIORITIES)}")
        for name, value in (
            ("--max-retries-failure", self.max_retries_failure),
            ("--max-retries-preemption", self.max_retries_preemption),
            ("--max-task-failures", self.max_task_failures),
            ("--timeout-seconds", self.timeout_seconds),
        ):
            if value < 0:
                raise LaunchConfigError(f"{name} must not be negative")
        overlap = set(self.env).intersection(self.secret_env_names)
        if overlap:
            raise LaunchConfigError(
                f"environment variables cannot be both public and secret: {', '.join(sorted(overlap))}"
            )
        invalid_names = [name for name in (*self.env, *self.secret_env_names) if not _ENV_NAME.fullmatch(name)]
        if invalid_names:
            raise LaunchConfigError(f"invalid environment variable names: {', '.join(invalid_names)}")

    def resolved_env(self, environ: dict[str, str] | None = None) -> dict[str, str]:
        """Resolve explicitly named secrets without mutating or printing them."""
        source = os.environ if environ is None else environ
        missing = [name for name in self.secret_env_names if name not in source]
        if missing:
            raise LaunchConfigError(f"missing requested secret environment variables: {', '.join(missing)}")
        return {**self.env, **{name: source[name] for name in self.secret_env_names}}

    def redacted_dict(self) -> dict[str, object]:
        """Return a dry-run representation which cannot disclose environment values."""
        rendered = asdict(self)
        rendered["cluster_config"] = str(self.cluster_config) if self.cluster_config else None
        rendered["command"] = list(self.command)
        rendered["env"] = sorted(self.env)
        rendered["secret_env_names"] = list(self.secret_env_names)
        rendered["setup_commands"] = list(self.setup_commands)
        rendered["task_command"] = list(self.task_command())
        return rendered

    def task_command(self) -> tuple[str, ...]:
        if self.nodes == 1:
            return self.command
        return (
            "python",
            "-m",
            "infra.iris.ray_runtime",
            "--rendezvous-dir",
            self.rendezvous_dir or "",
            "--",
            *self.command,
        )


class LaunchBackend(Protocol):
    """Submission boundary used by the CLI and unit tests."""

    def submit(self, spec: IrisLaunchSpec, workspace: Path, *, wait: bool) -> str:
        """Submit the specification and return its fully qualified Iris job ID."""
        ...


class IrisBackend:
    """Translate :class:`IrisLaunchSpec` into the optional Iris client API."""

    def submit(self, spec: IrisLaunchSpec, workspace: Path, *, wait: bool) -> str:
        try:
            from iris.cli.connect import open_iris_client  # noqa: PLC0415
            from iris.cluster.platforms.k8s.coreweave_topology import (  # noqa: PLC0415
                gpu_gang_coscheduling_level,
            )
            from iris.cluster.types import (  # noqa: PLC0415
                CoschedulingConfig,
                Entrypoint,
                EnvironmentSpec,
                ResourceSpec,
                gpu_device,
            )
            from iris.rpc import job_pb2  # noqa: PLC0415
            from rigging.timing import Duration  # noqa: PLC0415
        except ImportError as error:
            raise RuntimeError(
                "Iris is not installed. Run this launcher with the Iris package from marin-community/marin."
            ) from error

        priority_bands = {
            "production": job_pb2.PRIORITY_BAND_PRODUCTION,
            "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
            "batch": job_pb2.PRIORITY_BAND_BATCH,
        }
        timeout = Duration.from_seconds(spec.timeout_seconds) if spec.timeout_seconds else None
        environment = EnvironmentSpec(
            env_vars=spec.resolved_env(),
            # An empty list tells Iris to use the task image as-is. Slime's GPU
            # dependencies should be baked into that image; optional setup is explicit.
            setup_scripts=list(spec.setup_commands),
        )
        with open_iris_client(
            cluster_name=spec.cluster,
            config_file=spec.cluster_config,
            controller_url=spec.controller_url,
            workspace=workspace,
        ) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_command(*spec.task_command()),
                name=spec.job_name,
                resources=ResourceSpec(
                    cpu=spec.cpu,
                    memory=spec.memory,
                    disk=spec.disk,
                    device=gpu_device(spec.gpu_variant, spec.gpus_per_node),
                ),
                environment=environment,
                replicas=spec.nodes,
                coscheduling=(
                    CoschedulingConfig(
                        group_by=gpu_gang_coscheduling_level(spec.gpu_variant, spec.gpus_per_node, spec.nodes)
                    )
                    if spec.nodes > 1
                    else None
                ),
                max_retries_failure=spec.max_retries_failure,
                max_retries_preemption=spec.max_retries_preemption,
                max_task_failures=spec.max_task_failures,
                timeout=timeout,
                task_image=spec.task_image,
                priority_band=priority_bands[spec.priority],
                submit_argv=sys.argv,
            )
            job_id = str(job.job_id)
            print(f"Submitted Iris job {job_id}", flush=True)
            if wait:
                job.wait(timeout=float("inf"), stream_logs=True)
            return job_id


def _parse_env_assignment(value: str) -> tuple[str, str]:
    name, separator, assigned = value.partition("=")
    if not separator or not _ENV_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError("expected NAME=VALUE with a valid environment variable name")
    return name, assigned


def _parse_env_name(value: str) -> str:
    if not _ENV_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a valid environment variable name")
    return value


def _default_job_name() -> str:
    return datetime.now(timezone.utc).strftime("slime-%Y%m%d-%H%M%S")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a Slime GPU job to an Iris cluster")
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument("--cluster", help="named Iris cluster")
    endpoint.add_argument("--cluster-config", type=Path, help="Iris cluster configuration file")
    endpoint.add_argument("--controller-url", help="direct Iris controller URL")
    parser.add_argument("--job-name", default=None, help="unique Iris job name")
    parser.add_argument("--task-image", required=True, help="GPU image with Slime dependencies already installed")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES, help="number of Iris GPU tasks")
    parser.add_argument(
        "--rendezvous-dir",
        help="shared local or s3:// directory used to coordinate a multi-node Ray cluster",
    )
    parser.add_argument("--gpu-variant", default=DEFAULT_GPU_VARIANT)
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE)
    parser.add_argument("--cpu", type=float, default=DEFAULT_CPU)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--disk", default=DEFAULT_DISK)
    parser.add_argument("--priority", choices=SUPPORTED_PRIORITIES, default=DEFAULT_PRIORITY)
    parser.add_argument("--max-retries-failure", type=int, default=DEFAULT_MAX_RETRIES_FAILURE)
    parser.add_argument("--max-retries-preemption", type=int, default=DEFAULT_MAX_RETRIES_PREEMPTION)
    parser.add_argument("--max-task-failures", type=int, default=DEFAULT_MAX_TASK_FAILURES)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-task timeout; zero disables it",
    )
    parser.add_argument("--env", action="append", type=_parse_env_assignment, default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--secret-env",
        action="append",
        type=_parse_env_name,
        default=[],
        metavar="NAME",
        help="forward NAME from the launcher environment without putting its value in argv",
    )
    parser.add_argument(
        "--setup-command",
        action="append",
        default=[],
        help="task setup command; defaults to no setup because the image must be self-contained",
    )
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dry-run", action="store_true", help="validate and print a redacted request")
    parser.add_argument("--no-wait", action="store_true", help="return immediately after submission")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after '--'")
    return parser


def spec_from_args(args: argparse.Namespace) -> IrisLaunchSpec:
    command = tuple(args.command)
    if command[:1] == ("--",):
        command = command[1:]
    env: dict[str, str] = {}
    for name, value in args.env:
        if name in env:
            raise LaunchConfigError(f"environment variable was provided more than once: {name}")
        env[name] = value
    if len(args.secret_env) != len(set(args.secret_env)):
        raise LaunchConfigError("a secret environment variable was provided more than once")
    spec = IrisLaunchSpec(
        job_name=args.job_name or _default_job_name(),
        task_image=args.task_image,
        command=command,
        cluster=args.cluster,
        cluster_config=args.cluster_config,
        controller_url=args.controller_url,
        nodes=args.nodes,
        gpu_variant=args.gpu_variant,
        gpus_per_node=args.gpus_per_node,
        cpu=args.cpu,
        memory=args.memory,
        disk=args.disk,
        priority=args.priority,
        max_retries_failure=args.max_retries_failure,
        max_retries_preemption=args.max_retries_preemption,
        max_task_failures=args.max_task_failures,
        timeout_seconds=args.timeout_seconds,
        env=env,
        secret_env_names=tuple(args.secret_env),
        setup_commands=tuple(args.setup_command),
        rendezvous_dir=args.rendezvous_dir,
    )
    spec.validate()
    return spec


def run(argv: Sequence[str] | None = None, backend: LaunchBackend | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        spec = spec_from_args(args)
        workspace = args.workspace.resolve()
        if not workspace.is_dir():
            raise LaunchConfigError(f"workspace is not a directory: {workspace}")
        if args.dry_run:
            # Resolve secret presence during validation without exposing values.
            spec.resolved_env()
            print(json.dumps(spec.redacted_dict(), indent=2, sort_keys=True))
            return 0
        (backend or IrisBackend()).submit(spec, workspace, wait=not args.no_wait)
        return 0
    except (LaunchConfigError, RuntimeError) as error:
        parser.error(str(error))


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
