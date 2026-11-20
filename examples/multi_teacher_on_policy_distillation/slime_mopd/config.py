"""MOPD configuration: the teacher registry plus the domain-label mapping.

One prompt file carries a ``domain`` label per row, and teacher ids match those
labels, so the teacher registry doubles as the domain table: teacher ids *are*
domain names unless ``mopd.domains`` maps a label onto a differently named
teacher. There is deliberately no per-file dataset binding table -- one prompt
file, one label column, one teacher per label.

The config lives in a YAML file rather than CLI arguments because the teacher
count is data-driven. slime exposes two zero-core-change carriers:

- ``--custom-config-path <file>``: slime merges the YAML top-level keys into
  ``args`` (see ``slime/utils/arguments.py``); a top-level ``mopd`` key becomes
  ``args.mopd``.
- ``SLIME_MOPD_CONFIG=<file>``: environment fallback that needs no slime flag.

Both an ``mopd:`` wrapper and a bare mapping are accepted.

Minimal config::

    mopd:
      teachers:
        - {id: math,   backend: megatron, load: /models/teacher-math}
        - {id: code,   backend: megatron, load: /models/teacher-code}
        - {id: ifeval, backend: megatron, load: /models/teacher-if}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


class MopdConfigError(ValueError):
    """Raised when the MOPD configuration is missing or invalid."""


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` references in strings so one config fits every host."""

    if isinstance(value, str):
        return os.path.expandvars(value) if "$" in value else value
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


VALID_BACKENDS = ("sglang", "megatron")
VALID_LOGPROB_DTYPES = ("auto", "float32", "float16", "bfloat16")


@dataclass
class TeacherConfig:
    """A single teacher model endpoint or checkpoint."""

    id: str
    backend: str = "sglang"
    rm_url: str | None = None
    load: str | None = None
    ckpt_step: int | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, index: int, base_dir: str | None = None) -> TeacherConfig:
        if not isinstance(raw, dict):
            raise MopdConfigError(f"teachers[{index}] must be a mapping, got {type(raw).__name__}")
        teacher_id = raw.get("id") or raw.get("name")
        if not teacher_id:
            raise MopdConfigError(f"teachers[{index}] is missing required field 'id'")
        load = raw.get("load")
        if load and base_dir and not os.path.isabs(str(load)):
            load = os.path.join(base_dir, str(load))
        return cls(
            id=str(teacher_id),
            backend=str(raw.get("backend", "sglang")),
            rm_url=raw.get("rm_url"),
            load=str(load) if load else None,
            ckpt_step=raw.get("ckpt_step"),
        )


def _normalize_domains(raw: Any) -> dict[str, str]:
    """Normalize ``mopd.domains`` to ``{domain label: teacher id}``."""

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MopdConfigError("mopd.domains must be a mapping of domain label to teacher id")
    return {str(domain): str(teacher_id) for domain, teacher_id in raw.items()}


@dataclass
class MopdConfig:
    """Resolved MOPD configuration."""

    teachers: list[TeacherConfig]
    # domain label -> teacher id; empty means "teacher id == domain label".
    domains: dict[str, str] = field(default_factory=dict)
    # Last-resort fallback for samples whose domain has no teacher. Warns once
    # and counts; unset means such a sample is a hard error.
    default_teacher: str | None = None
    active_teacher_ids: list[str] | None = None
    max_concurrency: int = 64
    request_timeout_s: float = 120.0
    max_retries: int = 10
    # dtype of the stored teacher log-prob tensors; "auto" follows the model
    # params dtype (bf16/fp16) and falls back to float32.
    logprob_dtype: str = "auto"
    # Directory of the YAML file the config was loaded from (None when the
    # config came from an inline ``args.mopd`` mapping). Relative teacher
    # ``load`` paths are resolved against it.
    base_dir: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_raw(cls, raw: dict[str, Any], base_dir: str | None = None) -> MopdConfig:
        if not isinstance(raw, dict):
            raise MopdConfigError("MOPD config must be a mapping")
        raw = _expand_env(raw)
        if isinstance(raw.get("mopd"), dict):
            raw = raw["mopd"]

        teachers_raw = raw.get("teachers")
        if not teachers_raw:
            raise MopdConfigError("MOPD config must define a non-empty 'teachers' list")
        teachers = [TeacherConfig.from_raw(entry, index=i, base_dir=base_dir) for i, entry in enumerate(teachers_raw)]

        active = raw.get("active_teachers")
        if active is not None and not isinstance(active, list):
            raise MopdConfigError("mopd.active_teachers must be a list of teacher ids")

        # ``default_teacher`` is accepted at the top level; ``router`` is the
        # historical place for it and is still read so old configs keep working.
        router_raw = raw.get("router") or {}
        if not isinstance(router_raw, dict):
            raise MopdConfigError("mopd.router must be a mapping")
        default_teacher = raw.get("default_teacher", router_raw.get("default_teacher"))

        return cls(
            teachers=teachers,
            domains=_normalize_domains(raw.get("domains")),
            default_teacher=default_teacher,
            active_teacher_ids=[str(x) for x in active] if active is not None else None,
            max_concurrency=int(raw.get("max_concurrency", 64)),
            request_timeout_s=float(raw.get("request_timeout_s", 120.0)),
            max_retries=int(raw.get("max_retries", 10)),
            logprob_dtype=str(raw.get("logprob_dtype", "auto")),
        )

    def validate(self) -> None:
        if not self.teachers:
            raise MopdConfigError("MOPD config must define at least one teacher")

        ids = [teacher.id for teacher in self.teachers]
        duplicates = sorted({tid for tid in ids if ids.count(tid) > 1})
        if duplicates:
            raise MopdConfigError(f"duplicate teacher ids: {duplicates}")

        for teacher in self.teachers:
            if teacher.backend not in VALID_BACKENDS:
                raise MopdConfigError(
                    f"teacher {teacher.id!r} has unsupported backend {teacher.backend!r}; "
                    f"expected one of {VALID_BACKENDS}"
                )
            if (
                teacher.backend == "sglang"
                and teacher.rm_url is not None
                and not teacher.rm_url.startswith(("http://", "https://"))
            ):
                raise MopdConfigError(f"teacher {teacher.id!r} rm_url must be an http(s) URL")

        if self.logprob_dtype not in VALID_LOGPROB_DTYPES:
            raise MopdConfigError(
                f"unsupported mopd.logprob_dtype {self.logprob_dtype!r}; expected one of {VALID_LOGPROB_DTYPES}"
            )

        if self.default_teacher is not None and self.default_teacher not in ids:
            raise MopdConfigError(f"default_teacher {self.default_teacher!r} is not a known teacher id")

        for domain, teacher_id in self.domains.items():
            if teacher_id not in ids:
                raise MopdConfigError(f"domain {domain!r} references unknown teacher {teacher_id!r}")

        if self.active_teacher_ids is not None:
            unknown = sorted(set(self.active_teacher_ids) - set(ids))
            if unknown:
                raise MopdConfigError(f"active_teachers contains unknown ids: {unknown}")

    def teacher(self, teacher_id: str) -> TeacherConfig:
        for teacher in self.teachers:
            if teacher.id == teacher_id:
                return teacher
        raise MopdConfigError(f"unknown teacher id {teacher_id!r}")

    def has_teacher(self, teacher_id: str) -> bool:
        return any(teacher.id == teacher_id for teacher in self.teachers)

    def is_active(self, teacher_id: str) -> bool:
        """A teacher listed in the config is active unless excluded by `active_teachers`."""

        self.teacher(teacher_id)
        return self.active_teacher_ids is None or teacher_id in self.active_teacher_ids

    def active_teachers(self) -> list[TeacherConfig]:
        return [teacher for teacher in self.teachers if self.is_active(teacher.id)]

    def teacher_for_domain(self, domain: str) -> str | None:
        """Map a row's ``domain`` label onto a teacher id."""

        mapped = self.domains.get(domain)
        if mapped is not None:
            return mapped
        return domain if self.has_teacher(domain) else None


_CACHE: dict[tuple[str, Any], MopdConfig] = {}


def _config_source(args) -> tuple[str, Any]:
    inline = getattr(args, "mopd", None)
    if isinstance(inline, dict) and inline:
        return ("inline", inline)
    path = getattr(args, "mopd_config", None)
    if path:
        return ("path", str(path))
    env_path = os.environ.get("SLIME_MOPD_CONFIG")
    if env_path:
        return ("path", env_path)
    raise MopdConfigError(
        "MOPD config not found: set a top-level 'mopd' block via --custom-config-path, "
        "or set SLIME_MOPD_CONFIG to a YAML file"
    )


def load_mopd_config(args) -> MopdConfig:
    """Load and cache the MOPD config referenced by ``args`` or the environment."""

    kind, payload = _config_source(args)
    key = (kind, id(payload) if kind == "inline" else os.path.abspath(payload))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    base_dir = None
    if kind == "inline":
        raw = payload
        custom_path = getattr(args, "custom_config_path", None)
        if custom_path:
            base_dir = os.path.dirname(os.path.abspath(custom_path))
    else:
        path = payload
        if not os.path.exists(path):
            raise MopdConfigError(f"MOPD config file does not exist: {path}")
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        base_dir = os.path.dirname(os.path.abspath(payload))

    config = MopdConfig.from_raw(raw, base_dir=base_dir)
    config.base_dir = base_dir
    _CACHE[key] = config
    return config


def clear_config_cache() -> None:
    _CACHE.clear()
