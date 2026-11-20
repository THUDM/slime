"""Teacher routing from the row's ``domain`` label.

The prompt file carries a ``domain`` label per row (``math`` / ``code`` /
``ifeval`` in the example). ``slime_mopd.data_source`` stamps that label on every
sample as its ``source``, so routing is a two-step lookup with no hash router, no
per-file binding table and no path matching:

1. an explicit per-record annotation: ``sample.metadata["teacher"]``
2. the domain label on the sample (-> ``mopd.domains`` -> teacher id)
3. ``mopd.default_teacher`` (fallback; warns once and counts)

A sample with no annotation and a domain that maps to no teacher is **not**
silently assigned: if ``default_teacher`` is unset the resolution raises, and
the data source already rejects such files at startup. If it is set, the first
such sample logs one warning and a per-batch summary reports the total count
(see ``pop_default_fallback_count``).

The result is a single per-sample teacher id; training does not re-resolve
anything because the existing slime OPD loss is already per-sample.
"""

from __future__ import annotations

import logging
from typing import Any

from slime_mopd.config import MopdConfig, MopdConfigError, TeacherConfig

logger = logging.getLogger(__name__)

TEACHER_METADATA_KEYS = ("teacher", "mopd_teacher", "mopd_teacher_id")
DOMAIN_METADATA_KEYS = ("mopd_domain", "domain")


class RoutingError(ValueError):
    """Raised when a sample cannot be routed to an active teacher."""


class _DefaultFallbackStats:
    """Tracks unannotated samples that fell back to ``default_teacher``."""

    def __init__(self) -> None:
        self.count = 0
        self.warned = False

    def record(self, teacher_id: str) -> None:
        self.count += 1
        if not self.warned:
            self.warned = True
            logger.warning(
                "mopd: sample has no teacher annotation and its domain has no teacher; "
                "routing it to default_teacher %r. Further unannotated samples are counted, "
                "not logged; a per-batch summary follows.",
                teacher_id,
            )

    def drain(self) -> int:
        count, self.count = self.count, 0
        return count

    def reset(self) -> None:
        self.count = 0
        self.warned = False


_DEFAULT_FALLBACK = _DefaultFallbackStats()


def pop_default_fallback_count() -> int:
    """Return and reset the number of samples that used ``default_teacher``."""

    return _DEFAULT_FALLBACK.drain()


def reset_default_fallback_stats() -> None:
    """Clear the fallback counter and the once-only warning (for tests)."""

    _DEFAULT_FALLBACK.reset()


def sample_domain(sample: Any) -> str | None:
    """Return the domain label carried by a sample, if any.

    Rollout samples keep it in ``metadata["mopd_domain"]`` and, for the training
    actor, in the ``source_names`` field that slime forwards to each rank
    (surfaced there as ``metadata["source_name"]``).
    """

    metadata = getattr(sample, "metadata", None) or {}
    if isinstance(metadata, dict):
        for key in DOMAIN_METADATA_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
        source_name = metadata.get("source_name")
        if source_name:
            return str(source_name)
    source = getattr(sample, "source", None)
    if source:
        return str(source)
    return None


def explicit_teacher_id(sample: Any) -> str | None:
    """Return a per-record teacher annotation, if the data carries one."""

    metadata = getattr(sample, "metadata", None) or {}
    if isinstance(metadata, dict):
        for key in TEACHER_METADATA_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
    return None


class TrainSampleView:
    """Minimal sample view for train-side routing when only shipped fields exist.

    The Megatron teacher path does not receive per-record metadata, but slime
    ships ``source_names`` (and ``sample_indices``). The rollout side writes the
    resolved teacher id there, so the train side only has to read it back.
    """

    __slots__ = ("prompt", "metadata", "index")

    def __init__(self, prompt: Any = "", source_name: str | None = None, index: Any = None):
        self.prompt = prompt
        self.index = index
        self.metadata = {"source_name": source_name} if source_name is not None else {}


def resolve_teacher_id(config: MopdConfig, args: Any, sample: Any) -> str:
    """Resolve the teacher id for one sample, applying the precedence chain."""

    teacher_id = explicit_teacher_id(sample)

    if teacher_id is None:
        domain = sample_domain(sample)
        if domain is not None:
            teacher_id = config.teacher_for_domain(domain)

    if teacher_id is None and config.default_teacher is not None:
        teacher_id = config.default_teacher
        _DEFAULT_FALLBACK.record(teacher_id)

    if teacher_id is None:
        raise RoutingError(
            "cannot route sample: no metadata.teacher, domain label "
            f"{sample_domain(sample)!r} maps to no teacher, and no default_teacher"
        )
    return teacher_id


def resolve_teacher(config: MopdConfig, args: Any, sample: Any) -> TeacherConfig:
    """Resolve and validate one sample's active teacher config."""

    teacher_id = resolve_teacher_id(config, args, sample)
    try:
        config.teacher(teacher_id)
    except MopdConfigError as error:
        raise RoutingError(f"routed teacher id {teacher_id!r} is not defined in the registry") from error

    if not config.is_active(teacher_id):
        raise RoutingError(
            f"routed teacher {teacher_id!r} is inactive (not in active_teachers); "
            "fix the data annotation or registry instead of silently falling back"
        )
    return config.teacher(teacher_id)


def set_sample_teacher(sample: Any, teacher_id: str) -> None:
    """Cache the resolved id on the sample for downstream metrics."""

    try:
        sample.mopd_teacher_id = teacher_id
    except AttributeError:
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict):
            metadata["mopd_teacher_id"] = teacher_id
