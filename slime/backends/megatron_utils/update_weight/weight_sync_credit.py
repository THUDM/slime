from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightBucketReservation:
    """Credit held by one logical weight bucket."""

    weight_version: int
    bucket_id: int
    bucket_bytes: int


@dataclass(frozen=True, slots=True)
class WeightSyncLifecycleSnapshot:
    """Rank-local accounting for the active or most recently committed version."""

    active_version: int | None
    failed_reason: str | None
    inflight_buckets: int
    inflight_bytes: int
    transport_outstanding_bytes: int
    staging_resident_bytes: int
    pending_consumer_objects: int
    peak_inflight_buckets: int
    peak_inflight_bytes: int
    peak_transport_outstanding_bytes: int
    peak_staging_resident_bytes: int
    peak_pending_consumer_objects: int


@dataclass(slots=True)
class _BucketLifecycle:
    transport_bytes: int
    staging_bytes: int
    consumer_objects: int


class WeightSyncCreditController:
    """Track bounded, ordered weight buckets for one updater.

    A zero limit disables that credit dimension. When both limits are zero,
    callers keep the legacy synchronous transfer path while still using the
    version/order checks in this controller.
    """

    def __init__(self, max_inflight_buckets: int = 0, max_inflight_bytes: int = 0) -> None:
        if max_inflight_buckets < 0:
            raise ValueError("max_inflight_buckets must be non-negative")
        if max_inflight_bytes < 0:
            raise ValueError("max_inflight_bytes must be non-negative")

        self.max_inflight_buckets = max_inflight_buckets
        self.max_inflight_bytes = max_inflight_bytes
        self._active_version: int | None = None
        self._last_committed_version: int | None = None
        self._next_bucket_id = 0
        self._inflight: deque[WeightBucketReservation] = deque()
        self._inflight_bytes = 0
        self._bucket_lifecycles: dict[WeightBucketReservation, _BucketLifecycle] = {}
        self._transport_outstanding_bytes = 0
        self._staging_resident_bytes = 0
        self._persistent_staging_bytes = 0
        self._pending_consumer_objects = 0
        self._peak_inflight_buckets = 0
        self._peak_inflight_bytes = 0
        self._peak_transport_outstanding_bytes = 0
        self._peak_staging_resident_bytes = 0
        self._peak_pending_consumer_objects = 0
        self._failed_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.max_inflight_buckets > 0 or self.max_inflight_bytes > 0

    @property
    def inflight_buckets(self) -> int:
        return len(self._inflight)

    @property
    def inflight_bytes(self) -> int:
        return self._inflight_bytes

    @property
    def full(self) -> bool:
        return bool(
            (self.max_inflight_buckets and len(self._inflight) >= self.max_inflight_buckets)
            or (self.max_inflight_bytes and self._inflight_bytes >= self.max_inflight_bytes)
        )

    @property
    def snapshot(self) -> WeightSyncLifecycleSnapshot:
        return WeightSyncLifecycleSnapshot(
            active_version=self._active_version,
            failed_reason=self._failed_reason,
            inflight_buckets=len(self._inflight),
            inflight_bytes=self._inflight_bytes,
            transport_outstanding_bytes=self._transport_outstanding_bytes,
            staging_resident_bytes=self._staging_resident_bytes,
            pending_consumer_objects=self._pending_consumer_objects,
            peak_inflight_buckets=self._peak_inflight_buckets,
            peak_inflight_bytes=self._peak_inflight_bytes,
            peak_transport_outstanding_bytes=self._peak_transport_outstanding_bytes,
            peak_staging_resident_bytes=self._peak_staging_resident_bytes,
            peak_pending_consumer_objects=self._peak_pending_consumer_objects,
        )

    def metrics(self) -> dict[str, float]:
        """Return peak rank-local lifecycle counters for the current version."""
        snapshot = self.snapshot
        return {
            "perf/update_weights_peak_inflight_buckets": float(snapshot.peak_inflight_buckets),
            "perf/update_weights_peak_logical_inflight_bytes": float(snapshot.peak_inflight_bytes),
            "perf/update_weights_peak_transport_outstanding_bytes": float(snapshot.peak_transport_outstanding_bytes),
            "perf/update_weights_peak_staging_resident_bytes": float(snapshot.peak_staging_resident_bytes),
            "perf/update_weights_peak_pending_consumer_objects": float(snapshot.peak_pending_consumer_objects),
        }

    def begin_version(self, weight_version: int) -> None:
        if self._active_version is not None:
            raise RuntimeError(f"weight version {self._active_version} is still active")
        if self._inflight:
            raise RuntimeError("cannot begin a weight version with buckets still in flight")
        if self._bucket_lifecycles or self._transport_outstanding_bytes or self._staging_resident_bytes:
            raise RuntimeError("cannot begin a weight version with lifecycle resources still resident")
        if self._last_committed_version is not None and weight_version <= self._last_committed_version:
            raise ValueError(
                f"weight version {weight_version} must be newer than committed version {self._last_committed_version}"
            )

        self._active_version = weight_version
        self._next_bucket_id = 0
        self._failed_reason = None
        self._peak_inflight_buckets = 0
        self._peak_inflight_bytes = 0
        self._peak_transport_outstanding_bytes = 0
        self._peak_staging_resident_bytes = 0
        self._peak_pending_consumer_objects = 0

    def reserve(self, bucket_bytes: int) -> WeightBucketReservation | None:
        """Reserve the next bucket, or return ``None`` when credits are full."""
        if self._active_version is None:
            raise RuntimeError("begin_version must be called before reserving a bucket")
        if bucket_bytes < 0:
            raise ValueError("bucket_bytes must be non-negative")
        if self.max_inflight_bytes and bucket_bytes > self.max_inflight_bytes:
            raise ValueError(
                f"weight bucket requires {bucket_bytes} bytes, larger than "
                f"--update-weight-max-inflight-bytes={self.max_inflight_bytes}"
            )
        if self.max_inflight_buckets and len(self._inflight) >= self.max_inflight_buckets:
            return None
        if self.max_inflight_bytes and self._inflight_bytes + bucket_bytes > self.max_inflight_bytes:
            return None

        reservation = WeightBucketReservation(
            weight_version=self._active_version,
            bucket_id=self._next_bucket_id,
            bucket_bytes=bucket_bytes,
        )
        self._next_bucket_id += 1
        self._inflight.append(reservation)
        self._inflight_bytes += bucket_bytes
        self._peak_inflight_buckets = max(self._peak_inflight_buckets, len(self._inflight))
        self._peak_inflight_bytes = max(self._peak_inflight_bytes, self._inflight_bytes)
        return reservation

    def mark_launched(
        self,
        reservation: WeightBucketReservation,
        *,
        transport_bytes: int,
        staging_bytes: int,
        consumer_objects: int,
    ) -> None:
        """Record resources retained after a bucket has been submitted.

        These counters are evidence only: admission remains governed by logical
        bucket count and bytes. They make explicit when staging memory exceeds
        the logical byte credit instead of silently treating the two as equal.
        """
        self._require_inflight(reservation)
        if reservation in self._bucket_lifecycles:
            raise RuntimeError(f"weight bucket {reservation.bucket_id} was already launched")
        for name, value in (
            ("transport_bytes", transport_bytes),
            ("staging_bytes", staging_bytes),
            ("consumer_objects", consumer_objects),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self._bucket_lifecycles[reservation] = _BucketLifecycle(
            transport_bytes=transport_bytes,
            staging_bytes=staging_bytes,
            consumer_objects=consumer_objects,
        )
        self._transport_outstanding_bytes += transport_bytes
        self._staging_resident_bytes += staging_bytes
        self._pending_consumer_objects += consumer_objects
        self._peak_transport_outstanding_bytes = max(
            self._peak_transport_outstanding_bytes, self._transport_outstanding_bytes
        )
        self._peak_staging_resident_bytes = max(self._peak_staging_resident_bytes, self._staging_resident_bytes)
        self._peak_pending_consumer_objects = max(self._peak_pending_consumer_objects, self._pending_consumer_objects)

    def mark_next_wave(
        self,
        reservation: WeightBucketReservation,
        *,
        transport_bytes: int,
        staging_bytes: int,
        consumer_objects: int,
    ) -> None:
        """Observe another wave without releasing the logical bucket credit."""
        previous = self._require_lifecycle(reservation)
        if previous.transport_bytes or previous.staging_bytes or previous.consumer_objects:
            raise RuntimeError("cannot advance a weight wave before its resources complete")
        del self._bucket_lifecycles[reservation]
        try:
            self.mark_launched(
                reservation,
                transport_bytes=transport_bytes,
                staging_bytes=staging_bytes,
                consumer_objects=consumer_objects,
            )
        except BaseException:
            self._bucket_lifecycles[reservation] = previous
            raise

    def mark_transport_complete(self, reservation: WeightBucketReservation) -> None:
        """Release transport accounting exactly once after its wait boundary."""
        lifecycle = self._require_lifecycle(reservation)
        if lifecycle.transport_bytes:
            self._transport_outstanding_bytes -= lifecycle.transport_bytes
            lifecycle.transport_bytes = 0

    def mark_consumers_complete(self, reservation: WeightBucketReservation) -> None:
        """Release consumer-object accounting exactly once after acknowledgement."""
        lifecycle = self._require_lifecycle(reservation)
        if lifecycle.consumer_objects:
            self._pending_consumer_objects -= lifecycle.consumer_objects
            lifecycle.consumer_objects = 0

    def mark_staging_released(self, reservation: WeightBucketReservation) -> None:
        """Release staging accounting exactly once after producer references are dropped."""
        lifecycle = self._require_lifecycle(reservation)
        if lifecycle.staging_bytes:
            self._staging_resident_bytes -= lifecycle.staging_bytes
            lifecycle.staging_bytes = 0

    def set_persistent_staging_bytes(self, staging_bytes: int) -> None:
        """Observe reusable staging that lives outside individual bucket objects."""
        if staging_bytes < 0:
            raise ValueError("staging_bytes must be non-negative")
        self._staging_resident_bytes += staging_bytes - self._persistent_staging_bytes
        self._persistent_staging_bytes = staging_bytes
        self._peak_staging_resident_bytes = max(self._peak_staging_resident_bytes, self._staging_resident_bytes)

    def fail_version(self, weight_version: int, error: BaseException | str) -> None:
        """Poison a partially applied version so it can never be committed or reused."""
        if weight_version != self._active_version:
            raise RuntimeError(f"cannot fail inactive weight version {weight_version}")
        if self._failed_reason is None:
            self._failed_reason = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"

    def release(self, reservation: WeightBucketReservation) -> None:
        """Release the oldest bucket, preserving engine load order."""
        if not self._inflight:
            raise RuntimeError("no weight bucket credit is in flight")
        oldest = self._inflight[0]
        if reservation != oldest:
            raise RuntimeError(
                f"weight buckets must complete in order: expected bucket {oldest.bucket_id}, "
                f"got bucket {reservation.bucket_id}"
            )

        lifecycle = self._bucket_lifecycles.get(reservation)
        if lifecycle is not None:
            if lifecycle.transport_bytes or lifecycle.staging_bytes or lifecycle.consumer_objects:
                raise RuntimeError(
                    f"cannot release weight bucket {reservation.bucket_id} before transport, "
                    "consumers, and staging resources are complete"
                )
            del self._bucket_lifecycles[reservation]

        self._inflight.popleft()
        self._inflight_bytes -= reservation.bucket_bytes

    def commit_version(self, weight_version: int) -> None:
        if weight_version != self._active_version:
            raise RuntimeError(f"cannot commit inactive weight version {weight_version}")
        if self._failed_reason is not None:
            raise RuntimeError(f"cannot commit failed weight version {weight_version}: {self._failed_reason}")
        if self._inflight:
            raise RuntimeError(
                f"cannot commit weight version {weight_version} with {len(self._inflight)} bucket(s) in flight"
            )
        if (
            self._bucket_lifecycles
            or self._transport_outstanding_bytes
            or self._staging_resident_bytes
            or self._pending_consumer_objects
        ):
            raise RuntimeError(
                f"cannot commit weight version {weight_version} with lifecycle resources still resident"
            )

        self._last_committed_version = weight_version
        self._active_version = None

    def _require_inflight(self, reservation: WeightBucketReservation) -> None:
        if reservation not in self._inflight:
            raise RuntimeError(f"weight bucket {reservation.bucket_id} is not in flight")

    def _require_lifecycle(self, reservation: WeightBucketReservation) -> _BucketLifecycle:
        self._require_inflight(reservation)
        lifecycle = self._bucket_lifecycles.get(reservation)
        if lifecycle is None:
            raise RuntimeError(f"weight bucket {reservation.bucket_id} was not launched")
        return lifecycle
