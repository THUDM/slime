from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightBucketReservation:
    """Credit held by one logical weight bucket."""

    weight_version: int
    bucket_id: int
    bucket_bytes: int


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

    def begin_version(self, weight_version: int) -> None:
        if self._active_version is not None:
            raise RuntimeError(f"weight version {self._active_version} is still active")
        if self._inflight:
            raise RuntimeError("cannot begin a weight version with buckets still in flight")
        if self._last_committed_version is not None and weight_version <= self._last_committed_version:
            raise ValueError(
                f"weight version {weight_version} must be newer than committed version {self._last_committed_version}"
            )

        self._active_version = weight_version
        self._next_bucket_id = 0

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
        return reservation

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

        self._inflight.popleft()
        self._inflight_bytes -= reservation.bucket_bytes

    def commit_version(self, weight_version: int) -> None:
        if weight_version != self._active_version:
            raise RuntimeError(f"cannot commit inactive weight version {weight_version}")
        if self._inflight:
            raise RuntimeError(
                f"cannot commit weight version {weight_version} with {len(self._inflight)} bucket(s) in flight"
            )

        self._last_committed_version = weight_version
        self._active_version = None
