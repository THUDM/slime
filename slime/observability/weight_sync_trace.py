"""Opt-in tracing of the production updater's asynchronous bucket lifetime."""

from __future__ import annotations

from functools import partial, wraps

from .communication_timeline import (
    communication_context,
    communication_event,
    communication_phase,
    flush_communication_timeline,
    get_communication_timeline,
    iter_communication_buckets,
)


class WeightSyncTrace:
    """One version's transfer spans; never a replacement for credit accounting."""

    def __init__(self):
        self.pending = {}
        self.fields = {}
        self.next_bucket_id = 0

    def converted_chunks(self, chunks):
        for bucket_id, chunk in iter_communication_buckets(chunks, start_bucket_id=self.next_bucket_id):
            self.next_bucket_id = bucket_id + 1
            yield chunk

    def start(self, reservation, **metadata):
        bucket_id = reservation.bucket_id
        if bucket_id in self.pending:
            raise RuntimeError("duplicate active communication span for a weight bucket")
        fields = {"bucket_id": bucket_id, "message_bytes": reservation.bucket_bytes, **metadata}
        self.fields[bucket_id] = fields
        communication_event("weight_bucket_ready", **fields)
        phase = communication_phase("weight_bucket_send", **fields)
        phase.__enter__()
        self.pending[bucket_id] = phase

    def submitted(self, reservation):
        phase = self.pending.get(reservation.bucket_id)
        if phase is not None:
            phase.mark_api_return()

    def complete(self, reservation, error=None):
        phase = self.pending.pop(reservation.bucket_id, None)
        if phase is not None:
            phase.__exit__(type(error) if error else None, error, None)
            if error is None:
                communication_event(
                    "engine_bucket_receive",
                    **self.fields[reservation.bucket_id],
                    observation="trainer_transport_or_ipc_ack_complete",
                )

    def consumer(self, reservation):
        return communication_phase(
            "engine_load_weights",
            **self.fields[reservation.bucket_id],
            observation="trainer_engine_load_ack_wait",
        )

    def released(self, reservation):
        communication_event(
            "weight_bucket_reusable", bucket_id=reservation.bucket_id, message_bytes=reservation.bucket_bytes
        )
        self.fields.pop(reservation.bucket_id, None)

    def close(self, error):
        for phase in self.pending.values():
            phase.__exit__(type(error) if error else None, error, None)
        self.pending.clear()
        self.fields.clear()


def trace_weight_update(function=None, *, transport="nccl_or_cuda_ipc"):
    """Keep the disabled path free of trace objects, clocks and tensor scans."""

    if function is None:
        return partial(trace_weight_update, transport=transport)

    @wraps(function)
    def wrapped(self):
        if get_communication_timeline() is None:
            return function(self)
        trace = self._weight_sync_trace = WeightSyncTrace()
        with communication_context(weight_version=self.weight_version + 1, transport=transport):
            try:
                result = function(self)
                if trace.pending:
                    raise RuntimeError("weight update returned with incomplete transfer trace spans")
            except BaseException as error:
                trace.close(error)
                communication_event("weight_sync_failed", error_type=type(error).__name__)
                raise
            else:
                trace.close(None)
                communication_event("weight_sync_complete")
                return result
            finally:
                self._weight_sync_trace = None
                flush_communication_timeline(block=False)

    return wrapped
