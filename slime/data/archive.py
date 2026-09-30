"""Indexed rollout archives backed by immutable straw records.

An archive owns its data independently of queue consumption. Keys are scoped to
one archive and identify the sample versions saved in that archive.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict
from pathlib import Path


class RolloutArchive:
    def __init__(self, path, *, root=None):
        from straw.protocol import RecordSetRef
        from straw.store import SharedFilesystemStore
        from straw.tensor import MAX_PUBLICATION_BYTES, MAX_TENSOR_BYTES

        from slime.data.codec import CODECS, SampleCodec

        self.path = Path(path)
        self.index = json.loads(self.path.read_text())
        if self.index.get("format") != "slime.straw-debug" or self.index.get("version") != 1:
            raise ValueError(f"Unsupported straw rollout archive: {path}")
        self.manifest = RecordSetRef.from_dict(self.index["manifest"])
        self.store = SharedFilesystemStore(
            root or self.index["root"],
            self.manifest.manifest.segment.run_id,
            codecs=CODECS,
            max_record_bytes=MAX_TENSOR_BYTES,
            max_buffer_bytes=MAX_PUBLICATION_BYTES,
        )
        self.codec = SampleCodec(self.store)
        self.contents = self.codec.load(self.manifest)

    @classmethod
    def save(cls, path, samples, *, rollout_id, evaluation=False, args=None):
        from straw.protocol import digest
        from straw.reporting import write_report
        from straw.store import SharedFilesystemStore
        from straw.tensor import MAX_PUBLICATION_BYTES, MAX_TENSOR_BYTES

        from slime.data.codec import CODECS, SampleCodec
        from slime.data.transport import DiskPayloadRef, rollout_store

        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Rollout archives are immutable; choose a new path: {path}")
        shared = args is not None and args.rollout_data_transport == "straw"
        if shared:
            store, codec, lock = rollout_store(args)
        else:
            import threading

            store = SharedFilesystemStore(
                path.parent / "straw-data",
                "debug-rollout",
                codecs=CODECS,
                online_gc=True,
                max_record_bytes=MAX_TENSOR_BYTES,
                max_buffer_bytes=MAX_PUBLICATION_BYTES,
            )
            codec, lock = SampleCodec(store, args=args), threading.RLock()
        try:
            # Bounded chunks support indexed reads without a file per sample.
            # Lazy R3/SC references in the same pool share their existing bytes.
            chunks, entries = [], []
            with lock:
                for start in range(0, len(samples), 64):
                    chunk = samples[start : start + 64]
                    ref = codec.publish(chunk, submission_id=f"debug-chunk:{uuid.uuid4().hex}")
                    for offset, sample in enumerate(chunk):
                        context = getattr(sample, "_queue_receipt", None) or getattr(sample, "_queue_lease", None)
                        entries.append(
                            {
                                "sample_key": (
                                    f"sample:{sample.index}"
                                    if sample.index is not None
                                    else f"position:{start + offset}"
                                ),
                                "task_key": context["task_id"] if context else None,
                                "chunk": len(chunks),
                                "offset": offset,
                            }
                        )
                    chunks.append(DiskPayloadRef(ref, str(store.backend.root)))
                manifest = codec.publish(
                    {"chunks": chunks, "entries": entries}, submission_id=f"debug-index:{uuid.uuid4().hex}"
                )
                owner = f"debug:{path.resolve()}:{digest(asdict(manifest))}"
                # Ownership must commit before the externally visible index.
                store.retain(owner, [manifest])
                store.release_publications([manifest])
                write_report(
                    path,
                    {
                        "format": "slime.straw-debug",
                        "version": 1,
                        "rollout_id": rollout_id,
                        "evaluation": evaluation,
                        "root": str(store.backend.root),
                        "manifest": asdict(manifest),
                        "storage_owner": owner,
                    },
                )
        finally:
            if not shared:
                store.close()

    def keys(self):
        """Return sample/task keys in archive order, including duplicate sample IDs."""
        return [(entry["sample_key"], entry["task_key"]) for entry in self.contents["entries"]]

    def load_samples(self, *, sample_key=None, task_key=None):
        """Load all samples, or the intersection of the supplied archive keys.

        Compact/custom rollouts may emit several samples with the same index;
        therefore this API always returns a list. Tensor fields stay lazy.
        """
        selected = [
            entry
            for entry in self.contents["entries"]
            if (sample_key is None or entry["sample_key"] == sample_key)
            and (task_key is None or entry["task_key"] == task_key)
        ]
        if not selected and (sample_key is not None or task_key is not None):
            raise KeyError((sample_key, task_key))
        loaded, result = {}, []
        for entry in selected:
            chunk = entry["chunk"]
            if chunk not in loaded:
                loaded[chunk] = self.codec.load(self.contents["chunks"][chunk].manifest)
            sample = copy.copy(loaded[chunk][entry["offset"]])
            # An archive is training input, not authorization to mutate its
            # original queue. Keep generation provenance, strip live control.
            for name in (
                "_queue_lease",
                "_queue_receipt",
                "_queue_source_positions",
                "_queue_generation_start",
                "_queue_resume_origin",
            ):
                sample.__dict__.pop(name, None)
            result.append(sample)
        return result

    def export_pt(self, path):
        """Materialize a self-contained .pt dump, independent of this pool."""
        import torch

        from slime.data.tensor import materialize_tensor_refs

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            materialize_tensor_refs(
                {"rollout_id": self.index["rollout_id"], "samples": [s.to_dict() for s in self.load_samples()]}
            ),
            path,
        )

    def release(self):
        """Release archive retention after all its readers have finished.

        The index remains for inspection; GC may subsequently reclaim payloads
        not owned by another archive, queue or checkpoint.
        """
        self.store.release(self.index["storage_owner"])

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
