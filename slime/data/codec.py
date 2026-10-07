"""Explicit Sample/tensor codec for durable raw groups and training collections.

The representation is tagged JSON plus immutable typed byte records. It never
imports a class named by a payload and never executes pickle. Unknown Python
objects fail at publication. Replay tensors are stored lazily with their owning
Sample; continuations, training batches and checkpoints share immutable records.
"""

from __future__ import annotations

import base64
import dataclasses
import math
import sys

import numpy as np
import torch
from straw import Publication, Record
from straw.errors import CorruptData, InvalidReference, UnsupportedSchema
from straw.protocol import RecordSetRef, decode, encode
from straw.store import MAX_RECORDS
from straw.tensor import TensorRef as QueueTensorRef
from straw.tensor import tensor_record

from slime.utils.types import Sample

CODECS = ("bytes.v1", "json.v1", "slime.v1", "tensor.v1")


class SampleCodec:
    def __init__(self, store, *, args=None):
        if sys.byteorder != "little":
            raise UnsupportedSchema("Version 1 tensor codec requires a little-endian host")
        if not set(CODECS).issubset(store.codecs):
            raise ValueError("SampleCodec requires explicit slime.v1 and tensor.v1 store codecs")
        self.store = store
        self.args = args

    def _prepare_sample(self, sample):
        """Validate completed captures before publication; partial prefixes can resume."""
        validated = set()
        if self.args is None or sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
            return validated
        if getattr(self.args, "use_rollout_routing_replay", False):
            from slime.utils.routed_experts import validate_routed_experts_value

            routes = sample.rollout_routed_experts
            if routes is not None and (sample.loss_mask is None or any(sample.loss_mask)):
                if not isinstance(routes, QueueTensorRef):
                    routes = sample.materialize_rollout_routed_experts()
                validate_routed_experts_value(
                    routes, self.args, sample_index=sample.index, expected_rows=max(0, len(sample.tokens) - 1)
                )
                validated.add("rollout_routed_experts")
        if getattr(self.args, "use_score_centering", False):
            from slime.utils.score_centering import SAMPLER_TOPK_FIELDS, validate_sampler_top_p, validate_sampler_topk

            top_p = getattr(self.args, "rollout_top_p", 1.0) < 1
            fields = (
                ("rollout_top_p_token_ids", "rollout_top_p_token_offsets", "rollout_top_p_log_probs")
                if top_p
                else SAMPLER_TOPK_FIELDS
            )
            # Evaluation and prompt-only samples need no sampler capture.
            # Training conversion still requires enabled replay fields.
            if all(getattr(sample, key, None) is None for key in fields) and sample.response_length:
                return validated
            if top_p:
                values = [getattr(sample, key, None) for key in fields]
                if sample.response_length == 0 and all(value is None for value in values):
                    values = [
                        torch.empty(0, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.empty(0, dtype=torch.float32),
                    ]
                    for key, value in zip(fields, values, strict=True):
                        setattr(sample, key, value)
                validate_sampler_top_p(
                    *values,
                    sample.response_length,
                    loss_mask=sample.loss_mask,
                )
            else:
                validate_sampler_topk(sample, self.args.score_centering_top_k)
            validated.update(fields)
        return validated

    def publish(self, value, *, submission_id, metadata=None):
        publications = self.prepare(value, submission_id=submission_id, metadata=metadata)
        return self._publish(publications, submission_id=submission_id)[-1]

    def publish_many(self, values, *, submission_ids, metadata=None):
        publications, roots = [], []
        metadata = metadata if metadata is not None else [None] * len(values)
        for value, identity, fields in zip(values, submission_ids, metadata, strict=True):
            prepared = self.prepare(value, submission_id=identity, metadata=fields)
            offset = len(publications)
            for publication in prepared:
                publications.append(
                    Publication(
                        publication.records,
                        tuple(dep + offset if type(dep) is int else dep for dep in publication.dependencies),
                    )
                )
            roots.append(len(publications) - 1)
        refs = self._publish(publications, submission_id=f"bundle:{submission_ids[0]}")
        return [refs[i] for i in roots]

    def _publish(self, publications, *, submission_id):
        """Batch manifests; native Straw streams payloads with bounded scratch."""
        refs = []
        start = 0
        while start < len(publications):
            stop, count = start, 0
            while stop < len(publications):
                records = publications[stop].records
                if stop > start and count + len(records) + 1 > MAX_RECORDS:
                    break
                count += len(records) + 1
                stop += 1
            batch = [
                Publication(
                    publication.records,
                    tuple(
                        (refs[dep] if dep < start else dep - start) if type(dep) is int else dep
                        for dep in publication.dependencies
                    ),
                )
                for publication in publications[start:stop]
            ]
            refs.extend(self.store.publish_many(batch, submission_id=f"{submission_id}:part:{start}"))
            start = stop
        return refs

    def prepare(self, value, *, submission_id, metadata=None):
        blobs, aliases, dependencies = [], {}, []
        dependency_positions = {}
        tensor_sources = []
        token_count = 0

        def tensor(value, *, lazy=False, kind=None, validated=False):
            identity = (id(value), lazy, kind, validated)
            if identity in aliases:
                return aliases[identity]
            # Converted partial route chunks must stay alive while aliases are
            # indexed by object identity across all samples in this publication.
            tensor_sources.append(value)
            if isinstance(value, QueueTensorRef) and value.shares_storage(self.store):
                dependency, ordinal = value.share(
                    self.store,
                    submission_id=f"{submission_id}:retain:{len(dependencies)}",
                )
                if dependency not in dependency_positions:
                    dependency_positions[dependency] = len(dependencies)
                    dependencies.append(dependency)
                node = [
                    "tensor",
                    {
                        "dependency": dependency_positions[dependency],
                        "ordinal": ordinal,
                    },
                    list(value.shape),
                    value.dtype,
                    lazy,
                    value.kind,
                    validated or value.validated,
                ]
                aliases[identity] = node
                return node
            if isinstance(value, QueueTensorRef):
                kind, validated = kind or value.kind, validated or value.validated
                value = value.load()
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(np.array(value, copy=True))
            record = tensor_record(value, f"{submission_id}:tensor:{len(blobs)}", kind=kind)
            node = [
                "tensor",
                len(blobs),
                record.metadata["shape"],
                record.metadata["dtype"],
                lazy,
                kind,
                validated,
            ]
            aliases[identity] = node
            blobs.append(record)
            return node

        active = set()

        def visit(value):
            nonlocal token_count
            from slime.data.transport import DiskPayloadRef, RawRolloutRef, RolloutGroupRef, TrainBatchRef

            if isinstance(value, DiskPayloadRef):
                if value.manifest not in dependency_positions:
                    dependency_positions[value.manifest] = len(dependencies)
                    dependencies.append(value.manifest)
                dependency_index = dependency_positions[value.manifest]
                extra = {}
                if isinstance(value, RolloutGroupRef):
                    extra = {
                        "index": value.index,
                        "receipt": (dataclasses.asdict(value.receipt) if value.receipt else None),
                        "source_positions": value.source_positions,
                    }
                elif isinstance(value, RawRolloutRef):
                    extra = {
                        "receipt": dataclasses.asdict(value.receipt),
                        "metrics": value.metrics,
                    }
                elif isinstance(value, TrainBatchRef):
                    extra = {
                        "batch_id": value.batch_id,
                        "rank": value.rank,
                        "plan_digest": value.plan_digest,
                        "selection": value.selection,
                        "schedule": value.schedule,
                    }
                if value.path:
                    extra["path"] = value.path
                if value.sample_metadata is not None:
                    extra["sample_metadata"] = value.sample_metadata
                return [
                    "rollout-ref",
                    type(value).__name__,
                    dependency_index,
                    visit(extra),
                ]
            if value is None or type(value) in (str, bool, int):
                return value
            if type(value) is float:
                return value if math.isfinite(value) else ["float", repr(value)]
            if isinstance(value, (QueueTensorRef, torch.Tensor)):
                return tensor(value, lazy=isinstance(value, QueueTensorRef))
            if isinstance(value, np.ndarray):
                return ["ndarray", tensor(value)]
            if isinstance(value, np.generic):
                return visit(value.item())
            if isinstance(value, (bytes, bytearray)):
                return ["bytes", base64.b64encode(value).decode("ascii")]
            if isinstance(value, Sample.Status):
                return ["status", value.value]
            if isinstance(value, RecordSetRef):
                if value not in dependency_positions:
                    dependency_positions[value] = len(dependencies)
                    dependencies.append(value)
                return ["record-set", dependency_positions[value]]
            identity = id(value)
            if identity in active:
                raise TypeError("Cyclic custom fields are not supported by the durable Sample codec")
            active.add(identity)
            try:
                if isinstance(value, Sample):
                    # Include dynamic fields; unsupported values fail explicitly.
                    token_count += len(value.tokens)
                    validated = self._prepare_sample(value)
                    fields = []
                    for key, child in vars(value).items():
                        if child is not None and key in {
                            "rollout_routed_experts",
                            "rollout_topk_token_ids",
                            "rollout_topk_log_probs",
                            "rollout_top_p_token_ids",
                            "rollout_top_p_token_offsets",
                            "rollout_top_p_log_probs",
                        }:
                            if key == "rollout_routed_experts" and not isinstance(child, QueueTensorRef):
                                child = value.materialize_rollout_routed_experts(replace=False)
                            elif not isinstance(child, QueueTensorRef):
                                array = np.asarray(child)
                                # Partial/unvalidated captures must survive exactly;
                                # narrow only after validation, or type an empty field.
                                dtype = None
                                if key in validated or array.size == 0:
                                    dtype = torch.float32 if key.endswith("log_probs") else torch.int32
                                child = torch.as_tensor(array.copy(), dtype=dtype)
                            node = tensor(child, lazy=True, kind=key, validated=key in validated)
                        else:
                            node = visit(child)
                        fields.append([visit(key), node])
                    return ["sample", ["dict", fields]]
                if isinstance(value, Sample.SpecInfo):
                    return ["spec-info", visit(vars(value))]
                if isinstance(value, Sample.PrefixCacheInfo):
                    return ["prefix-cache-info", visit(vars(value))]
                from slime.rollout.filter_hub.base_types import DynamicFilterOutput

                if isinstance(value, DynamicFilterOutput):
                    return ["dynamic-filter", visit(vars(value))]
                if isinstance(value, dict):
                    return [
                        "dict",
                        [[visit(key), visit(item)] for key, item in value.items()],
                    ]
                if isinstance(value, (list, tuple)):
                    return [
                        "tuple" if isinstance(value, tuple) else "list",
                        [visit(item) for item in value],
                    ]
                # Preserve raw multimodal images losslessly using an explicit codec.
                from PIL import Image

                if isinstance(value, Image.Image):
                    import io

                    output = io.BytesIO()
                    value.save(output, format="PNG")
                    return [
                        "image-png",
                        base64.b64encode(output.getvalue()).decode("ascii"),
                    ]
                raise TypeError(
                    f"Unsupported durable Sample field type: {type(value).__module__}.{type(value).__qualname__}"
                )
            finally:
                active.remove(identity)

        tree = visit(value)
        publications = []
        if blobs:
            positions = {}
            start = 0
            while start < len(blobs):
                stop = min(len(blobs), start + MAX_RECORDS - 1)
                dependency_index = len(dependencies)
                dependencies.append(len(publications))
                publications.append(Publication(tuple(blobs[start:stop])))
                for index in range(start, stop):
                    positions[index] = {"dependency": dependency_index, "ordinal": index - start}
                start = stop
            for node in aliases.values():
                if isinstance(node[1], int):
                    node[1] = positions[node[1]]
        payload = encode({"version": 1, "tree": tree})
        publications.append(
            Publication(
                (
                    Record(
                        submission_id,
                        payload,
                        "slime.v1",
                        metadata or {},
                        tokens=token_count,
                    ),
                ),
                tuple(dependencies),
            )
        )
        return publications

    def load(self, ref, *, reader=None, selection=None):
        if reader is None:
            with self.store.read_session() as reader:
                return self.load(ref, reader=reader, selection=selection)
        if reader.run_id != self.store.run_id or reader.backend.root != self.store.backend.root:
            raise InvalidReference("Sample reader belongs to a different storage root or run")
        records = list(reader.read(ref))
        if len(records) != 1 or records[0].codec != "slime.v1":
            raise UnsupportedSchema("Expected one slime.v1 collection record")
        payload = decode(records[0].payload)
        if payload.get("version") != 1:
            raise UnsupportedSchema("Unsupported Sample codec version")
        aliases = {}
        tensor_sets = {}
        manifest = reader.manifest(ref)
        dependencies = [RecordSetRef.from_dict(value) for value in manifest["dependencies"]]

        def dependency(index):
            if type(index) is not int or not 0 <= index < len(dependencies):
                raise CorruptData("Sample codec dependency index is outside the manifest")
            return dependencies[index]

        def visit(node):
            if not isinstance(node, list):
                return node
            tag = node[0]
            if tag == "rollout-ref":
                from straw.protocol import CommitReceipt

                from slime.data.transport import DiskPayloadRef, RawRolloutRef, RolloutGroupRef, TrainBatchRef

                classes = {
                    cls.__name__: cls
                    for cls in (
                        DiskPayloadRef,
                        RawRolloutRef,
                        RolloutGroupRef,
                        TrainBatchRef,
                    )
                }
                if node[1] not in classes:
                    raise UnsupportedSchema(f"Unknown rollout reference type: {node[1]}")
                extra = visit(node[3])
                if extra.get("receipt"):
                    extra["receipt"] = CommitReceipt.from_dict(extra["receipt"])
                ref = dependency(node[2])
                return classes[node[1]](ref, str(self.store.backend.root), **extra)
            if tag == "dict":
                return {visit(key): visit(value) for key, value in node[1]}
            if tag in {"list", "tuple"}:
                value = [visit(item) for item in node[1]]
                return tuple(value) if tag == "tuple" else value
            if tag == "sample":
                values = visit(node[1])
                fields = {field.name for field in dataclasses.fields(Sample)}
                sample = Sample(**{key: value for key, value in values.items() if key in fields})
                for key, value in values.items():
                    if key not in fields:
                        setattr(sample, key, value)
                return sample
            if tag == "spec-info":
                return Sample.SpecInfo(**visit(node[1]))
            if tag == "prefix-cache-info":
                return Sample.PrefixCacheInfo(**visit(node[1]))
            if tag == "dynamic-filter":
                from slime.rollout.filter_hub.base_types import DynamicFilterOutput

                return DynamicFilterOutput(**visit(node[1]))
            if tag == "status":
                return Sample.Status(node[1])
            if tag == "float":
                if node[1] not in {"inf", "-inf", "nan"}:
                    raise CorruptData("Invalid special float")
                return float(node[1])
            if tag == "bytes":
                return base64.b64decode(node[1], validate=True)
            if tag == "record-set":
                value = dependency(node[1])
                reader.validate(value)
                return value
            if tag == "tensor":
                key = (encode(node[1]), node[4])
                if key not in aliases:
                    dep = node[1]["dependency"]
                    if dep not in tensor_sets:
                        tensor_sets[dep] = QueueTensorRef.from_record_set_many(
                            self.store, dependency(dep), reader=reader
                        )
                    ordinal = node[1]["ordinal"]
                    if type(ordinal) is not int or ordinal not in tensor_sets[dep]:
                        raise CorruptData("Tensor ordinal does not name a tensor in the publication")
                    value = dataclasses.replace(tensor_sets[dep][ordinal], validated=node[6])
                    if value.shape != tuple(node[2]) or value.dtype != node[3]:
                        raise CorruptData("Sample tensor descriptor disagrees with its storage record")
                    value = dataclasses.replace(value, kind=node[5])
                    aliases[key] = value if node[4] else value.load(reader=reader)
                return aliases[key]
            if tag == "ndarray":
                return visit(node[1]).numpy()
            if tag == "image-png":
                import io

                from PIL import Image

                image = Image.open(io.BytesIO(base64.b64decode(node[1], validate=True)))
                image.load()
                return image
            raise UnsupportedSchema(f"Unknown Sample codec tag: {tag!r}")

        tree = payload["tree"]
        if selection is None:
            return visit(tree)
        if tree[0] != "dict":
            raise ValueError("A column selection requires a dictionary payload")
        result = {}
        for key, node in tree[1]:
            if key not in selection:
                continue
            indices = selection[key]
            if indices is None:
                result[key] = visit(node)
            elif node[0] in {"list", "tuple"}:
                # Select descriptors before opening tensor records. Other DP
                # ranks' token/mask/R3 payloads are never read or materialized.
                result[key] = [visit(node[1][index]) for index in indices]
            else:
                result[key] = visit(node)[indices]
        return result
