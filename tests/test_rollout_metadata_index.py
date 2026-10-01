import json
from types import SimpleNamespace

import pytest
import torch
from straw import SharedFilesystemStore
from straw.tensor import TensorRef

from slime.data.archive import RolloutArchive
from slime.data.sample_metadata import describe_sample, validate_sample_metadata
from slime.utils.types import Sample


def sample():
    return Sample(
        index=0,
        group_index=0,
        tokens=[9, 1, 2],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[0.0, 0.0],
        status=Sample.Status.COMPLETED,
        rollout_routed_experts=torch.zeros((2, 3, 2), dtype=torch.int32),
        rollout_top_p_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
        rollout_top_p_log_probs=torch.zeros(2),
    )


def args():
    return SimpleNamespace(
        use_rollout_routing_replay=True, use_score_centering=True, num_layers=3, moe_router_topk=2, rollout_top_p=0.95
    )


def test_index_only_check_never_opens_straw_or_reads_tensors(tmp_path, monkeypatch):
    path = tmp_path / "rollout.straw.json"
    RolloutArchive.save(path, [sample()], rollout_id=0)

    def forbidden(*a, **kw):
        raise AssertionError("Metadata check opened a data store or tensor payload")

    monkeypatch.setattr(SharedFilesystemStore, "__init__", forbidden)
    monkeypatch.setattr(TensorRef, "load", forbidden)
    metadata = RolloutArchive.check_metadata(path, args())
    assert len(metadata) == 1
    assert metadata[0]["response_length"] == 2


@pytest.mark.parametrize("corruption", ["tokens", "mask", "routes", "offsets", "logps", "ids_dtype"])
def test_metadata_rejects_shape_and_dtype_mismatch(corruption):
    value = describe_sample(sample())
    if corruption == "tokens":
        value["tokens"] = 1
    if corruption == "mask":
        value["loss_mask"] = 1
    if corruption == "routes":
        value["tensors"]["rollout_routed_experts"]["shape"][1] = 4
    if corruption == "offsets":
        value["tensors"]["rollout_top_p_token_offsets"]["shape"] = [2]
    if corruption == "logps":
        value["tensors"]["rollout_top_p_log_probs"]["shape"] = [3]
    if corruption == "ids_dtype":
        value["tensors"]["rollout_top_p_token_ids"]["dtype"] = "float32"
    with pytest.raises(ValueError):
        validate_sample_metadata([value], args())


def test_legacy_index_requires_explicit_sidecar_and_checks_identity(tmp_path):
    path = tmp_path / "rollout.straw.json"
    RolloutArchive.save(path, [sample()], rollout_id=0)
    index = json.loads(path.read_text())
    metadata = index.pop("sample_metadata")
    path.write_text(json.dumps(index))
    with pytest.raises(FileNotFoundError):
        RolloutArchive.check_metadata(path, args())
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    document = {"manifest_digest": index["manifest"]["digest"], "sample_metadata": metadata}
    sidecar.write_text(json.dumps(document))
    assert len(RolloutArchive.check_metadata(path, args())) == 1
    document["manifest_digest"] = "wrong"
    sidecar.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="different archive"):
        RolloutArchive.check_metadata(path, args())


def test_metadata_counts_routed_chunks_without_materializing():
    value = sample()
    value.rollout_routed_experts = [
        torch.zeros((1, 3, 2), dtype=torch.int32),
        torch.zeros((2, 3, 2), dtype=torch.int32),
    ]
    metadata = describe_sample(value)
    assert metadata["tensors"]["rollout_routed_experts"] == {"shape": [3, 3, 2], "dtype": "int32"}
