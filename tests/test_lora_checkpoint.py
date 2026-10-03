"""Adapter-only checkpoint tests: save, load, round trip and every rejection path.

Runs on CPU. The production named-tensor iterator is redirected at plain
``named_parameters()`` because the production iterator needs a real Megatron
parallel state; the canonical-name convention is preserved so the checkpoint keys
are the same ones a real run produces.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import _lora_fakes  # noqa: F401  (installs the megatron stub before slime imports)
import pytest
import torch
from safetensors.torch import load_file, save_file
from _lora_fakes import FakeGPTModel, make_config, make_lora_args, patch_named_tensor_iterator

from slime.backends.megatron_utils.lora import (  # noqa: E402
    ADAPTER_CONFIG_FILE,
    ADAPTER_TRAINING_STATE_FILE,
    LoRACheckpointError,
    inject_lora,
    iter_lora_modules,
    load_lora_adapter,
    lora_state_dict,
    read_adapter_metadata,
    save_lora_adapter,
    validate_adapter_metadata,
)

NUM_GPUS = 0

RANK = 8


@pytest.fixture()
def lora_model(monkeypatch):
    patch_named_tensor_iterator(monkeypatch)
    torch.manual_seed(17)
    model = FakeGPTModel(num_layers=2, hidden=16, ffn=32)
    config = make_config(lora_rank=RANK, lora_alpha=float(2 * RANK))
    inject_lora(model, config)
    # Make the adapter non-trivial so a silently-dropped tensor cannot pass unnoticed.
    with torch.no_grad():
        for _, module in iter_lora_modules(model):
            module.lora_B.normal_(0.0, 0.1)
    return make_lora_args(lora_rank=RANK, lora_alpha=float(2 * RANK)), [model], config


# --------------------------------------------------------------------------- #
# state extraction
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_state_dict_contains_only_adapters(lora_model):
    args, model, _ = lora_model
    state = lora_state_dict(args, model)
    assert state, "expected adapter tensors"
    assert all(name.endswith((".lora_A", ".lora_B")) for name in state)
    assert all(name.startswith("module.module.decoder.layers.") for name in state)
    assert not any("weight" == name.rsplit(".", 1)[-1] for name in state)


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_save_writes_config_index_state_and_one_shard(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "step_5", config, rollout_id=5, policy_version="7")

    assert (path / ADAPTER_CONFIG_FILE).is_file()
    assert (path / ADAPTER_TRAINING_STATE_FILE).is_file()
    assert (path / "adapter_model.safetensors.index.json").is_file()
    assert (path / "adapter_model-tp0-pp0.safetensors").is_file()
    # No partial-write leftovers and no temporary directory survives.
    assert not list(tmp_path.glob(".*tmp-*"))
    assert not list(path.glob("*.partial"))
    assert sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) > 0

    metadata = read_adapter_metadata(path)
    assert metadata["rank"] == RANK
    assert metadata["alpha"] == float(2 * RANK)
    assert metadata["bias"] == "none"
    assert metadata["rollout_id"] == 5
    assert metadata["policy_version"] == "7"
    assert metadata["tensor_parallel_size"] == 1
    assert metadata["expert_tensor_parallel_size"] == 1
    assert metadata["target_modules"] == list(config.target_modules)


@pytest.mark.unit
def test_saved_shard_holds_no_base_weights(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    tensors = load_file(str(path / "adapter_model-tp0-pp0.safetensors"))
    assert tensors
    assert all(name.endswith((".lora_A", ".lora_B")) for name in tensors)
    base_numel = sum(param.numel() for name, param in model[0].named_parameters() if not param.requires_grad)
    assert sum(tensor.numel() for tensor in tensors.values()) < base_numel


@pytest.mark.unit
def test_training_state_documents_that_optimizer_state_is_absent(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=3, policy_version="3")
    payload = json.loads((path / ADAPTER_TRAINING_STATE_FILE).read_text(encoding="utf-8"))
    assert payload["contains_optimizer_state"] is False
    assert payload["contains_scheduler_state"] is False
    assert "--save" in payload["resume_note"]
    assert payload["rollout_id"] == 3 and payload["policy_version"] == "3"


@pytest.mark.unit
def test_save_replaces_an_existing_directory_atomically(lora_model, tmp_path):
    args, model, config = lora_model
    target = tmp_path / "adapter"
    save_lora_adapter(args, model, target, config, rollout_id=1)
    stale = target / "stale.txt"
    stale.write_text("junk", encoding="utf-8")
    save_lora_adapter(args, model, target, config, rollout_id=2)
    assert not stale.exists()
    assert read_adapter_metadata(target)["rollout_id"] == 2


# --------------------------------------------------------------------------- #
# load / round trip
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_save_load_round_trip_restores_the_adapter_exactly(lora_model, tmp_path, monkeypatch):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=9, policy_version="9")
    expected = {name: tensor.detach().clone() for name, tensor in lora_state_dict(args, model).items()}

    torch.manual_seed(999)
    fresh = FakeGPTModel(num_layers=2, hidden=16, ffn=32)
    inject_lora(fresh, config)
    with torch.no_grad():  # deliberately wrong values that must be overwritten
        for _, module in iter_lora_modules(fresh):
            module.lora_A.fill_(3.0)
            module.lora_B.fill_(4.0)

    training_state = load_lora_adapter(args, [fresh], path, config)
    restored = lora_state_dict(args, [fresh])
    assert set(restored) == set(expected)
    for name, tensor in restored.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert training_state["rollout_id"] == 9 and training_state["policy_version"] == "9"


@pytest.mark.unit
def test_round_trip_preserves_the_effective_policy(lora_model, tmp_path):
    """The point of the adapter checkpoint: the merged policy must be reproducible."""
    from slime.backends.megatron_utils.lora import lora_local_delta

    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    before = {name: lora_local_delta(module).clone() for name, module in iter_lora_modules(model)}

    torch.manual_seed(4242)
    fresh = FakeGPTModel(num_layers=2, hidden=16, ffn=32)
    inject_lora(fresh, config)
    load_lora_adapter(args, [fresh], path, config)
    after = {name: lora_local_delta(module) for name, module in iter_lora_modules(fresh)}

    assert set(before) == set(after)
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


@pytest.mark.unit
def test_missing_directory_is_rejected(lora_model, tmp_path):
    args, model, config = lora_model
    with pytest.raises(LoRACheckpointError, match="not a slime LoRA adapter checkpoint"):
        load_lora_adapter(args, model, tmp_path / "nope", config)


@pytest.mark.unit
def test_missing_shard_is_rejected(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    (path / "adapter_model-tp0-pp0.safetensors").unlink()
    with pytest.raises(LoRACheckpointError, match="missing from"):
        load_lora_adapter(args, model, path, config)


@pytest.mark.unit
def test_target_mismatch_is_rejected(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    other = make_config(lora_rank=RANK, lora_alpha=float(2 * RANK), lora_target_preset="dense_mlp")
    with pytest.raises(LoRACheckpointError, match="target_modules"):
        load_lora_adapter(args, model, path, other)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"rank": 16}, "rank"),
        ({"alpha": 999.0}, "alpha"),
        ({"format_version": 99}, "format_version"),
        ({"base_model_config_hash": "deadbeef"}, "base_model_config_hash"),
        ({"tensor_parallel_size": 8}, "tensor_parallel_size"),
        ({"pipeline_parallel_size": 4}, "pipeline_parallel_size"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size"),
    ],
)
def test_metadata_mismatch_is_rejected(lora_model, tmp_path, overrides, needle):
    import dataclasses

    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    metadata = read_adapter_metadata(path)
    metadata.update(overrides)
    if "base_model_config_hash" in overrides:
        config = dataclasses.replace(config, base_model_config_hash="a-different-hash")
    with pytest.raises(LoRACheckpointError) as excinfo:
        validate_adapter_metadata(metadata, config)
    assert needle in str(excinfo.value)


@pytest.mark.unit
def test_adapter_is_rejected_when_the_base_model_config_differs(lora_model, tmp_path):
    """An adapter must never be silently attached to a different base model."""
    import dataclasses

    args, model, config = lora_model
    config_a = dataclasses.replace(config, base_model_config_hash="hash-of-9B")
    path = save_lora_adapter(args, model, tmp_path / "adapter", config_a, rollout_id=0)
    config_b = dataclasses.replace(config, base_model_config_hash="hash-of-2B")
    with pytest.raises(LoRACheckpointError, match="different base model"):
        load_lora_adapter(args, model, path, config_b)


@pytest.mark.unit
def test_adapter_metadata_is_independent_of_expert_parallel_size(lora_model, tmp_path):
    """V1 has no expert adapters, so EP is provenance rather than a shard axis."""
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    metadata = read_adapter_metadata(path)
    metadata["expert_parallel_size"] = 8
    validate_adapter_metadata(metadata, config)


@pytest.mark.unit
def test_unknown_tensor_in_the_shard_is_rejected(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    shard = path / "adapter_model-tp0-pp0.safetensors"
    tensors = load_file(str(shard))
    tensors["module.module.decoder.layers.9.mlp.linear_fc1.lora_A"] = torch.zeros(RANK, 16)
    save_file(tensors, str(shard))
    with pytest.raises(LoRACheckpointError, match="unknown in checkpoint"):
        load_lora_adapter(args, model, path, config)


@pytest.mark.unit
def test_shape_mismatch_in_the_shard_is_rejected(lora_model, tmp_path):
    args, model, config = lora_model
    path = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=0)
    shard = path / "adapter_model-tp0-pp0.safetensors"
    tensors = load_file(str(shard))
    victim = next(name for name in tensors if name.endswith(".lora_A"))
    tensors[victim] = torch.zeros(RANK + 1, tensors[victim].shape[1])
    save_file(tensors, str(shard))
    with pytest.raises(LoRACheckpointError, match="Shape mismatch"):
        load_lora_adapter(args, model, path, config)


@pytest.mark.unit
def test_failed_publication_keeps_previous_checkpoint(lora_model, tmp_path, monkeypatch):
    from slime.backends.megatron_utils.lora import state

    args, model, config = lora_model
    target = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=1)
    previous = target.resolve()
    original_replace = state.os.replace

    def fail_publish(src, dst):
        if str(dst) == str(target):
            assert target.resolve() == previous
            raise OSError("simulated publication failure")
        return original_replace(src, dst)

    monkeypatch.setattr(state.os, "replace", fail_publish)
    with pytest.raises(LoRACheckpointError, match="Previous checkpoint remains"):
        save_lora_adapter(args, model, target, config, rollout_id=2)
    assert target.resolve() == previous
    assert read_adapter_metadata(target)["rollout_id"] == 1


@pytest.mark.unit
def test_load_pins_one_version_during_concurrent_publication(lora_model, tmp_path, monkeypatch):
    from slime.backends.megatron_utils.lora import state

    args, model, config = lora_model
    target = save_lora_adapter(args, model, tmp_path / "adapter", config, rollout_id=1)
    previous = target.resolve()
    expected = {name: p.clone() for name, p in lora_state_dict(args, model).items()}
    original_validate = state.validate_adapter_metadata

    def publish_after_reading_metadata(metadata, cfg):
        original_validate(metadata, cfg)
        with torch.no_grad():
            for p in lora_state_dict(args, model).values():
                p.add_(1)
        save_lora_adapter(args, model, target, config, rollout_id=2)

    monkeypatch.setattr(state, "validate_adapter_metadata", publish_after_reading_metadata)
    training_state = load_lora_adapter(args, model, target, config)
    assert training_state["rollout_id"] == 1
    assert target.resolve() != previous
    assert previous.is_dir()
    for name, p in lora_state_dict(args, model).items():
        torch.testing.assert_close(p, expected[name], rtol=0, atol=0)


@pytest.mark.unit
def test_save_never_deletes_a_legacy_checkpoint_directory(lora_model, tmp_path):
    args, model, config = lora_model
    target = tmp_path / "legacy"
    target.mkdir()
    sentinel = target / "checkpoint"
    sentinel.write_text("keep me")
    with pytest.raises(LoRACheckpointError, match="Cannot atomically overwrite"):
        save_lora_adapter(args, model, target, config)
    assert sentinel.read_text() == "keep me"


@pytest.mark.unit
def test_routed_adapter_roundtrip_and_legacy_layout_rejection(monkeypatch, tmp_path):
    from _lora_fakes import FakeMoEModel

    patch_named_tensor_iterator(monkeypatch)
    model = [FakeMoEModel(num_layers=1)]
    config = make_config(lora_rank=RANK, lora_target_preset="moe_routed_experts")
    args = make_lora_args(lora_rank=RANK, lora_target_preset="moe_routed_experts")
    inject_lora(model[0], config)
    with torch.no_grad():
        for _, module in iter_lora_modules(model):
            module.lora_B.normal_()
    expected = {name: tensor.clone() for name, tensor in lora_state_dict(args, model).items()}
    path = save_lora_adapter(args, model, tmp_path / "adapter", config)
    with torch.no_grad():
        for tensor in lora_state_dict(args, model).values():
            tensor.zero_()
    load_lora_adapter(args, model, path, config)
    for name, tensor in lora_state_dict(args, model).items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

    metadata = read_adapter_metadata(path)
    del metadata["expert_tensor_parallel_size"]
    (path / ADAPTER_CONFIG_FILE).write_text(json.dumps(metadata))
    with pytest.raises(LoRACheckpointError, match="no expert_tensor_parallel_size metadata"):
        load_lora_adapter(args, model, path, config)


@pytest.mark.unit
@pytest.mark.parametrize("available", [[], ["decoder.fc.lora_A"], ["decoder.fc.lora_A", "decoder.fc.lora_B"]])
def test_megatron_checkpoint_requires_all_adapter_keys(monkeypatch, tmp_path, available):
    from slime.backends.megatron_utils.lora.state import validate_megatron_lora_checkpoint

    serialization = types.ModuleType("megatron.core.dist_checkpointing.serialization")
    serialization.load_tensors_metadata = lambda path: dict.fromkeys(available)
    monkeypatch.setitem(sys.modules, serialization.__name__, serialization)
    model = [SimpleNamespace(sharded_state_dict=lambda: {
        "nested": [
            SimpleNamespace(key="decoder.fc.lora_A"),
            {"factor": SimpleNamespace(key="decoder.fc.lora_B")},
            SimpleNamespace(key="decoder.fc.weight"),
        ]
    })]
    if len(available) == 2:
        validate_megatron_lora_checkpoint(tmp_path, model)
    else:
        with pytest.raises(LoRACheckpointError, match="missing requested LoRA tensors"):
            validate_megatron_lora_checkpoint(tmp_path, model)


@pytest.mark.unit
def test_megatron_checkpoint_unreadable_metadata_fails_closed(monkeypatch, tmp_path):
    from slime.backends.megatron_utils.lora.state import validate_megatron_lora_checkpoint

    serialization = types.ModuleType("megatron.core.dist_checkpointing.serialization")

    def unreadable(path):
        raise FileNotFoundError(path)

    serialization.load_tensors_metadata = unreadable
    monkeypatch.setitem(sys.modules, serialization.__name__, serialization)
    with pytest.raises(LoRACheckpointError, match="Cannot inspect LoRA tensor metadata"):
        validate_megatron_lora_checkpoint(tmp_path, [])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
