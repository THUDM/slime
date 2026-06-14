"""Multi-rank test for Gemma4's layer_scalar rank-0-read + broadcast path.

The single-process tests in ``test_gemma4_provider.py`` confirm the
safetensors-read / PP-offset logic, but they never exercise the real
distributed path: ``_read_layer_scalars_from_safetensors`` runs only on
rank 0, then ``_broadcast_layer_scalars`` fans the dict out to the rest
via ``torch.distributed.broadcast_object_list``. A regression where
rank > 0 ends up with ``None`` or the default 1.0 would silently drift
activations on every forward pass — caught only much later by parity
tests.

This test spawns 2 gloo ranks (no CUDA required), has rank 0 fabricate a
safetensors ckpt in a tmpdir, then both ranks call ``_load_layer_scalars``
on a minimal inner model. We then assert the loaded scalars match on both
ranks.
"""

import json
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, master_port: int, ckpt_dir: str, out_dir: str):
    """Run on each spawned rank: init PG, build a fake inner model,
    call _load_layer_scalars, write the resulting scalars to a per-rank
    file so the parent process can diff them."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from slime_plugins.models import gemma4_provider as _provider
        import megatron.core.transformer.transformer_layer as tl

        inner = torch.nn.Module()
        inner.decoder = torch.nn.Module()
        layers = []
        for _ in range(3):
            layer = torch.nn.Module()
            layer.register_buffer("layer_scalar", torch.ones(1))
            layers.append(layer)
        inner.decoder.layers = torch.nn.ModuleList(layers)

        orig_offset = tl.get_transformer_layer_offset
        tl.get_transformer_layer_offset = lambda _cfg: 0
        try:
            _provider._load_layer_scalars(
                inner, ckpt_dir, config=type("C", (), {})()
            )
        finally:
            tl.get_transformer_layer_offset = orig_offset

        loaded = [layer.layer_scalar.item() for layer in inner.decoder.layers]
        out_path = os.path.join(out_dir, f"rank{rank}.json")
        with open(out_path, "w") as fp:
            json.dump({"rank": rank, "scalars": loaded}, fp)
    finally:
        dist.destroy_process_group()


def _write_fake_checkpoint(ckpt_dir: str, scalars: dict[int, float]) -> None:
    """Produce a safetensors file per layer + a matching
    model.safetensors.index.json. Matches
    _read_layer_scalars_from_safetensors' expectations.
    """
    from safetensors.torch import save_file

    weight_map = {}
    for layer_idx, value in scalars.items():
        tensor_name = f"model.language_model.layers.{layer_idx}.layer_scalar"
        fname = f"layer_{layer_idx}.safetensors"
        save_file(
            {tensor_name: torch.tensor([value], dtype=torch.float32)},
            os.path.join(ckpt_dir, fname),
        )
        weight_map[tensor_name] = fname

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "w") as fp:
        json.dump({"metadata": {}, "weight_map": weight_map}, fp)


def test_layer_scalars_broadcast_to_all_ranks():
    """Rank 0 reads scalars from disk, broadcasts to rank 1 via
    broadcast_object_list — both ranks must end up with identical
    layer_scalar values."""
    expected = {0: 0.5, 1: 1.25, 2: 2.0}

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, "ckpt")
        os.makedirs(ckpt_dir)
        _write_fake_checkpoint(ckpt_dir, expected)

        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        # Deterministic port; tests serialized per-module so no race.
        master_port = 29577

        mp.spawn(
            _worker,
            args=(2, master_port, ckpt_dir, out_dir),
            nprocs=2,
            join=True,
        )

        with open(os.path.join(out_dir, "rank0.json")) as fp:
            r0 = json.load(fp)
        with open(os.path.join(out_dir, "rank1.json")) as fp:
            r1 = json.load(fp)

    assert r0["rank"] == 0
    assert r1["rank"] == 1
    assert r0["scalars"] == pytest.approx([0.5, 1.25, 2.0])
    assert r1["scalars"] == pytest.approx([0.5, 1.25, 2.0]), (
        "rank 1 did not receive the broadcast scalars — check "
        "_broadcast_layer_scalars"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
