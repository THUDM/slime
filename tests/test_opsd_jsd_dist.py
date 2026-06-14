"""Distributed (tensor-parallel) test for the OPSD vocab-parallel JSD.

Spawns 2 gloo workers, shards the vocabulary across them, and checks that the
per-token JSD value *and the student-logit gradient* match a single-process dense
computation. This is the configuration (TP > 1) that exercises the cross-rank
gradient coupling through the softmax normalizer — a TP=1 test cannot catch a
wrong all-reduce backward.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from slime.utils.ppo_utils import compute_vocab_parallel_jsd

N, V = 4, 8


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _full_logits():
    g = torch.Generator().manual_seed(1234)
    student = torch.randn(N, V, dtype=torch.float64, generator=g)
    teacher = torch.randn(N, V, dtype=torch.float64, generator=g)
    return student, teacher


def _worker(rank: int, world_size: int, master_port: int, beta: float, result_dir: str) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    student, teacher = _full_logits()
    shard = V // world_size
    sl = slice(rank * shard, (rank + 1) * shard)
    s = student[:, sl].clone().requires_grad_(True)
    t = teacher[:, sl].clone()

    jsd = compute_vocab_parallel_jsd(s, t, beta, group)  # [N], replicated across ranks
    jsd.sum().backward()

    torch.save({"jsd": jsd.detach(), "grad": s.grad}, os.path.join(result_dir, f"rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_vocab_parallel_jsd_matches_dense_under_tp(beta, tmp_path):
    world_size = 2
    port = _free_port()
    mp.spawn(_worker, args=(world_size, port, beta, str(tmp_path)), nprocs=world_size, join=True)

    # Distributed results: jsd is replicated; reassemble the grad from vocab shards.
    rank_outs = [torch.load(os.path.join(str(tmp_path), f"rank{r}.pt")) for r in range(world_size)]
    jsd_dist = rank_outs[0]["jsd"]
    grad_dist = torch.cat([rank_outs[r]["grad"] for r in range(world_size)], dim=1)

    # Dense single-process reference (group=None) on the full vocab.
    student, teacher = _full_logits()
    s_full = student.clone().requires_grad_(True)
    jsd_dense = compute_vocab_parallel_jsd(s_full, teacher.clone(), beta, None)
    jsd_dense.sum().backward()

    assert torch.allclose(jsd_dist, jsd_dense, atol=1e-9), (jsd_dist - jsd_dense).abs().max()
    # The student-logit gradient is the key check: a wrong (identity) normalizer
    # backward under TP would corrupt this even though the forward value is correct.
    assert torch.allclose(grad_dist, s_full.grad, atol=1e-9), (grad_dist - s_full.grad).abs().max()
