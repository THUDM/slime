# P2P Shard Weight Sync

- [Why](#why)
- [Quick Start](#quick-start)
- [Prerequisites and Automatic Fallback](#prerequisites-and-automatic-fallback)
- [How It Works](#how-it-works)
- [Comparison with Default Broadcast](#comparison-with-default-broadcast)
- [FAQ](#faq)

## Why

In non-colocate mode, slime's default path **all_gathers** Megatron TP shards and **NCCL-broadcasts** full HF weights to SGLang rollout engines. For large models, gathering and rebroadcasting every parameter dominates the sync phase.

P2P shard sync lets each training TP rank send its local shard directly to the matching inference TP rank via `dist.send/recv`, skipping the full gather/broadcast.

## Quick Start

Use the **official slime Docker image** (the SGLang P2P patch is applied automatically at build time, after `sglang.patch` and `sglang-top_p.patch` — no manual steps).

Typical non-colocate setup for Qwen3-4B with Megatron TP=4 and one engine using 4 GPUs:

```bash
python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --tensor-model-parallel-size 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  --megatron-to-hf-mode bridge \
  --update-weight-mode full \
  --use-p2p-weight-update \
  --hf-checkpoint /path/to/Qwen3-4B \
  ...
```

Key flags:

| Flag | Meaning |
|---|---|
| `--use-p2p-weight-update` | Enable P2P shard sync (falls back to broadcast when preconditions fail) |
| `--update-weight-mode full` | P2P requires full weight sync |
| `--megatron-to-hf-mode bridge` | **Required** — P2P relies on Megatron Bridge weight layout |
| `--tensor-model-parallel-size` | Training TP; must match SGLang TP |
| `--rollout-num-gpus-per-engine` | GPUs per engine; SGLang TP = this value / `sglang_pp_size` |

## Prerequisites and Automatic Fallback

With `--use-p2p-weight-update`, slime checks the conditions below at startup. **If any fail, NCCL broadcast is used instead**, and rank 0 prints `[P2P] ... using NCCL broadcast weight update instead.`:

| Condition | Notes |
|---|---|
| Non-colocate | Colocate uses `UpdateWeightFromTensor`, not P2P |
| `--megatron-to-hf-mode bridge` | Raw mode does not support P2P |
| Shard-level conversion implemented | Currently **Qwen2 / Qwen3 dense**; MoE and other architectures fall back |
| Megatron TP == SGLang TP | Every rollout engine must align with training TP |
| SGLang PP == 1 | P2P send/recv pairing requires `sglang_pp_size=1`; otherwise falls back |

TP alignment:

```
Megatron TP  = tensor_model_parallel_size
SGLang TP    = rollout_num_gpus_per_engine / sglang_pp_size
```

## How It Works

1. **Vocab params** (embed / lm_head): small TP all_gather, strip Megatron padding, slice to SGLang shard boundaries.
2. **Other params**: shard-level Megatron→HF conversion (`convert_shard_to_hf`) without all_gather.
3. **Per bucket**: `all_gather_object` metadata → rank-0 HTTP notify SGLang → NCCL barrier → parallel `dist.send` → `ray.get` for load completion.

Implementation: `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_p2p.py`.

## Comparison with Default Broadcast

| | NCCL broadcast (default) | P2P shard |
|---|---|---|
| Trainer | TP all_gather → full HF weights | Each rank converts/sends its shard only |
| Communication | Broadcast full chunks | `dist.send/recv` per shard |
| Models | Any bridge-supported model | Currently Qwen2/Qwen3 dense + bridge |
| TP requirement | No strict alignment | Megatron TP == SGLang TP |

Omit `--use-p2p-weight-update` to use the default broadcast path.

## FAQ

**Q: P2P is enabled but logs show NCCL broadcast?**  
A: Check the rank-0 `[P2P]` message. Common causes: `--megatron-to-hf-mode raw`, unsupported/MoE models, TP mismatch, or `sglang_pp_size > 1`.

**Q: Relation to delta sync?**  
A: Mutually exclusive. P2P is under `--update-weight-mode full`; see [Delta Weight Sync](delta-weight-sync.md) for sparse updates.
