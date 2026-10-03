# LoRA

LoRA (Low-Rank Adaptation) freezes the base model and trains only a low-rank
update per adapted linear layer:

$$
W_{\mathrm{eff}} = W_{\mathrm{base}} + \frac{\alpha}{r} B A,
\qquad A \in \mathbb{R}^{r \times d_{\mathrm{in}}},\; B \in \mathbb{R}^{d_{\mathrm{out}} \times r}.
$$

$A$ is Kaiming-uniform initialised and $B$ is zero, so the adapted model starts
with a mathematically zero adapter delta. Low-precision rounding can still differ across kernels or merged execution. In slime, LoRA is off by default: existing
scripts need no new flags and their weight-sync payload stays byte-for-byte the
same.

Enable it with `--use-lora`:

```bash
--use-lora \
--lora-rank 32 \
--lora-alpha 64 \
--lora-target-preset moe_language_all
```

## Usage and run modes

Start with a working slime Megatron model configuration, HF base checkpoint,
dataset, and rollout configuration, then add LoRA arguments. Presets only select
modules; they do not add Megatron/HF conversion or SGLang support for a new model.
The following is an argument fragment for an existing training command, not a
standalone launch command.

```bash
# Fresh training from HF base weights, adapting dense attention and MLP.
--hf-checkpoint /models/base \
--load /models/base \
--use-lora \
--lora-target-preset dense_language \
--lora-rank 32 \
--lora-alpha 64 \
--lora-learning-rate 1e-5 \
--lora-weight-decay 0.0 \
--save /checkpoints/full \
--save-interval 20 \
--save-lora '/checkpoints/adapters/{rollout_id}'
```

`--hf-checkpoint` supplies model configuration, tokenizer, and conversion metadata;
`--load` selects the weights to load. For a fresh run, point both at the same HF
base. For resume, change `--load` to the full LoRA checkpoint and keep
`--hf-checkpoint` pointing at the matching HF base. Do not pass an adapter directory
to `--load`.

| Purpose | Loading arguments | Adapter / training-state behavior |
|---------|-------------------|-----------------------------------|
| Start from base weights | `--load HF_DIR`, no `--lora-load` | Automatically enters finetune mode; initializes a new adapter without restoring an old optimizer. |
| Start a new run with an adapter | `--load HF_DIR --lora-load ADAPTER_DIR` | Automatically enters finetune mode; loads the adapter with fresh training state. Explicit `--finetune` is also allowed. |
| Resume full training | `--load FULL_LORA_DIR`, no `--finetune` | Restores the adapter from the full checkpoint; optimizer/RNG restoration follows existing save/load flags. Ignores `--lora-load`. |
| Start a new run from full LoRA weights | `--load FULL_LORA_DIR --finetune` | Keeps the saved adapter weights without resuming the training state; an additional `--lora-load` replaces those adapter weights. |
| Release and recreate the actor | Add `--release-train` to the initial loading mode above | After saving, automatically resumes the full checkpoint from `--save` with finetune disabled; does not reload the original `--lora-load`. |

All rows require `--use-lora` and matching rank, alpha, and target settings. Do not
use `--no-save-optim` / `--no-load-optim` for full optimizer resume; an adapter export
alone cannot restore the optimizer. `--save-lora` adds an export, does not select a
loading mode, and does not replace `--save`.

`--release-train` also requires the existing lifecycle configuration:

```bash
--release-train \
--save /checkpoints/full \
--update-weight-mode full \
--update-weight-transport disk \
--update-weight-disk-dir /shared/rollout-weights
```

The disk directory must be visible to the trainer and rollout engines. In this
mode, an omitted `--save-interval` defaults to 1. This controls the actor lifecycle,
not adapter-only saving.

See [run-qwen3-30B-A3B-lora.sh](../../../scripts/run-qwen3-30B-A3B-lora.sh) for a
model-specific eight-GPU MoE launch example, not validation of all MoE models.
Adjust model, dataset, checkpoint paths, and resources before use. Its startup
cleanup terminates related services and Python processes; use a dedicated job environment.

## Reference policy

When KL/reference computation is enabled, the reference is always the frozen base
policy with adapters disabled. Loading a nonzero SFT adapter via `--lora-load`
does not make “base + initial SFT adapter” the reference. That reference mode is
unsupported; `--ref-load` and `--ref-update-interval` cannot change this behavior.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-lora` | Enable LoRA. Freezes the base model and trains adapters only. Unrelated to architectural low-rank parameters such as `--q-lora-rank`. |
| `--lora-rank` | Rank $r$ (default 64). Must be > 0, divisible by the TP size for row-parallel targets, and divisible by the expert-TP size for routed-expert fc2 targets (see [Parallelism](#parallelism)). |
| `--lora-alpha` | Finite positive scaling numerator (default 128.0); the effective scale is `alpha / rank`. |
| `--lora-dropout` | Must stay `0.0`. Any stochastic LoRA forward makes the training pass differ from the rollout pass and breaks on-policy RL. |
| `--lora-target-preset` | Named set of target-module regexes; see [Target presets](#target-presets). Mutually exclusive with `--lora-target-modules`. |
| `--lora-target-modules` | Explicit regexes, matched with `re.search` against module names of the unwrapped Megatron model. |
| `--lora-exclude-modules` | Regexes removed from the matched set. Defaults to embedding / output layer / vision tower. |
| `--lora-learning-rate` | LR for adapter parameters; explicitly supplied values must be finite and positive. Falls back to `--lr`. |
| `--lora-weight-decay` | Finite nonnegative weight decay for adapter parameters (default 0.0). Unlike `--lora-learning-rate`, this **always** overrides `--weight-decay`, so set this one rather than `--weight-decay` in LoRA mode. |
| `--lora-load` / `--save-lora` | Adapter-only checkpoint directories. `--save-lora` supports `{rollout_id}`. |
| `--lora-save-merged-hf` | Optional directory template for a merged, LoRA-free HF checkpoint. |
| `--lora-rollout-sync-mode` | `merged` (default) materialises `W_base + alpha/r * B @ A` at the sync boundary and reuses the standard full-weight path. |
| `--lora-allow-replicated-modules` | Regex allowlist for replicated linears at TP>1; only use for modules with identical inputs and gradients across TP replicas. |
| `--lora-bias` | Only `none` is supported; base biases remain frozen. |
| `--lora-debug` | Per-step gradient diagnostics. Adds tensor scans and device syncs; off by default. |

## Target presets

Presets are named regex sets, so you do not have to know the mcore module paths
for each architecture.

| Preset | Adapts |
|--------|--------|
| `dense_attention` | `linear_qkv`, `linear_proj` |
| `dense_mlp` | `mlp.linear_fc1`, `mlp.linear_fc2` |
| `dense_language` (default) | dense attention + dense MLP |
| `hybrid_attention` / `hybrid_language` | adds `self_attention.linear_attn` gated-delta-net projections |
| `moe_attention` | attention only, on a MoE model |
| `moe_shared_mlp` | `mlp.shared_experts.linear_fc1` / `linear_fc2` |
| `moe_routed_experts` | `mlp.experts.linear_fc1` / `linear_fc2` (grouped GEMM) |
| `moe_mlp` | shared + routed experts |
| `moe_language` | attention + shared experts (conservative MoE preset) |
| `moe_language_all` | attention + shared + routed experts |

Built-in presets exclude the MoE **router**, and router adaptation is not a supported use case: adapting the gate changes token-to-expert
assignment, which is a different intervention from adapting the experts.
The embedding, output layer, and the vision tower are excluded by default; asking
for vision LoRA is a hard error rather than a silent skip.

## Routed experts

Routed experts in Megatron are grouped-GEMM modules (`TEColumnParallelGroupedLinear`
/ `TERowParallelGroupedLinear`) whose weights are packed as `weight0 .. weight{n-1}`,
one per local expert. slime attaches **one adapter shared by every expert of a
layer**, sharing both A and B. More precisely, fc1 and fc2 each have their own
A/B pair, and different layers do not share factors. EP ranks hold replicas of
the same logical adapter, synchronized at initialization and by cross-EP gradient
summation, not independently trained per-rank adapters. ETP shards the factors
according to the parallel layout. Independent per-expert A/B pairs are unsupported.
This does not imply compatibility with native
adapter formats in other frameworks.

This matters for three reasons:

* **Parameter count does not scale with the expert count.** A 256-expert layer
  costs the same adapter as a dense layer.
* **The forward hook never needs the routing map.** The delta is simply
  $\frac{\alpha}{r} B A x$ over the whole permuted token batch, independent of
  which expert a token was dispatched to.
* **Merging is mathematically equivalent.** At the sync boundary the same delta is added to every
  `weight{i}`, so compare merged and adapter forward passes with appropriate low-precision tolerances.

Routed-expert LoRA is enabled by `moe_routed_experts`, `moe_mlp`, or
`moe_language_all`. Configure the model with the TE grouped-linear layout used
by `--moe-grouped-gemm`; other expert weight layouts are not automatically compatible.

## Parallelism

slime shards the adapter to match the base layer, so the adapter never introduces
an unaccounted collective:

| Target | `lora_A` | `lora_B` | Communication |
|--------|----------|----------|---------------|
| Column-parallel (`linear_qkv`, `linear_fc1`) | sharded on `dim 1` | sharded on `dim 0` | input copy, no output reduce |
| Row-parallel (`linear_proj`, `linear_fc2`) | sharded on `dim 1` | sharded on `dim 1` | all-reduce of the rank-$r$ intermediate |
| Routed expert fc1 (`expert_column`) | replicated | sharded on `dim 0` | none — the token dispatcher already gathered along the token axis |
| Routed expert fc2 (`expert_row`) | sharded on `dim 1` | sharded on `dim 1` | all-reduce over the **expert**-TP group only |

Consequences worth knowing:

* `--lora-rank` must be divisible by the tensor-parallel size for row-parallel
  targets, and by the expert tensor-parallel size for routed-expert fc2 targets. This
  is validated at injection time.
* Under expert parallelism (EP > 1), each EP rank routes different tokens through
  the same shared adapter, so its gradient is the **sum** over the expert-parallel
  group. slime marks expert adapters with `allreduce=False` so Megatron's DDP
  places them in the expert bucket and reduces them over the expert data-parallel
  group; the cross-EP summation is handled explicitly in the adapter's backward.
* Adapter checkpoints carry a distinct `replica_id` per EP replica, so the
  distributed-checkpoint writer picks exactly one writer per shard.
* At TP>1, LoRA refuses to inject into TP-replicated linears unless they are listed in
  `--lora-allow-replicated-modules`, because an unaccounted cross-TP gradient sync
  would silently desynchronise the replicas.

Expert adapter gradient buffers follow the expert topology when EP>1 or ETP!=TP,
and preserve stricter grouping attributes on the base expert weights. EP=1 alone
does not imply that ordinary DDP grouping is appropriate.

## Rollout synchronisation

With `--lora-rollout-sync-mode merged` (the default and currently the only
implemented mode), slime materialises `W_base + alpha/r * B @ A` for every adapted
tensor at the sync boundary and pushes ordinary full weights through the existing
NCCL / IPC / disk paths. The rollout engine never sees `lora_A` / `lora_B`, so:

* No SGLang-side LoRA support is required — including for routed experts, where
  fused MoE kernels would otherwise conflict with a separate adapter path.
* The merge is applied to a single coherent host snapshot and never mutates the
  training weights, so repeated syncs without a training step are idempotent.
* Sync reuses the existing full-weight update failure semantics. Atomic rollback
  across engines and buckets is not guaranteed; do not continue rollout after a
  partial update without restoring a consistent policy on all engines.

LoRA reduces trainable parameters, gradients, and optimizer state, but base weights
still need storage or the existing offload path. Merged synchronization updates
ordinary model weights and does not provide adapter-only transmission savings.
Measure memory and throughput with the intended parallelism, recomputation, merge,
and synchronization configuration.

`native_adapter` (pushing the raw adapter to the engine and letting SGLang apply
it) is reserved but explicitly **not implemented**; requesting it is a
fail-fast error rather than a silent fallback.

## Unsupported combinations

These are rejected during argument validation, model injection, or loading.
Module-specific restrictions can only be checked after model construction:

* MLA models (including custom target selections)
* `--train-backend fsdp` (LoRA is Megatron-only today)
* `--only-train-params-name-list` / `--freeze-params-name-list`
* `--use-critic` or `--advantage-estimator ppo`
* on-policy distillation (`--use-opd`, `--opd-teacher-load`)
* `--keep-old-actor`, `--ref-update-interval`, `--ref-load`
* `--lora-dropout` other than `0.0`, `--lora-bias` other than `none`
* vision-tower targets

DeepGEMM MoE forward replacement is rejected for layers with routed-expert LoRA
adapters: it bypasses the linear-module hooks. Disable that replacement for the
adapted layers or exclude their routed experts from the LoRA targets. This check
runs before the fused forward is installed.

## Checkpoints

`--save-lora DIR` writes an adapter-only checkpoint: `adapter_config.json`,
sharded `adapter_model-tp*-pp*.safetensors`, an index file, and `training_state.json`. Loading with
`--lora-load` validates rank, alpha, format version, base-model config hash, and
the TP/PP/ETP sizes; expert-parallel size is deliberately **not** part of the identity
because the shared adapter is EP-invariant. This applies to adapter-only files,
not arbitrary EP changes for full optimizer checkpoints. Publication is atomic: a failed shard
write leaves the previous checkpoint published.

Use `--lora-save-merged-hf` when you want a standalone HF checkpoint with the
adapter already folded into the base weights.

`--lora-load` initializes an adapter for a new run with `--finetune` (HF base
loading sets this mode automatically). When resuming a full Megatron checkpoint
without `--finetune`, its adapter and optimizer state take precedence and
`--lora-load` is ignored with a log message. This also applies to automatic actor
recreation under `--release-train`. Start a fresh run with `--load HF_DIR` and,
optionally, `--lora-load ADAPTER_DIR`. A Megatron checkpoint must already contain
all adapter tensors requested by the model, including under `--finetune`.
Base-only Megatron initialization and checkpoints without readable distributed
tensor metadata are unsupported. Missing adapter keys never trigger
a blanket `strict=False` fallback in slime; existing base-key checks are unchanged.

### File format and export limitations

Adapter directories use slime's custom Megatron shard format:

```text
adapter_config.json
adapter_model.safetensors.index.json
adapter_model-tp{tp_rank}-pp{pp_rank}.safetensors
training_state.json
```

Despite using safetensors and familiar filenames, these are not PEFT/HF adapters:
module names, factor sharding, and metadata differ. They cannot be passed directly
to `PeftModel.from_pretrained` or SGLang's native adapter loader, and `--lora-load`
cannot import external PEFT adapters directly. No format converter is provided.
`training_state.json` records rollout/policy information for logging; loading it
does not restore the optimizer, scheduler, or training progress. Resume training
from the full checkpoint produced by `--save`.

`--lora-save-merged-hf '/checkpoints/merged/{rollout_id}'` exports an ordinary HF
model for architectures supported by the existing HF conversion path. It retains
neither separate A/B factors nor the optimizer and cannot reconstruct the original
adapter. Using it as a new HF base starts a fresh zero-delta adapter and makes that
merged base the reference policy.

Adapter publication requires symlinks and atomic replacement on a shared filesystem.
Do not pre-create the final export directory: existing ordinary directories cannot
be overwritten. Use a new path, optionally containing `{rollout_id}`. When moving
a checkpoint, preserve the symlink's hidden version-directory target or copy the
complete contents with symlinks dereferenced; copying only the link is insufficient.

## Supported scope and limitations

Presets describe module paths, not model families. `hybrid_attention` and
`hybrid_language` only match the documented `linear_attn` projection layout;
use `--lora-target-modules` for other layouts. No preset automatically allows
TP-replicated layers. Use `--lora-allow-replicated-modules` explicitly only when
their inputs and gradients are known to be identical across TP replicas.
The former `qwen3_5_attention` / `qwen3_5_language` names are replaced by the
hybrid presets; use `dense_mlp` in place of `qwen3_5_mlp`.

Routed-expert LoRA requires expert-TP (ETP) to divide ordinary TP, with ETP rank
equal to TP rank modulo ETP. This keeps each TP checkpoint coordinate associated
with one expert shard; TP=4/ETP=1 remains supported, while TP=1/ETP=2 is rejected.
Other layouts are rejected during injection, before training.
Changing EP size still requires keeping TP/PP/ETP fixed. Legacy routed-expert
adapter checkpoints without ETP metadata must be re-exported from their original
training layout; their shard identity cannot be verified safely.

Custom regexes select supported linear modules; they do not add support for
arbitrary operators. At TP>1, column-parallel `gather_output=True` and row-parallel
`input_is_parallel=False` are unsupported. Column input width must be divisible
by TP. Fused normalization requires a recognized LayerNorm/RMSNorm with `eps`;
fused normalization in routed-expert modules is unsupported. No matched targets
or an unsupported matched module causes an error. Explicit
`--lora-exclude-modules` replaces the default exclusion list rather than extending it.

Each local PP/VPP model chunk must contain at least one target. Custom selections
that leave a chunk without adapters are not supported yet and fail at injection.

`--save-lora` and `--lora-save-merged-hf` run alongside full checkpoint saving.
They require `--save` and a positive `--save-interval`, or `--release-train` with
`--save`. They do not enable independent adapter-only saving. Full-checkpoint
resume ignores `--lora-load` even if its original directory no longer exists.

The base-model config hash checks architecture/config compatibility, not weight
identity. Keep the exact base weights used to train the adapter and record their
source/revision; an identically configured SFT checkpoint is not interchangeable.

## Validation and CI

Follow the repository [CI guide](../developer_guide/ci.md). The seven CPU test
files below are registered in `cpu-unittest` in `.github/workflows/pr-test.yml.j2`.
`test_lora_parallel_gpu.py` declares `NUM_GPUS = 2` and is registered in the
Megatron matrix (also reused by image validation). `_lora_fakes.py` is a helper,
not a standalone test entry. No additional workflow or dependency policy is needed.

CPU tests use PyTorch CPU/Gloo and the dependencies installed by the CPU workflow,
including pytest and safetensors. Run files separately, as CI does: the LoRA helper
installs process-local Megatron substitutes even if real Megatron is installed.
Do not combine these CPU tests and real Megatron tests into one pytest process.
Checkpoint tests fail on a missing safetensors dependency rather than silently skip.

```bash
# Run from the repository root, with the CI dependencies already available.
python .github/workflows/generate_github_workflows.py
pre-commit run --all-files --show-diff-on-failure

for name in \
  test_megatron_argument_validation \
  test_lora_config \
  test_megatron_lora \
  test_lora_checkpoint \
  test_lora_weight_sync \
  test_lora_lifecycle \
  test_lora_parallel
do
  SLIME_LORA_MULTI_GPU_TEST=0 python "tests/${name}.py" || exit 1
done
```

Commit the source template and generated workflow together if regeneration changes
them. Pre-commit may format files; inspect and commit those changes, then rerun until
it passes. Do not edit generated YAML by hand or relax lint rules to pass this suite.

In the IDC training image, with two visible CUDA GPUs and real Megatron/Transformer
Engine installed, run:

```bash
python tests/test_lora_parallel_gpu.py
```

The entry point spawns two workers itself; do not wrap it in torchrun. Explicit
execution fails if fewer than two GPUs are visible. General pytest discovery may
skip the test without GPUs; a skip is not GPU validation. Each worker requires real
Megatron and cannot silently substitute the CPU fake backend.

The GPU test covers FP32 linear-layer forward/backward parity at TP=2, with and
without sequence parallelism, using local and TE layers, plus distributed adapter
state checkpoint parity. It does not establish BF16 actor/DDP optimizer correctness,
MoE multi-GPU checkpoint correctness, or real SGLang synchronization. Mock sync tests
check tensor payloads, not multi-engine failure recovery.

For a PR, record the commit, training-image tag/digest, Megatron/TE/SGLang versions,
GPU count, parallel configuration, commands, and pass/fail/skip results. Separately
validate a short real LoRA RL run: adapter updates while the base stays frozen,
nonzero-adapter rollout parity, adapter save/reload, full optimizer resume, and
`--release-train` recreation. Attach those logs and report untested combinations.
Use `run-ci-changed` for changed tests or `run-ci-megatron` for the registered
Megatron suite when a maintainer enables the corresponding label; CPU jobs are automatic.

IDC validation must include EP=1 with TP>ETP and a real DDP optimizer step, in addition to EP>1.
