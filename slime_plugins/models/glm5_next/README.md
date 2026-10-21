# GLM-5.3 Flash / GLM5-next

Native Megatron integration for the `Glm5NextForConditionalGeneration` checkpoint.
The language model uses the same spec/provider entry point as the GLM5 plugin;
the CogViT wrapper follows the Qwen3.5 integration.

The 300B configuration has 45 layers: 34 KDA layers, 11 no-RoPE MLA layers with
learned four-token DSA pooling, four mHC residual streams, and 288 routed experts.
The first three layers use dense MLPs. Dense, routed, shared, and visual MLPs use
the checkpoint's SwiGLU clamp of 10.

## Environment and parallelism

Use the CUDA 13 image with upstream Megatron Core's TE module builders,
`flash-linear-attention` (`fla`), TileLang, and Transformers' GLM-OCR vision
components. SGLang must support GLM5-next, pooled DSA, KDA, routed-expert capture,
score centering, and top-p replay.

The current native attention implementation requires **TP=1**. For **CP>1**,
use **`--allgather-cp`**: the packed token stream is split into contiguous
intervals. MLA gathers compressed KV and indexer keys; KDA uses FLA's CP
convolution and recurrent-state exchange. The 64-H100 recipe defaults to CP=2,
PP=4 and EP=16, with expert TP=1. `CP_SIZE=1` selects the unsharded context
baseline. Packed sequence boundaries isolate both KDA state and sparse
attention. Indexer parameters are frozen. The recipe freezes the vision tower for text training;
the wrapper also accepts `pixel_values` / `image_grid_thw` and
`pixel_values_videos` / `video_grid_thw` for multimodal training.

## Three-step validation recipe

Assume the BF16 checkpoint has been downloaded to `/data/GLM-5.3-Flash` and the
dataset to `/data/dapo-math-17k/dapo-math-17k.jsonl`, with `/data` shared across
nodes. Start an eight-node Ray cluster with eight H100 GPUs per node, then run
from the repository root:

```bash
BASE_DIR=/data \
bash scripts/run-glm5.3-flash-300B.sh --check-weight-update-equal
```

Override `HF_CHECKPOINT` and `PROMPT_DATA` for other downloaded locations.
NCCL/Gloo select their interfaces automatically; set `SOCKET_IFNAME` if the
cluster requires a particular interface. `MEGATRON_PATH` defaults to the
standard image location `/root/Megatron-LM`.

The dataset uses `prompt` and `label` fields. The default reward is `deepscaler`;
use an appropriate reward for your data. `RM_TYPE=random` is only useful for a
smoke test with nonzero training gradients, not for assessing model quality.
The script enables R3, score centering, top-p=0.95 replay, and a nonzero learning
rate. It defaults to three rollouts and writes debug training data under
`RUN_DIR`.
Each rollout contains 16 prompts with eight samples each (global batch 128).
The default response limit is 32768 tokens, with a 65536-token context window.
Decode CUDA Graph is enabled for inference; training CUDA Graph remains disabled.
Entropy and KL regularization use their zero defaults.

`--check-weight-update-equal` snapshots the initial inference weights, resets
them, and compares every tensor after the initial Megatron-to-SGLang update.
This also checks that the expert routing optimization sends complete weights.
The direct HF loader keeps router correction biases in FP32 before copying.
Legacy `safe_gate=true` checkpoints are normalized to `gate_lower_bound=-5`
for both training and SGLang startup. Explicit gate bounds take precedence.
Use the updated `latest` or `v0.5.21` Megatron image patch: checkpoint reload must
refresh already-FP32 CPU optimizer copies as well as BF16 master parameters.
Without that fix, the first optimizer step overwrites loaded mHC/KDA parameters
with stale initialization values.

## Validation results

The full 300B checkpoint also completed three steps with **CP=2 and
`--allgather-cp`** on 64 H100s on 2026-10-16 (`local/glm53/run12-cp2`, Ray job
`raysubmit_FNCVTyJS6GAqXbz2`, status `SUCCEEDED`). All 64 inference ranks passed
the initial weight equality check. The run retained batch 128, R3, score
centering, top-p=0.95 replay, learning rate 1e-6, and the same graph settings.

| Step | Sample-mean logprob diff | Token-mean logprob diff | Gradient norm | Training time |
| --- | --- | --- | --- | --- |
| 1 | 0.00629394 | 0.00791693 | 0.76393992 | 452.3 s |
| 2 | 0.00618919 | 0.00837462 | 0.80803555 | 58.2 s |
| 3 | 0.00631780 | 0.00812524 | 0.65333384 | 63.4 s |

The first step included new FLA CP kernel compilation and autotuning; stack
samples captured compilation delaying other pipeline stages. Completed caches
were merged and broadcast to all eight nodes. Initial weight synchronization
took 18.2 seconds, and subsequent updates took 18.7, 25.8, and 24.2 seconds.
All 307,494 response tokens passed the top-p replay audit; responses remained
coherent after updates, with no truncation. The longest response was 5330 tokens.
CP=4 is covered by distributed attention/vision regression tests, while the
full-model three-step run used CP=2. These runs do not establish 32k-response
or full-model multimodal training performance.

The full 300B BF16 checkpoint completed three consecutive training steps on
64 H100s with CP=1 on 2026-10-16 (`local/glm53/run11`, Ray job
`raysubmit_VMBJeNVbE1yJxSSQ`, status `SUCCEEDED`). Each step used 128 samples,
R3, score centering, top-p=0.95 replay, and Adam at learning rate 1e-6.

| Step | Sample-mean logprob diff | Token-mean logprob diff | Gradient norm | Training time |
| --- | --- | --- | --- | --- |
| 1 | 0.00628354 | 0.00767620 | 0.63715422 | 107.8 s |
| 2 | 0.00660455 | 0.00810495 | 0.80081755 | 49.8 s |
| 3 | 0.00627287 | 0.00798890 | 0.83470374 | 52.0 s |

Chinese explanations, arithmetic, poetry, and English stories remained coherent
across the three rollouts. All 296,317 response tokens passed the top-p support
and selected-token logprob audit, with no truncated responses or nonfinite
logprobs. All 64 inference ranks captured and replayed decode CUDA Graph;
training used `cuda_graph_impl=none`, with entropy and KL coefficients both zero.
Weight synchronization after the steps took 20.1, 15.0, and 17.4 seconds.

The configured response limit was 32768; observed responses reached 5338 tokens.
This run used text prompts and a random smoke-test reward. Multimodal validation
separately covered the complete vision tower, image/video feature injection,
and backward through one real text layer. It does not establish full-model
32k-response or multimodal training performance.

The tested environment was PyTorch 2.13.0+cu130, Megatron Core
0.19.2+4b4acac9a, SGLang 0.5.21, Transformers 5.12.1, FLA 0.5.2, TileLang
0.1.12, and Triton 3.7.1. Local run artifacts contain
`metrics.json`, `validation.json`, `answers.json`, `top-p-audit.json`, and logs
from all eight nodes.

## Precision and optimization

The patch embedding replaces a convolution over complete nonoverlapping patches
with the equivalent GEMM, including the bias in the GEMM epilogue. Its output
matches SGLang and has lower error against FP64 than the original BF16 Conv3d.
KDA state parameters and mHC scale/bias parameters remain FP32; SwiGLU evaluates
its pointwise arithmetic in FP32 before storing BF16 output.

The shared sparse MLA backward kernel uses runtime sequence lengths, matching
the forward kernel. A new packed length reuses the compiled kernel instead of
compiling another specialization. Forward and gradients are checked against
FP64 attention, including partial tiles; dQ matches the former static kernel
exactly and dKV differences stay within its existing atomic-add variability.

Colocated weight updates use slime's existing expert routing path, including
the VLM's `language_model` parameter prefix. On the 64-H100 300B validation run,
this reduced transfer groups from 587 to 61 and total initial synchronization
from 97.9 to 21.5 seconds, including communication-group restoration, with all
64 inference ranks passing the initial weight equality check.

Focused regression tests:

```bash
PYTHONPATH=$PWD python tests/test_glm5_next_ops.py
PYTHONPATH=$PWD python tests/test_glm5_next_gpu.py
PYTHONPATH=$PWD python tests/test_glm5_next_cp.py  # requires four GPUs
PYTHONPATH=$PWD python tests/test_hf_to_megatron.py
PYTHONPATH=$PWD python tests/test_expert_routing.py
```

The CP tests cover CP=2/4, packed boundaries, all attention parameter gradients,
and image/video feature selection and gradients. In the strong-decay test,
FLA's BF16 backward has cancellation error even at CP=1; the FP32 gate
parameters are therefore also checked against an independent FP64 recurrence
to ensure context partitioning does not degrade accuracy.
