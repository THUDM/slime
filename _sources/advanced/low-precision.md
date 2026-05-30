# Low Precision Training and Rollout

Low precision in slime is primarily used to make rollout faster and more memory-efficient while keeping training numerically stable. For large MoE RL jobs, the recommended production path is:

> **BF16 training in Megatron + FP8 rollout/inference in SGLang**

Megatron keeps the trainable checkpoint in BF16/torch_dist format. SGLang serves an FP8 Hugging Face checkpoint for rollout. During weight updates, slime uses the quantization config in `--hf-checkpoint` to quantize updated BF16 weights before sending them to SGLang.

## Feature Maturity

| Feature | Status | Recommended Use |
|---|---|---|
| BF16 training + FP8 rollout/inference | Stable | Default path for large MoE RL recipes. Keeps training stable while reducing rollout memory and bandwidth. |
| FP8 KV cache in SGLang rollout | Stable when supported by your SGLang version/GPU stack | Increase KV cache capacity for long-context or agentic rollout by passing `--sglang-kv-cache-dtype fp8_e4m3`. |
| INT4 rollout / INT4 QAT | Beta | Use when rollout memory/throughput pressure is high and the model path has been validated. |
| FP8 training + FP8 rollout | Experimental | Useful for research on training/inference mismatch and throughput, but still has optimizer and checkpointing caveats. |

## BF16 Training with FP8 Rollout

This is the main production path in slime.

You can run FP8 rollout by setting `--hf-checkpoint` to a blockwise-quantized Hugging Face checkpoint. Convert a BF16 checkpoint with:

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_MODEL \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

Make sure the converted checkpoint's `config.json` contains the correct `quantization_config`. slime uses that config during weight updates, so the training side can remain BF16 while rollout receives FP8 weights.

Example:

```bash
# Megatron training checkpoint remains BF16 / torch_dist.
--ref-load /path/to/model_torch_dist

# SGLang rollout checkpoint is FP8 Hugging Face.
--hf-checkpoint /path/to/model-fp8-hf
```

## FP8 KV Cache for Rollout

For long-context, multi-turn, or agentic workloads, KV cache capacity is often the bottleneck. Because SGLang arguments are passed through by adding `--sglang-`, you can enable FP8 KV cache directly:

```bash
--sglang-kv-cache-dtype fp8_e4m3
```

This is a rollout-side setting. It does not change Megatron training precision; it increases effective SGLang KV cache capacity and can allow longer contexts or higher concurrency, subject to the accuracy/performance behavior of your SGLang version and GPU stack.

## FP8 Training with FP8 Rollout

slime also supports experimental FP8 training paths. We observed that FP8 training plus FP8 inference can improve inference throughput and reduce training/inference mismatch in some settings. More details are available in [this blog](https://lmsys.org/blog/2025-11-25-fp8-rl/).

### Quick Start

1. Convert your Hugging Face model weights to FP8 format using `tools/convert_hf_to_fp8.py`.

2. Add the FP8 training flags:

```bash
--fp8-format e4m3
--fp8-recipe blockwise
# --fp8-param-gather # optional; currently incompatible with CPU Adam
```

3. Ensure `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` is set. slime sets this to `1` by default for Ray actors.

4. Start an FP8 training example:

```bash
# Qwen3-4B FP8 training
bash scripts/low_precision/run-qwen3-4b-fp8.sh

# Qwen3-30B-A3B FP8 training (2 nodes)
bash scripts/low_precision/run-qwen3-30b-a3b-fp8.sh
```

### Implementation Notes

1. If an FP8 recipe is enabled, TransformerEngine layers are built in an FP8 context.
2. During training, weights and activations are quantized online to NVFP8 format, and cuBLAS FP8 GEMM is used for forward and backward GEMMs.
3. During RL weight updates, Megatron dequantizes FP8 weights to BF16, then slime quantizes the BF16 weights to FP8 and sends them to SGLang.
4. Checkpoints saved from the training engine are dequantized back to BF16 and saved as `torch_dist`.

Only `Linear` and `GroupLinear` layers in TransformerEngine use FP8. `embedding` and `lm_head` remain in their original precision. If `--fp8-param-gather` is not enabled, TransformerEngine weights remain stored in BF16 and are cast to FP8 only during `GEMM` or `GroupGEMM`.

### Known Caveat

`--fp8-param-gather` can save memory, but currently requires TransformerEngine `FusedAdam`, which conflicts with the CPU Adam offload path commonly used for large Megatron-LM RL jobs.

## INT4 QAT Training

INT4 STE (Straight-Through Estimator) training and INT4 inference can further reduce rollout memory and improve throughput. Treat this path as beta unless you have validated the target model and reward setup.

### Quick Start

1. Convert Hugging Face weights to INT4:

```bash
python tools/convert_hf_to_int4_direct.py \
  --model-dir /path/to/your/original/models \
  --save-dir /path/to/your/save/models
```

If you only need INT4 rollout, set `--hf-checkpoint` to the converted INT4 checkpoint.

2. Enable INT4 fake QAT:

```json
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"
```

`OPEN_TRAINING_INT4_GROUP_SIZE` should usually be:

- `128` for `moonlight-16B-A3B`, `qwen3-30B-A3B`, and `qwen3-235B-A22B-int4`;
- `32` for `kimi-k2-Thinking-int4`.

3. Launch an example:

```bash
# Moonlight-16B-A3B INT4 training
bash scripts/low_precision/run-moonlight-16B-A3B-int4.sh

# Qwen3-30B-A3B INT4 training
bash scripts/low_precision/run-qwen3-30B-A3B-int4.sh

# Qwen3-235B-A22B INT4 training (8 nodes)
bash scripts/low_precision/run-qwen3-235B-A22B-int4.sh

# Kimi-k2-Thinking INT4 training (32 nodes)
bash scripts/low_precision/run-kimi-k2-Thinking-int4.sh
```

For multi-node environments, start the Ray service according to your cluster configuration.
