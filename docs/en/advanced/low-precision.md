# Low Precision Training

- [FP8 rollout and FP8 training](#FP8-rollout-and-BF16-training)
- [FP8 rollout and FP8 training](#FP8-rollout-and-FP8-training)
- [INT4 QAT Training](#INT4-QAT-Training)

## FP8 rollout and BF16 training

You can run FP8 rollout simply by setting `--hf-checkpoint` with an blockwise quantized huggingface checkpoint, which can be converted by:

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_model \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

Please ensure that the converted checkpoint points to a directory where the `config.json` contains the correct `quantization_config` so that slime can automatically use FP8 quantization during weight updates.

## FP8 rollout and FP8 training

We also observed that under FP8 training and inference, it can achieve more efficient inference throughput and lower training-inference mismatch, resulting in more stable training. More details can be found in [this blog](https://lmsys.org/blog/2025-11-25-fp8-rl/).

### Quick Start

1. Convert your HuggingFace model weights to FP8 format using the above `tools/convert_hf_to_fp8.py`.

2. Setting up the running script: 

For training tasks, we need to add these flags:

```bash
--fp8-format e4m3
--fp8-recipe blockwise
# --fp8-param-gather # [optional] Currently incompatible with CPU Adam
```

Then ensure the `NVTE_FP8_BLOCK_SCALING_FP32_SCALES` environment variable is enabled.

Note that only `Linear` and `GroupLinear` layers in TransformerEngine use fp8 format. `embedding` and `lm_head` remain in their original precision. If `--fp8-param-gather` is not enabled, weights in TransformerEngine remain in bf16 format, only being cast to fp8 format during `GEMM` or `GroupGEMM` operations.

3. Start FP8 training with

```bash
# Qwen3-4B Int4 training
bash scripts/low_precision/run-qwen3-4b-fp8.sh

# Qwen3-30B-A3B (2 nodes)
bash scripts/low_precision/run-qwen3-30b-a3b-fp8.sh
```

4. Use the saved checkpoint for evaluation. 

Note that TransformerEngine does not specifically save FP8 quantized weights; the saved torch dist remains in original precision (usually bf16). If you want to evaluate under FP8, you need to convert the checkpoint from `torch_dist` to HuggingFace format, then convert to FP8 HuggingFace format.


### Quick Explanation

Here's a quick explanation of how FP8 training is currently implemented in slime:

1. Initialization: If FP8 recipe is enabled, layers will be built in FP8 context.

2. Training: During training, weights and activations are quantized online to nvfp8 format, and cuBLAS FP8 GEMM is called for various GEMM computations in forward and backward passes.

3. Weight updates: During RL weight updates, Megatron first dequantizes FP8 weights to bf16 format, then slime quantizes these bf16 weights to fp8 format and sends them to sglang. (This additional dequantization and quantization is not elegant, but we haven't modified the interface yet for framework compatibility.)

4. Save checkpoint: Similar to weight updates, if checkpoints need to be saved from the training engine, they will also be dequantized back to bf16 and saved to `torch_dist` format checkpoints.


### TODO

Currently, FP8 is far from being a complete feature and still has the following bugs, for examples:

- FP8 weights (`--fp8-param-gather`) can provide memory savings benefits, but currently FP8 weights must be used with TransformerEngine's FusedAdam, which conflicts with the commonly used Adam CPU offload technique in Megatron-LM.

## INT4 QAT Training

This guide provides examples for INT4 STE (Straight-Through Estimator) training and INT4 inference. Utilizing INT4 inference significantly improves throughput, thereby accelerating the training pipeline (specifically during the rollout generation phase).

### Quick Start

1. Convert HuggingFace Weights to INT4
Use the `tools/convert_hf_to_int4_direct.py` script to convert BF16 weights to INT4 format. Ensure that the `--hf-checkpoint` parameter points to a directory where `config.json` contains the correct `quantization_config`. slime will automatically utilize INT4 quantization during weight updates.

```bash
python tools/convert_hf_to_int4_direct.py \
  --model-dir /path/to/your/original/models \
  --save-dir /path/to/your/save/models \
```

Note: If you only hope to run with INT4 rollout, you only need to set the `--hf-checkpoint` to the converted INT4 checkpoint.

2. Start INT4 QAT Training

You need to configure the specific environment variables for quantization settings.

**Environment Variables:**

*   **`OPEN_TRAINING_INT4_FAKE_QAT_FLAG`**: Enables fake quantization operations for INT4 training.
*   **`OPEN_TRAINING_INT4_GROUP_SIZE`**: Specifies the block size (group size) for model quantization.
    *   Set to **128** for `moonlight-16B-A3B` 、 `qwen3-30B-A3B`and `qwen3-235B-A22B-int4`.
    *   Set to **32** for `kimi-k2-Thinking-int4`.

**Configuration Example:**

```json
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    ...
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"
```

**Launch Commands:**

```bash
# Moonlight-16B-A3B Int4 training
bash scripts/low_precision/run-moonlight-16B-A3B-int4.sh

# Qwen3‑30B‑A3B Int4 training
bash scripts/low_precision/run-qwen3‑30B‑A3B-int4.sh

# Qwen3-235B-A22B Int4 training (8 nodes)
bash scripts/low_precision/run-qwen3-235B-A22B-int4.sh

# Kimi-k2-Thinking Int4 training (32 nodes)
bash scripts/low_precision/run-kimi-k2-Thinking-int4.sh
```

- For multi-node environments, please start the Ray service according to your cluster configuration.
