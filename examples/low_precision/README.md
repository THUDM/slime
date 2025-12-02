## FP8 training examples

This is an example of FP8 training and FP8 inference. Under FP8 training and inference, it can achieve more efficient inference throughput and lower training-inference mismatch, resulting in more stable training.

### Files

* `run-qwen3-4b-fp8.sh`: example launch script with Qwen3‑4B in FP8.

* `run-qwen3-30b-a3b-fp8-two-nodes.sh`: example launch script for running Qwen3‑30B‑A3B in FP8 across two nodes.

### Quick Start

1. Check if your training script is properly configured. 

For training tasks, we need to add these flags:
```bash
--fp8-format e4m3
--fp8-recipe blockwise
# --fp8-param-gather # [optional] Currently incompatible with CPU Adam
```
Then ensure the `NVTE_FP8_BLOCK_SCALING_FP32_SCALES` environment variable is enabled.

Note that only `Linear` and `GroupLinear` layers in TransformerEngine use fp8 format. `embedding` and `lm_head` remain in their original precision. If `--fp8-param-gather` is not enabled, weights in TransformerEngine remain in bf16 format, only being cast to fp8 format during `GEMM` or `GroupGEMM` operations.

2. Convert your HuggingFace model weights to FP8 format. 

You can use `tools/convert_hf_to_fp8.py` to convert bf16 weights to fp8 format. Ensure that the `--hf-checkpoint` parameter points to a directory where the `config.json` contains the correct `quantization_config`. slime will automatically use FP8 quantization during weight updates. 

3. Start FP8 training.

```
cd slime

# Qwen3‑4B FP8 training (single node)
bash examples/low_precision/run-qwen3-4b-fp8.sh

# Qwen3‑30B‑A3B FP8 training (two nodes)
bash examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh
```
Following the above command will launch FP8 training. 

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

The slime team will continue to collaborate with the NVIDIA team to contribute more complete FP8 training infrastructure to the community.