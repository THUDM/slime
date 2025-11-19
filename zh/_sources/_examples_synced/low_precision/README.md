## FP8 training examples

This is an example of FP8 training and FP8 inference. Under FP8 training and inference, it can achieve more efficient inference throughput and lower training-inference mismatch, resulting in more stable training.

### Files

* `run-qwen3-4b-fp8.sh`: example launch script with Qwen3‑4B in FP8.

* `run-qwen3-30b-a3b-fp8-two-nodes.sh`: example launch script for running Qwen3‑30B‑A3B in FP8 across two nodes.

### Quick Start

1. [optional] Convert your HuggingFace weights to FP8 format. You can use `tools/convert_hf_to_fp8`, or directly write an FP8 format model config.

2. Start FP8 training

```
cd slime

# Qwen3‑4B FP8 training (single node)
bash examples/low_precision/run-qwen3-4b-fp8.sh

# Qwen3‑30B‑A3B FP8 training (two nodes)
bash examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh
```

Following the above command will launch FP8 training. According to slime's design, if the model under `--hf-checkpoint` is FP8, it will automatically use FP8 quantization in weight updates.

3. Use the saved checkpoint for evaluation

Note that TransformerEngine does not specifically save FP8 quantized weights; the saved torch dist remains in original precision (usually bf16). If you want to evaluate under FP8, you need to convert the checkpoint from `torch_dist` to HuggingFace format, then convert to FP8 HuggingFace format.


### Quick Explanation

Here's a quick explanation of how FP8 training is currently implemented in slime:

1. Initialization: If FP8 recipe is enabled, layers will be built in FP8 context.

2. Training: During training, weights and activations are quantized online to nvfp8 format, and cuBLAS FP8 GEMM is called for various GEMM computations in forward and backward passes.

3. Update weight: In RL weight updates, the training engine will attempt to save model weights. The saved results will be dequantized from FP8 to bf16, but since the config under `--hf-checkpoint` is FP8, slime will quantize this bf16.

4. Save checkpoint: Similar to weight updates, if checkpoints need to be saved from the training engine, they will also be dequantized back to bf16 and saved to `torch_dist` format checkpoints.


### TODO

Currently, FP8 is far from being a complete feature and still has the following bugs, for examples:

- FP8 weights (`--fp8-param-gather`) can provide memory savings benefits, but currently FP8 weights must be used with TransformerEngine's FusedAdam, which conflicts with the commonly used Adam CPU offload technique in Megatron-LM.

The slime team will continue to collaborate with the NVIDIA team to contribute more complete FP8 training infrastructure to the community.