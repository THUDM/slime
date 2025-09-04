# Speculative Decoding – Usage Guide

### Support Status

* ✅ **MTP layer for inference only (no training)**

  * ✅ Models with native MTP layers:

    * ✅ Mimo-7B-RL
    * 🧪 Deepseek-V3/R1
    * 🧪 GLM-4.5
  * 🚧 External draft models trained with SpecForge:

    * 🚧 [sglang-EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B)
    * 🚧 [Qwen3-235B-A22B-EAGLE3](https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3)
* ⏳ **MTP layer training with RL**

  * 🚧 Sequence packing with MTP layers is under development in Megatron.

### Usage

Add the following flags to `SGLANG_ARGS`:

```bash
# for speculative decoding
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

For details on parameter meanings and configuration, see the [SGLang speculative decoding documentation](https://docs.sglang.ai/advanced_features/speculative_decoding.html).

### Known Issues

#### [SGLang issue #9888](https://github.com/sgl-project/sglang/issues/9888) or [SGLang issue #9521](https://github.com/sgl-project/sglang/issues/9521)

* Error occurs during CUDA graph padding in the speculative decoding draft stage.
* Workarounds:

  1. Switch the inference backend to **fa3 Triton** (bug only occurs in **flashInfer**).
  2. Specify a broader range for `--sglang-cuda-graph-bs` to avoid batch sizes that trigger CUDA graph padding.
  3. Disable CUDA graph (not recommended due to significant performance loss).
  4. **Notice:** Disabling CUDA graph padding with `--sglang-disable-cuda-graph-padding` is currently ineffective for speculative decoding. See [SGLang `cuda_graph_runner.py`](tbd).
* For debugging, enable slime’s `--debug-rollout-only` flag to isolate rollout behavior from parameter updates or model offloading.

```bash
# If speculative decoding fails, this can help debug
--debug-rollout-only

# If flashInfer causes issues with speculative decoding, use fa3 or triton instead
--sglang-attention-backend fa3

# If CUDA graph fails due to padding, extend the CUDA graph batch size
--sglang-cuda-graph-bs $(seq 1 32) $(seq 40 8 64) $(seq 80 16 160)

# Improve performance by enlarging the running batch size limit
--sglang-max-running-requests 128
```

#### [SGLang issue #9481](https://github.com/sgl-project/sglang/issues/9481)

* Solution:

  1. Apply the latest SGLang patch.
  2. See [PR #9687](https://github.com/sgl-project/sglang/pull/9687) for reference changes.

#### [SGLang PR #9388](https://github.com/sgl-project/sglang/pull/9388)

* If using an external draft model results in **illegal memory access**, it may be caused by a context length mismatch between the draft and target models.
* Please update to **SGLang ≥ 0.5.1** (and update `sgl-kernel`) to apply this fix.
