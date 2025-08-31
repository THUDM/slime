# Speculative Decoding – Usage Guide

### Support Status

* ✅ MTP layer: **inference only**, not training yet

  * ✅ Models with a **native MTP layer**

    * ✅ Mimo-7B-RL
    * 🧪 DeepSeek-V3 / DeepSeek-R1
    * 🧪 GLM-4.5
  * ⏳ Draft models **trained with SpecForge**
* ⏳ MTP layer **training**

  * 🚧 Add sequence packing support for the MTP layer in **Megatron**

### How to Use

Add the following flags to `SGLANG_ARGS`:

```
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

For detailed parameter meanings and configuration, see SGLang’s speculative decoding [documentation](https://docs.sglang.ai/advanced_features/speculative_decoding.html).

### Known Issues

* In the **verify** phase of speculative decoding, there is a CUDA Graph **padding** bug that can surface as two kinds of errors: [SGLang #9521](https://github.com/sgl-project/sglang/issues/9521) and [SGLang #8336](https://github.com/sgl-project/sglang/issues/8336).

  * **Workarounds:**

    1. Increase `--sglang-cuda-graph-bs` to avoid CUDA Graph padding.
    2. Disable CUDA Graph padding via `--sglang-disable-cuda-graph-padding`.
    3. Disable CUDA Graph entirely (**not recommended**).
  * This issue exists with **fa3** and **FlashInfer**, so it’s **backend-agnostic**.
  * For debugging, try enabling Slime’s `--debug-rollout-only` to rule out effects from parameter updates or model offload.
  * The bug is **more severe inside RL frameworks** (vs. running SGLang alone) and often appears at the **start of a rollout**, likely related to large **batch size fluctuations** common in RL.
* FlashInfer’s speculative decoding has an additional CUDA Graph padding bug: [SGLang #9481](https://github.com/sgl-project/sglang/issues/9481).

  * **Workaround:** switch the attention backend with `--sglang-attention-backend fa3`.
