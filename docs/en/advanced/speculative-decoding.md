# Speculative Decoding

Speculative decoding is an important optimization for making faster rollout during RL training. Currently slime only supports speculative decoding without training.

For model with MTP layer (e.g. GLM-4.6, Deepseek-V3/R1), you can run with:

```bash
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

And for external draft model (e.g. draft models from [SpecForge](https://docs.sglang.ai/SpecForge/)), you need also pass:

```bash
--speculative-draft-model-path /your/draft/model/path
```

For details on parameter meanings and configuration, see the [SGLang speculative decoding documentation](https://docs.sglang.ai/advanced_features/speculative_decoding.html).
