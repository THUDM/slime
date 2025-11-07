# 投机采样


投机采样是加速 rollout 的重要优化手段，目前 slime 支持不通过训练更新 draft model 式的投机采样。

对于有 MTP 层支持的模型（例如，GLM-4.6、Deepseek-V3/R1），只需要添加：

```bash
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

如果要使用单独训练的 draft model（例如 [SpecForge](https://docs.sglang.ai/SpecForge/) 训练的），还需要额外设置：

```bash
--speculative-draft-model-path /your/draft/model/path
```

详细参数含义及配置方法，请参考 SGLang 的 speculative decoding [文档](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
