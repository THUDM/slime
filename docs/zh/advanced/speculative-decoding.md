# 投机采样

投机采样是加速 rollout 的重要优化手段。推理过程中不再让昂贵的 Target Model 逐个 token 进行 decode，而是先由一个轻量级的 draft model 先进行 decode，生成多个 token 后，再由大模型进行批量验证。

## 使用投机采样加速推理

对于有 MTP 层的模型（例如 GLM-4.6、Deepseek-V3/R1），只需要添加：

```bash
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

如果要使用单独训练的 draft model（例如 [SpecForge](https://docs.sglang.ai/SpecForge/) 训练的），还需要额外设置：

```bash
--sglang-speculative-draft-model-path /your/draft/model/path
```

详细参数含义及配置方法，请参考 SGLang 的 speculative decoding [文档](https://docs.sglang.ai/advanced_features/speculative_decoding.html)

## 在线 SFT draft model

随着 RL 流程的进行，draft model 和 target model 的采样概率差异逐渐增大，能通过验证的 draft token 逐渐减少，spec 甚至可能造成负收益。

目前，slime 支持了在 RL 流程中在线训练 MTP 层，随着训练的进行同步更新 draft model，稳定提高了采样速度，相关原理可参见 [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e)。使用方法如下：

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

注意 MTP 训练需要一个包含了 MTP 权重的 checkpoint，所以在将 huggingface checkpoint 转为 torch dist 时，也需要加上 `--mtp-num-layers 1`。

外部 draft model 的训练还在 WIP。
