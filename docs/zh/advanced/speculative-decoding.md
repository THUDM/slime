# 投机采样


Speculative decoding is an important optimization for making faster rollout during RL training. Currently slime only supports speculative decoding without training.

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

### 已知问题
[SGLang issue #9888](https://github.com/sgl-project/sglang/issues/9888) 或 [SGLang issue #9521](https://github.com/sgl-project/sglang/issues/9521)
- 报错发生在 speculative decoding draft 阶段的 cuda graph padding
- 解决方法: 
	1. 切换推理后端为 fa3 triton。该 bug 仅发生在 flashInfer 。
	2. 覆盖更宽的 `--sglang-cuda-graph-bs` 来避免某些 batch size 做 cuda graph padding
	3. 禁用 cuda graph（性能损失太大，不推荐）
	4. Notice：禁用 cuda graph padding `--sglang-disable-cuda-graph-padding` 目前对 speculative decoding 不生效。参考 [SGLang cuda_graph_runner.py](tbd)
- 如需 debug，可尝试开启 slime 的 `--debug-rollout-only` 参数，来排除参数更新或模型 offload 的影响
```bash
# if speculative decoding has bug, this can help debug
--debug-rollout-only

# If flashInfer has bug with speculative decoding, use fa3 or triton instead
--sglang-attention-backend fa3

# If bug exists when cuda graph do padding, extend the cuda graph batch size
--sglang-cuda-graph-bs $(seq 1 32) $(seq 40 8 64) $(seq 80 16 160)

# Improve performance by enlarging running batch size limit
--sglang-max-running-requests 128
```
[SGLang issue #9481](https://github.com/sgl-project/sglang/issues/9481)
- 解决方法：
	1. 应用最新的 sglang patch。
	2. 参考这个 pr 修改 sglang https://github.com/sgl-project/sglang/pull/9687 
[SGLang PR #9388](https://github.com/sgl-project/sglang/pull/9388)
- 如果使用外部 draft model 出现 illegal memory access，可能是由于 draft model 和 target model 的 context length 不匹配导致的 bug。
- 请更新 SGLang >= 0.5.1 来应用这个 PR。（并更新 `sgl-kernel`）
