# Speculative decoding 使用指南

### 支持情况
- ✅ mtp layer 仅推理，不训练
	- ✅ 拥有原生 mtp layer 的模型
		- ✅ Mimo-7B-RL
		- 🧪 Deepseek-V3/R1
		- 🧪 GLM-4.5
	- 🚧 SpecForge 训练的外部 draft model
		- ✅ [sglang-EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B)
		- 🚧 [Qwen3-235B-A22B-EAGLE3](https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3)
- ⏳ mtp layer 的 RL 训练
	- 🚧 在Megatron 支持 mtp layer 的 sequence packing
### 使用方法
在 SGLANG_ARGS 里添加如下参数
```bash
# for speculative decoding
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

#### Llama 3.1 8B EAGLE3 配置示例
对于使用 [sglang-EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B) 作为 draft model 的配置：

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8

   --sglang-speculative-algorithm EAGLE3
   --sglang-speculative-draft-model-path /root/sglang-EAGLE3-LLaMA3.1-Instruct-8B
   --sglang-speculative-num-steps 5
   --sglang-speculative-eagle-topk 8
   --sglang-speculative-num-draft-tokens 32

   --sglang-max-running-requests 48
   --sglang-dtype float16
   --sglang-context-length 2048  # 受 draft model 限制
   --sglang-attention-backend fa3
)
```

**重要配置说明：**
- `--sglang-dtype float16`：这里必须要配（单起 sglang 也需要）。因为 `sglang-EAGLE3-LLaMA3.1-Instruct-8B` 是 `float16`,`Llama-3.1-8B-Instruct` 是 `bfloat16`，二者不一致，如果不手动指定而是从各自的 config.json 里读取会报 `Capture cuda graph failed`
- `--sglang-context-length 2048`：`sglang-EAGLE3-LLaMA3.1-Instruct-8B` 的`max_position_embeddings=2048`，目前 sglang 的实现强制要求 target model 和 draft model 的content length 一致 
- `--sglang-attention-backend fa3`：避免 flashInfer 的问题（https://github.com/sgl-project/sglang/issues/9888）
- 完整配置示例可参考 `scripts/run-eagle3-llama3.1-8B.sh`

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

#### Llama 3.1 8B EAGLE3 的问题 
- **Context Length 限制**：`sglang-EAGLE3-LLaMA3.1-Instruct-8B` draft model 的 context length 仅支持 2048 tokens，导致 target model 的 context 也必须限制为 2048。 所以目前需要限制 prompt + response 的总长度不能超过 2048 tokens。

- **Draft Model 训练限制**：当前实现不支持 draft model 本身的训练，draft model 只能用于推理。

- **性能问题**：在当前配置下，speculative decoding 并未提供性能收益，甚至可能出现负收益。