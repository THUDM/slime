# Speculative decoding 使用指南

### 支持情况
- ✅ mtp layer 仅推理，不训练
	- ✅ 拥有原生 mtp layer 的模型
		- ✅ Mimo-7B-RL
		- 🧪 Deepseek-V3/R1
		- 🧪 GLM-4.5
	- ⏳ SpecForge 训练的 draft model
- ⏳ mtp layer 训练
	- 🚧 在Megatron 支持 mtp layer 的 sequence packing
### 使用方法
在 SGLANG_ARGS 里添加如下参数
```
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```
详细参数含义及配置方法，请参考 SGLang 的 speculative decoding [文档](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
### 已知问题
- 目前在 speculative decoding 的 verify 阶段，cuda graph 的 padding 存在 bug。会出现两种可能的报错。[SGLang #9521](https://github.com/sgl-project/sglang/issues/9521) 和 [SGLang #8336](https://github.com/sgl-project/sglang/issues/8336)。
	- 解决方法: 
		1. 扩大 `--sglang-cuda-graph-bs` 来避免 cuda graph padding
		2. 禁用 cuda graph padding `--sglang-disable-cuda-graph-padding`
		3. 禁用 cuda graph（不推荐）
	- fa3 和 flashInfer 都存在该问题，与推理后端无关。
	- 如需 debug，可尝试开启 slime 的 `--debug-rollout-only` 参数，来排除参数更新或模型 offload 的影响
	- 该 bug 在 RL 框架内较严重（相比单跑 SGLang），且集中在某轮 rollout 的起始阶段发生。可能与 RL 场景 batch 波动较大有关。
- flashInfer 的 speculative decoding 存在另一个 cuda graph padding 的 bug。[SGLang #9481](https://github.com/sgl-project/sglang/issues/9481)
	- 解决方法：
		- 1. 切换推理后端 `--sglang-attention-backend fa3`