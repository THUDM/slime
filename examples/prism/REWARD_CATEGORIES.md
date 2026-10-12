---
orphan: true
---

# Prism 样本奖励类别

`examples.prism.rewards.reward` 是这套示例自己的奖励入口。类别函数和
`CATEGORY_REWARDS` 映射均在 `examples/prism/rewards.py`，基础评分客户端复用
`slime_plugins.tts_reward`。数据中的 domain、Prism task type 与奖励类别分别保留。

| `metadata.reward_category` | 固定启用的评分分量 |
| --- | --- |
| `base` | WER/CER |
| `voiceclone` | WER/CER + SIM |
| `instruction` | WER/CER + JudgeCLAP |

统一入口要求每条 prompt 显式填写类别，未知或缺失类别会报错。每个 GRPO prompt group
的候选继承同一类别；批量调用也检查这一约束。
类别不会根据 `domain` 或 `task_type` 自动推断。

## 配置与启动

沿用现有 `REWARD_CONFIG` 的 `services` 和 `weights` 格式。每个类别只调用自己启用
的服务，所需分量必须有服务绑定和正权重；其他类别的服务可以省略。不同类别共享同一
评分服务的 fingerprint 缓存和并发上限。

例如，用现有配置工具将三个系数显式设为 `1.0`：

```bash
"$TRAIN_PYTHON" -m examples.prism.configure \
  --student "$HF_CHECKPOINT" --codec "$CODEC_MODEL" --output "$RUN_CONFIG" \
  --reward wer http://asr-host:8101 1.0 \
  --reward sim http://sim-host:8102 1.0 \
  --reward judge_clap http://judge-host:8103 1.0

export OMNI_CONFIG="$RUN_CONFIG/student.yaml"
export REWARD_CONFIG="$RUN_CONFIG/rewards.yaml"
export PROMPT_DATA=/data/prism-category-prompts.jsonl
bash examples/prism/run-grpo.sh \
  --custom-rm-path examples.prism.rewards.reward
```

此时三类分别得到 `WER_reward`、`WER_reward + SIM_reward`、
`WER_reward + JudgeCLAP_reward`。WER/CER 分量沿用 `1 / (1 + error_rate)`；SIM
沿用非负 cosine similarity；JudgeCLAP 沿用全局属性分数平均。修改 `weights` 可以
调整分量系数，但类别要求的分量不能通过零权重关闭。reward 总和不再额外归一化，
后续组优势仍由原 GRPO 流程计算。

参见 `category-prompts.example.jsonl`，填写实际参考音频路径后作为数据模板使用。
每行填写 `metadata.reward_category`，其他 Prism 输入沿用已有协议：

- `base`：需要 `transcript_target` 和 `language`，不需要 SIM 参考或 JudgeCLAP 属性。
- `voiceclone`：还需要有效的音色参考。示例显式填写 Full VoiceClone 的完整参考属性映射。
- `instruction`：还需要可被 JudgeCLAP 评分的全局属性指令；该类别不会调用 SIM。

这些输入会严格验证，技术故障继续抛错。合法零帧 stop 的所选奖励分量均记录为明确的
`zero_frame` 结果，reward 为零。分类路由同时支持单样本和混合类别的样本列表；
列表返回值保持输入顺序，并等待已启动的评分完成后传播故障。

## 记录与检查

评分结果继续写入 `metadata.omni_reward`，包含选中分量的 raw/transformed 值、服务
身份、实际权重和 total，并补充 `category`。样本自身的 domain 和 task type 保持其
原有语义。新增类别时，在示例模块中增加奖励函数及映射即可。

```bash
"$TRAIN_PYTHON" -m pytest \
  tests/test_prism_reward_categories.py tests/test_tts_reward_plugin.py -q
```

测试使用冻结评分协议的 CPU 替身，覆盖分量选择、混合批次、共享并发上限、配置与
输入验证、零帧 stop 和故障传播。实际混合任务的 GPU 训练与语音质量收益需另行验收。
