# 在策略蒸馏 (On-Policy Distillation)

在策略蒸馏 (OPD) 让学生模型在自己的 rollout 数据上训练，同时匹配教师模型的 token 级 log-probability，从而实现从大模型到小模型的知识传递。OPD 与 advantage estimator 正交——它作为 KL 惩罚项叠加在任意 estimator（GRPO、PPO、REINFORCE++ 等）之上。

## 关键参数

| 参数 | 说明 |
|------|------|
| `--use-opd` | 启用在策略蒸馏。使用 OPD 的必需标志。 |
| `--opd-type` | OPD 类型：`sglang` 或 `megatron`。启用 `--use-opd` 时必须设置。 |
| `--opd-kl-coef` | OPD KL 惩罚系数（默认值：1.0）。控制蒸馏信号相对于 RL advantage 的权重。 |
| `--opd-teacher-load` | 教师模型的 Megatron checkpoint 路径。`--opd-type=megatron` 时**必须**设置，`--opd-type=sglang` 时**不可**设置。 |
| `--opd-teacher-ckpt-step` | 可选的教师模型 checkpoint 步数。 |

## 原理

OPD 通过减去一个 KL 惩罚项来修改 advantage 计算，鼓励学生匹配教师的输出分布：

$$
\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t
$$

其中 $A_t$ 是基础 estimator（如 GRPO）的原始 advantage，$\lambda_{\text{opd}}$ 是 `--opd-kl-coef`，$D_{\text{KL}}$ 是 token 级的逆 KL 散度。

因此 OPD 可以与任何 advantage estimator 组合使用，包括 GRPO、PPO、REINFORCE++ 和 GSPO。

## 两种教师模式

### SGLang 模式 (`--opd-type sglang`)

教师模型运行在外部 SGLang 服务器上，教师的 log-probs 在 rollout 阶段获取。

**适用场景**：教师与学生架构不同，或教师模型太大无法与训练模型同时加载。

**工作流程**：
1. 外部 SGLang 服务器运行教师模型。
2. 在 rollout 阶段，自定义 reward 函数（`slime.rollout.on_policy_distillation.reward_func`）将每个样本发送到教师服务器以获取 token 级 log-probs。
3. 自定义后处理函数（`slime.rollout.on_policy_distillation.post_process_rewards`）将教师 log-probs 裁剪到 response 范围并存储到 `sample.teacher_log_probs` 中。
4. 在训练阶段，从存储的教师 log-probs 计算 KL 惩罚并应用到 advantages 上。

**配置**：
```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
```

### Megatron 模式 (`--opd-type megatron`)

教师模型通过 `--opd-teacher-load` 直接加载到 Megatron 中，教师的 log-probs 在训练前向传播阶段计算。

**适用场景**：教师与学生/参考模型架构相同，且能放入 GPU 显存。

**工作流程**：
1. 教师模型在初始化时作为额外的 Megatron 模型加载。
2. 在训练前向传播阶段，教师模型为每个样本计算 log-probs。
3. 内联计算 KL 惩罚并应用到 advantages。

**配置**：
```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_torch_dist
```

> **注意**：教师 checkpoint 必须是 Megatron 格式（`torch_dist` 或 `torch`）。可以使用 `tools/convert_hf_to_torch_dist.py` 从 HuggingFace 格式转换。

## 运行示例

完整的示例脚本在 `examples/on_policy_distillation/` 中：

### SGLang 教师

```bash
# 1. 下载模型和数据
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2. 转换学生模型
cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist

# 3. 运行
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

### Megatron 教师

```bash
# 1. 将学生和教师模型都转换为 Megatron 格式
# 2. 运行
bash examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
```

## 初步结果

使用 Qwen3-8B-Base 模型在 [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) 数据集的一部分上进行 SFT，然后在剩余数据上用 Qwen3-32B 教师进行在策略蒸馏，Math500 评测结果如下：

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |
