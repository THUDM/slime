# 在策略蒸馏 (On-Policy Distillation)

在策略蒸馏 (OPD) 使用学生当前策略采样的 response token 来训练学生。对于学生轨迹中访问到的每个前缀，固定的教师模型为同一个 next token 评分，从而沿学生自己的轨迹提供稠密的 token 级学习信号。在 slime 中，这一信号以逆 KL 的采样估计惩罚 advantage，因此可以与 GRPO、PPO、REINFORCE++ 等 advantage estimator 组合；当任务奖励为零时，同一机制就是纯蒸馏。

## 关键参数

| 参数 | 说明 |
|------|------|
| `--use-opd` | 启用在策略蒸馏。使用 OPD 的必需标志。 |
| `--opd-type` | OPD 类型：`sglang` 或 `megatron`。启用 `--use-opd` 时必须设置。 |
| `--opd-kl-coef` | OPD KL 惩罚系数（默认值：1.0）。控制蒸馏信号相对于 RL advantage 的权重。 |
| `--opd-teacher-load` | 教师模型的 Megatron checkpoint 路径。`--opd-type=megatron` 时**必须**设置，`--opd-type=sglang` 时**不可**设置。 |
| `--opd-teacher-ckpt-step` | 可选的教师模型 checkpoint 步数。 |

## 原理

记 $\pi_\theta$ 为学生策略，$\pi_T$ 为教师策略，$h_t$ 为学生生成轨迹中采样 token $a_t$ 之前的历史。按照 [Thinking Machines Lab 给出的定义](https://thinkingmachines.ai/blog/on-policy-distillation/)，token 级逆 KL 为

$$
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot \mid h_t)\,\|\,\pi_T(\cdot \mid h_t)\right)
= \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid h_t)}\!\left[
\log \pi_\theta(a_t \mid h_t) - \log \pi_T(a_t \mid h_t)
\right].
$$

这里的顺序很重要：KL 的第一个参数是学生分布，期望同样对学生分布取值。教师不生成训练轨迹，而是评估学生实际采样的 token。

slime 不会遍历完整词表来精确计算这个期望。对于每个采样 token，它使用如下 Monte Carlo 贡献：

$$
\hat d_t = \log \pi_\theta(a_t \mid h_t) - \log \pi_T(a_t \mid h_t),
\qquad a_t \sim \pi_\theta(\cdot \mid h_t),
$$

然后修改基础 advantage：

$$
\hat A_t = A_t - \lambda_{\mathrm{opd}}\hat d_t.
$$

其中 $A_t$ 来自所配置的 estimator（纯蒸馏时为零），$\lambda_{\mathrm{opd}}$ 是 `--opd-kl-coef`。尽管 KL 的期望非负，单个样本的 $\hat d_t$ 仍可能为负。策略损失使用修改后的 $\hat A_t$，因此 OPD 项与 GRPO、PPO、REINFORCE++、GSPO 等 advantage estimator 的选择相互独立。

## 两种教师模式

### SGLang 模式 (`--opd-type sglang`)

教师模型运行在外部 SGLang 服务器上，教师的 log-probs 在 rollout 阶段获取。

**适用场景**：教师与学生架构不同，或教师模型太大无法与训练模型同时加载。由于教师需要为学生的原始 token ID 评分，两者仍须使用兼容的 tokenizer 和词表。

**工作流程**：
1. 外部 SGLang 服务器运行教师模型。
2. 在 rollout 阶段，自定义 reward 函数（`slime.rollout.on_policy_distillation.reward_func`）将学生采样的 token ID 发送给教师服务器，并获取教师对这些相同 token 的 log-probability。
3. 自定义后处理函数（`slime.rollout.on_policy_distillation.post_process_rewards`）将教师 log-probs 裁剪到 response 范围并存储到 `sample.teacher_log_probs` 中。
4. 在训练阶段，slime 从基础 advantage 中减去按 `--opd-kl-coef` 缩放后的采样 log-probability 差值。

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
