# 从策略梯度到训推一致性

[返回实验室](../index.rst)。精度、确定性、异步、PD 和 HiCache 见[系统篇](rl-systems.md)。

下面按照仓库实现推导各个目标，区分数学恒等式、有偏 surrogate 和工程近似。论文链接标注原始工作；实现链接说明向导具体导出的变体。

## 统一符号

$x$ 表示 prompt，$y=(a_1,\ldots,a_T)$ 表示回答，$h_t=(x,a_{<t})$ 表示 prefix。奖励 $R(x,y)$ 在求导时不依赖参数。必须区分三种策略：

| 符号 | 含义 | 来源 |
|---|---|---|
| $q(a\mid h)$ | 真正产生 token 的行为分布 | SGLang，包含量化、采样变换以及可能过期的权重 |
| $p_0(a\mid h)$ | 更新前冻结的训练策略 | Megatron 重新计算的 old log-probs |
| $p_\theta(a\mid h)$ | 当前可训练策略 | Megatron 的可微 forward |
| $\operatorname{sg}(z)$ | 数值为 $z$、梯度为零 | 对权重、奖励和修正系数停止梯度 |

固定 prefix，定义 score 为 $s_a=\nabla_\theta\log p_\theta(a\mid h)$。在概率可微、归一化且可交换求和与求导的前提下：

$$
\sum_a p_\theta(a\mid h)s_a
=\sum_a\nabla_\theta p_\theta(a\mid h)
=\nabla_\theta 1=0.
$$

从 $J(\theta)=\sum_y p_\theta(y\mid x)R(x,y)$ 出发，应用 $\nabla p=p\nabla\log p$ 和自回归分解：

$$
\nabla J
=\mathbb E_{y\sim p_\theta}[R\nabla\log p_\theta(y\mid x)]
=\mathbb E_{y\sim p_\theta}\left[R\sum_t s_{a_t}\right].
$$

在 **$p_\theta$ 分布下**，与当前 action 无关的 baseline $b(h_t)$ 和 score 相乘后的期望为零，因此可以降低方差而不改变这个期望。换成 $q$ 采样，一般就不能这样消去。两个引擎加载了同名 checkpoint，并不证明训练是 on-policy。

## GRPO

原论文：[DeepSeekMath](https://arxiv.org/abs/2402.03300)；裁剪目标：[PPO](https://arxiv.org/abs/1707.06347)。

对同一 prompt 采样 $G$ 个回答，用组内统计构造 advantage：

$$
\bar R=\frac1G\sum_i R_i,\quad
\sigma_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\quad
A_i=\frac{R_i-\bar R}{\sigma_R+\epsilon}.
$$

同一回答中的 response token 共用这个 advantage。标准差是否使用样本修正以及具体 epsilon 取决于实现；关闭标准差归一化后，$A_i=R_i-\bar R$。组均值包含当前样本，不能直接套用“独立 baseline 无偏”的证明。独立采样时，仅减均值会让 on-policy 梯度期望缩放为 $1-1/G$；再除标准差还引入随机尺度。

令 $r_{i,t}(\theta)=p_\theta(a_{i,t}\mid h_{i,t})/p_0(a_{i,t}\mid h_{i,t})$。忽略可选 KL 项，要最大化的 clipped surrogate 为：

$$
J_{\rm clip}=\frac1G\sum_i\frac1{T_i}\sum_t
\min\left(r_{i,t}A_i,
\operatorname{clip}(r_{i,t},1-\epsilon_-,1+\epsilon_+)A_i\right).
$$

$A_i>0$ 时，上界限制继续提高 token 概率的收益；$A_i<0$ 时，下界限制继续降低概率的收益。在 $\theta=\theta_0$ 且未触发 clipping 时，$\nabla r=r\nabla\log p_\theta$ 还原为策略梯度方向。这是局部 surrogate，不是任意过期数据上的恒等式。

向导使用 `--advantage-estimator grpo`、每个 prompt 8 次采样；不开 SC 时采用 PPO clipping。`--calculate-per-token-loss` 将聚合方式改成“总 loss / 总 response token 数”，与“先求每个回答均值，再平均回答”并不相同。

## TIS

原始工作：[On the Rollout-Training Mismatch in Modern RL Systems](https://openreview.net/forum?id=8MHqvb4lK9) 和[作者技术文章](https://fengyao.notion.site/off-policy-rl)。下面的代数也对应仓库内置 callback。

固定 prefix，只要 $p_0(a)f(a)\ne0$ 的位置都有 $q(a)>0$，就有换测度恒等式：

$$
\mathbb E_{p_0}[f(a)]
=\sum_a q(a)\frac{p_0(a)}{q(a)}f(a)
=\mathbb E_q[\rho_a f(a)],\quad
\rho_a=\exp(\log p_0(a)-\log q(a)).
$$

PPO 的更新比值 $r=p_\theta/p_0$ 与行为修正比值 $\rho=p_0/q$ 负责不同问题。未裁剪时两者乘积为 $p_\theta/q$。slime 将 PPO loss 乘上停止梯度的 $w(\rho)=\operatorname{clip}(\rho,L,U)$，向导取 $L=0,U=2$。

减去精确表达式，就能直接得到偏差：

$$
\mathbb E_q[w(\rho)f]-\mathbb E_{p_0}[f]
=\mathbb E_q[(w(\rho)-\rho)f].
$$

$L=0$ 时，偏差为 $-\mathbb E_q[(\rho-U)f\mathbf1_{\rho>U}]$。限制权重能抑制离群值，却不能保持无偏；$L>0$ 时还会抬高小权重。

这个恒等式固定了 prefix，没有修正 prefix 本身的分布。完整序列的换测度需要 $p_0(y)/q(y)=\prod_t\rho_t$，方差可能很大。逐 token TIS 是实用 surrogate，不能宣称等价于完整乘积。

实现：[`vanilla_tis_function`](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py)、[`importance_weights`](https://github.com/THUDM/slime/blob/main/slime/utils/ppo_utils.py)。REINFORCE（包括 SC）使用停止梯度的**当前**训练策略作为修正分子，而不是 $p_0$。

## ICE-POP

原论文：[Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://arxiv.org/abs/2510.18855)。

向导选择 slime 的 `icepop_function`，其行为修正规则是：

$$
w_{\rm ice}(\rho)=\rho\mathbf1_{L\le\rho\le U},\qquad L=0.5,\quad U=2.
$$

区间内保留重要性权重，区间外贡献为零。因此相对于同一个固定 prefix 的目标：

$$
\mathbb E_q[w_{\rm ice}(\rho)f]-\mathbb E_{p_0}[f]
=-\mathbb E_q[\rho f\mathbf1_{\rho<L\lor\rho>U}].
$$

例如 $\rho=4$ 时，向导中的 TIS 给权重 2，IcePop 给权重 0。两者不是同一种估计器。Callback 将权重置零，**不会**按保留下来的 token 数重新归一化 loss；PPO clipping 仍可以独立作用于 $r$。论文完整训练流程还有其他部分，启用这个 callback 并不等于复现整篇论文。

导出参数为 `--use-tis --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function --tis-clip-low 0.5 --tis-clip 2`。观察 `tis`、`tis_clipfrac`、reward 以及 mask 后是否还有足够的学习信号。

## R3

原论文：[Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)。

把 MoE 层写为 $F(x)=\sum_{i\in I(x)}g_i(x)E_i(x)$，其中 $I=\operatorname{TopK}(u(x))$ 是离散 expert 集合。Top-k 边界附近，微小扰动也可能换掉一个专家。加减同一个中间项，可将输出差异分解为：

$$
F_{I_q}(x_q)-F_{I_p}(x_p)
=\underbrace{F_{I_q}(x_q)-F_{I_q}(x_p)}_{\text{相同路由下的数值与输入差异}}
+\underbrace{F_{I_q}(x_p)-F_{I_p}(x_p)}_{\text{专家选择差异}}.
$$

训练时固定为 rollout 记录的 $I_q$，就能在**此层、此比较中**消去第二项，但第一项仍然存在。以 softmax gate 为例：

$$
g_i=\frac{\mathbf1_{i\in I_q}\exp u_i}{\sum_{j\in I_q}\exp u_j},\qquad
\frac{\partial g_i}{\partial u_j}=g_i(\mathbf1_{i=j}-g_j),\quad i,j\in I_q.
$$

重放的是专家身份，不是 detach 的专家输出；梯度仍可流过选中的 gate 和 expert。GLM 使用 sigmoid 路由及自己的归一化与缩放，以上 softmax 只是示例，实际必须保留模型的 gate 规则。

向导添加 `--use-rollout-routing-replay`。slime 管理的 SGLang 自动返回路由；external 引擎需自行启用 `--enable-return-routed-experts`。存储随 token 数 × MoE 层数 × 选中专家数增长。R3 可独立组合 TIS/SC，对 dense 模型没有意义。实现见 [`sglang_engine.py`](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/sglang_engine.py) 和 [`megatron_utils`](https://github.com/THUDM/slime/tree/main/slime/backends/megatron_utils) 的 replay 逻辑。

## SC

原论文：[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)。以下按照 slime 的 [`score_centering_correction`](https://github.com/THUDM/slime/blob/main/slime/utils/score_centering.py) 和 loss 实现展开推导。

### 从非零均值到中心化 score

固定 prefix，令 $\mu=\sum_a q_as_a$。即使两侧 checkpoint 相同，一般仍有 $\mu\ne0$。对停止梯度的 advantage $A$，应用协方差分解：

$$
\mathbb E_q[As]=\mathbb E_q[A]\mu+\operatorname{Cov}_q(A,s).
$$

令 $\tilde s_a=s_a-\mu$，则 $\mathbb E_q[\tilde s]=0$，因此：

$$
\mathbb E_q[A\tilde s]=\operatorname{Cov}_q(A,s).
$$

Prefix 条件下的均值项被消去了。但协方差仍在 $q$ 下计算，这并不证明得到了 $p_\theta$ 下的无偏梯度。对完整回答做组内 reward centering，也不保证每个 prefix 上都有 $\mathbb E[A\mid h]=0$。

### 与重要性修正组合

令 $w_a=f(p_a/q_a)$，$f$ 可以是原始比值、TIS、mask 或常数 1。定义 $\mu_w=\sum_aq_aw_as_a$，对**加权 score**进行中心化：

$$
\tilde s_a^w=w_as_a-\mu_w,\qquad \mathbb E_q[\tilde s^w]=0.
$$

实现这个局部梯度方向的一种最小化 loss 是：

$$
\ell=-\operatorname{sg}(A)\left[
\operatorname{sg}(w_a)\log p_a-
\sum_v\operatorname{sg}(q_vw_v)\log p_v\right].
$$

只对 $\log p$ 求导，得到 $-\nabla\ell=A(w_as_a-\mu_w)$。如果让梯度流过 $w$，就会出现额外项。完整 support 且 $w=p/q$ 时，$\mu_w=\sum p s=0$；经过裁剪或 mask 后通常不再为零。

### Top-k 如何避免保存整个词表

slime 保存 sampler 的 head $H$，向导默认 128 个 token。这些概率保留**完整词表上的质量**，不按 head 重新归一化。定义：

$$
P_H=\sum_{v\in H}p_v,\quad Q_H=\sum_{v\in H}q_v,\quad
\alpha=\frac{1-Q_H}{1-P_H},\quad
\hat q_v=\begin{cases}q_v&v\in H\\\alpha p_v&v\notin H.\end{cases}
$$

该近似的概率和为一。利用 $\sum_vp_vs_v=0$，把 tail 项替换掉：

$$
\mu_{\hat q}=\sum_Hq_vs_v+\alpha\sum_{v\notin H}p_vs_v
=\sum_H(q_v-\alpha p_v)s_v.
$$

Tail 中 $p_v/\hat q_v=1/\alpha$，因此加权形式为：

$$
\mu_{\hat q,w}=\sum_{v\in H}
[q_vf(p_v/q_v)-\alpha f(1/\alpha)p_v]s_v.
$$

这正是代码中的 head residual。完整分布的中心化是精确的，使用重建 tail 时则是近似；很小的 tail mass 还需要数值保护。可以用一个 logit 小例子验证：$p=(0.6,0.4)$、$q=(0.5,0.5)$、$s_a=e_a-p$，未中心化的均值为 $q-p=(-0.1,0.1)$；减去它之后 score 的期望为零。

### 必须匹配真实采样分布

向导固定 temperature=1、top-p=1、top-k=-1，并使用 `--pg-loss-type reinforce`、`--disable-grpo-std-normalization`、`--calculate-per-token-loss`。不能把 SC 直接塞进 GSPO / CISPO / PPO clipping；此路径不支持 streaming、约束解码、采样惩罚和逐请求修改 temperature。

slime 还支持记录 top-p support $S$：将训练概率也归一化到 $S$，使用所有记录的 $q_v$。此时 $\mu_w=\sum_{v\in S}q_vw_v\nabla\log p_v^S$ 无需 tail 近似。这只相对于记录的 support 精确，不等同于未截断策略；数据量可能远大于 top-128。详见[采样与兼容条件](../get_started/usage.md#score-centering)。

## 如何验证选择

固定 prompt 数据、seed、回复长度预算和评估集。从短基线开始，每次改一个机制，同时记录 reward、`train_rollout_logprob_abs_diff`、适用时的 `tis_clipfrac`，以及异步的 `staleness/mean`、`staleness/max`。差异更小不自动意味着学得更好。保存固定 batch 做 [debug replay](../developer_guide/debug.md)，将优化器行为与采样变化分开定位。
