# From policy gradients to train–rollout consistency

[Return to the experiment lab](../index.rst). Read [the systems companion](rl-systems.md) for precision, determinism, asynchronous execution, PD, and HiCache.

The equations below are a tutorial derivation of the objectives implemented in this repository. They distinguish identities, biased surrogates, and engineering approximations. Paper links identify the original work; the implementation links explain the particular variant exported by the lab.

## A common language

Let $x$ be a prompt, $y=(a_1,\ldots,a_T)$ a response, and $h_t=(x,a_{<t})$ a prefix. Let $R(x,y)$ be a reward independent of the parameters during differentiation. Use three different policies:

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $q(a\mid h)$ | Behavior distribution that actually sampled the token | SGLang, including quantization, sampling transforms, and potentially older weights |
| $p_0(a\mid h)$ | Frozen trainer before the optimizer update | Megatron's recomputed old log-probs |
| $p_\theta(a\mid h)$ | Current trainable policy | Megatron's differentiable forward |
| $\operatorname{sg}(z)$ | Stop-gradient: value $z$, derivative zero | Detaching weights, rewards, and correction coefficients |

At a fixed prefix define the score $s_a=\nabla_\theta\log p_\theta(a\mid h)$. Assuming normalized differentiable probabilities and legitimate interchange of summation and differentiation:

$$
\sum_a p_\theta(a\mid h)s_a
=\sum_a\nabla_\theta p_\theta(a\mid h)
=\nabla_\theta 1=0.
$$

Starting with $J(\theta)=\sum_y p_\theta(y\mid x)R(x,y)$, apply $\nabla p=p\nabla\log p$ and autoregressive factorization:

$$
\nabla J
=\mathbb E_{y\sim p_\theta}\left[R\nabla\log p_\theta(y\mid x)\right]
=\mathbb E_{y\sim p_\theta}\left[R\sum_{t=1}^{T}s_{a_t}\right].
$$

An action-independent baseline $b(h_t)$ has zero expectation against the score **under $p_\theta$**, so it can reduce variance without changing that expectation. Sampling from $q$ generally breaks this cancellation. This is why identical checkpoint names do not prove an on-policy update.

## GRPO

Original work: [DeepSeekMath](https://arxiv.org/abs/2402.03300); clipping objective: [PPO](https://arxiv.org/abs/1707.06347).

For $G$ responses to one prompt, form group statistics and a response advantage:

$$
\bar R=\frac1G\sum_{i=1}^G R_i,\quad
\sigma_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\quad
A_i=\frac{R_i-\bar R}{\sigma_R+\epsilon}.
$$

The response advantage is shared across its response tokens. The exact standard-deviation convention and numerical epsilon are implementation details; disabling standard-deviation normalization gives $A_i=R_i-\bar R$. Group baselines include the sampled response itself, so the usual unbiased-baseline argument does not apply literally. For independent responses, mean centering alone scales the expected on-policy gradient by $1-1/G$; standard-deviation normalization adds a random scale.

Define $r_{i,t}(\theta)=p_\theta(a_{i,t}\mid h_{i,t})/p_0(a_{i,t}\mid h_{i,t})$. A clipped surrogate to maximize, omitting optional KL terms, is

$$
J_{\rm clip}=\frac1G\sum_i\frac1{T_i}\sum_t
\min\left(r_{i,t} A_i,
\operatorname{clip}(r_{i,t},1-\epsilon_-,1+\epsilon_+)A_i\right).
$$

For positive $A_i$, the upper clip limits incentive to increase token probability. For negative $A_i$, the lower clip limits incentive to decrease it. At $\theta=\theta_0$, inside the clip, $\nabla r=r\nabla\log p_\theta$ reduces to the policy-gradient direction. This is a local surrogate, not a general identity for arbitrarily stale data.

The lab emits `--advantage-estimator grpo`, eight samples per prompt, and PPO clipping unless SC is enabled. `--calculate-per-token-loss` changes the outer reduction to total loss divided by total response-token count; it is not the same weighting as averaging each response's mean.

## TIS

Original work: [On the Rollout-Training Mismatch in Modern RL Systems](https://openreview.net/forum?id=8MHqvb4lK9), with the [authors' technical article](https://fengyao.notion.site/off-policy-rl). The following algebra also describes the repository's built-in callback.

At a fixed prefix, provided $q(a)>0$ wherever $p_0(a)f(a)\ne0$, change of measure gives

$$
\mathbb E_{a\sim p_0}[f(a)]
=\sum_a q(a)\frac{p_0(a)}{q(a)}f(a)
=\mathbb E_{a\sim q}[\rho_a f(a)],\qquad
\rho_a=\exp(\log p_0(a)-\log q(a)).
$$

The PPO policy-update ratio $r=p_\theta/p_0$ and behavior-correction ratio $\rho=p_0/q$ serve different purposes. Before clipping, their product is $p_\theta/q$. slime multiplies the PPO loss by detached $w(\rho)=\operatorname{clip}(\rho,L,U)$; the lab uses $L=0,U=2$.

Derive its bias by subtracting the exact expression:

$$
\mathbb E_q[w(\rho)f]-\mathbb E_{p_0}[f]
=\mathbb E_q[(w(\rho)-\rho)f].
$$

For $L=0$, this becomes $-\mathbb E_q[(\rho-U)f\,\mathbf1_{\rho>U}]$. Capping weights limits outliers but does not give unbiased importance sampling. With nonzero $L$, small weights are also increased.

This fixed-prefix identity does not correct the distribution of prefixes. Full-sequence change of measure would use $p_0(y)/q(y)=\prod_t\rho_t$, often with high variance. Token-wise TIS is a practical surrogate, not an exact replacement for that product.

Implementation: [`vanilla_tis_function`](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py), [`importance_weights`](https://github.com/THUDM/slime/blob/main/slime/utils/ppo_utils.py). REINFORCE, including SC, uses detached **current** trainer probabilities in the correction numerator instead of $p_0$.

## ICE-POP

Original work: [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://arxiv.org/abs/2510.18855).

The lab selects slime's built-in `icepop_function`. Its behavior-correction rule is

$$
w_{\rm ice}(\rho)=\rho\,\mathbf1_{L\le\rho\le U},\qquad L=0.5,\quad U=2.
$$

Inside the interval this equals importance sampling. Outside it, the contribution is zero. Its bias relative to the same fixed-prefix target is therefore

$$
\mathbb E_q[w_{\rm ice}(\rho)f]-\mathbb E_{p_0}[f]
=-\mathbb E_q[\rho f\,\mathbf1_{\rho<L\;\lor\;\rho>U}].
$$

A token with $\rho=4$ receives weight 2 under the lab's TIS rule and weight 0 under IcePop. These are distinct estimators. The callback sets weights to zero; it does **not** renormalize the loss by the number of retained tokens. PPO clipping can still act on $r$ independently. The paper's complete training recipe contains other components; selecting this callback does not reproduce the whole paper.

Exported settings: `--use-tis --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function --tis-clip-low 0.5 --tis-clip 2`. Observe `tis`, `tis_clipfrac`, reward, and effective signal after masking.

## R3

Original work: [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370).

Write a MoE layer as $F(x)=\sum_{i\in I(x)}g_i(x)E_i(x)$, with discrete expert set $I=\operatorname{TopK}(u(x))$. A perturbation near a top-k boundary can swap an expert even when its magnitude is tiny. Decompose the output difference by adding and subtracting the same intermediate term:

$$
F_{I_q}(x_q)-F_{I_p}(x_p)
=\underbrace{F_{I_q}(x_q)-F_{I_q}(x_p)}_{\text{same-route numerical/input difference}}
+\underbrace{F_{I_q}(x_p)-F_{I_p}(x_p)}_{\text{expert-selection difference}}.
$$

Replay sets the trainer route to recorded $I_q$, eliminating the second term **at this layer for this comparison**. It does not eliminate the first. For a softmax-gated example,

$$
g_i=\frac{\mathbf1_{i\in I_q}\exp u_i}{\sum_{j\in I_q}\exp u_j},\qquad
\frac{\partial g_i}{\partial u_j}=g_i(\mathbf1_{i=j}-g_j),\quad i,j\in I_q.
$$

Replay expert identities, not detached expert outputs. Gradients can still flow through selected gates and experts. GLM uses sigmoid-based routing and its own normalization/scaling, so this softmax example is illustrative; keep the architecture's actual gate rule.

The lab emits `--use-rollout-routing-replay`; slime-managed SGLang requests routed experts automatically. External engines must be launched with `--enable-return-routed-experts`. Storage grows with tokens × MoE layers × selected experts. R3 is independent of TIS/SC and is not meaningful for a dense model. Implementation: [`sglang_engine.py`](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/sglang_engine.py) and the replay code in [`megatron_utils`](https://github.com/THUDM/slime/tree/main/slime/backends/megatron_utils).

## SC

Original work: [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807). The derivation here follows the algebra of slime's [`score_centering_correction`](https://github.com/THUDM/slime/blob/main/slime/utils/score_centering.py) and its loss implementation.

### From an unwanted mean to a centered score

Fix a prefix and let $\mu=\sum_a q_a s_a$. In general $\mu\ne0$, even with the same checkpoint on both sides. For a detached advantage $A$, covariance decomposition gives

$$
\mathbb E_q[A s]=\mathbb E_q[A]\mu+\operatorname{Cov}_q(A,s).
$$

Use $\tilde s_a=s_a-\mu$. Then $\mathbb E_q[\tilde s]=0$, and

$$
\mathbb E_q[A\tilde s]=\operatorname{Cov}_q(A,s).
$$

The prefix-conditional mean term vanishes. The covariance is still under $q$, so this does not establish an unbiased gradient under $p_\theta$. Group-centered rewards across full responses do not ensure $\mathbb E[A\mid h]=0$ at every prefix.

### Compose with a correction weight

Let $w_a=f(p_a/q_a)$, with $f$ either identity, TIS, masking, or 1. Define $\mu_w=\sum_a q_aw_as_a$ and center the **weighted** score:

$$
\tilde s^w_a=w_as_a-\mu_w,\qquad
\mathbb E_q[\tilde s^w]=0.
$$

An implementable minimization loss with exactly this local gradient direction is

$$
\ell=-\operatorname{sg}(A)\left[
\operatorname{sg}(w_a)\log p_a
-\sum_v\operatorname{sg}(q_vw_v)\log p_v\right].
$$

Differentiate only $\log p$: $-\nabla\ell=A(w_as_a-\mu_w)$. Differentiating through $w$ would add unwanted derivatives. With exact $w=p/q$ and full support, $\mu_w=\sum p s=0$; clipped or masked weights generally make it nonzero.

### Why top-k does not require storing the full vocabulary

slime records the sampler's head $H$ (128 tokens in the lab). Its probabilities retain **full-vocabulary mass**, not a head-only renormalization. Set

$$
P_H=\sum_{v\in H}p_v,\quad Q_H=\sum_{v\in H}q_v,\quad
\alpha=\frac{1-Q_H}{1-P_H},\quad
\hat q_v=\begin{cases}q_v&v\in H\\\alpha p_v&v\notin H.\end{cases}
$$

This approximation sums to one. Since $\sum_v p_vs_v=0$, the tail score can be replaced algebraically:

$$
\mu_{\hat q}=\sum_{H}q_vs_v+\alpha\sum_{v\notin H}p_vs_v
=\sum_H(q_v-\alpha p_v)s_v.
$$

In the tail $p_v/\hat q_v=1/\alpha$, so with $w=f(p/q)$:

$$
\mu_{\hat q,w}=\sum_{v\in H}
\left[q_v f(p_v/q_v)-\alpha f(1/\alpha)p_v\right]s_v.
$$

This is the head-only residual used in the implementation. Full-distribution centering is exact; centering with the reconstructed tail is an approximation, and small tail masses need numerical protection. A minimal logit check uses $p=(0.6,0.4)$, $q=(0.5,0.5)$ and $s_a=e_a-p$: the uncentered mean is $q-p=(-0.1,0.1)$; subtracting this mean makes the expected centered score zero.

### Match the sampler's actual distribution

The lab fixes temperature 1, top-p 1, top-k -1 and selects `--pg-loss-type reinforce`, `--disable-grpo-std-normalization`, `--calculate-per-token-loss`. SC cannot be substituted into GSPO/CISPO/PPO clipping. Streaming, constrained decoding, sampling penalties, and per-request temperature changes are unsupported here.

slime also supports recorded top-p support $S$: normalize trainer probabilities on $S$ and use all recorded $q_v$ on that support. Then $\mu_w=\sum_{v\in S}q_vw_v\nabla\log p^S_v$ needs no tail approximation. This is exact relative to the recorded support, not to an untruncated policy; storage can greatly exceed top-128. See [sampling and compatibility details](../get_started/usage.md#score-centering).

## What to measure

Hold prompt data, seed, response budget, and evaluation fixed. Begin with the short baseline, change one mechanism at a time, and record reward alongside `train_rollout_logprob_abs_diff`, `tis_clipfrac` when applicable, and async `staleness/mean` / `staleness/max`. Smaller mismatch alone does not prove better learning. Save a fixed batch for [debug replay](../developer_guide/debug.md) to distinguish optimizer behavior from sampling variation.
