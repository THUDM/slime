# Training-Inference Mismatch Correction on `pg_loss`
Training-Inference Mismatch Correction through **Importance Sampling**.



## Summary
This function is used to solve **training-inference mismatch** through algorithmic adapations, e.g. TIS, MIS. (Reference: [training-inference mismatch]())

We included 3 rollout correction algorithms, (1) decoupled, 3-policies PPO with training-inference importance sampling, (2) direct policy overwriting in standard PPO  (3) pure REINFORCE loss (with out PPO clipping) with training-inference importance sampling. You may use **loss algorithm selection** APIs `--use_rollout_log_probs` and `--use-tis` to select one of the rollout correction loss (details in **III. Algorithms**).

When training-inference importance sampling is enabled (`--use-tis`), You can also specify the **TIS settings** with selection on `tis_mode = {truncate, mask, clip}`, and `tis_level = {token, sequence, geometric}`. We will also provide some recommended settings on each mode. (details in **IV. recommended settings**)


## I. Algorithms

### 0. [Baseline: No Mismatch Correction] Standard PPO
This is the basic PPO algorithm with potentially training-inference mismatch issue when the output of SGLang and Megatron does not exactly match.
$$
L_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{Megatron}}}(y \mid x)} A_t,\;
    \operatorname{clip}\!\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{Megatron}}}(y \mid x)},
      1 - \epsilon,\;
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$


### 1. Bypassing PPO importance sampling  
In this method, we directly use rollout engine's log probs as the old policy in offline PPO's importance sampling, instead of the recomputed log_probs in training engine.

$$
L_{\text{PPO\_bypass}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)} A_t,\;
    \operatorname{clip}\!\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)},
      1 - \epsilon,\;
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

Advantages: 
- Efficiency: skip log_prob recomputation on training engine. Reduce one forward pass's computation.



### 2. Decoupled, 3-policies PPO importance sampling  

[Decoupled PPO](https://arxiv.org/pdf/2110.00641) achieves batch independent PPO by decoupling two roles: Proximal Policy (anchor policy for PPO clipping, control update size) and Behavior Policy (for off-policy correction in importance sampling). Therefore, there are totally 3 roles engaged in this mode, **target policy** $\pi_\theta$, **proximal policy** $\pi_{\textcolor{blue}{\text{old}}}$ and **behavior policy** $\pi_{\textcolor{red}{\text{SGLang}}}$. $\pi_{\textcolor{blue}{\text{old}}}$ is recomputed with Megatron at the beginning of each training step.



$$
L_{\text{PPO\_bypass}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
    \frac{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)}
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)} A_t,\;
    \operatorname{clip}\!\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)},
      1 - \epsilon,\;
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

Advantages: 
- Achieves batch size invariance and efficient stale data utilization
- Enables accurate off-policy metrics monitoring


### 3. REINFORCE + pure importance sampling

REINFORCE + pure IS is a simple and efficient method by directly apply TIS/MIS to REINFORCE loss, without PPO clipping.

$$
L_{\text{REINFORCE\_IS}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
    \frac{\pi_{\theta}(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)}
    \cdot \Sigma_t \log \pi_\theta \cdot A_t
\right].
$$


Advantages: 
- Efficiency: skip log_prob recomputation on training engine. Reduce one forward pass's computation.


## II. APIs on algorithms

You may choose from above algorithms by specifying arguments below:

`--use_rollout_log_probs`: True if only use `rollout_log_probs` to compute the loss, bypassing old_log_probs calculated by training engine;

`--use-tis`: True if apply TIS/MIS to loss.

| `use_rollout_log_probs` | `use_tis` | Algorithm | Policies |Compute old_log_probs | Batch Invariant | Recommended TIS Mode |
|-----------------|-------------|-----------|--------------|---------------|-----------------|----------------------|
| False | False | Standard PPO (Algorithm 0) | 2 ($\pi_\theta$, $\pi_{\textcolor{blue}{\text{old}}}$)|Yes | No | N/A |
| True | False | Bypassing PPO (Algorithm 1) | 2 ($\pi_\theta$, $\pi_{\textcolor{red}{\text{SGLang}}}$) |🚀 Skipped | No | N/A |
| False | True | Decoupled PPO (Algorithm 2) | 3 ($\pi_\theta$, $\pi_{\textcolor{blue}{\text{old}}}$, $\pi_{\textcolor{red}{\text{SGLang}}}$)  |Yes  | Yes | token/seq/geo |
| True | True | REINFORCE+IS (Algorithm 3) | 2 ($\pi_\theta$, $\pi_{\textcolor{red}{\text{SGLang}}}$) |🚀 Skipped | No | seq |




## III. Recommended Settings on TIS/MIS [In Construction]

When choose to use importance sampling for mismatch correction (`use-tis` enabled, Algorithm 2 & 3), you may specify the IS modes and applied levels. 

We provided 3 modes: **truncate**, **clip** and **mask**.

Truncate and clip is applied to **importance sampling weight**; 
Mask is applied through **rejection sampling**.

[Some examples here]
```
[Input]
[truncate mode output]
[clip mode output]
[mask mode output]
```

And three levels: **token**, **sequence**, **geometric**.

**Token Level (default)**:

Computes importance weights independently for each token:
$w_i = \exp\left(\log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i)\right)$

Characteristics: Biased but computationally simple, suitable for most scenarios

**Sequence Level**:

Uses the product of all token weights as the sequence weight:
$w_{\text{seq}} = \exp\left( \sum_i \left( \log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i) \right) \right)$

Characteristics: Unbiased but high variance, suitable for sequence-level optimization

**Geometric Level**:

Uses geometric mean to compute sequence weight:
$
w_{\text{seq}} = \exp\left( \frac{1}{n} \sum_{i=1}^{n} \left( \log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i) \right) \right)
$

Characteristics: Biased but low variance, balances bias and variance

**Recommendations**:

[A table here]



## Reference

We thank the materials below for their excellent findings and theories.
1. [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)
2. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
3. [Mathematical Formulations of Rollout Correction Methods in verl](https://github.com/szrlee/verl/blob/yingru/rollout_correction/docs/advance/rollout_corr_math.md)

