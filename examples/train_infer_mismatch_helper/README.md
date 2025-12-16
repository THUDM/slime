# Rollout Correction Methods

Rollout correction (e.g, TIS, MIS) through algorithmic methods.


## Quick Takeaway

This function is used to solve offline scenarios through algorithmic adaptations, e.g. TIS/MIS.

We included 3 rollout correction algorithms:

1. decoupled, 3-policies PPO with rollout importance sampling
2. direct rollout policy overwriting in the standard PPO
3. pure REINFORCE loss (without PPO clipping) with rollout importance sampling


`--use-tis`: use this flag to **turn on TIS/MIS** for rollout correction (details in **Algorithms**).
You may specify the **IS/RS configs** with a config file using `--custom-config-path`.

`--use-rollout-logprobs`: When use this flag, the logprobs will **not** be recomputed by training engine - rollout log probs will be directly used in PPO/GRPO loss.

`--get-mismatch-metrics`: When you don't want to add TIS/MIS, but still want to monitor the mismatch-related metrics (e.g. rollout-training KL). It will **only return mismatch metrics** but not change the loss in any way.


## Algorithms

We give examples of the algorithms for solving the training-inference mismatch issue.

### [Baseline: No Mismatch Correction] Standard PPO

This is the basic PPO algorithm with potentially training-inference mismatch issue when the output of SGLang and Megatron does not exactly match.

$$
L_{\text{PPO}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{Megatron}}}(y \mid x)} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{Megatron}}}(y \mid x)},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

### Bypassing PPO importance sampling

Like REINFORCE, we directly use the rollout engine's log probs as the old policy in offline PPO's importance sampling, rather than the recomputed log-probs from the training engine.

$$
L_{\text{PPO-bypass}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

Advantages: 

- Efficiency: skip `log_prob` recomputation on training engine. Reduce one expensive forward pass on all the generated trajectories.

### Decoupled, 3-policy PPO Importance Sampling  

[Decoupled PPO](https://arxiv.org/pdf/2110.00641) achieves batch-independent PPO by decoupling two roles: Proximal Policy (anchor policy for PPO clipping, control update size) and Behavior Policy (for off-policy correction in importance sampling). Therefore, there are totally 3 roles engaged in this mode, **target policy** $\pi_\theta$, **proximal policy** $\pi_{\textcolor{blue}{\text{old}}}$, and **behavior policy** $\pi_{\textcolor{red}{\text{SGLang}}}$. $\pi_{\textcolor{blue}{\text{old}}}$ is recomputed with Megatron at the beginning of each training step.

$$
L_{\text{PPO-decoupled}}(\theta)
= - \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\textcolor{red}{\text{SGLang}}}} \left[
    \frac{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)}{\pi_{\textcolor{red}{\text{SGLang}}}(y \mid x)}
  \min \left(
    \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)} A_t,
    \mathrm{clip}\left(
      \frac{\pi_\theta(y \mid x)}{\pi_{\textcolor{blue}{\text{old}}}(y \mid x)},
      1 - \epsilon,
      1 + \epsilon
    \right) A_t
  \right)
\right].
$$

Advantages:

- Achieves batch size invariance and efficient stale data utilization
- Enables accurate off-policy metrics monitoring

## APIs of Algorithms

You may choose from above algorithms by specifying arguments below:

`--use-rollout-logprobs`: True if only use `rollout_log_probs` to compute the loss, bypassing old_log_probs calculated by training engine;

`--use-rollout-correction`: True if apply importance sampling/rejection sampling to loss.

| `use_rollout_logprobs` | `use_rollout_correction` | Algorithm | Policies |Compute old_log_probs | Batch Invariant | Recommended TIS Mode |
|-----------------|-------------|-----------|--------------|---------------|-----------------|----------------------|
| False | False | Standard PPO (Algorithm 0) | 2 ($\pi_\theta$, $\pi_{\textcolor{blue}{\text{old}}}$)|Yes | No | N/A |
| True | False | Bypassing PPO (Algorithm 3) | 2 ($\pi_\theta$, $\pi_{\textcolor{red}{\text{SGLang}}}$) |🚀 Skipped | No | N/A |
| False | True | Decoupled PPO (Algorithm 2) | 3 ($\pi_\theta$, $\pi_{\textcolor{blue}{\text{old}}}$, $\pi_{\textcolor{red}{\text{SGLang}}}$)  |Yes  | Yes | token/seq/geo |

## Configs and Recommended Settings

When choosing to use importance sampling or rejection sampling for mismatch correction (`use-rollout-correction` enabled, Algorithm 2 & 3), you may specify the IS modes and applied levels. 

### Arguments

`use-tis`: Enable importance sampling. The IS weight will be multiplied by the policy gradient loss. 

- `--tis-mode`: Mode for IS. Allowed mode: **truncate**, **clip**.
- `--tis-lower-bound`, `--tis-upper-bound`: Bounds for IS weights.
- `--tis-level`: Allowed levels: **token**, **sequence**, **geometric**. See explanations below.
- `--tis-batch-normalize`: Normalize IS weights to mean=1.0 across batch


`use-rs`: Enable rejection sampling. When choosing to use rejection sampling, the tokens/sequences with an IS weight out of threshold will be directly masked. Those rejected tokens/sequences will not be considered for loss averaging.

- `--rs-lower-bound`, `--rs-upper-bound`: Bounds for RS
- `--rs-level`: Allowed levels: **token**, **sequence**, **geometric**. See explanations below.
- `--rs-veto-threshold`: Sequence-level rejection threshold for catastrophic mismatches

### Importance Sampling

For both IS and RS, we provided 3 levels: **token**, **sequence**, **geometric**.

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
$w_{\text{seq}} = \exp\left( \frac{1}{n} \sum_{i=1}^{n} \left( \log \pi_{\text{train}}(x_i) - \log \pi_{\text{rollout}}(x_i) \right) \right)$

Characteristics: Biased but low variance, balances bias and variance

### Rejection Sampling

**Token Level**: Reject tokens with IS weight out of threshold

**Sequence Level:** Reject sequences with mean IS weight out of threshold

**Geometric Level:** Reject sequences with geometric mean IS weight out of threshold

- Extremely selective: Requires near-perfect policy match
- High rejection rate: Only suitable for very slight distribution shifts

**Veto Mechanism**:

Veto mechanism rejects sequences with catastrophically low token probabilities.
Reject entire sequence if $\exists t \in T$ such that $\rho_t < C_{\text{veto}}$

- Prevents catastrophic updates from tokens with near-zero probability under $\pi_{\text{old}}$
- Independent of IS/RS settings

*Typical values: $10^{-4}$ to $10^{-6}$*

## Mismatch Metrics

When rollout log probabilities are available, SLIME automatically tracks comprehensive metrics to monitor training-inference mismatch and importance sampling weights. These metrics help diagnose policy divergence and guide hyperparameter tuning.

### Mismatch Monitoring Metrics

These metrics quantify the difference between training and rollout policies. They are computed automatically when `rollout_log_probs` are provided, regardless of whether TIS/MIS correction is enabled.

| Metric Name | Description |
|------------|-------------|
| `mismatch_training_log_ppl` | Negative mean log probability under training policy: $-\mathbb{E}[\log \pi_{\text{train}}]$ |
| `mismatch_training_ppl` | Perplexity of training policy: $\exp(-\mathbb{E}[\log \pi_{\text{train}}])$ |
| `mismatch_rollout_log_ppl` | Negative mean log probability under rollout policy: $-\mathbb{E}[\log \pi_{\text{rollout}}]$ |
| `mismatch_rollout_ppl` | Perplexity of rollout policy: $\exp(-\mathbb{E}[\log \pi_{\text{rollout}}])$ |
| `mismatch_kl` | Forward KL divergence estimator: $\mathbb{E}[\log \pi_{\text{rollout}} - \log \pi_{\text{train}}]$ |
| `mismatch_k3_kl` | K3 KL estimator: $\mathbb{E}[\exp(r) - r - 1]$ where $r = \log \pi_{\text{train}} - \log \pi_{\text{rollout}}$ |
| `mismatch_log_ppl_diff` | Log perplexity difference|
| `mismatch_log_ppl_abs_diff` | Absolute log perplexity difference |
| `mismatch_ppl_ratio` | Perplexity ratio |
| `train_rollout_logprob_abs_diff` | Token-level absolute log probability difference |

**Usage**: These metrics help you monitor policy drift. Large values indicate a significant mismatch between the training and rollout engines.

### IS/RS Correction Metrics

These metrics track importance sampling weights and corrections. They are only computed when `--use-tis` is enabled.

When using `--custom-tis-function-path` pointing to MIS implementation (e.g., `mis.py`), additional fine-grained metrics become available:

| Metric Name | Description | Required Args | Optional Control Args |
|------------|-------------|---------------|----------------------|
| `ois` | On-policy importance sampling ratio: $\exp(\log \pi_{\text{train}} - \log \pi_{\text{old}})$ | `--use-tis` | Only for Algorithm 2 (Decoupled PPO) |
| `mis_mean_is_weight_before_clip` | Raw IS weights before any correction: $\exp(\text{log-ratio})$ | `--use-tis` | `--mis-level` (token/sequence/geometric) |
| `mis_ratio_mean_after_mis` | IS weights after correction (bounded or masked) | `--use-tis` | `--mis-mode`, bounds |
| `mis_truncate_fraction` | Fraction of weights truncated (mode-specific) | `--use-tis`, `--mis-mode=truncate` | `--mis-upper-bound` |
| `mis_clip_fraction_low` | Fraction of weights clipped below lower bound | `--use-tis`, `--mis-mode=clip` | `--mis-lower-bound`, `--mis-upper-bound` |
| `mis_clip_fraction_high` | Fraction of weights clipped above upper bound | `--use-tis`, `--mis-mode=clip` | `--mis-lower-bound`, `--mis-upper-bound` |
| `mis_mask_fraction_low` | Fraction of tokens rejected (below lower bound) | `--use-tis`, `--mis-mode=mask` | `--mis-lower-bound`, `--mis-upper-bound` |
| `mis_mask_fraction_high` | Fraction of tokens rejected (above upper bound) | `--use-tis`, `--mis-mode=mask` | `--mis-lower-bound`, `--mis-upper-bound` |
| `mis_catastrophic_token_fraction` | Fraction of catastrophic tokens (veto-specific) | `--use-tis`, `--mis-veto-threshold` set | Sequence-level rejection |
| `mis_catastrophic_seq_fraction` | Fraction of sequences with catastrophic tokens | `--use-tis`, `--mis-veto-threshold` set | Sequence-level rejection |
| `mis_batch_norm_factor` | Batch normalization factor applied to weights | `--use-tis`, `--mis-batch-normalize` | Normalizes mean to 1.0 |

## Reference

We thank the materials below for their excellent findings and theories.

1. [Mathematical Formulations of Rollout Correction Methods in verl (Yingru Li)](https://github.com/szrlee/verl/blob/yingru/rollout_correction/docs/advance/rollout_corr_math.md).
2. [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)
3. [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
