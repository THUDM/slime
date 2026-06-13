# CISPO Custom Loss

This example implements CISPO (Clipped IS-weight Policy Optimization) with
slime's existing custom loss interface. It keeps CISPO out of
`--advantage-estimator`: use a normal estimator such as `grpo`, then replace the
policy objective with this custom loss.

```bash
--advantage-estimator grpo
--loss-type custom_loss
--custom-loss-function-path examples.cispo.cispo_loss.cispo_loss_function
--custom-config-path examples/cispo/cispo.yaml
--use-rollout-logprobs
--calculate-per-token-loss
```

The CISPO objective maximizes a detached clipped ratio times the target policy
log-probability:

```python
ratio = torch.exp(target_logprobs - sampling_logprobs)
clipped_ratio = torch.clamp(ratio, clip_low_threshold, clip_high_threshold)
loss = -(clipped_ratio.detach() * advantages * target_logprobs)
```

The bundled config follows the one-sided CISPO setting:

```yaml
loss_config:
  cispo:
    clip_low_threshold: 0.0
    clip_high_threshold: 4.0
```

These are Tinker-style absolute ratio thresholds, not epsilon values.

`--use-rollout-logprobs` is recommended so the CISPO denominator uses the actual
sampler log-probs from `rollout_log_probs`, matching Tinker's
`sampling_logprobs` input.

`--calculate-per-token-loss` is recommended when matching Tinker's token-level
loss aggregation.

This example intentionally does not combine CISPO with `--use-tis`. TIS applies
an additional importance-weighting or masking pass, while CISPO's detached
clipped ratio is already part of the objective.

# References
1. MiniMax-M1: https://arxiv.org/abs/2506.13585
2. ScaleRL: https://arxiv.org/abs/2510.13786
3. Tinker CISPO: https://tinker-docs.thinkingmachines.ai/tinker/losses/cispo/
