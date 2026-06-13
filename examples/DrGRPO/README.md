# Dr.GRPO

Dr.GRPO's constant-divisor loss normalization (arXiv:2503.20783, Eq. 2; also used by
DeepSWE) is built into slime — no custom code is needed:

```bash
--pg-loss-divisor 40960   # a constant, e.g. the max context length
```

When set, pg_loss is aggregated as `sum(token_loss * loss_mask) / divisor` instead of the
default sum of per-sample active-token means, removing the length bias of per-sample
normalization. Other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) keep the default
reducer.

For normalizations that need more than a constant divisor, see the *Custom pg_loss
Reducer* section in [docs/en/get_started/customization.md](../../docs/en/get_started/customization.md).
