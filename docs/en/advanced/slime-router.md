# slime router

slime router has been removed from slime.

When slime needs to launch a local router for rollout, it now always uses `sglang_router` built from [zhuzilin/sgl-router](https://github.com/zhuzilin/sgl-router).

## Migration

- `--use-slime-router` is deprecated and ignored.
- `--slime-router-timeout` is deprecated and ignored.
- `--slime-router-max-connections` is deprecated and ignored.
- `--slime-router-health-check-failure-threshold` is deprecated and ignored.

## What to use instead

- Use slime's default `sglang_router` path.
- For router capabilities and deployment details, see the [SGLang Model Gateway documentation](https://docs.sglang.io/advanced_features/sgl_model_gateway.html).
