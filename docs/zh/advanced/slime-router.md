# slime router

slime 中的 slime router 已被移除。

现在，当 slime 需要为 rollout 启动本地 router 时，会统一使用由 [zhuzilin/sgl-router](https://github.com/zhuzilin/sgl-router) 构建的 `sglang_router`。

## 迁移说明

- `--use-slime-router` 已废弃并被忽略。
- `--slime-router-timeout` 已废弃并被忽略。
- `--slime-router-max-connections` 已废弃并被忽略。
- `--slime-router-health-check-failure-threshold` 已废弃并被忽略。

## 替代方案

- 直接使用 slime 默认的 `sglang_router` 路径。
- Router 的能力和部署方式请参考 [SGLang Model Gateway 官方文档](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)。
