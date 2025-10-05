# 容灾

为了保证长期稳定的 RL 训练，slime 会默认开始一定程度的容灾机制。这里主要介绍一下 slime 中容灾的一些设计思路。

## rollout 容灾

slime 会在 rollout 过程中，定期向所有 SGLang server 发送心跳请求（`/health_generate`），如果心跳超时，则会停止这个 SGLang server。并在这轮 rollout 完成之后进行重启和正确的参数更新。

- `--rollout-health-check-first-wait`：由于一些大的 MoE 模型在第一次运行时需要处理一些编译，我们会在第一次 rollout 前等待 `rollout_health_check_first_wait` 秒再开始发送心跳，默认为 300s；
- `--rollout-health-check-interval`：心跳检查间隔，默认为 10s；
- `--rollout-health-check-timeout`：心跳超时限额，默认为 5s。
