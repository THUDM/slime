# 容灾

为了保证长期稳定的 RL 训练，slime 会默认开始一定程度的容灾机制。这里主要介绍一下 slime 中容灾的一些设计思路。

可以通过 `--use-fault-tolerance` 开启容灾机制。

## rollout 容灾

slime 会在 rollout 过程中，定期向所有 SGLang server 发送心跳请求（`/health_generate`），如果心跳超时，则会停止这个 SGLang server 并**立即重启**。

### 即时重启与远程权重加载

当容灾功能启用且某个 rollout engine 失败时，slime 会：
1. **立即杀死**失败的 engine
2. **立即查询** router 获取活跃的 workers
3. **立即重启**新的 engine，通过 SGLang 的 transfer engine 直接从活跃的 engine 加载权重

这种即时容灾机制确保了训练过程的连续性 - 失败的 engine 会被立即替换，而不需要等待当前 rollout 轮次完成。

新 engine 使用以下参数从活跃 engine 加载权重：
- `--load-format remote_instance`
- `--remote-instance-weight-loader-backend transfer_engine`
- `--remote-instance-weight-loader-seed-instance-ip <active_worker_ip>`
- `--remote-instance-weight-loader-seed-instance-service-port <active_worker_port>`

这利用了 SGLang 内置的从现有 engine 引导新 engine 的能力，避免了从存储加载模型权重的开销，实现快速恢复。

### 配置参数

- `--rollout-health-check-first-wait`：由于一些大的 MoE 模型在第一次运行时需要处理一些编译，我们会在第一次 rollout 前等待 `rollout_health_check_first_wait` 秒再开始发送心跳，默认为 300s；
- `--rollout-health-check-interval`：心跳检查间隔，默认为 10s；
- `--rollout-health-check-timeout`：心跳超时限额，默认为 5s。
