# PD 分离

PD Disaggregation 将 SGLang rollout 中的 Prefill worker 和 Decode worker 拆开部署。它特别适合 multi-turn、long-context 和 agentic RL：这些 workload 中，prompt processing 和 token generation 的计算/显存特征往往完全不同。

## 什么时候使用

建议在以下场景使用 PD 分离：

- rollout context 很长，或会随着多轮交互持续增长；
- decode 阶段占据主要 rollout 时间；
- multi-turn session 需要更好的 prefix-cache locality；
- prefill 和 decode 需要不同 TP、显存或 runtime 设置；
- 希望 rollout topology 更接近生产 serving，而不是单一 uniform inference group。

对于短单轮任务，默认 regular SGLang engine layout 通常更简单。

## 配置路径

slime 支持两种 PD 配置方式。

### 简单路径：`--prefill-num-servers`

如果只有单个 actor model，并且只需要简单 PD layout，可以设置：

```bash
--prefill-num-servers 1
```

这是轻量路径，适合只想拆开 prefill/decode、但不需要分别调每个 group 的场景。

### 高级路径：`--sglang-config`

生产级 rollout topology 推荐使用 [SGLang Config](sglang-config.md)。它可以独立配置 prefill 和 decode group，也能表达 EPD-style layout、heterogeneous server group、multi-model serving 和 per-group SGLang override。

示例：

```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: prefill
        num_gpus: 4
        num_gpus_per_engine: 2
        overrides:
          chunked_prefill_size: 8192
      - worker_type: decode
        num_gpus: 12
        num_gpus_per_engine: 4
        overrides:
          mem_fraction_static: 0.88
```

启动：

```bash
python train.py \
  --sglang-config sglang_pd.yaml \
  --rollout-num-gpus 16 \
  ...
```

## 为什么这对 RL 重要

RL rollout 往往不是一批短 completion。Agentic 和 verifier-based workload 常见特征包括：

- 来自 tool/environment history 的长 prompt；
- 每个 sample 多轮交互；
- decode latency long tail；
- session-local prefix cache 机会；
- actor、reference、reward、judge model 资源需求不同。

PD 让 slime 在不改变 training loop 的情况下，使用更贴合真实 serving workload 的 rollout topology。

## 运维注意事项

- 新的复杂部署优先使用 `--sglang-config`，而不是 `--prefill-num-servers`。
- multi-turn agent 建议开启 router session affinity，使同一 sample 的多轮请求可以复用 prefix cache。见 [Session-Affinity Routing](sglang-config.md#session-affinity-routing-for-multi-turn-agents)。
- `--rollout-num-gpus` 应等于 SGLang config 中描述的 GPU 总数。
- 不要在同一个 model entry 中混用 `regular` worker 和 `prefill`/`decode` worker。
- 当 prompt processing 和 token generation 的瓶颈不同时，分别调 prefill 和 decode 的 TP。

## 相关文档

- [SGLang Config](sglang-config.md)
- [Agentic RL Training Roadmap](../get_started/agent.md)
- [Trace Viewer](../developer_guide/trace.md)
