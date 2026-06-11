# External Rollout Engines 配置路线图

External rollout engine 指的是：SGLang engine 不由 slime 训练任务启动，而是由外部系统预先部署和管理；slime 只在训练时连接这些 engine，注册 router，并在需要时同步训练后的 actor 权重。

这篇文档是一个导航页。它帮助你判断什么时候该用 `--rollout-external-engine-addrs`，什么时候该用 `--rollout-http-endpoint-url`，什么时候该继续使用 `--sglang-config`，以及 external 场景下该选择 full checkpoint update from disk 还是 delta update。

## 从哪里开始

| 目标 | 推荐入口 |
| :--- | :--- |
| engine 已经由外部系统启动，只想让 slime 连接并做 rollout | `--rollout-external-engine-addrs` |
| rollout serving 是单一 HTTP URL 背后的弹性集群，没有稳定的 per-engine 句柄 | `--rollout-http-endpoint-url` |
| serving 侧主动拉取发布的权重版本，而不是接收直接的 update RPC | `--update-weight-delta-publish-only`，见 [Publish-Only 磁盘 Delta](delta-weight-sync.md#publish-only-磁盘-delta) |
| engine 仍由 slime 启动，但需要 PD 分离、多模型、异构 server group 或 per-group overrides | [SGLang Config](sglang-config.md) |
| 训练器和 external engine 可以建立 NCCL group | 默认的 `--update-weight-mode full --update-weight-transport nccl` |
| 训练器和 external engine 不能建立 NCCL group，但能共享同一路径的文件系统 | `--update-weight-mode full --update-weight-transport disk` |
| 大模型跨集群或跨数据中心同步，full checkpoint 太重 | `--update-weight-mode delta --update-weight-transport disk` |
| rollout serving 想使用独立 SGLang 环境，甚至不同型号或不同厂家的 GPU | external engine + disk transport |
| 想验证 delta wire/apply 逻辑，但仍在同一数据中心内 | `--update-weight-mode delta --update-weight-transport nccl` |
| 需要 reference、reward、tool-side model 等冻结模型 | 优先用 [SGLang Config](sglang-config.md#3-多模型服务) 的 `update_weights: false` |

## External Engine 做了什么

使用 external engine 时，先独立启动 SGLang server：

```bash
python -m sglang.launch_server --model-path /path/to/model --port 10090 ...
python -m sglang.launch_server --model-path /path/to/model --port 10091 ...
```

训练任务里传入这些地址：

```bash
python train.py \
  --rollout-external-engine-addrs host1:10090 host2:10091 \
  ...
```

slime 会请求每个 engine 的 `/server_info` 或 `/get_server_info`，推断 engine 的 GPU 数、TP/PP 信息和 worker 类型（`regular`、`prefill`、`decode`）。如果没有提供 `--sglang-router-ip/--sglang-router-port`，slime 会启动自己的 router，并把这些 external engine 注册进去。

这条路径适合 serving 生命周期由训练任务外部管理的部署：例如独立的推理集群、跨 Ray 集群部署、手工预热的 SGLang engine，或由其他编排系统管理的 rollout service。

## Opaque HTTP Rollout Endpoint

`--rollout-external-engine-addrs` 仍然假设 SGLang engine 有稳定地址：slime 会逐个查询 `/server_info`，把每个 engine 注册到 router，并向已知 engine 句柄推送权重更新。有些部署无法提供这种契约——例如单一 URL 背后的 serverless 或自动扩缩容推理集群，worker 随时增减，也不暴露任何 worker 管理 API。这种情况下让 slime 直接指向 endpoint：

```bash
python train.py \
  --rollout-http-endpoint-url https://rollout.example.com \
  ...
```

在这个模式下，slime 不启动任何 engine 和 router，对 endpoint 的假设只有生成路由：rollout 请求 POST 到 `{url}/generate`，自定义 rollout function 里的 `get_model_url(args, ...)` 也解析到该 endpoint。placement group 中不会分配 rollout GPU，`/server_info` 永远不会被查询，slime 的 fault tolerance 也不管理这个集群——故障恢复由 endpoint 运营方负责。`--rollout-http-endpoint-url` 与 `--rollout-external-engine-addrs` 互斥。

两个配套参数让默认 SGLang rollout 适配没有 router API 的 endpoint：

- `--rollout-http-endpoint-abort-strategy {cancel-only,router-workers}`：控制两次 rollout 之间 `abort` 的行为。`cancel-only`（设置 endpoint URL 时的默认值）只取消 slime 本地待完成的生成任务，不调用 router 的 worker 列表或 per-worker abort API。`router-workers` 保留原有基于 router 的 abort，在其他情况下仍是默认值。注意 `cancel-only` 不收集 partial sample，因此与 `--partial-rollout` 不兼容。
- `--custom-rollout-request-hook-path`：可选 hook，在默认 SGLang `/generate` 请求发出前调用。签名为 `def hook(args, sample, request) -> None | dict`。`request` dict 包含 `url`、`payload`、`headers`、`max_retries`、`retry_sleep`、`rollout_id` 和 `evaluation`；可以原地修改，也可以返回一个 dict 覆盖字段。

请求级权重约束应通过这个 hook 添加。例如 hook 可以加入 `"weight_version": {"exact_version": <ready_version>}` 或 `"weight_version": {"min_required_version": <minimum_version>}`，并调整 `max_retries`/`retry_sleep`。这些字段用于 opaque router 把请求路由到落后副本时尽早失败并重试，避免浪费 rollout compute；它们不定义 SLIME 的 off-policy 或 staleness 语义，真正的有效版本仍由训练调度和 loss/correction 路径决定。

至于权重同步，弹性集群通常也无法接收 per-engine 的 `update_weights_from_disk` RPC。可以把 endpoint 与 publish-only delta 同步组合使用：训练端通过自定义 hook 发布每个完整的权重版本，serving 侧按自己的节奏消费——见 [Publish-Only 磁盘 Delta](delta-weight-sync.md#publish-only-磁盘-delta)。如果请求级最低版本重试已经足够，保留 publish-only 的默认流水线模式即可；如果 publish hook 会轮询 rollout 集群状态、并且你希望下一次 rollout dispatch 等待该就绪阈值，可以设置 `--update-weight-delta-publish-wait=sync`。

## 与 `--sglang-config` 的关系

`--rollout-external-engine-addrs` 和 `--sglang-config` 互斥，因为它们负责不同的边界：

- `--sglang-config`：slime 负责 engine 生命周期。你用 YAML 描述 topology，slime 启动 server group、router，并管理多模型和选择性权重更新。
- `--rollout-external-engine-addrs`：外部系统负责 engine 生命周期。slime 只发现已启动的 engine，接入 router，并把它们当作默认 rollout model。

如果你的主要需求是多模型 serving、reference/reward 冻结模型、PD 分离或异构组配置，优先使用 `--sglang-config`。如果 engine 已经在训练任务外部部署好，再使用 external engine。

## 环境与硬件解耦

External engine 的一个重要含义是：SGLang serving 侧不需要使用 slime 训练任务的 Python 环境、Megatron 环境或 Ray runtime。它可以运行在单独的 SGLang 容器、独立集群或其他编排系统里；slime 只依赖 HTTP endpoint、`/server_info` 信息，以及所选权重同步方式需要的通信路径。

当使用 disk transport 时，权重通过共享文件系统上的 HF checkpoint 或 safetensors delta 传递，再由 SGLang 通过 `update_weights_from_disk` 热加载。这条路径不要求训练 GPU 和 rollout GPU 是同一型号，甚至不要求是同一厂家；只要 SGLang 本身支持该硬件后端、模型格式和精度配置即可。例如训练可以在一组 GPU 上运行，rollout serving 可以放在另一组不同型号或不同厂家的 GPU 上。

如果使用 NCCL transport，则仍然需要满足 NCCL 通信和硬件兼容性要求。跨厂家、跨不兼容网络或跨数据中心部署通常应选择 `--update-weight-transport disk`。

## Update From Disk

full checkpoint update from disk 是 external 场景最简单的兜底路径：

```bash
--update-weight-mode full
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/full-updates
```

每次权重同步时，训练端会在 `--update-weight-disk-dir` 下写一个完整 HF checkpoint 目录，例如 `weight_v000123/`，然后通过 HTTP 调用每个 SGLang engine 的 `update_weights_from_disk`，让 engine 在不重启进程的情况下重新加载 checkpoint。

这个模式的优点是控制面简单：不要求训练器和 engine 建 NCCL group，只要求二者能看到同一个共享文件系统路径。缺点也直接：每次同步都写完整 actor 权重，对大模型和高频同步来说非常重。

调试时可以加：

```bash
--update-weight-disk-keep-files
```

这样 slime 不会在 engine 确认加载后清理完整 checkpoint 目录，方便检查写出的 HF checkpoint。

## Update With Delta

delta update 面向大模型、跨集群或跨数据中心训推解耦。它不写完整 checkpoint，而是在训练端保留上一次同步后的 pinned CPU snapshot，逐字节检测变化，只发送变化位置和值。

跨集群 / 共享文件系统推荐：

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd
--update-weight-disk-dir /shared/fs/delta-updates
```

在 disk transport 下，每次同步会写一组稀疏 safetensors 到 `weight_v{N:06d}/`，然后调用 `update_weights_from_disk(load_format="delta")`。SGLang 侧只把变化位置覆写到当前权重上，不变位置保持原值。

在同一数据中心内做实现验证或带宽不紧张时，也可以用 NCCL transport：

```bash
--update-weight-mode delta
--update-weight-transport nccl
--update-weight-encoding indices
```

编码如何选择、delta wire layout、接收端 selective overwrite 以及调优参数见 [Delta 权重同步](delta-weight-sync.md)。

## 部署检查清单

- external engine 的 HTTP 地址必须能从训练任务访问。
- external engine 可以使用独立 SGLang 环境；不需要安装 slime 或 Megatron 训练环境。
- disk transport 支持训练和 rollout 使用不同型号或不同厂家的 GPU，前提是 SGLang 支持对应硬件和模型格式。
- disk transport 要求训练端和 SGLang engine 看到同一个 `--update-weight-disk-dir` 路径；路径只在训练端可见是不够的。
- external engine 当前不支持 slime 的 fault tolerance 恢复流程；engine 生命周期由外部系统负责。`--rollout-http-endpoint-url` 背后的集群同理。
- `--sglang-config` 与 `--rollout-external-engine-addrs` 互斥；`--rollout-external-engine-addrs` 与 `--rollout-http-endpoint-url` 也互斥。
- opaque HTTP endpoint 只需要提供生成路由；slime 不会调用任何 worker 管理 API。如果集群无法接收直接的权重更新 RPC，请使用 publish-only delta 同步。
- delta mode 不支持 `--colocate`，因为 colocated 权重同步通过 CUDA IPC 传句柄，delta 编码不会节省实际传输量。

## 参考工作

[Cursor Research Team 的 Composer 2 技术报告](https://arxiv.org/html/2603.24477v2) 公开描述了一个相近的生产形态：训练和 rollout generation 高度异步，Cursor 与 Fireworks AI 合作运行 RL inference；每个训练 step 都把更新后的权重写到共享 S3，并用 delta compression 降低传输量，不同区域的 inference 集群再从共享 delta chain 下载并重建权重。

slime 的 external engine、update from disk 和 delta disk transport 面向同一类基础设施问题：训练与推理解耦后，权重同步必须能跨进程、跨集群甚至跨数据中心工作，同时不能让训练主循环被完整模型传输拖住。
