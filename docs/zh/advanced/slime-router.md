# Slime Router

Slime 提供一个可选的 SlimeRouter，用于 rollout / data generation 阶段。它是一个轻量级的 HTTP router/proxy，位于一个或多个 SGLang worker server 前，补齐一些 training-oriented 能力——这些并不是 serving-focused router 的主要目标。

---

## 1. 什么是 SlimeRouter？

SlimeRouter 是一个小型 FastAPI 服务，主要能力包括：

- 注册 worker（SGLang HTTP server）到本地池
- 路由请求到选定的 worker（简单的 least-inflight load balancing）
- 代理任意路径到选定的 worker（例如 `/generate`）
- 定期 health checks，并隔离不健康的 worker
- 支持 middleware plugins（通过 `--slime-router-middleware-paths`）实现 rollout 特定处理（例如 caching、request/response transform）

在 slime 架构中，router 是 rollout 系统（"SGLang + router"）的一部分：负责生成样本并将其推入数据缓冲区。

### 启动方式

在分布式训练中，当未提供 `--sglang-router-ip` 时，slime 会自动启动一个 router：

- 如果设置了 `--use-slime-router`，slime 启动 SlimeRouter
- 否则，slime 启动 SGLang Model Gateway

---

## 2. 为什么需要 SlimeRouter

与 production inference 不同，RL rollout 往往需要捕获用于训练的额外 metadata：token-level logprobs、loss masks，以及（对 MoE 模型）expert routing decisions。SlimeRouter 通过 middleware system 和 passthrough proxy 设计提供这些能力。

### 2.1 Radix-tree cache（透明的 token 管理）

> 当你的 rollout 流程是 text-in/text-out、并且很难可靠地保存 token IDs 时，适合用 radix-tree cache；如果你已经能自己控制 token-in/token-out（例如 search r1、multiturn VLM 这些 example），通常不需要 radix-tree cache。

text-in text-out 接口可能导致 token retokenization mismatches：训练阶段重新 tokenize 文本，得到的 token 序列可能与 rollout 阶段不同，从而破坏 PPO/GRPO 这类方法所需的 per-token alignment。

radix-tree cache 可以透明地解决这个问题：它拦截 text-based request，对其进行 tokenize，并将 trajectory（text、token IDs、logprobs、loss masks）按文本前缀作为 key 存储。rollout 结束后，调用 `/retrieve_from_text` 就能取回与 rollout 完全一致的 token 序列及其对齐的 metadata，无需修改现有 rollout 代码。

实现上，radix-tree cache 会做几件事：

- 接收 text 以及 tokens/metadata，并写入 radix tree
- 通过 longest-prefix matching 复用已缓存的 token 序列（使后续流程可以走 token-in/token-out）
- rollout 过程中持续插入新的 text continuation（同一 prompt 下可有多条 trajectory，例如 GRPO）
- 定期清理 stale nodes，控制内存占用

当你有 text-based rollout 代码、想获得 token-level 精度但又不想重写，或者在 GRPO 场景中多个 trajectory 共享相同 prompt 前缀时，建议使用 radix-tree cache。

### 2.2 Rollout routing replay (R3) for MoE

对 MoE 模型，slime 支持 rollout routing replay (R3)：在 rollout 期间记录 expert routing decisions，并在训练期间 replay，以提升训练稳定性。

#### SGLang 端

SGLang 侧通过以下机制提供 expert routing capture：

- `--enable-return-routed-experts`：启用路由捕获的服务器参数
- `RoutedExpertsCapturer`：在前向传播期间捕获每个 MoE 层的 `topk_ids`（选定的专家 ID）
- `return_routed_experts`：检索路由数据的请求参数
- 在响应 `meta_info` 中返回 `routed_experts`：一个形状为 `[seq_len - 1, num_layers, top_k]` 的 expert ID tensor

#### Slime 端

Slime 侧消费路由数据，并在训练中完成 replay：

- `--use-slime-router --use-rollout-routing-replay`：启用 R3 需要同时设置这两个标志
- Rollout 发送 `return_routed_experts=True` 并将结果存储在 `sample.rollout_routed_experts` 中
- 训练调用 `fill_routing_replay()` 将路由数据加载到 `RoutingReplay` 对象中
- forward pass 期间，直接 replay 记录的 routing decisions，而不是重新计算

#### 为什么需要 SlimeRouter

我们需要 SlimeRouter，是因为当请求设置 `return_routed_experts=true` 时，SGLang worker 会在响应里返回路由信息（`meta_info.routed_experts`），而 SlimeRouter 会端到端保留这个字段。SGLang Model Gateway 会用固定 schema 重建响应，可能会丢掉这类额外 metadata（细节见第 3 节）。

---

## 3. 与 SGLang Model Gateway 的区别

SlimeRouter 与 SGLang Model Gateway 都能将请求路由到 worker，但它们面向的目标不同、优化方向也不同。

### 主要区别

SlimeRouter 是一个轻量级的 Python/FastAPI proxy，作为 SGLang worker 的 passthrough proxy。这种 passthrough 设计使得 RL 特定功能成为可能，例如 radix-tree trajectory caching 和 R3（需要保留原始 response metadata，如 `routed_experts`）。

SGLang Model Gateway 是一个高性能 Rust router，面向大规模 inference 优化：async non-blocking routing、高级 fault tolerance（retries、circuit breakers）、多种 load balancing policy（包括 cache-aware routing），以及 PD disaggregation 支持。但它会用固定 schema 重建响应，因此不保留 slime 的 R3 流程所需 metadata。

更多关于 SGLang Model Gateway 的信息，请参阅[官方文档](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)。

### 如何选择

- 当你需要 R3 或 radix-tree cache 时，使用 SlimeRouter
- 其他情况使用 SGLang Model Gateway（推荐默认选项）

