# slime router

slime 提供一个可选的 slime router，用于 rollout / data generation 阶段。它是一个轻量级的 HTTP router/proxy，位于一个或多个 SGLang worker server 前，补齐一些 training-oriented 能力——这些并不是 serving-focused router 的主要目标。

---

## 1. 什么是 slime router？

slime router 是一个小型 FastAPI 服务，主要能力包括：

- 注册 worker（SGLang HTTP server）到本地池，支持 **prefill / decode / regular** worker 类型
- 路由请求到选定的 worker——支持 least-inflight 负载均衡和 **PD 双发路由**
- 流式代理请求到选定的 worker（例如 `/generate`），不缓冲完整 response body，提高高并发下的吞吐
- 定期 health checks，并隔离不健康的 worker

在 slime 架构中，router 是 rollout 系统（"SGLang + router"）的一部分：负责生成样本并将其推入数据缓冲区。

### 启动方式

在分布式训练中，当未提供 `--sglang-router-ip` 时，slime 会自动启动一个 router：

- 如果设置了 `--use-slime-router`，slime 启动 slime router
- 否则，slime 启动 SGLang Model Gateway

---

## 2. 为什么需要 slime router

与 production inference 不同，RL rollout 往往需要捕获用于训练的额外 metadata：token-level logprobs 和 loss masks。slime router 通过 passthrough proxy 设计提供这些能力。

### 2.1 PD 分离（Prefill-Decode 分离）

slime router 支持 **Prefill-Decode (PD) 分离**。当 prefill 和 decode worker 注册后，router 会自动启用 PD 模式：

- Worker 注册时携带 `worker_type`（`prefill`、`decode` 或 `regular`），通过 `POST /workers` 端点。
- 对每个请求，router 通过 least-inflight 负载均衡选择一对 (prefill, decode) worker，向请求体注入 bootstrap 信息（`bootstrap_host`、`bootstrap_port`、`bootstrap_room`），然后将同一个修改后的请求**并发发送**给两个 worker。
- Decode worker 的响应返回给调用方。实际的 KV-cache 传输由 worker 通过 bootstrap 连接内部协调完成。
- 如果没有 prefill/decode worker 存在，router 回退到标准的单 worker 路由。

这与 SGLang Model Gateway 的 PD router 所使用的双发方式一致。

---

## 3. 与 SGLang Model Gateway 的区别

slime router 与 SGLang Model Gateway 都能将请求路由到 worker，但它们面向的目标不同、优化方向也不同。

### 主要区别

slime router 是一个轻量级的 Python/FastAPI proxy，作为 SGLang worker 的 passthrough proxy。

SGLang Model Gateway 是一个高性能 Rust router，面向大规模 inference 优化：async non-blocking routing、高级 fault tolerance（retries、circuit breakers）、多种 load balancing policy（包括 cache-aware routing），以及 PD disaggregation 支持。

两个 router 都支持 R3（rollout routing replay）用于 MoE 模型。

更多关于 SGLang Model Gateway 的信息，请参阅[官方文档](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)。

### 如何选择

- 当你需要保留 metadata 的 PD 分离时，使用 slime router
- 其他情况使用 SGLang Model Gateway（推荐默认选项）
