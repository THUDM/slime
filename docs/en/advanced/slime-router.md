# slime router

slime includes an optional slime router used during rollout / data generation. It is a lightweight HTTP router/proxy that sits in front of one or more SGLang worker servers and adds training-oriented capabilities that are not the main goal of serving-focused routers.

---

## 1. What is slime router?

slime router is a small FastAPI service that:

- Registers workers (SGLang HTTP servers) into a local pool, with support for **prefill / decode / regular** worker types
- Routes requests to a selected worker via least-inflight load balancing or **PD dual-dispatch routing**
- Streams proxied responses (e.g. `/generate`) without buffering the full body, improving throughput under high concurrency
- Runs periodic health checks and quarantines unhealthy workers

In slime's architecture, the router is part of the rollout system ("SGLang + router") that generates samples and pushes them into the data buffer.

### How it is launched

In distributed training, slime will start a router automatically when `--sglang-router-ip` is not provided:

- If `--use-slime-router` is set, slime starts slime router
- Otherwise, slime starts SGLang Model Gateway

---

## 2. Why we need slime router

Unlike production inference, RL rollout needs to capture additional metadata for training: token-level logprobs, loss masks, and (for MoE models) expert routing decisions. slime router provides these capabilities through its passthrough proxy design.

### 2.1 Rollout routing replay (R3) for MoE

For MoE models, slime supports rollout routing replay (R3): record expert routing decisions during rollout and replay them during training to improve stability.

#### SGLang side

SGLang provides expert routing capture via:

- `--enable-return-routed-experts`: server argument to enable routing capture
- `RoutedExpertsCapturer`: captures `topk_ids` (selected expert IDs) at each MoE layer during forward pass
- `return_routed_experts`: request parameter to retrieve routing data
- Returns `routed_experts` in response `meta_info` - a `[seq_len - 1, num_layers, top_k]` tensor of expert IDs

#### slime side

slime consumes the routing data and replays it during training:

- `--use-slime-router --use-rollout-routing-replay`: both flags required to enable R3
- Rollout sends `return_routed_experts=True` and stores results in `sample.rollout_routed_experts`
- Training calls `fill_routing_replay()` to load routing data into `RoutingReplay` objects
- During forward pass, recorded routing decisions are replayed instead of recomputed

#### Why slime router is needed

We need slime router because the SGLang worker returns routed experts in the response (`meta_info.routed_experts`) when the request sets `return_routed_experts=true`, and slime router preserves this field end-to-end. SGLang Model Gateway may drop this extra metadata when it reconstructs responses with a fixed schema (see section 3).

### 2.2 PD disaggregation

slime router supports **Prefill-Decode (PD) disaggregation**. When prefill and decode workers are registered, the router automatically enables PD mode:

- Workers register themselves with a `worker_type` (`prefill`, `decode`, or `regular`) via the `POST /workers` endpoint.
- For each request, the router picks a (prefill, decode) worker pair via least-inflight load balancing, injects bootstrap information (`bootstrap_host`, `bootstrap_port`, `bootstrap_room`) into the request body, and sends the same modified request to **both** workers concurrently.
- The decode worker's response is returned to the caller. The actual KV-cache transfer between workers is coordinated internally via the bootstrap connection.
- If no prefill/decode workers exist, the router falls back to standard single-worker routing.

This mirrors the dual-dispatch approach used by SGLang Model Gateway's PD router.

---

## 3. Differences vs SGLang Model Gateway

slime router and SGLang Model Gateway can both route requests to workers, but they are optimized for different goals.

### Key differences

slime router is a lightweight Python/FastAPI proxy that acts as a passthrough to SGLang workers. This passthrough design enables RL-specific features like R3 (which require preserving raw response metadata like `routed_experts`).

SGLang Model Gateway is a high-performance Rust-based router optimized for large-scale inference: async non-blocking routing, advanced fault tolerance (retries, circuit breakers), multiple load balancing policies (including cache-aware routing), and PD disaggregation support. However, it reconstructs responses with a fixed schema, so it does not preserve the metadata needed for slime's R3 flow.

For more details on SGLang Model Gateway, see the [official documentation](https://docs.sglang.io/advanced_features/sgl_model_gateway.html).

### When to use which

- Use slime router when you need R3 or PD disaggregation with metadata preservation
- Use SGLang Model Gateway for everything else (recommended default)
