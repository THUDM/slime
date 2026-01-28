# Slime Router

Slime includes an optional SlimeRouter used during rollout / data generation. It is a lightweight HTTP router/proxy that sits in front of one or more SGLang worker servers and adds training-oriented capabilities that are not the main goal of serving-focused routers.

---

## 1. What is SlimeRouter?

SlimeRouter is a small FastAPI service that:

- Registers workers (SGLang HTTP servers) into a local pool
- Routes requests to a selected worker (simple least-inflight load balancing)
- Proxies arbitrary paths to the selected worker (e.g. `/generate`)
- Runs periodic health checks and quarantines unhealthy workers
- Supports middleware plugins (via `--slime-router-middleware-paths`) to implement rollout-specific processing (e.g. caching, request/response transforms)

In slime's architecture, the router is part of the rollout system ("SGLang + router") that generates samples and pushes them into the data buffer.

### How it is launched

In distributed training, slime will start a router automatically when `--sglang-router-ip` is not provided:

- If `--use-slime-router` is set, slime starts SlimeRouter
- Otherwise, slime starts SGLang Model Gateway

---

## 2. Why we need SlimeRouter

Unlike production inference, RL rollout needs to capture additional metadata for training: token-level logprobs, loss masks, and (for MoE models) expert routing decisions. SlimeRouter provides these capabilities through its middleware system and passthrough proxy design.

### 2.1 Radix-tree cache (transparent token management)

> Use this when your rollout pipeline is text-in/text-out and you cannot reliably persist token IDs; if you already control token-in/token-out (e.g. search r1, multiturn VLM examples), you likely don't need the radix-tree cache.

Text-in text-out interfaces can cause token retokenization mismatches - re-tokenizing text at training time may produce different token sequences than rollout, breaking per-token alignment needed for PPO/GRPO losses.

The radix-tree cache solves this transparently: it intercepts text-based requests, tokenizes them, and stores trajectories (text, token IDs, logprobs, loss masks) keyed by the text prefix. After rollout finishes, calling `/retrieve_from_text` returns the exact token sequence with aligned metadata, without requiring any changes to your rollout code.

Implementation-wise, the radix-tree cache:

- Accepts text plus tokens/metadata and stores them in a radix tree
- Uses longest-prefix matching to reuse cached token sequences (enabling token-in/token-out downstream)
- Allows insertion of new text continuations as rollout proceeds (multiple trajectories per prompt, e.g. GRPO)
- Periodically cleans up stale nodes to control memory usage

Use the radix cache when you have text-based rollout code and want token-level precision without rewriting, or when running GRPO with multiple trajectories sharing the same prompt prefix.

### 2.2 Rollout routing replay (R3) for MoE

For MoE models, slime supports rollout routing replay (R3): record expert routing decisions during rollout and replay them during training to improve stability.

#### SGLang side

SGLang provides expert routing capture via:

- `--enable-return-routed-experts`: server argument to enable routing capture
- `RoutedExpertsCapturer`: captures `topk_ids` (selected expert IDs) at each MoE layer during forward pass
- `return_routed_experts`: request parameter to retrieve routing data
- Returns `routed_experts` in response `meta_info` - a `[seq_len - 1, num_layers, top_k]` tensor of expert IDs

#### Slime side

Slime consumes the routing data and replays it during training:

- `--use-slime-router --use-rollout-routing-replay`: both flags required to enable R3
- Rollout sends `return_routed_experts=True` and stores results in `sample.rollout_routed_experts`
- Training calls `fill_routing_replay()` to load routing data into `RoutingReplay` objects
- During forward pass, recorded routing decisions are replayed instead of recomputed

#### Why SlimeRouter is needed

We need SlimeRouter because the SGLang worker returns routed experts in the response (`meta_info.routed_experts`) when the request sets `return_routed_experts=true`, and SlimeRouter preserves this field end-to-end. SGLang Model Gateway may drop this extra metadata when it reconstructs responses with a fixed schema (see section 3).

---

## 3. Differences vs SGLang Model Gateway

SlimeRouter and SGLang Model Gateway can both route requests to workers, but they are optimized for different goals.

### Key differences

SlimeRouter is a lightweight Python/FastAPI proxy that acts as a passthrough to SGLang workers. This passthrough design enables RL-specific features like radix-tree trajectory caching and R3 (which require preserving raw response metadata like `routed_experts`).

SGLang Model Gateway is a high-performance Rust-based router optimized for large-scale inference: async non-blocking routing, advanced fault tolerance (retries, circuit breakers), multiple load balancing policies (including cache-aware routing), and PD disaggregation support. However, it reconstructs responses with a fixed schema, so it does not preserve the metadata needed for slime's R3 flow.

For more details on SGLang Model Gateway, see the [official documentation](https://docs.sglang.io/advanced_features/sgl_model_gateway.html).

### When to use which

- Use SlimeRouter when you need R3 or radix-tree caching
- Use SGLang Model Gateway for everything else (recommended default)

