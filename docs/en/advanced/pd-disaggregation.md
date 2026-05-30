# PD Disaggregation

PD Disaggregation separates Prefill and Decode workers in SGLang rollout. This is especially useful for multi-turn, long-context, and agentic RL workloads where prompt processing and token generation have very different compute and memory profiles.

## When to Use

Use PD Disaggregation when:

- rollout contexts are long or grow across turns;
- decode dominates rollout time;
- prefix-cache locality matters for multi-turn sessions;
- prefill and decode need different TP, memory, or runtime settings;
- you want an SGLang serving topology that is closer to production serving rather than a single uniform inference group.

For short single-turn tasks, the default regular SGLang engine layout is usually simpler.

## Configuration Paths

slime supports two ways to configure PD.

### Simple Path: `--prefill-num-servers`

For a single actor model with a simple PD layout, set:

```bash
--prefill-num-servers 1
```

This is the lightweight path used by simple scripts. It is convenient when you only need to split prefill/decode without tuning each group separately.

### Advanced Path: `--sglang-config`

For production rollout topologies, use [SGLang Config](sglang-config.md). It lets you configure prefill and decode groups independently, and can also express EPD-style layouts, heterogeneous server groups, multi-model serving, and per-group SGLang overrides.

Example:

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

Launch with:

```bash
python train.py \
  --sglang-config sglang_pd.yaml \
  --rollout-num-gpus 16 \
  ...
```

## Why This Matters for RL

RL rollout is often not a uniform batch of short completions. Agentic and verifier-based workloads commonly have:

- long prompts from tool/environment history;
- multiple turns per sample;
- long-tail decode latency;
- session-local prefix cache opportunities;
- different resource needs for actor, reference, reward, or judge models.

PD lets slime keep the training loop unchanged while using a rollout topology that matches the actual serving workload.

## Operational Notes

- For new complex deployments, prefer `--sglang-config` over `--prefill-num-servers`.
- Use router session affinity for multi-turn agents so turns from the same sample can reuse prefix cache. See [Session-Affinity Routing](sglang-config.md#session-affinity-routing-for-multi-turn-agents).
- Keep `--rollout-num-gpus` equal to the total GPUs described by the SGLang config.
- Do not mix `regular` workers with `prefill`/`decode` workers inside the same model entry.
- Tune prefill and decode TP separately when prompt processing and token generation have different bottlenecks.

## Related Docs

- [SGLang Config](sglang-config.md)
- [Agentic RL Training Roadmap](../get_started/agent.md)
- [Trace Viewer](../developer_guide/trace.md)
