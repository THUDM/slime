# External Rollout Engines Roadmap

An external rollout engine is an SGLang engine that is not launched by the slime training job. Another system deploys and owns the engine lifecycle; slime connects to those engines during training, registers a router, and syncs updated actor weights when needed.

This page is a roadmap. Use it to decide when to use `--rollout-external-engine-addrs`, when to use `--rollout-http-endpoint-url`, when to stay with `--sglang-config`, and which weight-update path to pick for external deployments.

## Where To Start

| Goal | Recommended entry point |
| :--- | :--- |
| Engines are already launched externally and slime should only connect for rollout | `--rollout-external-engine-addrs` |
| Rollout serving is an elastic fleet behind a single HTTP URL, with no stable per-engine handles | `--rollout-http-endpoint-url` |
| The serving side pulls published weight versions instead of receiving direct update RPCs | `--update-weight-delta-publish-only`, see [Publish-Only Disk Delta](delta-weight-sync.md#publish-only-disk-delta) |
| slime should still launch engines, but you need PD disaggregation, multi-model serving, heterogeneous server groups, or per-group overrides | [SGLang Config](sglang-config.md) |
| Trainer and external engines can form an NCCL group | Default `--update-weight-mode full --update-weight-transport nccl` |
| Trainer and external engines cannot form an NCCL group, but can see the same filesystem path | `--update-weight-mode full --update-weight-transport disk` |
| Full checkpoints are too heavy for large-model cross-cluster or cross-DC sync | `--update-weight-mode delta --update-weight-transport disk` |
| Rollout serving can use an independent SGLang environment, or even different GPU models/vendors | external engines + disk transport |
| You want to validate delta wire/apply logic inside one datacenter | `--update-weight-mode delta --update-weight-transport nccl` |
| You need frozen reference, reward, or tool-side models | Prefer `update_weights: false` in [SGLang Config](sglang-config.md#3-multi-model-serving) |

## What External Engine Does

First launch SGLang servers independently:

```bash
python -m sglang.launch_server --model-path /path/to/model --port 10090 ...
python -m sglang.launch_server --model-path /path/to/model --port 10091 ...
```

Then pass those addresses to the training job:

```bash
python train.py \
  --rollout-external-engine-addrs host1:10090 host2:10091 \
  ...
```

slime queries each engine's `/server_info` or `/get_server_info` endpoint and infers GPU counts, TP/PP information, and worker type (`regular`, `prefill`, or `decode`). If `--sglang-router-ip/--sglang-router-port` is not provided, slime launches its own router and registers the external engines with it.

This path fits deployments where serving is owned outside the training job: a separate inference cluster, a separate Ray cluster, manually warmed SGLang engines, or a rollout service managed by another orchestrator.

## Opaque HTTP Rollout Endpoint

`--rollout-external-engine-addrs` still assumes SGLang engines with stable addresses: slime queries `/server_info` per engine, registers each one with a router, and pushes weight updates to known engine handles. Some deployments cannot offer that contract — for example a serverless or autoscaled inference fleet behind one URL, where workers come and go and no worker-management API is exposed. For those, point slime at the endpoint directly:

```bash
python train.py \
  --rollout-http-endpoint-url https://rollout.example.com \
  ...
```

In this mode slime launches no engines and no router, and assumes nothing about the endpoint beyond the generation route: rollout requests POST to `{url}/generate`, and `get_model_url(args, ...)` in custom rollout functions resolves to the endpoint as well. No rollout GPUs are allocated in the placement group, `/server_info` is never queried, and slime fault tolerance does not manage the fleet — recovery is the endpoint operator's job. `--rollout-http-endpoint-url` and `--rollout-external-engine-addrs` are mutually exclusive.

Two companion flags adapt the default SGLang rollout to an endpoint that lacks router APIs:

- `--rollout-http-endpoint-abort-strategy {cancel-only,router-workers}`: how `abort` behaves between rollouts. `cancel-only` (the default when an endpoint URL is set) cancels slime's local pending generation tasks without calling the router's worker-list or per-worker abort APIs. `router-workers` keeps the existing router-based abort and remains the default otherwise. Note that `cancel-only` does not collect partial samples, so it does not compose with `--partial-rollout`.
- `--custom-rollout-request-hook-path`: optional hook called before each default SGLang `/generate` request. Signature: `def hook(args, sample, request) -> None | dict`. The `request` dict contains `url`, `payload`, `headers`, `max_retries`, `retry_sleep`, `rollout_id`, and `evaluation`; mutate it in place or return a dict of updates.

Use the request hook for rollout-endpoint admission control. For example, a hook may attach `"weight_version": {"exact_version": <ready_version>}` or `"weight_version": {"min_required_version": <minimum_version>}` and increase `max_retries`/`retry_sleep`. Those request fields avoid wasted rollout compute when an opaque router sends the request to a replica that has not loaded a usable version yet. They do not define SLIME's off-policy or staleness semantics; the trainer schedule and loss/correction path still decide which versions are valid.

For weight sync, an elastic fleet usually cannot receive per-engine `update_weights_from_disk` RPCs either. Combine the endpoint with publish-only delta sync, where the trainer publishes each complete weight version through a custom hook and the serving side consumes it on its own schedule — see [Publish-Only Disk Delta](delta-weight-sync.md#publish-only-disk-delta). If request-level minimum-version retry is enough, leave publish-only in its default pipelined mode. If the publish hook polls rollout-fleet status and you want the next rollout dispatch to wait for that readiness threshold, set `--update-weight-delta-publish-wait=sync`.

## Relationship With `--sglang-config`

`--rollout-external-engine-addrs` and `--sglang-config` are mutually exclusive because they own different boundaries:

- `--sglang-config`: slime owns the engine lifecycle. The YAML describes the topology, and slime launches server groups, routers, multi-model serving, and selective weight updates.
- `--rollout-external-engine-addrs`: an external system owns the engine lifecycle. slime discovers already-running engines, attaches them to a router, and treats them as the default rollout model.

If your main requirement is multi-model serving, frozen reference/reward models, PD disaggregation, or heterogeneous group configuration, prefer `--sglang-config`. Use external engines when the engines are already deployed outside the training job.

## Environment And Hardware Decoupling

An important implication of external engines is that the SGLang serving side does not need to use the slime training job's Python environment, Megatron environment, or Ray runtime. It can run in a separate SGLang container, an independent cluster, or another orchestration system. slime only depends on the HTTP endpoint, `/server_info`, and the communication path required by the selected weight-sync transport.

With disk transport, weights move through HF checkpoints or safetensors deltas on a shared filesystem, and SGLang hot-loads them through `update_weights_from_disk`. This path does not require the training GPUs and rollout GPUs to be the same model, or even from the same vendor, as long as SGLang supports that hardware backend, model format, and precision configuration. For example, training can run on one GPU fleet while rollout serving runs on another fleet with different GPU models or vendors.

With NCCL transport, the usual NCCL communication and hardware-compatibility requirements still apply. For cross-vendor, incompatible-network, or cross-datacenter deployments, prefer `--update-weight-transport disk`.

## Update From Disk

Full-checkpoint update from disk is the simplest fallback path for external deployments:

```bash
--update-weight-mode full
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/full-updates
```

At every weight sync, the trainer writes a complete HF checkpoint directory under `--update-weight-disk-dir`, such as `weight_v000123/`, then calls each SGLang engine's `update_weights_from_disk` endpoint over HTTP so the engine reloads the checkpoint without a process restart.

This mode has a simple control plane: it does not require an NCCL group between trainer and engines. It only requires both sides to see the same shared filesystem path. The tradeoff is size: every sync writes the full actor weights, which is expensive for large models or frequent updates.

For debugging, add:

```bash
--update-weight-disk-keep-files
```

This keeps the full-checkpoint directories after engines acknowledge the load.

## Update With Delta

Delta update targets large-model training/inference disaggregation across clusters or datacenters. Instead of writing a full checkpoint, the trainer keeps a pinned-CPU snapshot of the previous sync, detects byte-level changes, and sends only changed positions and values.

Recommended for cross-cluster / shared-filesystem deployments:

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd
--update-weight-disk-dir /shared/fs/delta-updates
```

With disk transport, each sync writes sparse safetensors under `weight_v{N:06d}/`, then calls `update_weights_from_disk(load_format="delta")`. SGLang overwrites only changed positions in the current weights; unchanged positions stay in place.

For intra-datacenter validation or bandwidth-rich environments, NCCL transport is also available:

```bash
--update-weight-mode delta
--update-weight-transport nccl
--update-weight-encoding indices
```

For encoding choices, wire layout, receiver-side selective overwrite, and tuning parameters, see [Delta Weight Sync](delta-weight-sync.md).

## Deployment Checklist

- External engine HTTP addresses must be reachable from the training job.
- External engines can use an independent SGLang environment; they do not need the slime or Megatron training environment.
- Disk transport supports different GPU models or vendors between training and rollout, as long as SGLang supports the target hardware and model format.
- Disk transport requires trainer and SGLang engines to see the same `--update-weight-disk-dir` path; a path visible only to the trainer is not enough.
- External engines are not recovered by slime fault tolerance; their lifecycle belongs to the external deployment system. The same applies to fleets behind `--rollout-http-endpoint-url`.
- `--sglang-config` and `--rollout-external-engine-addrs` are mutually exclusive, as are `--rollout-external-engine-addrs` and `--rollout-http-endpoint-url`.
- An opaque HTTP endpoint only needs to serve the generation route; worker-management APIs are never called. If the fleet cannot accept direct weight-update RPCs, use publish-only delta sync.
- Delta mode does not support `--colocate`, because colocated sync uses CUDA IPC handles and delta encoding does not reduce the actual transfer.

## Related Work

The [Composer 2 technical report by the Cursor Research Team](https://arxiv.org/html/2603.24477v2) describes a similar production shape: training and rollout generation run asynchronously, Cursor partners with Fireworks AI for RL inference, updated weights are written to shared S3 every training step, delta compression reduces transfer size, and inference clusters in different regions download and reconstruct weights from a shared delta chain.

slime's external engines, update from disk, and delta disk transport address the same infrastructure problem: once training and inference are disaggregated, weight sync must work across processes, clusters, and even datacenters without letting full-model transfer dominate the training loop.
