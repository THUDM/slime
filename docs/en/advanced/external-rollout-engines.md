# External Rollout Engines Roadmap

An external rollout engine is an SGLang engine that is not launched by the slime training job. Another system deploys and owns the engine lifecycle; slime connects to those engines during training, registers a router, and syncs updated actor weights when needed.

This page is a roadmap. Use it to decide when to use `--rollout-external-engine-addrs`, when to stay with `--sglang-config`, and which weight-update path to pick for external deployments.

## Where To Start

| Goal | Recommended entry point |
| :--- | :--- |
| Engines are already launched externally and slime should only connect for rollout | `--rollout-external-engine-addrs` |
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

Delta update targets large-model training/inference disaggregation across clusters or datacenters. Instead of writing a full checkpoint every sync, the trainer keeps a CPU snapshot of the previous sync, diffs each parameter against it, and publishes only the changed bytes; every rollout host applies the delta into its local checkpoint and reloads via the vanilla `update_weights_from_disk` endpoint.

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/delta-updates
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt
```

See [Delta Weight Sync](delta-weight-sync.md) for the mechanism, encodings, integrity checks, and shared-filesystem visibility hooks.

## Deployment Checklist

- External engine HTTP addresses must be reachable from the training job.
- External engines can use an independent SGLang environment; they do not need the slime or Megatron training environment.
- Disk transport supports different GPU models or vendors between training and rollout, as long as SGLang supports the target hardware and model format.
- Disk transport requires trainer and SGLang engines to see the same `--update-weight-disk-dir` path; a path visible only to the trainer is not enough.
- External engines are not recovered by slime fault tolerance; their lifecycle belongs to the external deployment system.
- `--sglang-config` and `--rollout-external-engine-addrs` are mutually exclusive.
- Delta mode does not support `--colocate`, because colocated sync uses CUDA IPC handles and delta encoding does not reduce the actual transfer.

## Related Work

The [Composer 2 technical report by the Cursor Research Team](https://arxiv.org/html/2603.24477v2) describes a similar production shape: training and rollout generation run asynchronously, Cursor partners with Fireworks AI for RL inference, updated weights are written to shared S3 every training step, delta compression reduces transfer size, and inference clusters in different regions download and reconstruct weights from a shared delta chain.

slime's external engines, update from disk, and delta disk transport address the same infrastructure problem: once training and inference are disaggregated, weight sync must work across processes, clusters, and even datacenters without letting full-model transfer dominate the training loop.
