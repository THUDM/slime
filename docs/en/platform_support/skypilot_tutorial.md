# SkyPilot

[SkyPilot](https://github.com/skypilot-org/skypilot) is an open-source framework for running workloads on Kubernetes or any cloud. This tutorial shows how to launch multi-node slime training with SkyPilot: node provisioning, Ray cluster startup, and job submission are described in a single YAML, replacing the per-node `ray start` steps from the [Quick Start](../get_started/quick_start.md).

It covers two setups:

- **Multi-node training on one cluster** — the standard setup from the Quick Start's multi-node section.
- **Disaggregated training and inference** — the trainer and SGLang engines run as separate, gang-scheduled jobs that scale independently.

This page is maintained by the SkyPilot maintainers.

## Prerequisites

Install SkyPilot with the extras for your infrastructure and confirm it can reach it:

```bash
pip install "skypilot[kubernetes]"   # or [aws], [gcp], ... — see SkyPilot docs
sky check
```

The examples below use the `slimerl/slime:latest` Docker image from the Quick Start, so no additional environment setup is needed inside the nodes.

## Multi-Node Training on One Cluster

The Quick Start starts a Ray cluster by running `ray start` on every node, then submits training with `ray job submit` from node 0. The following SkyPilot task performs the same steps: it provisions `num_nodes` nodes with GPUs, starts the Ray head and workers, and submits the job. Environment variables like `SKYPILOT_NODE_RANK` and `SKYPILOT_NODE_IPS` are injected by SkyPilot on every node.

```yaml
# slime-multinode.yaml
resources:
  infra: kubernetes          # or aws / gcp / any infra configured in `sky check`
  accelerators: H100:8
  image_id: docker:slimerl/slime:latest

num_nodes: 2

workdir: .                   # ship your training scripts to every node

run: |
  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    ray start --head --node-ip-address ${MASTER_ADDR} \
      --num-gpus 8 --disable-usage-stats

    # Wait until every node has joined the Ray cluster.
    while [ "$(ray list nodes --format json | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')" -lt "$SKYPILOT_NUM_NODES" ]; do
      sleep 5
    done

    ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{
        "env_vars": {
          "PYTHONPATH": "/root/Megatron-LM/"
        }
      }' \
      -- python3 train.py \
      --... # Megatron/SGLang/slime arguments, as in the Quick Start
  else
    sleep 10
    ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
  fi
```

Launch it with:

```bash
sky launch -c slime-train slime-multinode.yaml
```

SkyPilot provisions the nodes (creating them if needed), runs `run` on each node, and streams the logs. `sky down slime-train` tears the cluster down. The same YAML can be launched as a managed job with `sky jobs launch`, which adds automatic recovery from node failures.

## Disaggregated Training and Inference

slime supports connecting the trainer to SGLang engines launched by an external system (`--rollout-external-engine-addrs`, see [External Rollout Engines](../advanced/external-rollout-engines.md)). With a SkyPilot **Job Group**, the trainer and each engine are separate jobs in one YAML that are gang-scheduled together and reach each other by stable hostname (`<job>.<group>`), so the fleet of engines can be sized independently of the trainer:

```yaml
# One Job Group: 1 trainer + 1 SGLang engine (add more engine jobs to scale).
---
name: slime-rl
execution: parallel
primary_tasks: [trainer]
---
name: sglang
resources:
  infra: kubernetes
  accelerators: H100:1
  image_id: docker:slimerl/slime:latest
volumes:
  /shared/policy: slime-policy   # shared RWX volume for disk weight sync
run: |
  # Serve SGLang; the trainer reaches this engine at sglang.<group>.
  ...
---
name: trainer
resources:
  infra: kubernetes
  accelerators: H100:4
  image_id: docker:slimerl/slime:latest
volumes:
  /shared/policy: slime-policy
envs:
  SGLANG_MEMBERS: "sglang"       # engine job names; add sglang-2, sglang-3, ...
run: |
  # Start Ray + slime with --rollout-external-engine-addrs pointed at the engines,
  # publishing weights per step via the shared volume (delta or full) or NCCL.
  ...
```

After each optimizer step the trainer publishes updated weights and the engines reload them — over the shared volume with `--update-weight-transport disk` (optionally `--update-weight-mode delta` to ship only changed bytes), or over NCCL ([Delta Weight Sync](../advanced/delta-weight-sync.md)).

## End-to-End Example: Agentic Coding RL

A complete, runnable version of the disaggregated setup lives in the SkyPilot repository:

**[slime on SkyPilot Job Groups](https://github.com/skypilot-org/skypilot/tree/master/llm/slime)** — trains a coding agent (Qwen3-14B) on SWE-smith with slime: a Megatron trainer job plus 1–3 SGLang engine jobs in one Job Group, agent rollouts executing untrusted code in sandboxed pods, and disk-based delta weight sync between the jobs. The example includes launch YAMLs, all setup/run scripts, and benchmark results for scaling the inference fleet (1 → 3 engines cuts async step time from about 1200 s to about 660 s on the example workload).

Issues with the SkyPilot setups on this page can be reported to the [SkyPilot repository](https://github.com/skypilot-org/skypilot/issues).
