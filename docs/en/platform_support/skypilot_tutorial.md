# SkyPilot

[SkyPilot](https://github.com/skypilot-org/skypilot) is an open-source framework for running workloads on Kubernetes or any cloud. This tutorial shows how to launch multi-node slime training with SkyPilot: node provisioning, Ray cluster startup, and job submission are described in a single YAML, replacing the per-node `ray start` steps from the [Quick Start](../get_started/quick_start.md).

It covers two setups, both running the Quick Start's Qwen3-4B GRPO recipe (`scripts/run-qwen3-4B.sh`) on the DAPO-math dataset:

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

The Quick Start starts a Ray cluster by running `ray start` on every node, then submits training with `ray job submit` from node 0. The following SkyPilot task performs the same steps: it provisions `num_nodes` nodes with GPUs, downloads and converts the model on each node, starts the Ray head and workers, and submits the job. Environment variables like `SKYPILOT_NODE_RANK` and `SKYPILOT_NODE_IPS` are injected by SkyPilot on every node.

<details>
<summary><code>slime-multinode.yaml</code></summary>

```yaml
# slime-multinode.yaml
resources:
  infra: kubernetes          # or aws / gcp / any infra configured in `sky check`
  accelerators: H100:4
  image_id: docker:slimerl/slime:latest

num_nodes: 2

setup: |
  pip install -q -U "huggingface_hub[cli]"
  [ -d /root/Qwen3-4B ] || hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
  [ -d /root/dapo-math-17k ] || hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
  [ -d /root/aime-2024 ] || hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
  # Convert the HF checkpoint to Megatron torch_dist format (each node needs a local copy).
  if [ ! -d /root/Qwen3-4B_torch_dist ]; then
    cd /root/slime
    source scripts/models/qwen3-4B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      ${MODEL_ARGS[@]} --hf-checkpoint /root/Qwen3-4B --save /root/Qwen3-4B_torch_dist
  fi

run: |
  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
  if [ "$SKYPILOT_NODE_RANK" != "0" ]; then
    # Worker nodes join the Ray cluster. --block keeps the worker's Ray daemons in the
    # foreground for the whole run (an exiting run command would get them reaped) and
    # returns once the head shuts down at the end of training.
    sleep 10
    ray start --address=${MASTER_ADDR}:6379 --num-gpus ${SKYPILOT_NUM_GPUS_PER_NODE} --disable-usage-stats \
      --dashboard-agent-listen-port 52366 --metrics-export-port 8091 --block
    exit 0
  fi

  # Start Ray from /root/slime: job entrypoints run in the head's working directory.
  cd /root/slime
  source scripts/models/qwen3-4B.sh

  # Non-default agent/metrics ports: SkyPilot's runtime on the node runs its own Ray.
  ray start --head --node-ip-address ${MASTER_ADDR} \
    --num-gpus ${SKYPILOT_NUM_GPUS_PER_NODE} --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --dashboard-agent-listen-port 52366 --metrics-export-port 8091

  # Wait until every node has joined the Ray cluster.
  until python3 -c "import ray, sys; ray.init(address='${MASTER_ADDR}:6379', logging_level='error'); sys.exit(0 if len([n for n in ray.nodes() if n['Alive']]) >= ${SKYPILOT_NUM_NODES} else 1)"; do sleep 5; done

  # Wait for Ray's job agent to be ready to accept submissions.
  until ray job submit --address="http://127.0.0.1:8265" --no-wait -- true >/dev/null 2>&1; do
    echo "waiting for the Ray job agent..."; sleep 5
  done
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
    -- python3 /root/slime/train.py \
    --actor-num-nodes ${SKYPILOT_NUM_NODES} \
    --actor-num-gpus-per-node ${SKYPILOT_NUM_GPUS_PER_NODE} \
    --num-gpus-per-node ${SKYPILOT_NUM_GPUS_PER_NODE} \
    --colocate \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --ref-load /root/Qwen3-4B_torch_dist \
    --load /root/Qwen3-4B_slime/ \
    --save /root/Qwen3-4B_slime/ \
    --save-interval 20 \
    --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rollout-shuffle \
    --rm-type deepscaler \
    --num-rollout 3000 \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1 \
    --global-batch-size 256 \
    --balance-data \
    --eval-interval 20 \
    --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl \
    --n-samples-per-eval-prompt 16 \
    --eval-max-response-len 16384 \
    --eval-top-p 1 \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.00 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 9216 \
    --rollout-num-gpus-per-engine 2 \
    --sglang-mem-fraction-static 0.7 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash
```

</details>

The training arguments are the Quick Start's Qwen3-4B recipe; the only adjustments are the topology flags (`--actor-num-nodes`, `--actor-num-gpus-per-node`, and `--num-gpus-per-node`), whose values come from the SkyPilot-injected environment. `--num-gpus-per-node` matters on nodes with fewer than 8 GPUs: slime's colocated engine mapping assumes 8 per node unless told otherwise. Launch it with:

```bash
sky launch -c slime-train slime-multinode.yaml
```

SkyPilot provisions the nodes (creating them if needed), runs `setup` and `run` on each node, and streams the logs. `sky down slime-train` tears the cluster down. The task assumes a fresh cluster: to re-run training, recreate the cluster (`sky down slime-train && sky launch -c slime-train ...`) rather than re-launching onto one whose Ray daemons are still running. The same YAML can be launched as a managed job with `sky jobs launch`, which adds automatic recovery from node failures.

## Disaggregated Training and Inference

slime supports connecting the trainer to SGLang engines launched by an external system (`--rollout-external-engine-addrs`, see [External Rollout Engines](../advanced/external-rollout-engines.md)). With a SkyPilot **Job Group**, the trainer and each engine are separate jobs in one YAML that are gang-scheduled together and reach each other by stable hostname (`<job>-0.<group>`), so the fleet of engines can be sized independently of the trainer.

The trainer publishes updated weights after each optimizer step and the engines reload them from a shared `ReadWriteMany` volume (`--update-weight-transport disk`). Create the volume once:

```yaml
# policy-volume.yaml
name: slime-policy
type: k8s-pvc
size: 100Gi
infra: kubernetes
config:
  access_mode: ReadWriteMany
```

```bash
sky volumes apply policy-volume.yaml
```

Then launch the Job Group:

<details>
<summary><code>slime-jobgroup.yaml</code></summary>

```yaml
# slime-jobgroup.yaml
---
name: slime-rl
execution: parallel
primary_tasks: [trainer]     # the group succeeds/fails with the trainer
inter_connection: true       # place all jobs on one cluster so they can reach each other
termination_delay: 60s
---
name: sglang
resources:
  infra: kubernetes
  accelerators: H100:1
  image_id: docker:slimerl/slime:latest
volumes:
  /shared/policy: slime-policy
setup: |
  pip install -q -U "huggingface_hub[cli]"
  [ -d /root/Qwen3-4B ] || hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
run: |
  # One SGLang server; the trainer reaches it at sglang-0.<jobgroup>:30000.
  python -m sglang.launch_server --model-path /root/Qwen3-4B --tp 1 \
    --host 0.0.0.0 --port 30000 --mem-fraction-static 0.7
---
name: trainer
resources:
  infra: kubernetes
  accelerators: H100:2
  image_id: docker:slimerl/slime:latest
volumes:
  /shared/policy: slime-policy
setup: |
  pip install -q -U "huggingface_hub[cli]"
  [ -d /root/Qwen3-4B ] || hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
  [ -d /root/dapo-math-17k ] || hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
  [ -d /root/aime-2024 ] || hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
  # Convert the HF checkpoint to Megatron torch_dist format.
  if [ ! -d /root/Qwen3-4B_torch_dist ]; then
    cd /root/slime
    source scripts/models/qwen3-4B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      ${MODEL_ARGS[@]} --hf-checkpoint /root/Qwen3-4B --save /root/Qwen3-4B_torch_dist
  fi
run: |
  # Wait for the engine job to serve (jobs in a group provision independently).
  ENGINE_ADDR="sglang-0.${SKYPILOT_JOBGROUP_NAME}:30000"
  until curl -sf "http://${ENGINE_ADDR}/health" >/dev/null; do
    echo "waiting for engine ${ENGINE_ADDR}..."; sleep 10
  done
  echo "engine healthy: ${ENGINE_ADDR}"

  # Start Ray from /root/slime: job entrypoints run in the head's working directory.
  cd /root/slime
  source scripts/models/qwen3-4B.sh

  # Non-default agent/metrics ports: SkyPilot's runtime on the node runs its own Ray.
  ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --dashboard-agent-listen-port 52366 --metrics-export-port 8091

  # Wait for Ray's job agent to be ready to accept submissions.
  until ray job submit --address="http://127.0.0.1:8265" --no-wait -- true >/dev/null 2>&1; do
    echo "waiting for the Ray job agent..."; sleep 5
  done
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
    -- python3 /root/slime/train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-external-engine-addrs ${ENGINE_ADDR} \
    --update-weight-mode full \
    --update-weight-transport disk \
    --update-weight-disk-dir /shared/policy \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --ref-load /root/Qwen3-4B_torch_dist \
    --load /root/Qwen3-4B_slime/ \
    --save /root/Qwen3-4B_slime/ \
    --save-interval 20 \
    --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rollout-shuffle \
    --rm-type deepscaler \
    --num-rollout 3000 \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1 \
    --global-batch-size 256 \
    --balance-data \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.00 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 9216 \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.7 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash
```

</details>

```bash
sky jobs launch -n slime-rl slime-jobgroup.yaml
```

To scale the inference fleet, add more engine jobs (`sglang-2`, `sglang-3`, ...) to the YAML and append their addresses to `--rollout-external-engine-addrs`. For large models, `--update-weight-mode delta` ships only the changed bytes ([Delta Weight Sync](../advanced/delta-weight-sync.md)); NCCL transport (`--update-weight-transport nccl`) avoids the shared volume entirely.

## End-to-End Example: Agentic Coding RL

A complete agentic RL version of the disaggregated setup lives in the SkyPilot repository:

**[slime on SkyPilot Job Groups](https://github.com/skypilot-org/skypilot/tree/master/llm/slime)** — trains a coding agent (Qwen3-14B) on SWE-smith with slime: a Megatron trainer job plus 1–3 SGLang engine jobs in one Job Group, agent rollouts executing untrusted code in sandboxed pods, and disk-based delta weight sync between the jobs. The example includes launch YAMLs, all setup/run scripts, and benchmark results for scaling the inference fleet (1 → 3 engines cuts async step time from about 1200 s to about 660 s on the example workload).

Issues with the SkyPilot setups on this page can be reported to the [SkyPilot repository](https://github.com/skypilot-org/skypilot/issues).
