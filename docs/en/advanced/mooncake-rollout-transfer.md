# Mooncake Rollout Data Transfer

slime normally passes rollout data from the rollout manager to the trainer through
Ray's object store. Mooncake Store is an alternative for this handoff. It keeps the
rollout dictionary contract unchanged while moving the payload through Mooncake's
data plane.

This setting only controls rollout data transfer:

- `--rollout-data-transport` selects how rollout data reaches the trainer.
- Weight synchronization is configured separately by the `--update-weight-*`
  options.
- SGLang KV-cache disaggregation uses its own transfer settings.

Both `train.py` and `train_async.py` support the Mooncake path.

## Requirements

Before starting a slime job:

- Use the same slime revision and Mooncake wheel on every Ray node. The Mooncake
  package must provide the structured-object `put`/`get` APIs used by slime.
- Start `mooncake_master`, or provide a Mooncake HA endpoint. slime connects as a
  client and does not manage the endpoint lifecycle.
- Use data-network addresses that are reachable from every Ray node.
- Reserve enough host memory for `MOONCAKE_GLOBAL_SEGMENT_SIZE` and
  `MOONCAKE_LOCAL_BUFFER_SIZE`.
- For RDMA, expose the RDMA device to the runtime, allow memory locking, and set the
  node-local device name before starting Ray.
- Make the model, Megatron checkpoint, dataset, and Python environment available on
  every node that may run the corresponding worker.

Follow the [Mooncake installation guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html)
for packages and platform requirements.

## Configure the backend

Choose TCP or RDMA before starting Ray. TCP only needs a routable data network.
RDMA also needs a local device on each node; device names may differ across nodes.

Add this option to an existing slime recipe:

```bash
--rollout-data-transport mooncake
```

Mooncake reads its connection settings from the environment. Export them before
starting Ray so that Ray workers inherit the node-local values:

```bash
export MOONCAKE_MASTER="<mooncake-endpoint>:50051"
export MOONCAKE_PROTOCOL="<tcp-or-rdma>"
export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2gb"
export MOONCAKE_LOCAL_BUFFER_SIZE="2gb"

# Optional when the Ray node IP is already the desired data-network address.
# export MOONCAKE_LOCAL_HOSTNAME="<local-data-network-ip>"

# RDMA only. Set the device attached to this node's data-network address.
# export MOONCAKE_DEVICE="<local-rdma-device>"
```

`MOONCAKE_LOCAL_HOSTNAME` and `MOONCAKE_DEVICE` are node-local. Do not put one
node's values into a cluster-wide Ray runtime environment.

## Two-node walkthrough

The following example runs two synchronous rollout and training iterations with
Qwen3-4B. It uses one eight-GPU node for training and one eight-GPU node for rollout.
Set the variables for your cluster, then complete the steps in order.

Both nodes must use the same Python environment, slime revision, and Mooncake wheel.

### 1. Choose the protocol and set node-local values

On the head node:

```bash
export HEAD_IP="<head-data-network-ip>"
export MOONCAKE_MASTER="${HEAD_IP}:50051"
export MOONCAKE_PROTOCOL="<tcp-or-rdma>"
export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2gb"
export MOONCAKE_LOCAL_BUFFER_SIZE="2gb"
export MOONCAKE_LOCAL_HOSTNAME="${HEAD_IP}"

# RDMA only.
# export MOONCAKE_DEVICE="<head-rdma-device>"
```

On the worker node:

```bash
export HEAD_IP="<head-data-network-ip>"
export WORKER_IP="<worker-data-network-ip>"
export MOONCAKE_MASTER="${HEAD_IP}:50051"
export MOONCAKE_PROTOCOL="<tcp-or-rdma>"
export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2gb"
export MOONCAKE_LOCAL_BUFFER_SIZE="2gb"
export MOONCAKE_LOCAL_HOSTNAME="${WORKER_IP}"

# RDMA only.
# export MOONCAKE_DEVICE="<worker-rdma-device>"
```

The 2 GiB settings keep this small example easy to run. For a production job,
size both values for the rollout payloads and the number of partitions that may
remain in flight at the same time.

### 2. Start Mooncake and Ray on the head

Activate the slime environment, then run:

```bash
mooncake_master --rpc_address=0.0.0.0 --rpc_port=50051 \
  >mooncake_master.log 2>&1 &

ray stop --force
ray start --head \
  --node-ip-address="${HEAD_IP}" \
  --port=6379 \
  --num-gpus=8 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265
```

An existing Mooncake master can be replaced with a configured HA endpoint; in that
case, do not launch another local master.

### 3. Join the worker

Activate the same slime environment on the worker, then run:

```bash
ray stop --force
ray start \
  --address="${HEAD_IP}:6379" \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus=8 \
  --disable-usage-stats
```

Check `ray status` on the head before submitting training. The cluster should report
16 GPUs.

### 4. Submit training from the head

Set paths that are valid on the nodes where the corresponding workers run:

```bash
export SLIME_HOME="<path-to-slime>"
export MEGATRON_HOME="<path-to-Megatron-LM>"
export HF_CHECKPOINT="<path-to-Qwen3-4B>"
export REF_LOAD="<path-to-Qwen3-4B-Megatron-checkpoint>"
export PROMPT_DATA="<path-to-training-data.jsonl>"
export RAY_DASHBOARD_ADDR="http://127.0.0.1:8265"

cd "${SLIME_HOME}"
source scripts/models/qwen3-4B.sh

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_HOME}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"MOONCAKE_MASTER\": \"${MOONCAKE_MASTER}\",
    \"MOONCAKE_PROTOCOL\": \"${MOONCAKE_PROTOCOL}\",
    \"MOONCAKE_TE_META_DATA_SERVER\": \"${MOONCAKE_TE_META_DATA_SERVER}\",
    \"MOONCAKE_GLOBAL_SEGMENT_SIZE\": \"${MOONCAKE_GLOBAL_SEGMENT_SIZE}\",
    \"MOONCAKE_LOCAL_BUFFER_SIZE\": \"${MOONCAKE_LOCAL_BUFFER_SIZE}\"
  }
}"

ray job submit --address="${RAY_DASHBOARD_ADDR}" \
  --working-dir="${SLIME_HOME}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --ref-load "${REF_LOAD}" \
  --load "${REF_LOAD}" \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --start-rollout-id 0 \
  --num-rollout 2 \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 256 \
  --rollout-temperature 1 \
  --global-batch-size 4 \
  --balance-data \
  --rollout-data-transport mooncake \
  --tensor-model-parallel-size 8 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 1024 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine 8 \
  --sglang-mem-fraction-static 0.35 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --no-gradient-accumulation-fusion \
  --bf16 \
  --distributed-backend nccl
```

For fully asynchronous training, keep the same Mooncake environment and transport
option, then use the regular async entrypoint and settings:

```diff
- -- python3 train.py ...
+ -- python3 train_async.py ...
```

See the [fully asynchronous example](../_examples_synced/fully_async/README.md) for
the remaining async arguments.

## Configuration reference

| Setting | Default | Purpose |
|---|---|---|
| `--rollout-data-transport` | `object-store` | Set to `mooncake` to use Mooncake for rollout data. |
| `MOONCAKE_MASTER` | none | Address of `mooncake_master` or the HA metadata endpoint. |
| `MOONCAKE_LOCAL_HOSTNAME` | Ray node IP | Data-network address advertised by the local client. |
| `MOONCAKE_TE_META_DATA_SERVER` | `P2PHANDSHAKE` | Transfer Engine metadata service. |
| `MOONCAKE_PROTOCOL` | `rdma` | Transfer protocol, normally `tcp` or `rdma`. |
| `MOONCAKE_DEVICE` | auto-discovery | Local RDMA device name. |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `8gb` | Store capacity contributed by rollout writers. |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | `32gb` | Local transfer and staging capacity. |

The trainer-side GET clients do not contribute a global Store segment. slime still
uses `MOONCAKE_LOCAL_BUFFER_SIZE` there for transfer and pool-backed results, and
releases those buffers after the training step consumes them.

## Troubleshooting

- **Mooncake import fails during argument parsing:** install a compatible Mooncake
  wheel in the Python environment used by every Ray worker.
- **Store setup fails:** verify `MOONCAKE_MASTER`, endpoint reachability, and that all
  nodes use the same Mooncake version.
- **RDMA setup fails:** verify `MOONCAKE_DEVICE`, locked-memory limits, device access,
  and that `MOONCAKE_LOCAL_HOSTNAME` belongs to the selected RDMA network.
- **Allocation fails:** increase available host memory or lower the two Mooncake size
  settings. Account for concurrent rollout partitions and in-flight steps.
