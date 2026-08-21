# 使用 Mooncake 传输 Rollout Data

slime 默认通过 Ray object store 将 rollout data 从 rollout manager 传给 trainer。
Mooncake Store 可以替换这段传输路径：rollout dict 的接口和训练语义保持不变，
数据 payload 则通过 Mooncake data plane 传输。

这个配置只影响 rollout data：

- `--rollout-data-transport` 选择 rollout data 到达 trainer 的方式；
- 模型权重同步由 `--update-weight-*` 参数单独配置；
- SGLang KV cache disaggregation 也使用独立的传输配置。

同步入口 `train.py` 和异步入口 `train_async.py` 都支持 Mooncake。

## 环境要求

启动 slime 任务前，请确认：

- 所有 Ray 节点使用相同的 slime revision 和 Mooncake wheel。Mooncake 包需要
  提供 slime 使用的 structured-object `put`/`get` 接口；
- 已经启动 `mooncake_master`，或者准备好 Mooncake HA endpoint。slime 只作为
  client 连接，不负责管理 endpoint 生命周期；
- 所有 Ray 节点都能访问用于数据传输的网络地址；
- 为 `MOONCAKE_GLOBAL_SEGMENT_SIZE` 和 `MOONCAKE_LOCAL_BUFFER_SIZE`
  预留足够的主机内存；
- 使用 RDMA 时，运行环境能够访问 RDMA device、允许锁定内存，并且在启动
  Ray 前配置好每个节点自己的 device name；
- 模型、Megatron checkpoint、数据集和 Python 环境在使用它们的节点上均可用。

Mooncake 的包和平台要求请参考
[安装文档](https://kvcache-ai.github.io/Mooncake/getting_started/build.html)。

## 配置传输后端

启动 Ray 前先选择 TCP 或 RDMA。TCP 只要求数据网络可达；RDMA 还需要每个
节点配置本地 device，不同节点的 device name 可以不同。

在已有 slime recipe 中增加：

```bash
--rollout-data-transport mooncake
```

Mooncake 从环境变量中读取连接配置。请在启动 Ray 前导出这些变量，使 Ray
worker 继承节点本地配置：

```bash
export MOONCAKE_MASTER="<mooncake-endpoint>:50051"
export MOONCAKE_PROTOCOL="<tcp-or-rdma>"
export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2gb"
export MOONCAKE_LOCAL_BUFFER_SIZE="2gb"

# Ray node IP 已经是数据网络地址时可以省略。
# export MOONCAKE_LOCAL_HOSTNAME="<local-data-network-ip>"

# 仅 RDMA 需要。填写当前节点数据网卡对应的 device。
# export MOONCAKE_DEVICE="<local-rdma-device>"
```

`MOONCAKE_LOCAL_HOSTNAME` 和 `MOONCAKE_DEVICE` 是节点本地配置。不要把
某一个节点的值写入整个集群共用的 Ray runtime environment。

## 双机运行示例

下面的示例使用 Qwen3-4B 完成两轮同步 rollout 和训练。一个八卡节点用于
训练，另一个八卡节点用于 rollout。请先根据集群环境设置变量，再按顺序执行。

两个节点必须使用相同的 Python 环境、slime revision 和 Mooncake wheel。

### 1. 选择协议并配置节点变量

在 head 节点执行：

```bash
export HEAD_IP="<head-data-network-ip>"
export MOONCAKE_MASTER="${HEAD_IP}:50051"
export MOONCAKE_PROTOCOL="<tcp-or-rdma>"
export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2gb"
export MOONCAKE_LOCAL_BUFFER_SIZE="2gb"
export MOONCAKE_LOCAL_HOSTNAME="${HEAD_IP}"

# 仅 RDMA 需要。
# export MOONCAKE_DEVICE="<head-rdma-device>"
```

在 worker 节点执行：

```bash
export HEAD_IP="<head-data-network-ip>"
export WORKER_IP="<worker-data-network-ip>"
export MOONCAKE_MASTER="${HEAD_IP}:50051"
export MOONCAKE_PROTOCOL="<tcp-or-rdma>"
export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="2gb"
export MOONCAKE_LOCAL_BUFFER_SIZE="2gb"
export MOONCAKE_LOCAL_HOSTNAME="${WORKER_IP}"

# 仅 RDMA 需要。
# export MOONCAKE_DEVICE="<worker-rdma-device>"
```

这里使用 2 GiB 是为了方便运行这个小规模示例。生产任务需要根据 rollout
payload 大小和同时处于 in-flight 状态的 partition 数量重新评估这两个值。

### 2. 在 head 节点启动 Mooncake 和 Ray

激活 slime Python 环境后执行：

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

如果使用已有的 Mooncake master 或 HA endpoint，不要再启动本地 master。

### 3. 将 worker 加入 Ray 集群

在 worker 节点激活相同的 Python 环境，然后执行：

```bash
ray stop --force
ray start \
  --address="${HEAD_IP}:6379" \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus=8 \
  --disable-usage-stats
```

提交训练前，在 head 节点运行 `ray status`，确认集群中有 16 张 GPU。

### 4. 从 head 节点提交训练

以下路径必须在实际使用它们的节点上有效：

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

使用 fully async 训练时，保留相同的 Mooncake 环境变量和传输参数，切换到
slime 原有的异步入口和配置：

```diff
- -- python3 train.py ...
+ -- python3 train_async.py ...
```

其余异步参数请参考 [fully async 示例](../_examples_synced/fully_async/README.md)。

## 配置项说明

| 配置 | 默认值 | 作用 |
|---|---|---|
| `--rollout-data-transport` | `object-store` | 设为 `mooncake` 后使用 Mooncake 传输 rollout data。 |
| `MOONCAKE_MASTER` | 无 | `mooncake_master` 或 HA metadata endpoint 的地址。 |
| `MOONCAKE_LOCAL_HOSTNAME` | Ray node IP | 当前 client 对外发布的数据网络地址。 |
| `MOONCAKE_TE_META_DATA_SERVER` | `P2PHANDSHAKE` | Transfer Engine metadata service。 |
| `MOONCAKE_PROTOCOL` | `rdma` | 传输协议，通常使用 `tcp` 或 `rdma`。 |
| `MOONCAKE_DEVICE` | 自动发现 | 当前节点的 RDMA device name。 |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `8gb` | rollout writer 贡献的 Store 容量。 |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | `32gb` | 本地传输和 staging buffer 容量。 |

trainer 侧的 GET client 不会贡献 global Store segment，但仍会使用
`MOONCAKE_LOCAL_BUFFER_SIZE` 完成传输和管理 pool-backed result。训练消费完
数据后，slime 会释放这些 buffer。

## 常见问题

- **参数解析时无法导入 Mooncake：**所有 Ray worker 的 Python 环境都需要安装
  兼容的 Mooncake wheel；
- **Store setup 失败：**检查 `MOONCAKE_MASTER`、endpoint 可达性，以及所有
  节点使用的 Mooncake 版本是否一致；
- **RDMA 初始化失败：**检查 `MOONCAKE_DEVICE`、locked-memory limit、device
  权限，以及 `MOONCAKE_LOCAL_HOSTNAME` 是否属于所选 RDMA 网络；
- **内存分配失败：**释放主机内存或下调两个 Mooncake size 参数，并为并发的
  rollout partition 和 in-flight step 预留容量。
