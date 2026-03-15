#!/bin/bash
# ========================================================================
# Slime KIE GRPO Multi-Node Training Script
# ========================================================================
# LTP 平台多机多卡 slime 训练脚本

set -eox pipefail

# ------------------------------------------------------------------------
# 环境初始化
# ------------------------------------------------------------------------
echo "========================================="
echo "开始执行 Slime 多机训练任务"
echo "Job Name: ${FULL_JOB_NAME}"
echo "Pod Name: ${POD_NAME}"
echo "========================================="

# 清理残留进程 - 加强版
echo "清理残留进程..."
pkill -9 -f "sglang" || true
pkill -9 -f "ray::" || true
pkill -9 -f "raylet" || true
pkill -9 -f "gcs_server" || true
pkill -9 -f "plasma" || true
ray stop --force 2>/dev/null || true
sleep 3

# 清理 Ray 临时文件和 socket
echo "清理 Ray 临时文件..."
rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf /dev/shm/ray_* 2>/dev/null || true
sleep 2

# 检查并释放端口 6379 和 8265
for port in 6379 8265; do
    pid=$(lsof -t -i:${port} 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        echo "端口 ${port} 被占用，正在释放 (PID: $pid)..."
        kill -9 $pid 2>/dev/null || true
    fi
done
sleep 2

# 设置环境变量
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPU 配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ------------------------------------------------------------------------
# 从 config.yaml 读取的环境变量 (LTP 平台自动转为大写)
# ------------------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf"}
TRAIN_DATA=${TRAIN_DATA:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/kie_rl_train.parquet"}
SAVE_PATH=${SAVE_PATH:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/internvl_kie"}

# 集群配置
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
COLOCATE=${COLOCATE:-"true"}

# 训练配置
TRAIN_BACKEND=${TRAIN_BACKEND:-"fsdp"}
NUM_ROLLOUT=${NUM_ROLLOUT:-100}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}

# GRPO 配置
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-"grpo"}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}
KL_COEF=${KL_COEF:-0.00}
ENTROPY_COEF=${ENTROPY_COEF:-0.01}
EPS_CLIP=${EPS_CLIP:-0.2}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.28}

# 优化器配置
OPTIMIZER=${OPTIMIZER:-"adam"}
LR=${LR:-"5e-7"}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-"cosine"}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}

# SGLang 配置
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.6}

# Reward 配置
CUSTOM_RM_PATH=${CUSTOM_RM_PATH:-"slime.rollout.rm_hub.kie_reward.kie_reward"}

# WandB 配置
USE_WANDB=${USE_WANDB:-"true"}
WANDB_PROJECT=${WANDB_PROJECT:-"internvl-kie-grpo"}
WANDB_GROUP=${WANDB_GROUP:-"internvl3.5-4b-kie"}
WANDB_KEY=${WANDB_KEY:-""}
WANDB_MODE=${WANDB_MODE:-"offline"}

# 保存配置
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

# ------------------------------------------------------------------------
# 创建输出目录
# ------------------------------------------------------------------------
mkdir -p ${SAVE_PATH}
LOGS_PATH="${SAVE_PATH}/logs"
mkdir -p ${LOGS_PATH}

# ------------------------------------------------------------------------
# 获取 Master 地址 (PyTorchJob 自动设置)
# ------------------------------------------------------------------------
# PyTorchJob 会设置 MASTER_ADDR 和 MASTER_PORT
# 如果使用 hostNetwork，MASTER_POD_IP 是 Master 节点的实际 IP
if [[ -n "${MASTER_POD_IP}" ]]; then
    export MASTER_ADDR=${MASTER_POD_IP}
fi
MASTER_PORT=${MASTER_PORT:-6022}
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# ------------------------------------------------------------------------
# Ray 集群启动
# ------------------------------------------------------------------------
# 计算节点数
NNODES=$((NUM_WORKERS + 1))

# 获取当前节点 IP (强制使用内网 10.x IP)
CURRENT_NODE_IP=$(hostname -i | tr ' ' '\n' | grep '^10\.' | head -1)
if [[ -z "${CURRENT_NODE_IP}" ]]; then
    # 如果没有 10.x IP，使用第一个 IP并警告
    CURRENT_NODE_IP=$(hostname -i | awk '{print $1}')
    echo "⚠️  警告: 未找到 10.x 内网 IP，使用: ${CURRENT_NODE_IP}"
else
    echo "✓ 使用内网 IP: ${CURRENT_NODE_IP}"
fi

# 根据当前节点 IP 自动选择网卡，避免把 GLOO/NCCL 绑到不存在的设备名
SOCKET_IFNAME=""
if command -v ip >/dev/null 2>&1; then
    SOCKET_IFNAME=$(ip -o -4 addr show | awk -v ip="${CURRENT_NODE_IP}" '$4 ~ ("^" ip "/") {print $2; exit}')
elif [[ -x /sbin/ip ]]; then
    SOCKET_IFNAME=$(/sbin/ip -o -4 addr show | awk -v ip="${CURRENT_NODE_IP}" '$4 ~ ("^" ip "/") {print $2; exit}')
elif command -v ifconfig >/dev/null 2>&1; then
    SOCKET_IFNAME=$(ifconfig | awk -v ip="${CURRENT_NODE_IP}" '
        /^[^ \t]/ {iface=$1; sub(/:$/, "", iface)}
        $0 ~ ip {print iface; exit}
    ')
else
    echo "⚠️  容器内未找到 ip/ifconfig，跳过网卡自动识别"
fi
if [[ -z "${SOCKET_IFNAME}" ]]; then
    echo "⚠️  未能根据 ${CURRENT_NODE_IP} 自动识别网卡，保留系统默认网卡选择"
else
    export NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
    export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
    echo "✓ 使用网络接口: ${SOCKET_IFNAME}"
fi

# IB 配置仅在显式提供时启用，避免默认值与当前环境冲突
if [[ -n "${NCCL_IB_HCA}" ]]; then
    export NCCL_IB_HCA
fi
if [[ -n "${NCCL_IB_GID_INDEX}" ]]; then
    export NCCL_IB_GID_INDEX
fi

# 判断当前节点角色 - PyTorchJob pod 名称格式: jobname-master-0 或 jobname-worker-N
if [[ "$POD_NAME" == *"-master-"* ]]; then
    echo "📍 当前节点是 Master (Ray Head)"
    NODE_RANK=0

    # 启动 Ray Head 节点
    echo "启动 Ray Head 节点..."
    ray start --head \
        --node-ip-address ${CURRENT_NODE_IP} \
        --port 6379 \
        --num-gpus ${GPUS_PER_NODE} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    # 等待 Worker 节点加入
    echo "等待 Worker 节点加入 Ray 集群..."
    sleep 60

    # 检查 Ray 集群状态
    ray status

else
    echo "📍 当前节点是 Worker 节点 ($POD_NAME)"
    # 从 pod 名称提取 worker 编号
    worker_num=$(echo "$POD_NAME" | grep -oE 'worker-[0-9]+' | grep -oE '[0-9]+')
    NODE_RANK=$((worker_num + 1))

    # 等待 Master 节点启动 Ray Head
    echo "等待 Master 节点启动 Ray Head..."
    sleep 30

    # 加入 Ray 集群
    echo "加入 Ray 集群 (Master: ${MASTER_ADDR}:6379)..."

    # 重试加入集群
    MAX_RETRIES=10
    RETRY_INTERVAL=10
    for i in $(seq 1 $MAX_RETRIES); do
        if ray start --address=${MASTER_ADDR}:6379 \
            --num-gpus ${GPUS_PER_NODE} \
            --node-ip-address ${CURRENT_NODE_IP} \
            --disable-usage-stats; then
            echo "成功加入 Ray 集群"
            break
        else
            echo "加入集群失败，重试 $i/$MAX_RETRIES..."
            sleep $RETRY_INTERVAL
        fi
    done

    if ! ray status >/dev/null 2>&1; then
        echo "Worker 节点未能加入 Ray 集群，退出"
        exit 1
    fi

    # Worker 节点等待训练完成
    echo "Worker 节点已加入集群，等待训练任务..."

    # Worker 节点前台等待 - 通过 tail 跟踪 ray 日志保持前台运行
    RAY_LOG_DIR="/tmp/ray/session_latest/logs"

    # 等待日志目录创建
    while [[ ! -d "${RAY_LOG_DIR}" ]]; do
        echo "等待 Ray 日志目录创建..."
        sleep 5
    done

    # 前台跟踪日志，同时检查 raylet 进程
    echo "开始跟踪 Ray 日志..."
    tail -f ${RAY_LOG_DIR}/raylet.out ${RAY_LOG_DIR}/raylet.err 2>/dev/null &
    TAIL_PID=$!

    # 等待 raylet 进程结束
    while pgrep -f "raylet" > /dev/null; do
        sleep 30
    done

    echo "Ray 进程已结束，Worker 节点退出"
    kill $TAIL_PID 2>/dev/null || true
    exit 0
fi

# ------------------------------------------------------------------------
# 构建训练参数 (仅 Master 节点执行)
# ------------------------------------------------------------------------
CKPT_ARGS="--hf-checkpoint ${MODEL_PATH}"

ROLLOUT_ARGS=(
    "--prompt-data ${TRAIN_DATA}"
    "--input-key problem"
    "--label-key answer"
    "--apply-chat-template"
    "--rollout-shuffle"
    "--num-rollout ${NUM_ROLLOUT}"
    "--rollout-batch-size ${ROLLOUT_BATCH_SIZE}"
    "--n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}"
    "--rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}"
    "--rollout-temperature ${ROLLOUT_TEMPERATURE}"
    "--global-batch-size ${GLOBAL_BATCH_SIZE}"
    "--rollout-stop-token-ids 151645"
)

# 多模态配置 - 注意 JSON 中不能有空格，否则会被 shell 拆分
MULTIMODAL_ARGS='--multimodal-keys {"image":"images"}'

# Reward 配置
REWARD_ARGS="--custom-rm-path ${CUSTOM_RM_PATH}"

# FSDP 训练后端配置
FSDP_ARGS=(
    "--train-backend ${TRAIN_BACKEND}"
    "--gradient-checkpointing"
    "--update-weight-buffer-size 536870912"
)

# GRPO 算法配置
GRPO_ARGS=(
    "--advantage-estimator ${ADVANTAGE_ESTIMATOR}"
    "--kl-loss-coef ${KL_LOSS_COEF}"
    "--kl-loss-type ${KL_LOSS_TYPE}"
    "--kl-coef ${KL_COEF}"
    "--entropy-coef ${ENTROPY_COEF}"
    "--eps-clip ${EPS_CLIP}"
    "--eps-clip-high ${EPS_CLIP_HIGH}"
)

# 优化器配置
OPTIMIZER_ARGS=(
    "--optimizer ${OPTIMIZER}"
    "--lr ${LR}"
    "--lr-decay-style ${LR_DECAY_STYLE}"
    "--weight-decay ${WEIGHT_DECAY}"
    "--adam-beta1 ${ADAM_BETA1}"
    "--adam-beta2 ${ADAM_BETA2}"
)

# SGLang 配置
SGLANG_ARGS=(
    "--rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    "--sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}"
    "--sglang-decode-log-interval 500"
    "--sglang-enable-metrics"
    "--attn-implementation flash_attention_2"
    "--sglang-cuda-graph-max-bs 32"
)

# 保存配置
SAVE_ARGS=(
    "--save ${SAVE_PATH}"
    "--save-interval ${SAVE_INTERVAL}"
)

# WandB 配置
WANDB_ARGS=""
if [[ "${USE_WANDB}" == "true" ]]; then
    WANDB_ARGS="--use-wandb --wandb-project ${WANDB_PROJECT} --wandb-group ${WANDB_GROUP} --wandb-mode ${WANDB_MODE}"
    if [[ -n "${WANDB_KEY}" ]]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb-key ${WANDB_KEY}"
    fi
fi

# 集群配置
CLUSTER_ARGS=(
    "--actor-num-nodes ${ACTOR_NUM_NODES}"
    "--actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}"
)

if [[ "${COLOCATE}" == "true" ]]; then
    CLUSTER_ARGS+=("--colocate")
fi

# ------------------------------------------------------------------------
# 提交训练任务
# ------------------------------------------------------------------------
echo "提交 Slime 训练任务..."

# 设置 slime 路径 - 使用共享存储上的修改版本
SLIME_PATH=${SLIME_PATH:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime"}

# FSDP 后端不需要设置 CUDA_DEVICE_MAX_CONNECTIONS (参考 command_utils.py)
# 不设置 working_dir，因为代码在共享存储上，所有节点都能访问
# PyTorch 分布式 & NCCL 配置（参考之前 torchrun 配置）
RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "'${SLIME_PATH}'",
    "MASTER_ADDR": "'${MASTER_ADDR}'",
    "MASTER_PORT": "'${MASTER_PORT}'",
    "WORLD_SIZE": "'${ACTOR_NUM_NODES}'",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "NCCL_DEBUG": "WARN",'
if [[ -n "${SOCKET_IFNAME}" ]]; then
RUNTIME_ENV_JSON+='
    "NCCL_SOCKET_IFNAME": "'${SOCKET_IFNAME}'",
    "GLOO_SOCKET_IFNAME": "'${SOCKET_IFNAME}'",'
fi
RUNTIME_ENV_JSON+='
    "no_proxy": "127.0.0.1,'${MASTER_ADDR}'"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${SLIME_PATH}/train.py \
    ${CKPT_ARGS} \
    ${ROLLOUT_ARGS[@]} \
    ${MULTIMODAL_ARGS} \
    ${REWARD_ARGS} \
    ${FSDP_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${SAVE_ARGS[@]} \
    ${WANDB_ARGS} \
    ${CLUSTER_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/train_${HOSTNAME}_$(date +%Y%m%d-%H%M%S).log
