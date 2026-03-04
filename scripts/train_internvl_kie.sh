#!/bin/bash
#
# InternVL3.5-4B KIE 训练启动脚本
#

set -e
export WANDB_ENTITY="Qianfan-VL"
export WANDB_PROJECT="slime-dev"
export WANDB_NAME="qianfanocr-kie-slime-test1"
#export WANDB_MODE=offline
export WANDB_API_KEY=285f5c49b9ab1d920af3d2e84df63461a74921ae
export WANDB_BASE_URL=https://wandb.store
export https_proxy=http://agent.baidu.com:8891
export http_proxy=http://agent.baidu.com:8891


# ============== 配置 ==============
MODEL_PATH="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/InternVL3_5-4B-HF"
DATA_PATH="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/kie_train.parquet"
OUTPUT_DIR="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/internvl_kie"
NUM_GPUS=8

# ============== 颜色输出 ==============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============== 清理残留进程 ==============
echo "=========================================="
echo "  清理残留进程和临时文件"
echo "=========================================="
echo ""

echo_info "停止 Ray、SGLang、Redis 相关进程..."
pkill -9 -f ray 2>/dev/null || true
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f redis 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
sleep 2

echo_info "清理 Ray 临时文件..."
rm -rf /tmp/ray* 2>/dev/null || true
rm -rf /dev/shm/ray* 2>/dev/null || true
rm -rf /tmp/redis* 2>/dev/null || true
sleep 1

echo_info "残留进程清理完成 ✓"
echo ""

# ============== 启动前检查 ==============
echo "=========================================="
echo "  InternVL3.5-4B KIE 训练启动检查"
echo "=========================================="
echo ""

# 1. 清除代理设置
echo_info "清除代理设置..."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
echo_info "代理已清除 ✓"
echo ""

# 2. 检查数据文件
echo_info "检查训练数据..."
if [ -f "$DATA_PATH" ]; then
    DATA_SIZE=$(ls -lh "$DATA_PATH" | awk '{print $5}')
    echo_info "数据文件存在: $DATA_PATH ($DATA_SIZE) ✓"
else
    echo_error "数据文件不存在: $DATA_PATH"
    echo_warn "请先运行数据转换脚本:"
    echo "  python scripts/convert_kie_data.py -i <input_files> -o $DATA_PATH -f parquet"
    exit 1
fi
echo ""

# 3. 检查模型路径
echo_info "检查模型路径..."
if [ -d "$MODEL_PATH" ]; then
    if [ -f "$MODEL_PATH/config.json" ]; then
        echo_info "模型路径有效: $MODEL_PATH ✓"
    else
        echo_error "模型路径存在但缺少 config.json"
        exit 1
    fi
else
    echo_error "模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo ""

# 4. 检查 GPU
echo_info "检查 GPU 可用性..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo_info "检测到 $GPU_COUNT 个 GPU"
    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        echo_warn "可用 GPU ($GPU_COUNT) 少于配置 ($NUM_GPUS)，将使用 $GPU_COUNT 个 GPU"
        NUM_GPUS=$GPU_COUNT
    fi
else
    echo_error "nvidia-smi 不可用，请检查 CUDA 环境"
    exit 1
fi
echo ""

# 5. 检查输出目录
echo_info "检查输出目录..."
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo_info "创建输出目录: $OUTPUT_DIR"
else
    echo_info "输出目录已存在: $OUTPUT_DIR"
fi
echo ""

# 6. 检查 WandB (可选)
echo_info "检查 WandB 配置..."
if [ -n "$WANDB_API_KEY" ]; then
    echo_info "WandB 已配置 ✓"
    WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-name $WANDB_NAME"
else
    echo_warn "WandB 未配置，将不记录到 WandB"
    WANDB_ARGS=""
fi
echo ""

# ============== 所有检查通过 ==============
echo "=========================================="
echo -e "  ${GREEN}所有检查通过，准备启动训练${NC}"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  - 模型: $MODEL_PATH"
echo "  - 数据: $DATA_PATH"
echo "  - 输出: $OUTPUT_DIR"
echo "  - GPU 数量: $NUM_GPUS"
echo ""

# 等待用户确认
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."
echo ""

# ============== 启动训练 ==============
echo_info "启动训练..."

cd /mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime

python examples/internvl_kie/train_internvl_kie.py

echo ""
echo_info "训练完成!"
