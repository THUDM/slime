#!/bin/bash
# 启动 SGLang 服务 (InternVL3.5-4B)
#
# 使用方法:
#   bash scripts/start_sglang_internvl.sh
#   bash scripts/start_sglang_internvl.sh --port 30001
#   bash scripts/start_sglang_internvl.sh --tp 2
MODEL_PATH="/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303//iter_0004600_hf"
PORT=30000
TP=1
MEM_FRACTION=0.8

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --mem)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "启动 SGLang 服务"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "TP: $TP"
echo "显存比例: $MEM_FRACTION"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 启动 SGLang
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --mem-fraction-static "$MEM_FRACTION" \
    --trust-remote-code \
    --log-level info \
    --chat-template chatml
