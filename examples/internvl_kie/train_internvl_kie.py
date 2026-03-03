"""
InternVL3.5-4B KIE (Key Information Extraction) 训练脚本

使用 SLIME 框架进行 GRPO 训练
"""
import os
import slime.utils.external_utils.command_utils as U

# ============== 配置 ==============
NUM_GPUS = 8
MODEL_PATH = "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/InternVL3_5-4B"

# 数据路径（转换后的）
TRAIN_DATA = "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/kie_train.parquet"

# 原始数据路径
RAW_DATA_FILES = [
    "/mnt/cfs_bj_mt/workspace/zhengmingming/qianfan_ocr/ocrbench/v2/sft_synthesized_v3/wildreceipt_synthesized_v3_json.jsonl",
    "/mnt/cfs_bj_mt/workspace/zhengmingming/qianfan_ocr/ocrbench/ccocr/synthetic/SIBR_ccocr_synthesized_json.jsonl",
    "/mnt/cfs_bj_mt/workspace/zhengmingming/qianfan_ocr/ocrbench/v2/M6Doc/m6doc_synthesized_json.jsonl",
    "/mnt/cfs_bj_mt/workspace/zhengmingming/qianfan_ocr/ocrbench/v2/sft_synthesized/fund_synthesized_fixed_json_v2.jsonl",
]


def prepare_data():
    """准备训练数据"""
    import subprocess

    # 创建数据目录
    os.makedirs("/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data", exist_ok=True)

    # 检查是否已经转换
    if os.path.exists(TRAIN_DATA):
        print(f"Training data already exists: {TRAIN_DATA}")
        return

    # 转换数据
    cmd = [
        "python", "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/scripts/convert_kie_data.py",
        "-i", *RAW_DATA_FILES,
        "-o", TRAIN_DATA,
        "-f", "parquet",
        "-v"
    ]
    print(f"Converting data: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def execute():
    """执行训练"""

    # 模型检查点
    ckpt_args = f"--hf-checkpoint {MODEL_PATH} "

    # Rollout 配置
    rollout_args = (
        f"--prompt-data {TRAIN_DATA} "
        "--input-key problem "
        "--label-key answer "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 100 "           # 训练轮数
        "--rollout-batch-size 16 "     # 每轮生成的样本数
        "--n-samples-per-prompt 4 "    # 每个 prompt 采样次数
        "--rollout-max-response-len 2048 "  # 最大响应长度
        "--rollout-temperature 0.7 "   # 采样温度
        "--global-batch-size 64 "      # 全局 batch size
    )

    # 多模态配置
    multimodal_args = '--multimodal-keys \'{"image": "images"}\' '

    # 自定义奖励函数 - KIE 专用
    reward_args = "--custom-rm-path slime.rollout.rm_hub.kie_reward.kie_reward "

    # FSDP 训练后端配置
    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
    )

    # GRPO 算法配置
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.01 "         # KL 散度损失系数
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.01 "         # 熵正则化
        "--eps-clip 0.2 "              # PPO clip 范围
        "--eps-clip-high 0.28 "
    )

    # 优化器配置
    optimizer_args = (
        "--optimizer adam "
        "--lr 5e-7 "                   # 学习率（VLM 建议较小）
        "--lr-decay-style cosine "     # 学习率衰减
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # SGLang 推理引擎配置
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 500 "
        "--sglang-enable-metrics "
        "--attn-implementation flash_attention_2 "
        "--sglang-cuda-graph-max-bs 32 "
    )

    # 保存配置
    save_args = (
        "--save /mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/internvl_kie "
        "--save-interval 10 "          # 每 10 轮保存一次
    )

    # 日志配置
    wandb_args = (
        "--wandb-project internvl-kie-grpo "
        "--wandb-name internvl3.5-4b-kie "
    )

    # GPU 配置
    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "                  # 训练和推理共用 GPU
    )

    # 组合所有参数
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{multimodal_args} "
        f"{reward_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{fsdp_args} "
        f"{sglang_args} "
        f"{save_args} "
        f"{wandb_args} "
        f"{misc_args} "
    )

    extra_env_vars = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    # 1. 准备数据
    prepare_data()

    # 2. 清理代理设置
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    # 3. 执行训练
    execute()
