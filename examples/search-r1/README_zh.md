# Search-R1 lite

[English](./README.md)

这里是一个对 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的简单复现，以及是一个在 slime 中使用多轮对话和工具调用的样例。

## 配置环境

使用 `slimerl/slime:latest` 镜像，并初始化 Search-R1 需要的环境：

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e . --no-deps
# for Search R1
pip install chardet
```

下载并准备训练数据：

```bash
cd /root/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/

# 设置工作目录
WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# 处理多个数据集的搜索格式训练文件
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# （可选）处理多个数据集的搜索格式测试文件
# 注意：最终文件未经过打乱
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

**注意：** 如果您计划使用本地搜索后端，请参阅[附录](#附录配置本地检索器)了解如何设置本地检索服务器。

初始化 Qwen2.5-3B 模型：

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

## 配置说明

### 搜索后端配置

`generate_with_search.py` 文件支持**本地搜索**和 **Google 搜索**两种后端。通过 `SEARCH_R1_CONFIGS` 字典进行配置：

```python
SEARCH_R1_CONFIGS = {
    # ============== 通用配置 ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,

    # ============== 搜索后端选择 ==============
    "search_backend": "local",  # 选项："local" 或 "google"

    # ============== 本地搜索配置 ==============
    # (仅当 search_backend="local" 时使用)
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # 本地检索服务器的 URL
        "proxy": None,
    },

    # ============== Google 搜索配置 ==============
    # (仅当 search_backend="google" 时使用)
    "google": {
        "api_key": "your_api_key_here",  # 替换为你的 serper.dev API key
        "snippet_only": True,
        "proxy": None,
    },

    # ============== 日志概率收集 ==============
    "return_logprob": True,  # 设置为 True 以收集日志概率（TIS 所需）

    # ============== 奖励模型配置 ==============
    "format_score": 0.2,
}
```

#### 使用本地搜索

1. 设置 `"search_backend": "local"`
2. 在 `"local"` 部分配置本地检索服务器 URL
3. 运行训练脚本前先启动本地搜索服务器

#### 使用 Google 搜索

1. 设置 `"search_backend": "google"`
2. 在 `"google"` 部分配置你的 serper.dev API key
3. 从 [serper.dev](https://serper.dev) 获取 API key

### 启用 TIS（轨迹重要性采样）

TIS 需要收集日志概率。启用 TIS 的步骤：

**1. 在 `generate_with_search.py` 中：**
```python
SEARCH_R1_CONFIGS = {
    # ... 其他配置
    "return_logprob": True,  # TIS 必须设置为 True
}
```

**2. 在 `run_qwen2.5_3B.sh` 中：**

在 `GRPO_ARGS` 中取消注释 TIS 相关参数：
```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # 取消注释以启用 TIS
   --use-tis
)
```

并在 `CUSTOM_ARGS` 中取消注释 TIS 配置路径：
```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # 取消注释以启用 TIS
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

**重要注意事项：**
- TIS 需要在 `SEARCH_R1_CONFIGS` 中设置 `return_logprob=True`
- 收集日志概率时，响应后处理会自动禁用以保持 token/logp 对齐
- TIS 会增加计算开销，但可以提高训练效率

## 运行脚本

```bash
cd slime/
bash examples/search-r1/run_qwen2.5_3B.sh
```

## 代码结构

为了实现多轮 + 工具调用，在 slime 中只需要实现一个自定义的数据生成函数，以及一个任务所需的 reward model，对应启动脚本中的这 2 个配置项：

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
```

也就是 `generate_with_search.py` 中的 `generate` 和 `reward_func` 两个函数。

## 附录：配置本地检索器

本节提供详细的本地密集检索器设置说明，用于本地搜索后端。

### 前置条件

本地检索器需要单独的 conda 环境，以避免与训练环境冲突。它使用 GPU 进行高效检索。

### 步骤 1：安装 Conda

如果您还没有安装 conda，运行以下命令：

```bash
# 下载并安装 conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc

# 接受 conda 服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 步骤 2：创建检索器环境

创建并激活一个 Python 3.10 的 conda 环境：

```bash
# 创建环境
conda create -n retriever python=3.10 -y
conda activate retriever

# 安装带 CUDA 支持的 PyTorch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装所需的包
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

### 步骤 3：下载索引和语料库

**注意：** 本地检索文件体积较大。下载需要约 **60-70 GB** 空间，解压后约 **132 GB**。请确保有足够的磁盘空间。

```bash
# 设置保存路径
save_path=/root/Index

# 下载索引和语料库文件
python /root/slime/examples/search-r1/local_dense_retriever/download.py --save_path $save_path

# 合并分割的索引文件
cat $save_path/part_* > $save_path/e5_Flat.index

# 解压语料库
gzip -d $save_path/wiki-18.jsonl.gz
```

### 步骤 4：启动本地检索服务器

```bash
# 如果遇到 "conda not found" 错误，运行：
# source ~/miniconda3/etc/profile.d/conda.sh
# conda init
# source ~/.bashrc

# 激活检索器环境
conda activate retriever

# 设置路径
save_path=/root/Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# 启动检索服务器
python /root/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
```

**重要注意事项：**
- 首次启动会下载模型和加载索引，可能需要几分钟
- 正常启动时间（不包括下载）：1-2 分钟
- 每张 GPU 显存占用：约 5-7 GB
- 本地搜索引擎的 Python 进程不会随着 shell 关闭而终止
- 重启服务器：使用 `lsof -i :8000` 找到 PID，然后 kill 并重启

### 步骤 5：启动训练

确保您**不在** retriever conda 环境中。如果在，请运行 `conda deactivate`。

```bash
cd /root/slime

# 设置您的 wandb key（可选）
export WANDB_KEY="your_wandb_key_here"

# 如果 ray 进程卡住，尝试：
# rm -rf /root/.cache
# rm -rf /root/.*

# 运行训练脚本
bash /root/slime/examples/search-r1/run_qwen2.5_3B.sh
```

### 故障排查

**Ray 进程卡住：**
```bash
rm -rf /root/.cache
# 如果仍然卡住：
rm -rf /root/.*
```

**Conda 环境问题：**
- 确保在运行训练前退出 retriever 环境
- 验证训练使用的是基础 Python 环境

**检索服务器无响应：**
- 检查服务器是否运行：`lsof -i :8000`
- 验证 GPU 可用性：`nvidia-smi`
- 检查日志查看错误信息
