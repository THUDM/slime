# Search-R1 lite

[English](./README.md)

这里是一个对 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的简单复现，以及是一个在 slime 中使用多轮对话和工具调用的样例。

## 配置环境

使用 `slimerl/slime:latest` 镜像，并初始化 Search-R1 需要的环境：

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .
# for Search R1
pip install chardet
```

请参照 Search-R1 中提供的脚本下载数据：

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/
python scripts/data_process/nq_search.py --local_dir /root/nq_search/
```

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
