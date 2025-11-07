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

## 运行脚本

需要将你的 serper.dev API 配置在 `generate_with_search.py` 中：

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 3,
    "topk": 3,
    "google_api_key": "YOUR_API_KEY",  # Replace with your actual API key
    "snippet_only": True,  # Set to True to only return snippets
    "proxy": None,  # Set to your proxy if needed
    "search_concurrency": 256,
    # rm
    "format_score": 0.2,
}
```

并运行：

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
