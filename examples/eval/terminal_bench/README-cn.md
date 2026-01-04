# Terminal Bench 评估集成

本目录将 Terminal Bench (TB) 封装为 Slime 的评估委托（Eval Delegate）。评估过程在宿主机（Host）上通过 `tb` CLI 执行，Slime 负责读取并汇总各项指标，包括 `accuracy`、`n_resolved`、`n_unresolved`、`pass_at_k/*` 以及 Token 统计数据（如 `total_input_tokens_mean/median` 和 `total_output_tokens_mean/median`）。

## 运行架构

* **Slime 内部**：运行训练/评估主循环；调用 TB delegate client。
* **宿主机（Host）**：运行 TB delegate server (`tb_server.py`)，由其执行 `tb run ...`。
* **Server逻辑**：读取最新的 TB JSON 结果并将各项指标返回给 Slime。

## 1) 获取代码 (宿主机)

```bash
mkdir slime-tb
cd slime-tb
git clone https://github.com/THUDM/slime.git
git clone https://github.com/laude-institute/terminal-bench
```

## 2) 启动 Slime 容器

```bash
docker run \
  -itd \
  --gpus all \
  --shm-size 32g \
  --network host \
  --ipc=host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -v /mnt/data/.cache:/root/.cache \
  -v $(pwd):/shared/slime-tb \
  --name <slime_container_name> \
  slimerl/slime:latest \
  /bin/bash
```

## 3) 进入 Slime 容器

```bash
docker exec -it <slime_container_name> /bin/bash
```

## 4) 配置 Terminal Bench 环境 (宿主机)

在运行 `tb_server.py` 的宿主机上执行：

```bash
# 在宿主机终端执行（非 Docker 内部）
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install terminal-bench/.
uv pip install -r slime/examples/eval/terminal_bench/requirements.txt
```

*如果仓库路径不是 `./slime` 和 `./terminal-bench`，请根据实际路径调整。*

## 5) 启动 Terminal Bench server

在宿主机上启动（即 `tb` 命令可用的环境）：

```bash
python slime/examples/eval/terminal_bench/tb_server.py \
  --host 0.0.0.0 --port 9051 \
  --output-root tb_eval_output
```

**该脚本的功能：**

* 默认设置 `OPENAI_API_KEY=EMPTY`。
* 执行 `tb run -a terminus-2 -m openai/<model> ... --n-concurrent 8`。
* 等待运行完成后，返回 `accuracy`、`pass_at_k` 以及 Token 消耗等统计数据。

## 6) 运行评估脚本 (示例)

如果使用提供的 Qwen 评估启动脚本 (`run-eval-tb-qwen.sh`)，请按以下步骤操作：

**更新路径**：将 `eval_tb_example.yaml` 中的 `dataset_path` 修改为宿主机上 `terminal-bench/tasks` 的**绝对路径**（注意不是 Docker 内部路径）。

**下载模型**：在 Slime 容器内下载 HuggingFace 权重：
```bash
huggingface-cli download open-thoughts/OpenThinker-Agent-v1 \
--local-dir /root/.cache/OpenThinker-Agent-v1
```

**格式转换**：将 HuggingFace 权重转换为 Slime 的 torch distributed 格式。在 Slime 根目录下执行：
```bash
cd /shared/slime-tb/slime
source scripts/models/qwen3-8B.sh

export PYTHONPATH=/root/Megatron-LM:/shared/slime-tb/slime

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/.cache/OpenThinker-Agent-v1 \
  --save /root/.cache/OpenThinker-Agent-v1_torch_dist
```

**开始评估**：在 Slime 容器内运行：
```bash
bash slime/examples/eval/scripts/run-eval-tb-qwen.sh 2>&1 | tee run.log
```

*为了快速测试，可以在 `eval_tb_example.yaml` 中通过 `task_ids` 指定特定任务，或通过 `n_tasks` 限制评估任务的数量。*

## 7) 常见问题

当在 Docker 容器中使用 `--network host` 运行 Slime 时，Ray 可能由于与宿主机共享网络而出现端口冲突。

这会导致 Ray 启动失败，或报 Redis/会话相关错误。通常可以在启动 Ray head 时显式指定未占用端口来解决，比如设置非默认的 `--port` 和 `--dashboard-port`。

有时甚至会导致 Ray job 提交失败，提示没有可用 agent 接受任务。这通常是 dashboard agent 或 runtime env agent 的端口也发生冲突。此时可在启动 Ray 时指定这些端口（如 `--dashboard-agent-listen-port`、`--dashboard-agent-grpc-port`、`--runtime-env-agent-port`）来解决。

如果 TB server无法通过 sglang router 连接到 Slime（`InternalServerError`），请检查 router 端口（例如 30005）实际监听的地址，并更新 `eval_tb_example.yaml` 中的 `api_base`：

```bash
ss -lntp | grep 30005
```

TB server开始接受请求后，可能会在输出中看到 `Parser warnings`、`Context length exceeded`、`Command 1 should end with newline`、`Harness execution failed`等。这些是Terminal Bench 的警告，如果正常运行可以忽略。