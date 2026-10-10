# CI（持续集成）

提交或更新 PR、向 `main` 推送代码，以及手动触发工作流时，slime 都会运行 CPU 测试。GPU 端到端测试通过 PR 标签触发，在自托管的 GPU 机器上运行真实的 Megatron 训练和 SGLang 推理。日常改动可以先用 CPU 测试检查，再按改动范围选择 GPU 测试。

## 如何触发测试

| 触发方式 | CI 任务 | 运行环境 | 覆盖范围 |
|---|---|---|---|
| 自动运行 | `cpu-unittest` | CPU | 参数校验、批次调度、指标、奖励计算、样本处理、checkpoint 工具和扩展接口。 |
| 自动运行 | `agent-test` | CPU | Agent 适配器，额外安装所需的模型服务 SDK。 |
| `run-ci-sglang-config` | `e2e-test-sglang-config` | CPU/GPU | SGLang 部署配置，包括多模型、不同引擎布局和显存卸载后的故障恢复。 |
| `run-ci-megatron` | `e2e-test-megatron` | GPU | Megatron 训练，包括 Dense、MoE、PPO、MTP、OPD、全异步 rollout、PD/Mooncake、调试数据重放和 codex/sunabako agent 训练。 |
| `run-ci-precision` | `e2e-test-precision` | CPU/GPU | 数值精度，以及不同并行配置下的结果一致性。 |
| `run-ci-ckpt` | `e2e-test-ckpt` | GPU | Checkpoint 保存和加载，包括 CPU/GPU 优化器状态和异步保存。 |
| `run-ci-image` | `e2e-test-image` | GPU | 在 `slimerl/slime-test:latest` 镜像上运行与 `run-ci-megatron` 相同的测试。 |
| `run-ci-changed` | `e2e-test-changed` | CPU/GPU | 只运行本次新增或修改的测试，GPU 数量由文件中的 `NUM_GPUS` 决定。 |

也可以在 GitHub Actions 页面通过 `workflow_dispatch` 手动运行。手动触发会运行已注册的 CPU 和 GPU 任务；向 `main` 推送代码只自动运行 CPU 任务。

工作流定义在 `.github/workflows/pr-test.yml`，由 `.github/workflows/pr-test.yml.j2` 生成。修改测试列表或触发条件时，应编辑模板，再生成工作流文件。

## 测试如何运行

### CPU 测试

CPU 任务运行在 GitHub 托管的 `ubuntu-latest` 环境中，安装 CPU 版 PyTorch 和测试依赖，再执行 `python tests/<test_file>.py`。它们不使用 Docker，也不申请 GPU。

`cpu-unittest` 主要检查：

- Megatron 参数和 Hugging Face 模型配置是否合法；
- DP/CP 批次调度，以及 CP 划分前后的 loss 是否一致；
- 指标上报和分布式指标汇总；
- math、GPQA、F1、DeepScaler、DAPO 等奖励计算；
- `Sample`、rollout 数据校验和 agent 轨迹合并；
- Hugging Face checkpoint 保存，以及自定义 rollout、生成函数和运行时 hook 的接口约定。

Agent 适配器测试放在独立的 `agent-test` 任务中，因为它们还需要 `openai`、`openai-agents`、`anthropic` 等 SDK。

CPU 测试列表中带有 `straw: true` 的条目，会从 PyPI 安装最新版 `straw-queue` wheel。测试不需要 Rust 工具链或 straw 源码仓库。`test_optional_straw.py` 则刻意不安装 straw，检查默认数据传输仍可运行，以及显式选择 straw 时是否给出清晰的安装提示。

常用的本地运行方式：

```bash
python tests/test_agent/test_trajectory_manager_branching.py
python -m pytest tests/test_megatron_argument_validation.py tests/plugin_contracts/test_plugin_generate_contracts.py
```

### GPU 端到端测试

GPU 任务运行在自托管机器上，每项测试依次执行以下步骤：

1. 启动 Docker 容器，通常使用 `slimerl/slime:latest`；镜像验证使用 `slimerl/slime-test:latest`。
2. 从 PyPI 安装最新版 `straw-queue` wheel，并通过 `pip install -e . --no-deps` 安装当前版本的 slime。
3. 通过 `tests/ci/gpu_lock_exec.py --count <num_gpus>` 申请所需 GPU。
4. 执行 `python tests/<test_file>.py`。

测试文件通常用 `prepare()` 下载模型和数据集，用 `execute()` 构建训练参数并调用 `U.execute_train(...)`。

### 只运行改动的测试

添加 `run-ci-changed` 标签后，CI 会相对于 `origin/main` 查找新增或修改的 `tests/test_*.py` 和 `tests/plugin_contracts/test_*.py`，并为每个文件创建一个测试任务。

GPU 数量取自文件顶层的 `NUM_GPUS = <N>`。没有声明时默认申请 8 张卡，因此只需要 CPU 的测试应写明：

```python
NUM_GPUS = 0
```

这类任务仍在自托管机器的 Docker 容器中执行，但 `NUM_GPUS = 0` 时不会申请 GPU。

## 数据传输与恢复测试

### Agent 训练

`tests/test_agent_sunabako_codex_e2e.py` 使用 **Codex 0.162.1**、**Claude Code 2.1.296**、
Qwen3.8-27B 和两个独立镜像中的小米任务：

- `format-code-task-000003`：给 smol-evm 补上缺失的 `SHL` 左移指令。
- `format-code-task-000045`：让 BootBot 保留对象形式的 quick replies。

每一步都使用这两道题，**每题 4 条，共 8 个独立 agent 并行**，分别在新沙箱中评分，
然后执行一次 GRPO 优化器更新。**第一步使用 Codex，第二步使用 Claude Code**，
第二步使用更新后的模型，对相同两道题重新采样。两个 CLI 共用 trace 拼接和训练路径。
总共运行 16 条轨迹、训练 2 步，不按 reward 过滤或额外重采。
两道题的原始代码都必须无法通过官方测试；失败轨迹保留 reward 0，与 reward 1 的轨迹
一起参与训练。每道题的 4 条结果独立归一化，使用组内均值与样本标准差加 epsilon；
整组同分时，advantage 正常为 0。

测试会从实际训练张量中核对每个 token 的 advantage，并记录 8 个训练 rank 上的两次
优化器调用。梯度与参数必须有限；非零梯度必须带来参数变化。采样使用 temperature=1、
`top_p=0.95`，禁用 top-k，启用 SC，并校验原始 token、mask、sampler 概率及 nucleus
replay 数据。Qwen3.8-27B 是 dense 模型，因此关闭 R3。
推理通过 SGLang EAGLE 启用 checkpoint 中的 MTP head（3 个 draft step、4 个 draft token），
并检查日志中确实出现已接受的 draft token。
上下文额度为 64K，并给 draft token 预留空间。Claude Code 使用与 example 相同的
六个代码工具：Bash、Read、Edit、Write、Glob 和 Grep。

每个 agent 上限 600 秒，GitHub job 连同准备阶段上限 35 分钟。
连续的模型调用会保留原始 token 和采样分布，合并成训练轨迹；历史实际改写时才分支。
CI 与正式 example 一样，训练每条真实轨迹中的全部片段，保持 token、logprob 和
reward 原样；`agents/<sample-index>/agent-full.pt` 保存完整轨迹。
agent 只收到原始题目和仓库，隐藏测试保留在独立评分沙箱中。

模型、选中的两个任务镜像、两个 CLI 安装包和 TileLang/Triton 编译结果均使用缓存。
镜像下载支持断点续传与完整性校验，`--prepare-only` 在申请 GPU 锁之前准备资源。
首次下载可能超过计时 CI 的上限。代理变量会传入容器，本机 Ray 和沙箱流量绕过代理。
`run-ci-megatron` 包含这项 GPU e2e，`run-ci-changed` 也能发现它，自动 `agent-test`
任务覆盖 CPU 合约检查。

`tests/ci/setup_agent_e2e.sh` 按 `examples/coding_agent_rl/requirements-sunabako.txt`
安装 PyPI 发布的 `sunabako==0.1.1` wheel。该版本通过 `uid_range_size` 支持 native 用户，
本地节点为 8 个沙箱分别保留 65,536 个 UID/GID，
state 挂载到 `/workspace`，保证映射后的用户能遍历父目录。测试在现有特权 Docker
容器中运行，无需内层 Docker daemon 或 PRoot。RSS 模式只用于有界功能验证，
**不证明总内存硬限制**；生产环境仍需可写、已委派的 cgroup，缺失时拒绝启动。

已有沙箱集群时，安装相同 requirements 后运行：

```bash
HF_CHECKPOINT=/path/to/Qwen3.8-27B \
RAY_AUTH_MODE=token \
SUNABAKO_CLUSTER=/path/to/cluster.json \
SUNABAKO_IMAGES=/path/to/images.json \
ADAPTER_PUBLIC_HOST=<training-node-ip> \
python tests/test_agent_sunabako_codex_e2e.py
```

镜像映射必须在每台沙箱节点包含两个任务，并提供至少 8 个并发沙箱的容量。只在无硬 cgroup 的功能测试中，显式设置
`SUNABAKO_ALLOW_TEST_MEMORY=1`。可用 `SLIME_AGENT_TEST_DATA` 复用已下载的小米 parquet
和镜像映射，用 `SLIME_AGENT_CODEX_NATIVE_TARBALL` 和 `SLIME_AGENT_CC_NATIVE_TARBALL`
复用官方平台安装包，无需在沙箱中下载工具链。
`SLIME_AGENT_TEST_RUN_DIR` 必须是新目录，会保留 CLI 日志、评分输出、rollout/train
张量、参数更新证据与 `result.json`。测试与其他 E2E 一样调用 `U.execute_train()`，
通过 `extra_env_vars` 传入 agent 环境。Ray CLI 直接向 GitHub Actions 控制台输出训练日志；
训练结束后将该任务日志保存为 `train.log`，用于检查 MTP 和训练指标。
GitHub Actions 在成功或失败后都会上传这些产物。

### straw

使用 straw 的端到端测试会显式设置 `--rollout-data-transport straw`，确保本地执行、`run-ci-changed` 和固定测试列表使用同一种传输方式。覆盖范围包括：

- R3：`test_qwen3_30B_A3B_r3.py` 和 `test_moonlight_16B_A3B_r3.py`。
- SC：`test_qwen2.5_0.5B_score_centering.py`，分别检查 top-k 和 top-p 数据。
- 全异步 rollout、fanout、PPO、MTP、PD/Mooncake、分布式 SGLang、混合显存卸载后的故障恢复、调试数据重放，以及释放训练资源后继续 rollout。
- Checkpoint 保存与加载：不同阶段共用 straw 存储池，检查队列和训练状态能否恢复。`test_straw_checkpoint_fork.py` 还检查指定恢复步骤、多次回退、自动选择分支和调试数据重放。

R3、SC 和全异步测试同时开启在线 GC。普通 straw 测试使用独立的临时目录，并在结束后清理。单机 GPU 测试使用本地文件系统配置；多机 JuiceFS 的持久性需要单独验证。其他端到端测试继续覆盖 Ray object-store 和 NIXL 传输。

`test_straw_fully_async_recovery.py` 是自动运行的 CPU 集成测试。它用 SIGKILL 终止一个包含两个本地 Ray 节点的任务，再启动新进程，从同一个文件系统队列恢复；在线 GC 开启和关闭两种情况都会验证。测试使用小规模 R3/SC 数据，以及结果固定的模拟推理和打分函数。安装兼容的 straw wheel 后，可以本地运行：

```bash
PYTHONPATH=. python tests/test_straw_fully_async_recovery.py
```

### PipelineRL

`test_qwen2.5_0.5B_pipeline_rl.py` 使用 4 张 GPU，运行全异步 rollout 和三个实际的 GRPO 训练步骤。它检查同一个 HTTP 请求能否在权重更新期间继续生成，也检查训练是否确实改变了策略权重。

固定测试列表包含三种配置：`--flush-cache-interval 0` 配合 NCCL 或完整权重落盘同步，以及 `--flush-cache-interval 2` 配合 NCCL 定期刷新缓存。CPU 测试 `test_pipeline_rl.py` 检查刷新周期和发给 SGLang 的控制请求。这些测试检查功能是否正确，不评估学习效果或吞吐提升。

### Megatron 手动重启

`test_qwen2.5_0.5B_training_recovery.py` 使用 4 张 GPU，在同一个 Ray 集群中先后运行两次训练任务。第一次使用 TP=1，故意触发真实的 CUDA OOM；确认训练任务退出后推理服务仍能响应，再改成 TP=2 重新提交，此时 DP 大小也会改变。

测试检查是否复用了健康的 SGLang 进程、路由器和 GPU 资源，重放的批次内容是否一致，以及训练调度器进度、非零且有限的梯度和最终 checkpoint。固定测试列表包括：

| 数据保存方式 | RolloutManager 状态 | 检查内容 |
|---|---|---|
| straw，开启在线 GC | 保持存活 | 重新连接训练进程，重放已经训练但尚未保存到 checkpoint 的批次。 |
| straw，开启在线 GC | 失败后被杀掉 | 新 manager 接回原推理集群，并重放同样的批次。 |
| straw，已有模型和优化器 checkpoint，使用 Megatron YAML 配置 | 训练过程中被杀掉 | 从 checkpoint 恢复，核对配置与恢复状态。 |
| Rollout 调试文件 | 失败后被杀掉 | 从调试文件恢复数据，新 manager 接回原推理集群。 |
| straw，使用 disk-delta 同步权重 | 失败后被杀掉 | 以恢复后的权重发布新的完整基准，再继续 delta 更新。 |
| straw，使用 PD/Mooncake 推理 | 失败后被杀掉 | 卡住 prefill actor，在连接重置超时后替换它，并保留健康的 decode actor。 |

`test_qwen3_30B_A3B_training_recovery.py` 使用 8 张 GPU，在 MoE 模型、R3 和 stateless Adam 配置下覆盖 OOM、checkpoint 恢复及 manager 丢失。它不保存优化器张量，但会检查 scheduler 进度，比对 TP/DP 改变前后的持久化路由字节，并在恢复后完成训练。Dense 测试覆盖普通 Adam 的优化器 checkpoint 恢复。

无论是否传入兼容参数 `--use-fault-tolerance`，内部推理健康检查都会启用。CPU 测试还包括：`test_training_recovery.py` 的配置与 checkpoint 边界检查，`test_disk_delta_recovery.py` 的权重更新应答丢失，以及 `test_rollout_manager_recovery.py` 中真实 Ray manager 的 SIGKILL 和转换应答丢失。

### Rollout 收尾时清理失效引擎

`test_qwen2.5_0.5B_rollout_health.py` 在 rollout 收尾前停掉真实的 SGLang HTTP 服务，但保留它在路由器中的注册信息。两个四卡场景分别保留或杀掉对应的 Ray actor，检查收尾是否有超时限制、返回训练前是否注销失效服务、更新权重时能否恢复引擎，以及训练能否保存最终 checkpoint。

两种情况都不传 `--use-fault-tolerance`，并将后台检查间隔和首次等待时间设为 600 秒，以验证收尾检查会立即执行，不必等待后台定时检查。

## 添加测试

### 添加 CPU 测试

参考相邻文件，将测试放在 `tests/test_*.py`、`tests/utils/test_*.py` 或 `tests/plugin_contracts/test_*.py` 下。如果文件会被 `run-ci-changed` 运行，需要声明顶层 `NUM_GPUS = 0`。

CI 会直接执行测试文件，因此使用 pytest 的文件需要提供入口：

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

需要自动运行的测试，还应注册到 `.github/workflows/pr-test.yml.j2` 的 `cpu-unittest` 或 `agent-test` 列表中，再重新生成工作流。

### 添加 GPU 端到端测试

1. 创建 `tests/test_<your_test_name>.py`，沿用现有的 `prepare()` / `execute()` 结构。
2. 用顶层 `NUM_GPUS = <N>` 声明所需 GPU 数量。
3. 在 `prepare()` 中下载模型和数据集。
4. 在 `execute()` 中构建参数并调用 `U.execute_train(...)`。
5. 将测试注册到 `.github/workflows/pr-test.yml.j2` 中合适的 GPU 任务，再重新生成工作流。

示例：

```python
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")

def execute():
    # 构建参数字符串并调用 U.execute_train(...)
    ...

if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
```

## 生成工作流

不要直接编辑生成的 `.github/workflows/pr-test.yml`。修改 `.github/workflows/pr-test.yml.j2` 后，运行：

```bash
python .github/workflows/generate_github_workflows.py
```

提交时要同时包含模板和生成后的工作流文件。

## PR 选择哪些检查

- 参数解析、奖励计算、批次调度、样本、轨迹或扩展接口改动：先运行对应的 CPU 测试。
- SGLang 部署或推理引擎布局改动：添加 `run-ci-sglang-config`。
- Megatron 训练、loss、checkpoint 转换或模型训练配置改动：添加 `run-ci-megatron`；涉及数值或保存恢复时，再加 `run-ci-precision` 或 `run-ci-ckpt`。
- Docker 镜像或依赖改动：添加 `run-ci-image`。它运行完整的 Megatron 测试列表，消耗的 GPU 时间较多。
- 新增或修改测试：添加 `run-ci-changed`，直接验证改动的测试文件。
