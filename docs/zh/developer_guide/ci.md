# CI（持续集成）

slime CI 分成两层：

1. **默认运行的 CPU 正确性测试**：每个 PR、每次 push 到 `main`、以及手动 `workflow_dispatch` 都会运行。
2. **通过 label 触发的 GPU end-to-end 测试**：在自托管 GPU runner 上验证真实的 Megatron + SGLang training/rollout 路径。

这个拆分是有意为之。大部分 correctness invariant 应该在不等待 GPU 集群的情况下快速检查；真正依赖完整训练和 rollout 的行为，则由 GPU e2e job 覆盖。

## 工作原理

workflow 定义在 `.github/workflows/pr-test.yml`，它由 `.github/workflows/pr-test.yml.j2` 自动生成。

### CPU Jobs

CPU job 运行在 GitHub-hosted `ubuntu-latest` runner 上：

- `cpu-unittest` 安装 CPU PyTorch 和轻量依赖，然后通过 `python tests/<test_file>.py` 运行注册的 unit/contract tests。
- `agent-adapter-test` 使用同样的方式运行 agent adapter tests，并额外安装 `openai`、`openai-agents`、`anthropic` 等 SDK 依赖。

CPU job 不使用 Docker，不申请 GPU，也不会调用 `tests/ci/gpu_lock_exec.py`。

### GPU E2E Jobs

GPU job 运行在自托管 GPU runner 上。每个 job 会：

1. 启动 Docker container，通常使用 `slimerl/slime:latest`；镜像验证使用 `slimerl/slime-test:latest`。
2. 通过 `pip install -e . --no-deps` 安装 slime。
3. 通过 `tests/ci/gpu_lock_exec.py --count <num_gpus>` 申请所需 GPU。
4. 执行注册的测试文件：`python tests/<test_file>.py`。

GPU 测试通常遵循 e2e 模式：`prepare()` 下载模型和数据集，`execute()` 构建 CLI 参数并调用 `U.execute_train(...)`。

### Changed-Test Job

`run-ci-changed` 会动态检测相对于 `origin/main` 新增或修改的 `tests/test_*.py` 和 `tests/plugin_contracts/test_*.py` 文件。

对每个 changed test file，它会读取文件顶层的 `NUM_GPUS = <N>` 常量并构建 matrix。如果缺少 `NUM_GPUS`，CI 会默认使用 `8`，因此 CPU-only test 应该声明：

```python
NUM_GPUS = 0
```

changed-test job 本身走 self-hosted Docker 路径。当 `NUM_GPUS = 0` 时，它会直接运行测试，不申请 GPU。

## CI Jobs 与触发方式

| Trigger | Job | 类型 | 说明 |
|---|---|---|---|
| 自动运行 | `cpu-unittest` | CPU | 默认运行的 unit/contract tests，覆盖 argument validation、schedule、reward、sample、rollout validation、checkpoint utilities 和 plugin contracts。 |
| 自动运行 | `agent-adapter-test` | CPU | 默认运行的 agent adapter tests，包含额外 provider SDK 依赖。 |
| `run-ci-sglang-config` | `e2e-test-sglang-config` | GPU | SGLang config 测试，覆盖高级 rollout engine deployment 和 mixed/offload 场景。 |
| `run-ci-megatron` | `e2e-test-megatron` | GPU | 核心 Megatron 训练测试，覆盖 dense、MoE、PPO、MTP、OPD、async rollout、PD/Mooncake 和 debug replay 路径。 |
| `run-ci-precision` | `e2e-test-precision` | GPU | 数值精度和并行一致性检查。 |
| `run-ci-ckpt` | `e2e-test-ckpt` | GPU | Checkpoint save/load 正确性，包括 CPU/GPU optimizer state 和 async save。 |
| `run-ci-image` | `e2e-test-image` | GPU | 在 `slimerl/slime-test:latest` 上运行与 `run-ci-megatron` 相同的 matrix。 |
| `run-ci-changed` | `e2e-test-changed` | Mixed | 只运行 changed tests，并使用每个文件中的 `NUM_GPUS`。 |

也可以在 Actions 页面通过 `workflow_dispatch` 手动验证；它会按照 workflow 条件运行注册的 jobs。

## CPU Unit Tests

CPU suite 是 correctness 的第一道防线，用来在进入昂贵 GPU run 之前捕获 silent RL infrastructure bugs。

当前注册的 CPU suite 覆盖：

- Megatron argument 和 HF config validation；
- DP/CP scheduling utilities 和 CP loss invariance；
- metric reporting 和 distributed metric aggregation；
- math、GPQA、F1、DeepScaler、DAPO-style math 等 reward-model grading utilities；
- `Sample` 行为、rollout validation 和 agent trajectory merging；
- HF checkpoint saver 行为；
- rollout function、generate function、runtime hook 和 path loading 的 customization hook contracts。

Agent adapter tests 单独放在一个 CPU job 中，因为它们需要额外 SDK 依赖。

常用本地命令：

```bash
python tests/test_agent_trajectory.py
python -m pytest tests/test_megatron_argument_validation.py tests/plugin_contracts/test_plugin_generate_contracts.py
```

## GPU E2E Tests

GPU e2e tests 验证 CPU tests 无法覆盖的集成训练/rollout 行为：

- `run-ci-sglang-config`：高级 SGLang deployment path，包括 config-based engine layouts。
- `run-ci-megatron`：主要 Megatron backend coverage，包括 dense/MoE recipe、async rollout、OPD、PPO-style path、PD/Mooncake 和 debug rollout-then-train replay。
- `run-ci-precision`：不同并行设置下的数值一致性。
- `run-ci-ckpt`：checkpoint save/load 组合和 async save。
- `run-ci-image`：与 `run-ci-megatron` 相同的 matrix，但运行在 release/test image 上。

日常 PR 优先使用 targeted labels。`run-ci-image` 消耗 GPU 时间较多，应谨慎使用。

## 编写新测试

### CPU Tests

对于 CPU-only tests：

1. 按照附近文件的模式，将测试放在 `tests/test_*.py`、`tests/utils/test_*.py` 或 `tests/plugin_contracts/test_*.py` 下。
2. 如果这个文件可能被 `run-ci-changed` 运行，添加顶层 `NUM_GPUS = 0`。
3. 让文件可以直接执行：

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

4. 如果测试需要永久进入 CI matrix，在 `.github/workflows/pr-test.yml.j2` 的 `cpu-unittest` 或 `agent-adapter-test` job 中注册，然后重新生成 workflow。

### GPU E2E Tests

对于 GPU e2e tests：

1. 创建 `tests/test_<your_test_name>.py`，遵循已有的 `prepare()` / `execute()` 模式。
2. 用 `NUM_GPUS = <N>` 声明所需 GPU 数量。
3. 在 `prepare()` 中下载所需模型和数据集。
4. 在 `execute()` 中构建参数并调用 `U.execute_train(...)`。
5. 在 `.github/workflows/pr-test.yml.j2` 的合适 GPU job 中注册测试，然后重新生成 workflow。

示例骨架：

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

## Workflow 生成

workflow 文件 `pr-test.yml` 由 Jinja2 模板 `pr-test.yml.j2` 自动生成。不要直接编辑 `pr-test.yml`。

如果要修改固定 CI matrix：

1. 编辑 `.github/workflows/pr-test.yml.j2`。
2. 运行：

```bash
python .github/workflows/generate_github_workflows.py
```

3. 同时提交 `.github/workflows/pr-test.yml.j2` 和生成的 `.github/workflows/pr-test.yml`。

## PR 应该选择哪些检查

- 纯 argument parsing、reward、schedule、sample、trajectory 或 hook-contract 改动：优先依赖 CPU tests。
- SGLang topology 或 rollout engine deployment 改动：使用 `run-ci-sglang-config`。
- Megatron training、loss、checkpoint conversion 或 model recipe 改动：使用 `run-ci-megatron`；必要时加 `run-ci-precision` 或 `run-ci-ckpt`。
- Docker image 或 dependency 改动：使用 `run-ci-image`。
- 新增或修改测试：使用 `run-ci-changed` 做 targeted validation。
