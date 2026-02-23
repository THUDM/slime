# CI（持续集成）

slime 使用 GitHub Actions 进行 CI。测试通过 **PR label** 触发——给 PR 添加特定 label 即可运行对应的测试套件。

## 工作原理

工作流定义在 `.github/workflows/pr-test.yml`（由 `pr-test.yml.j2` 自动生成）。每个 CI 任务会：

1. 在自托管 GPU runner 上以 Docker 容器（`slimerl/slime:latest`）运行。
2. 通过 `pip install -e . --no-deps` 安装 slime。
3. 通过 `tests/ci/gpu_lock_exec.py --count <num_gpus>` 获取所需数量的 GPU。
4. 执行测试文件：`python tests/<test_file>.py`。

每个测试文件遵循统一的模式：`prepare()` 函数下载模型和数据集，`execute()` 函数构建命令行参数并调用 `U.execute_train(...)`。

## CI Labels

给 PR 添加 label 即可触发对应的测试套件：

| Label | Job | 说明 |
|---|---|---|
| `run-ci-short` | `e2e-test-short` | Qwen2.5-0.5B 轻量级冒烟测试（4 GPU），用于快速反馈。 |
| `run-ci-fsdp` | `e2e-test-fsdp` | FSDP 后端测试（true on-policy、VL、megatron-fsdp 对齐）。 |
| `run-ci-megatron` | `e2e-test-megatron` | 核心 Megatron 训练测试，覆盖 Dense、MoE、PPO、MTP、OPD 等。 |
| `run-ci-precision` | `e2e-test-precision` | 数值精度校验（并行一致性检查、megatron-fsdp 对齐）。 |
| `run-ci-ckpt` | `e2e-test-ckpt` | Checkpoint 保存/加载正确性（同步和异步保存）。 |
| `run-ci-image` | `e2e-test-image` | 在 `slimerl/slime-test:latest` 镜像上运行**全部**测试（用于镜像验证）。 |
| `run-ci-changed` | `e2e-test-changed` | **动态**检测 PR 中新增或修改的测试文件，仅运行这些测试。 |

所有 label 也可通过 `workflow_dispatch`（在 Actions 页面手动触发）来运行。

## 重点 Label 说明

### `run-ci-changed` — 仅运行新增或修改的测试

这是开发中最常用的 label。当你新增或修改了测试文件时，只需给 PR 添加 `run-ci-changed`，CI 会自动：

1. **检测**相对于 `origin/main` 新增或修改的 `tests/test_*.py` 文件（通过 `git diff --diff-filter=AM`）。
2. **提取**每个测试文件中的 `NUM_GPUS` 值。
3. **构建**动态 GitHub Actions matrix，并行运行每个测试。

这意味着你不需要手动在 workflow 中注册新测试——只需确保测试文件顶部有 `NUM_GPUS = <N>` 常量，`run-ci-changed` 就会自动识别并运行。

**示例**：如果你的 PR 新增了 `tests/test_qwen3_8B_opd_sglang.py`（其中 `NUM_GPUS = 8`），添加 `run-ci-changed` label 后会自动在 8 张 GPU 上运行该测试。

### `run-ci-image` — 在测试镜像上运行全部测试

这会在 `slimerl/slime-test:latest` Docker 镜像上运行**所有**已注册的测试。适用于：

- 验证新构建的 Docker 镜像是否可用。
- 在合并前做全面的测试检查。

由于包含所有测试，GPU 占用时间较长——日常开发请优先使用更有针对性的 label。

### `run-ci-megatron` — 核心 Megatron 测试

这是验证 Megatron 后端改动的主要 label，覆盖：

- Dense 模型：GLM4-9B、Qwen3-4B（PPO）
- MoE 模型：Qwen3-30B-A3B（有/无 DeepEP + FP8）、Moonlight-16B-A3B
- 特殊场景：MiMo-7B MTP、Qwen2.5-0.5B debug rollout-then-train、OPD（sglang teacher 模式）

所有测试使用 8 张 GPU。如果你正在修改 Megatron 训练逻辑、loss 计算或 checkpoint 转换，应该使用这个 label。

## 编写新测试

1. 创建 `tests/test_<your_test_name>.py`，遵循标准模式：

```python
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4  # 此常量会被 run-ci-changed 自动读取

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    # 按需下载数据集 ...

def execute():
    # 构建参数字符串并调用 U.execute_train(...)
    ...

if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
```

2. **快速验证**：直接推送测试文件，给 PR 添加 `run-ci-changed` label，测试会被自动检测并运行。

3. **注册到固定 label 组**：编辑 `.github/workflows/pr-test.yml.j2`，在对应 job 的 `tests` 列表中添加条目，然后重新生成：

```bash
cd .github/workflows && python generate_github_workflows.py
```

记得同时提交 `.j2` 和生成的 `.yml` 文件。

## Workflow 生成

工作流文件 `pr-test.yml` 是从 Jinja2 模板 `pr-test.yml.j2` 自动生成的。**不要直接编辑 `pr-test.yml`**。修改步骤：

1. 编辑 `.github/workflows/pr-test.yml.j2`。
2. 运行 `python .github/workflows/generate_github_workflows.py`。
3. 同时提交两个文件。
