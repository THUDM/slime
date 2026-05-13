# `sandbox_env`

`slime` 在 SWE-rebench 和 `inspire-sandbox` 上执行训练 rollout 的实现目录。

Scaffold 协议（command_factory + template build）已经独立到
`/avalanche/share_workspace/agentic_protocol/`，eval 和 train 共享同一份代码，
不再有路径/版本分叉。

## 主路径

```
SWE-rebench task JSON
        │
        ├──► agentic_protocol.template_build.build_swe_rebench
        │       构建 scaffold-enabled template manifest（共享入口）
        │
        ├──► build_rebench_runtime_data.py
        │       生成 slime 训练/验证 JSONL，并把 template/user/env 写入 metadata
        │
        └──► slime/train_async.py
                --rollout-function-path examples.sandbox_env.swe_rollout.generate_rollout
                        │
                        ├── inspire_sandbox.Sandbox.create
                        ├── sandbox 内 wstunnel server 启动
                        ├── host 侧 wstunnel client 暴露本地 model proxy
                        ├── SWE_AGENT_HARNESS 选择 scaffold 执行
                        └── rebench.py 运行 eval 并计算 reward
```

## 关键配置

| 变量 | 说明 |
|---|---|
| `SWE_AGENT_HARNESS` | scaffold 选择，支持 `qwen_code`、`claude_code`、`openhands`、`open_code`、`opencode`、`codex`、`codex_cli`。默认 `qwen_code`。 |
| `AGENTIC_PROTOCOL_ROOT` | sandbox 内 scaffold 根目录。默认 `/__avaeval_agentic_protocol_v1__`（与 zf 一致）。重新 build template 后才能切换。 |
| `SWE_MODEL_PROXY_PORT` | sandbox 内 OpenAI-compatible model proxy 端口，默认 `30001`。 |
| `SWE_WSTUNNEL_SERVER_PORT` | sandbox 内 wstunnel server 端口，默认 `19090`。 |
| `SWE_GROUP_CONCURRENCY` | rollout group 并发数量，默认等于 `SWE_ROLLOUT_BATCH_SIZE`。 |
| `SWE_SAMPLE_CONCURRENCY` | sample 并发数量，默认等于 `SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT`。 |
| `SWE_LOG_ROOT` | 每个 sample 的 sandbox/eval/tunnel 日志目录。 |

## PYTHONPATH 接入

slime 训练脚本会把 `${AVALANCHE_ROOT}/share_workspace` 注入到 ray runtime
env 的 `PYTHONPATH`，使 `import agentic_protocol.command_factory` 在 rollout
worker 中可用。本地手动跑工具脚本时也需要：

```bash
export PYTHONPATH=/avalanche/share_workspace:${PYTHONPATH:-}
```

## 推荐阅读顺序

1. `swe_rollout.py`：训练入口，只看 Slime 如何调度 rollout。
2. `sandbox_runtime.py`：查看 `SWE_*` 配置、sandbox env、日志、sandbox IO 和单样本生命周期。
3. `sglang_openai_proxy.py`：查看 scaffold 请求如何转成 SGLang 生成，并如何记录训练轨迹。
4. `rebench.py`：查看 reward 和 log parser 规则。
5. `agentic_protocol/` (在 `/avalanche/share_workspace/`)：scaffold 命令、template install、layout。

## 文件职责

按依赖方向自上而下排列。

### Slime-aware

| 文件 | 职责 |
|---|---|
| `swe_rollout.py` | Slime 注册入口：维护 tokenizer/mask 状态、调度 rollout group、把单样本结果转换成 `Sample`。不直接操作 sandbox。 |

### Rollout runtime

| 文件 | 职责 |
|---|---|
| `sandbox_runtime.py` | sandbox 运行时总入口：`SWE_*` 配置、日志、Inspire sandbox IO、wstunnel、scaffold preflight 和单样本生命周期。无 Slime/torch 依赖；scaffold 命令来自共享 `agentic_protocol.command_factory`。 |
| `sglang_openai_proxy.py` | host 侧 OpenAI-compatible proxy；把 scaffold 的请求转成 SGLang 生成，处理工具调用协议，并记录训练轨迹。 |
| `rebench.py` | rebench 任务的纯逻辑（eval 脚本渲染、log parser、`FAIL_TO_PASS` / `PASS_TO_PASS` 评分）+ sandbox 内 IO 层。 |

### Data and tools

| 文件 | 职责 |
|---|---|
| `build_rebench_runtime_data.py` | 把 SWE-rebench task JSON 转成 slime 可消费的 JSONL，注入 template alias / image user / image env。 |
| `tools/mini_rollout_test.py` | 不调用模型，直接应用 gold patch 并运行 rebench eval，用于验证 sandbox/eval/reward 链路。 |
| `tools/debug_inspire_image_env.py` | 调试单个 image/template 的用户、环境变量和工作目录。 |
| `tools/golden_eval_failed_cases.py` | 批量检查 gold patch eval 失败样本。 |
| `tools/benchmark_sandbox_concurrency.py` | 分阶段压测 sandbox 创建、preflight、wstunnel、Nex proxy 和 agent CLI 并发。 |
| `tools/build_agent_tools_bundle_pinned.py` | 生成固定版本的 `agentic_protocol` 工具 bundle，用于重建 template。 |
| `tools/respec_templates_to_gc2.py` | 批量把已有 template 重新指定为 `G_C2` 规格，并输出可恢复 JSONL manifest。 |
| `tools/monitor_gc2_prod.py` | 解析 GC2 生产训练日志，生成 markdown/json 监控摘要。 |
| `tools/verify_step_loss_mask.py` | 检查 `sample_artifacts.json` 中 proxy 生成与 harness 注入消息的 `step_loss_mask`。 |

### Shared (in /avalanche/share_workspace/agentic_protocol/)

| 模块 | 职责 |
|---|---|
| `command_factory/layout.py` | sandbox 内 scaffold 目录约定，`AGENTIC_PROTOCOL_ROOT` 可由 env 覆盖。 |
| `command_factory/{qwen,claude,openhands,opencode,codex,npm_agent,uv,node,wstunnel}.py` | 每个 scaffold 的 readiness command、runtime command、template install command。 |
| `command_factory/registry.py` | `SWE_AGENT_HARNESS` 名称归一化和 factory 选择。 |
| `template_build/build_swe_rebench.py` | 从 SWE-rebench manifest 批量构建 Inspire template 的入口。 |
| `template_build/build_swe_verified_inspire_template.py` | SWE-Verified 单镜像 template 入口（eval 在用）。 |
| `template_build/build_terminal_bench_inspire_template.py` | Terminal-Bench template 入口（eval 在用）。 |
| `template_build/inspire_template_build.py` | 共享 args、validation、build orchestration。 |

## 常用命令

生成 scaffold template manifest（默认 root 是 `/__avaeval_agentic_protocol_v1__`）：

```bash
source ./login.sh
export PYTHONPATH=/avalanche/share_workspace:${PYTHONPATH:-}
python3 -m agentic_protocol.template_build.build_swe_rebench \
  --source-manifest /avalanche/zf_workspace/eval/data/swe_rebench_v2/data/prefetch_image_template_success.jsonl \
  --success-manifest slime/examples/sandbox_env/data_output/swe_rebench_scaffold_template_success.jsonl \
  --failure-manifest slime/examples/sandbox_env/data_output/swe_rebench_scaffold_template_failure.jsonl \
  --instance-id pion__mediadevices-75 \
  --agent-harness-name qwen_code \
  --agent-frameworks qwen_code
```

`--max-instances N` 或重复传入 `--instance-id` 可一次 build 多个。

生成 slime runtime JSONL：

```bash
python3 slime/examples/sandbox_env/build_rebench_runtime_data.py \
  --tasks-json /abs/path/to/train-00000-of-00001.json \
  --train-dest /abs/path/to/swe_train.jsonl \
  --val-dest /abs/path/to/swe_val.jsonl \
  --consumable-template-manifest /abs/path/to/prefetch_image_template_success.jsonl \
  --require-consumable-templates \
  --conversation-prompt
```

运行 4 node 训练脚本：

```bash
export SWE_CONSUMABLE_TEMPLATE_MANIFEST="/abs/path/to/prefetch_image_template_success.jsonl"
export SWE_REQUIRE_CONSUMABLE_TEMPLATES=1
export SWE_AGENT_HARNESS=qwen_code
bash slime/examples/sandbox_env/scripts/train/run_qwen3_5_35b_a3b_swe_inspire_4node.sh
```

单机 debug 脚本：

```bash
SWE_NUM_ROLLOUT=1 \
SWE_ROLLOUT_BATCH_SIZE=1 \
SWE_SAMPLES_PER_PROMPT=1 \
SWE_DEBUG_ROLLOUT_ONLY=1 \
bash slime/examples/sandbox_env/scripts/train/run_qwen3_5_0_8b_swe_inspire_8gpu_debug.sh
```

## 验证

`tools/mini_rollout_test.py` 跳过模型生成，直接应用 gold patch 并运行 rebench eval。它覆盖 sandbox 创建、workspace 准备、patch 应用和 reward 解析：

```bash
export PYTHONPATH=/avalanche/share_workspace:${PYTHONPATH:-}
SWE_STARTUP_TIMEOUT=600 \
SWE_WAIT_TIMEOUT=1800 \
SWE_SANDBOX_START_RETRY_TIMES=2 \
SWE_SANDBOX_START_RETRY_INTERVAL=5 \
python3 slime/examples/sandbox_env/tools/mini_rollout_test.py
```
