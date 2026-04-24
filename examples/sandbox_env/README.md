# slime sandbox_env

这个目录现在只保留 `template` 路径，目标是：

- `slime` 负责 rollout / training
- sandbox backend 默认走 `inspire-sandbox`
- OpenSWE 数据优先复用已预取的 template
- 没有预取 template 时，仍可在 adapter 内部从已有 image 创建临时 template

已经移除的旧路径：

- image build / push / save archive
- image prefetch / image load
- runtime 侧的 consumable image manifest 过滤

当前主文件：

- [rock_inspire_adapter.py](./rock_inspire_adapter.py): ROCK Sandbox 到 `inspire_sandbox` 的适配层
- [rock_swe_rollout.py](./rock_swe_rollout.py): `slime` 与 ROCK agent runtime 的 rollout bridge
- [build_openswe_base_templates.py](./build_openswe_base_templates.py): 构建 `openswe-base-py*` 基础模板
- [prefetch_inspire_templates.py](./prefetch_inspire_templates.py): 把 OpenSWE Dockerfile 翻译成 per-task template 并批量预取
- [build_swe_runtime_data.py](./build_swe_runtime_data.py): 生成训练/验证 runtime jsonl
- [run_prefetch_10.sh](./run_prefetch_10.sh): 小规模 template 预取入口
- [run_qwen3_0_6b_swe_inspire_debug.sh](./run_qwen3_0_6b_swe_inspire_debug.sh): 本地 debug
- [run_qwen3_5_0_8b_swe_inspire_8gpu_debug.sh](./run_qwen3_5_0_8b_swe_inspire_8gpu_debug.sh): 8 卡单机 debug
- [run_qwen3_30b_a3b_swe_inspire_1node.sh](./run_qwen3_30b_a3b_swe_inspire_1node.sh): 单机训练
- [run_qwen3_5_35b_a3b_swe_inspire_4node.sh](./run_qwen3_5_35b_a3b_swe_inspire_4node.sh): 多机训练

## Required Env

`login.sh` 里至少需要：

```bash
export SBX_API_KEY=...
export SBX_API_URL=...
```

如果要做 template 预取，通常还需要：

```bash
export INSP_GITHUB_TOKEN=...
```

## Template Flow

推荐路径：

1. 先构建基础模板：

```bash
python3 slime/examples/sandbox_env/build_openswe_base_templates.py
```

2. 再预取任务模板：

```bash
bash slime/examples/sandbox_env/run_prefetch_10.sh
```

或直接：

```bash
python3 slime/examples/sandbox_env/prefetch_inspire_templates.py \
  --oss-jsonl /abs/path/to/openswe_oss.jsonl \
  --max-per-source 100 \
  --parallelism 10 \
  --spec G_C2 \
  --out-dir /abs/path/to/template_prefetch/run_001
```

输出中最重要的是：

- `prefetch_template_success.jsonl`
- `prefetch_template_failure.jsonl`

训练脚本会读取 `prefetch_template_success.jsonl`，把 `instance_id -> inspire_template` 写进 runtime metadata。

## Runtime Data

`build_swe_runtime_data.py` 现在只支持 template manifest：

```bash
python3 slime/examples/sandbox_env/build_swe_runtime_data.py \
  --oss-jsonl /abs/path/to/openswe_oss.jsonl \
  --train-dest /tmp/swe_train.jsonl \
  --consumable-template-manifest /abs/path/to/prefetch_template_success.jsonl \
  --require-consumable-templates
```

如果没有传 template manifest，runtime row 仍会保留 OpenSWE 的 repo/base_commit/dockerfile 信息；`inspire` backend 会优先用 `metadata.inspire_template`，否则退化为“从已有 image 创建 template”，但不会再走旧的 image build / prefetch 代码。

## Launch

本地 smoke：

```bash
bash slime/examples/sandbox_env/run_qwen3_0_6b_swe_inspire_debug.sh
bash slime/examples/sandbox_env/run_qwen3_5_0_8b_swe_inspire_8gpu_debug.sh
```

常用覆盖项：

```bash
export ROCK_SWE_SANDBOX_BACKEND=inspire
export ROCK_INSPIRE_SPEC=G_C2
export ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST=/abs/path/to/prefetch_template_success.jsonl
export ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES=1
```

如果切回 `rock` backend，仍需提供：

```bash
export ROCK_SWE_SANDBOX_BACKEND=rock
export ROCK_SWE_BASE_URL=http://127.0.0.1:19080
```

## Notes

- `run_prefetch_10.sh` 已改为 template 预取，不再处理 image store。
- 训练/提交脚本只传递 template manifest 相关参数。
- `ROCK_INSPIRE_SPEC` 仍是 `inspire-sandbox` 资源选择的主开关。
