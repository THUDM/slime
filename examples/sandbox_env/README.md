# `sandbox_env`

这个目录是 `slime` 上跑 SWE-rebench + `inspire-sandbox` 的主实现，当前维护的主路径是：

1. 从公开 task image 预构建 Inspire template
2. 把 template manifest 补齐镜像默认用户和环境变量
3. 生成 rollout / train 用的 runtime jsonl
4. 用 `rock_swe_rollout.py` 通过 ROCK agent runtime 在 sandbox 里执行任务

## Core Files

- [rock_inspire_adapter.py](./rock_inspire_adapter.py): 把 ROCK sandbox 接口适配到 `inspire-sandbox`
- [rock_swe_rollout.py](./rock_swe_rollout.py): `slime` rollout 与 SWE sandbox 执行主桥接层
- [prefetch_image_templates.py](./prefetch_image_templates.py): 从 SWE-rebench 的公开 `image_name` 预构建 Inspire template
- [populate_image_default_users.py](./populate_image_default_users.py): 回填 template manifest 里的 `docker_image_default_user` 和 `docker_image_env`
- [build_rebench_runtime_data.py](./build_rebench_runtime_data.py): 生成训练/验证使用的 runtime jsonl

内部辅助模块：

- [runtime_env_paths.py](./runtime_env_paths.py): agent runtime 安装路径和占位符展开
- [rebench_eval.py](./rebench_eval.py): SWE-rebench eval script 渲染、log parser 加载、reward 计算
- [rollout_logging.py](./rollout_logging.py): sample 日志目录和快照落盘
- [anti_call_llm_helper.py](./anti_call_llm_helper.py): 注入 sandbox 内的 anti-call helper 代码

配置和启动入口：

- [rock_agent_qwen_rebench_template.yaml](./rock_agent_qwen_rebench_template.yaml)
- [rock_agent_iflow_rebench_template.yaml](./rock_agent_iflow_rebench_template.yaml)
- [run_qwen3_5_35b_a3b_swe_inspire_4node.sh](./run_qwen3_5_35b_a3b_swe_inspire_4node.sh)
- [run_qwen3_5_35b_a3b_swe_inspire_6node.sh](./run_qwen3_5_35b_a3b_swe_inspire_6node.sh)
- [run_qwen3_5_35b_a3b_swe_inspire_8node.sh](./run_qwen3_5_35b_a3b_swe_inspire_8node.sh)

## Required Env

通常先 `source login.sh`。至少需要：

```bash
export SBX_API_KEY=...
export SBX_API_URL=...
```

如果训练脚本或 rollout 需要显式指定后端，也会用到：

```bash
export ROCK_SWE_SANDBOX_BACKEND=inspire
export ROCK_INSPIRE_SPEC=G_C2
```

## Main Flow

### 1. Prefetch Templates

从 SWE-rebench 的公开镜像直接构建 Inspire template：

```bash
python3 slime/examples/sandbox_env/prefetch_image_templates.py \
  --tasks-json /abs/path/to/train-00000-of-00001.json \
  --agent-config slime/examples/sandbox_env/rock_agent_qwen_rebench_template.yaml \
  --success-manifest /abs/path/to/prefetch_image_template_success.jsonl \
  --failure-manifest /abs/path/to/prefetch_image_template_failure.jsonl \
  --max-instances 100 \
  --parallelism 8 \
  --spec G_C4
```

核心输出：

- `prefetch_image_template_success.jsonl`
- `prefetch_image_template_failure.jsonl`

`success` manifest 里会记录 `instance_id -> inspire_template`，后续 runtime data 和训练脚本都依赖它。

### 2. Populate Image User / Env

prefetch 完成后，补齐镜像默认用户和默认环境变量：

```bash
python3 slime/examples/sandbox_env/populate_image_default_users.py \
  --success-manifest /abs/path/to/prefetch_image_template_success.jsonl \
  --failure-manifest /abs/path/to/prefetch_image_template_failure.jsonl \
  --parallelism 4
```

这一步会把下面这些字段写回 `success` manifest：

- `docker_image_default_user`
- `docker_image_env`
- 以及对应的 `*_raw` / `*_checked_at`

`rock_swe_rollout.py` 会用这些字段决定 sandbox 默认用户和镜像环境补充。

### 3. Build Runtime Data

生成 `slime` 使用的 runtime jsonl：

```bash
python3 slime/examples/sandbox_env/build_rebench_runtime_data.py \
  --tasks-json /abs/path/to/train-00000-of-00001.json \
  --train-dest /abs/path/to/swe_train.jsonl \
  --val-dest /abs/path/to/swe_val.jsonl \
  --consumable-template-manifest /abs/path/to/prefetch_image_template_success.jsonl \
  --require-consumable-templates
```

运行后，每条 sample 的 `metadata` 里会包含：

- `inspire_template`
- `docker_image_default_user`
- `docker_image_env`
- `repo` / `repo_workdir` / `base_commit`
- `install_config` / `FAIL_TO_PASS` / `PASS_TO_PASS`

### 4. Launch Training / Rollout

常用入口是多机训练脚本，例如：

```bash
bash slime/examples/sandbox_env/run_qwen3_5_35b_a3b_swe_inspire_4node.sh
```

这些脚本默认会读取：

- `ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST`
- `ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES`
- `ROCK_SWE_AGENT_CONFIG_PATH`
- `ROCK_SWE_SANDBOX_BACKEND`
- `ROCK_INSPIRE_SPEC`

如果你已经生成了新的 manifest，最常见的覆盖方式是：

```bash
export ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST=/abs/path/to/prefetch_image_template_success.jsonl
export ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES=1
export ROCK_SWE_AGENT_CONFIG_PATH=slime/examples/sandbox_env/rock_agent_qwen_rebench_template.yaml
export ROCK_SWE_SANDBOX_BACKEND=inspire
export ROCK_INSPIRE_SPEC=G_C4
```

## Notes

- 当前主流程围绕 template manifest，而不是旧的 image build / image prefetch 流程。
- rollout 默认优先使用 `metadata.inspire_template`；没有 template 时，adapter 仍可从已有 image 创建临时 template。
- `rock_swe_rollout.py` 现在只保留主编排逻辑；rebench 评测和 sample 日志已经拆到独立模块，后续维护时优先在对应小模块里改。
