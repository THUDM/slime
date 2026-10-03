# LoRA

LoRA（Low-Rank Adaptation）冻结基座模型，只训练每个被适配线性层上的低秩增量：

$$
W_{\mathrm{eff}} = W_{\mathrm{base}} + \frac{\alpha}{r} B A,
\qquad A \in \mathbb{R}^{r \times d_{\mathrm{in}}},\; B \in \mathbb{R}^{d_{\mathrm{out}} \times r}.
$$

$A$ 使用 Kaiming-uniform 初始化，$B$ 初始化为零，因此初始 adapter 的数学增量为零；跨内核或合并前后的低精度计算仍可能存在舍入差异。
slime 中 LoRA 默认关闭：已有脚本无需新增任何参数，其权重同步 payload 也保持逐字节不变。

通过 `--use-lora` 开启：

```bash
--use-lora \
--lora-rank 32 \
--lora-alpha 64 \
--lora-target-preset moe_language_all
```

## 使用方法与运行模式

先准备能在 slime 中运行的 Megatron 模型配置、HF 底座、数据集和 rollout 配置，
再加入 LoRA 参数。preset 只选择模块，不会自动完成新模型的 Megatron/HF 权重转换
或 SGLang 支持。下面是添加到现有训练命令的参数片段，不是独立启动命令。

```bash
# 新训练：从 HF 底座初始化，训练 dense attention + MLP。
--hf-checkpoint /models/base \
--load /models/base \
--use-lora \
--lora-target-preset dense_language \
--lora-rank 32 \
--lora-alpha 64 \
--lora-learning-rate 1e-5 \
--lora-weight-decay 0.0 \
--save /checkpoints/full \
--save-interval 20 \
--save-lora '/checkpoints/adapters/{rollout_id}'
```

`--hf-checkpoint` 提供模型配置、tokenizer 和转换所需信息；`--load` 指定实际加载的
权重。新训练时两者指向同一份 HF 底座；恢复时 `--load` 改为完整 LoRA checkpoint，
`--hf-checkpoint` 仍指向匹配的 HF 底座。不要将 adapter 目录传给 `--load`。

| 目的 | 加载参数组合 | adapter / 训练状态行为 |
|------|--------------|-----------------------|
| 从底座开始新训练 | `--load HF_DIR`，不指定 `--lora-load` | 自动进入 finetune 模式，初始化新 adapter，不恢复旧 optimizer。 |
| 用已有 adapter 开始新训练 | `--load HF_DIR --lora-load ADAPTER_DIR` | 自动进入 finetune 模式；加载 adapter，重新开始训练状态。也可显式加 `--finetune`。 |
| 恢复完整训练 | `--load FULL_LORA_DIR`，不加 `--finetune` | 从完整 checkpoint 恢复 adapter；optimizer/RNG 是否恢复遵循原有保存、加载参数。`--lora-load` 被忽略。 |
| 从完整 LoRA 权重开始新训练 | `--load FULL_LORA_DIR --finetune` | 保留其中的 adapter 权重，但不作为训练断点续跑；如另给 `--lora-load`，该 adapter 会覆盖加载的 adapter。 |
| 释放并重建 actor | 在上述初始加载组合上加 `--release-train` | 保存后自动改为从 `--save` 恢复完整 checkpoint，关闭 finetune；不会反复加载最初的 `--lora-load`。 |

所有行都需要 `--use-lora` 及匹配的 rank、alpha 和 target 配置。完整断点续训时
不要使用 `--no-save-optim` / `--no-load-optim`；仅有 adapter 导出不足以恢复 optimizer。
`--save-lora` 是附加导出选项，不决定加载模式，也不替代 `--save`。

使用 `--release-train` 还必须遵守既有约束：

```bash
--release-train \
--save /checkpoints/full \
--update-weight-mode full \
--update-weight-transport disk \
--update-weight-disk-dir /shared/rollout-weights
```

磁盘同步目录必须对 trainer 和 rollout 引擎可见；未设置 `--save-interval` 时该模式
默认设为 1。它是 actor 生命周期选项，不是 adapter-only 保存模式。

MoE 示例见 [run-qwen3-30B-A3B-lora.sh](../../../scripts/run-qwen3-30B-A3B-lora.sh)。
这是特定模型的八卡启动示例，不代表所有 MoE 都已验证。使用前修改模型、数据、
保存路径和资源参数；脚本开头会终止相关服务及 Python 进程，只应在专用作业环境使用。

## Reference policy

启用 KL/reference 计算时，reference 始终是冻结底座（禁用 adapter 后的策略）。
即使用 `--lora-load` 加载了非零的 SFT adapter，reference 也不是“底座 + 初始 SFT
adapter”。若需要后者，当前版本不支持；也不能用 `--ref-load` 或
`--ref-update-interval` 改变这一语义。

## 关键参数

| 参数 | 说明 |
|------|------|
| `--use-lora` | 开启 LoRA。冻结基座模型，只训练 adapter。与 `--q-lora-rank` 这类架构自带的低秩参数无关。 |
| `--lora-rank` | 秩 $r$（默认 64）。必须 > 0；选中 row-parallel 目标时还必须能被 TP size 整除，选中 routed-expert fc2 目标时必须能被 expert-TP size 整除（见[并行](#并行)）。 |
| `--lora-alpha` | 有限正数，缩放分子（默认 128.0），实际缩放为 `alpha / rank`。 |
| `--lora-dropout` | 必须保持 `0.0`。任何随机性都会让训练前向与 rollout 前向不一致，破坏 on-policy RL。 |
| `--lora-target-preset` | 具名的目标模块正则集合，见[目标预设](#目标预设)。与 `--lora-target-modules` 互斥。 |
| `--lora-target-modules` | 显式正则，对未包装的 Megatron 模型的模块名做 `re.search`。 |
| `--lora-exclude-modules` | 从匹配集合中剔除的正则。默认剔除 embedding / output layer / 视觉塔。 |
| `--lora-learning-rate` | adapter 参数的学习率，显式设置时必须为有限正数；未设置时回退到 `--lr`。 |
| `--lora-weight-decay` | adapter 参数的有限非负 weight decay（默认 0.0）。与 `--lora-learning-rate` 不同，它**总是**覆盖 `--weight-decay`，所以 LoRA 模式下请设置这个参数而不是 `--weight-decay`。 |
| `--lora-load` / `--save-lora` | 仅含 adapter 的 checkpoint 目录。`--save-lora` 支持 `{rollout_id}`。 |
| `--lora-save-merged-hf` | 可选：导出已合并（不含 LoRA）的 HF checkpoint 目录模板。 |
| `--lora-rollout-sync-mode` | `merged`（默认）在同步边界物化 `W_base + alpha/r * B @ A`，复用标准全量权重通路。 |
| `--lora-allow-replicated-modules` | TP>1 时显式放行复制线性层的正则；仅用于已确认各 TP 副本输入、梯度一致的模块。 |
| `--lora-bias` | 仅支持 `none`，不训练底座 bias。 |
| `--lora-debug` | 每步梯度诊断。会引入张量扫描和设备同步，默认关闭。 |

## 目标预设

预设是具名的正则集合，避免你为每种架构去记 mcore 的模块路径。

| 预设 | 适配对象 |
|------|----------|
| `dense_attention` | `linear_qkv`、`linear_proj` |
| `dense_mlp` | `mlp.linear_fc1`、`mlp.linear_fc2` |
| `dense_language`（默认） | dense attention + dense MLP |
| `hybrid_attention` / `hybrid_language` | 额外包含 `self_attention.linear_attn` 下的 gated-delta-net 投影 |
| `moe_attention` | MoE 模型上仅 attention |
| `moe_shared_mlp` | `mlp.shared_experts.linear_fc1` / `linear_fc2` |
| `moe_routed_experts` | `mlp.experts.linear_fc1` / `linear_fc2`（grouped GEMM） |
| `moe_mlp` | shared + routed experts |
| `moe_language` | attention + shared experts（保守的 MoE 预设） |
| `moe_language_all` | attention + shared + routed experts |

内置 preset 不选择 MoE 的 **router**，当前也不将 router 适配作为受支持用法：适配 gate 会改变 token 到 expert 的分配，这与适配 expert 本身是两种不同的干预。
embedding、output layer 和视觉塔默认被排除；请求视觉塔 LoRA 会直接报错，而不是静默跳过。

## Routed experts

Megatron 中的 routed experts 是 grouped-GEMM 模块（`TEColumnParallelGroupedLinear` /
`TERowParallelGroupedLinear`），权重按 `weight0 .. weight{n-1}` 打包，每个本地 expert 一份。
slime 为每层挂载**一个被所有 expert 共享的 adapter**，A、B 两个因子均共享。
准确地说，每层 fc1、fc2 各有一组 A/B，不同层不共享。不同 EP rank 保存同一逻辑
adapter 的副本，通过初始化同步和跨 EP 梯度求和保持一致；不是每个 rank 独立训练
一个 adapter。ETP 则按并行规则切分因子。不支持每个 expert 独立的 A/B。
这不代表与其他框架的原生 adapter 格式兼容。

这一点的三个直接好处：

* **参数量不随 expert 数增长。** 256 个 expert 的层与一个 dense 层的 adapter 开销相同。
* **前向 hook 无需 routing map。** 增量就是在整批 permuted token 上做
  $\frac{\alpha}{r} B A x$，与 token 被分派到哪个 expert 无关。
* **合并在数学上等价。** 在同步边界，同一个增量被加到每个 `weight{i}` 上，低精度下应按数值容差比较合并与 adapter 前向。

通过 `moe_routed_experts`、`moe_mlp` 或 `moe_language_all` 启用，模型配置需使用
`--moe-grouped-gemm` 所对应的 TE grouped linear 布局。其他 expert 权重布局不自动兼容。

## 并行

slime 让 adapter 的切分方式与基座层一致，因此 adapter 不会引入任何未被记账的通信：

| 目标 | `lora_A` | `lora_B` | 通信 |
|------|----------|----------|------|
| Column-parallel（`linear_qkv`、`linear_fc1`） | 按 `dim 1` 切分 | 按 `dim 0` 切分 | 输入 copy，输出不 reduce |
| Row-parallel（`linear_proj`、`linear_fc2`） | 按 `dim 1` 切分 | 按 `dim 1` 切分 | 对 rank-$r$ 中间量做 all-reduce |
| Routed expert fc1（`expert_column`） | 复制 | 按 `dim 0` 切分 | 无——token dispatcher 已沿 token 轴 gather 过 |
| Routed expert fc2（`expert_row`） | 按 `dim 1` 切分 | 按 `dim 1` 切分 | 仅在 **expert**-TP 组内 all-reduce |

由此带来的几点：

* row-parallel 目标要求 `--lora-rank` 能被 TP size 整除，routed-expert fc2 目标要求能被
  expert-TP size 整除。该约束在注入时校验。
* 在 expert 并行（EP > 1）下，每个 EP rank 用不同的 token 走同一个共享 adapter，
  因此其梯度是 expert-parallel 组内的**求和**。slime 给 expert adapter 标记
  `allreduce=False`，让 Megatron DDP 把它们放进 expert bucket 并在 expert data-parallel
  组内规约；跨 EP 的求和则在 adapter 的 backward 中显式完成。
* adapter checkpoint 为每个 EP 副本赋予不同的 `replica_id`，保证分布式 checkpoint
  的每个分片恰好只有一个 writer。
* 除非在 `--lora-allow-replicated-modules` 中显式列出，TP>1 时 LoRA 拒绝注入 TP 复制的线性层，
  因为一次未被记账的跨 TP 梯度同步会让副本静默失步。

专家 adapter 在 EP>1 或 ETP!=TP 时使用专家拓扑的梯度缓冲区，并保留底座专家
参数上更严格的分组标记；因此 EP=1 并不意味着一定使用普通 DDP 分组。

## Rollout 同步

在 `--lora-rollout-sync-mode merged`（默认，也是目前唯一实现的模式）下，slime 在同步边界
为每个被适配张量物化 `W_base + alpha/r * B @ A`，并通过既有的 NCCL / IPC / 磁盘通路推送普通全量权重。
推理引擎永远看不到 `lora_A` / `lora_B`，因此：

* 不需要 SGLang 侧的 LoRA 支持——包括 routed experts，否则 fused MoE kernel 会与独立的 adapter 路径冲突。
* 合并作用在一份一致的 host 快照上，绝不修改训练权重，所以不训练时重复同步是幂等的。
* 同步沿用既有全量权重更新的失败处理，不保证跨引擎、跨 bucket 的原子回滚。
  若发生部分更新，必须先恢复所有引擎的策略一致性，再继续 rollout。

LoRA 减少可训练参数、梯度和 optimizer 状态，但底座仍需常驻或沿原有流程卸载；
merged 同步仍更新普通模型权重，不会获得 adapter-only 传输的带宽收益。实际显存
与吞吐改善取决于并行、重计算、合并和同步成本，需要实测。

`native_adapter`（把原始 adapter 推给引擎、由 SGLang 施加）是保留选项，但**明确未实现**；
请求它会直接报错，而不是静默回退。

## 不支持的组合

下列配置在参数校验、模型注入或加载阶段拒绝；具体模块结构必须等模型构建后才能检查：

* MLA 模型（包括自定义 target）
* `--train-backend fsdp`（LoRA 目前仅支持 Megatron）
* `--only-train-params-name-list` / `--freeze-params-name-list`
* `--use-critic` 或 `--advantage-estimator ppo`
* on-policy distillation（`--use-opd`、`--opd-teacher-load`）
* `--keep-old-actor`、`--ref-update-interval`、`--ref-load`
* `--lora-dropout` 非 `0.0`、`--lora-bias` 非 `none`
* 视觉塔目标

对于含 routed-expert LoRA adapter 的层，禁止启用 DeepGEMM MoE forward 替换，
因为它会绕过线性层的 hook。请关闭这些层的融合替换，或将其 routed experts 从
LoRA target 中排除。该检查在安装融合 forward 前执行。

## Checkpoint

`--save-lora DIR` 写出仅含 adapter 的 checkpoint：`adapter_config.json`、分片的
`adapter_model-tp*-pp*.safetensors`、索引文件以及 `training_state.json`。用 `--lora-load` 加载时会校验
rank、alpha、format version、基座模型 config hash 以及 TP/PP/ETP size；expert-parallel size
刻意**不**参与身份校验，因为共享 adapter 对 EP 不变（这里指 adapter-only 格式，不承诺完整 optimizer checkpoint 可任意改变 EP）。发布是原子的：分片写入失败时，
上一份 checkpoint 仍保持已发布状态。

若需要一份把 adapter 折叠进基座权重的独立 HF checkpoint，使用 `--lora-save-merged-hf`。

`--lora-load` 仅用于 `--finetune` 模式下的新训练初始化（从 HF 底座加载时会自动
设置该模式）。不带 `--finetune` 恢复完整 Megatron checkpoint 时，以其中的 adapter
和 optimizer 状态为准，忽略 `--lora-load` 并记录日志；`--release-train` 自动重建
actor 时同样如此。新训练请使用 `--load HF_DIR`，需要时再加
`--lora-load ADAPTER_DIR`。Megatron checkpoint 必须包含当前模型需要的全部
adapter 张量，即使使用 `--finetune` 也不例外。不支持从无 adapter 的
Megatron 底座初始化，也不支持无法读取分布式张量元数据的 checkpoint。
slime 不会因 adapter 缺失而全局启用 `strict=False`，原有底座参数检查保持不变。

### 文件格式与导出限制

adapter 目录使用 slime 自定义的 Megatron 分片格式，包含：

```text
adapter_config.json
adapter_model.safetensors.index.json
adapter_model-tp{tp_rank}-pp{pp_rank}.safetensors
training_state.json
```

虽然使用 safetensors 和类似的文件名，它不是 PEFT/HF adapter：模块命名、因子切分和
元数据协议不同，不能直接交给 `PeftModel.from_pretrained` 或 SGLang 原生 adapter
加载器，也不能直接用 `--lora-load` 导入外部 PEFT adapter。当前没有格式转换器。
`training_state.json` 仅记录 rollout/policy 等信息，加载时用于日志，不恢复 optimizer、
scheduler 或训练进度。完整续训请使用 `--save` 生成的完整 checkpoint。

`--lora-save-merged-hf '/checkpoints/merged/{rollout_id}'` 导出普通 HF 模型，适用于
已有 HF 转换支持的架构；它不保留独立 A/B 和 optimizer，不能还原为原 adapter。
把它作为新的 HF 底座训练时，新 adapter 从零增量开始，reference 也变为该合并底座。

adapter 发布依赖共享文件系统上的符号链接及原子替换。不要提前创建最终导出目录：
已有的普通目录会被拒绝覆盖；可使用包含 `{rollout_id}` 的新路径。搬运 checkpoint 时
需同时保留链接指向的隐藏版本目录，或解引用链接复制完整内容，不能只复制链接。

## 支持范围与限制

preset 按模块路径定义，不绑定模型系列。`hybrid_attention` 和 `hybrid_language`
只匹配上述 `linear_attn` 投影结构；其他结构请使用 `--lora-target-modules`。
任何 preset 都不会自动放行 TP 复制层。只有确认各 TP 副本的输入和梯度一致时，
才应通过 `--lora-allow-replicated-modules` 显式放行。
原 `qwen3_5_attention` / `qwen3_5_language` 改用对应的 hybrid preset；
原 `qwen3_5_mlp` 改用 `dense_mlp`。

routed-expert LoRA 要求 expert-TP（ETP）整除普通 TP，且 ETP rank 等于 TP rank
对 ETP 大小取模，确保每个 TP checkpoint 坐标只对应一个专家分片。TP=4/ETP=1
仍支持，TP=1/ETP=2 会拒绝。其他布局在注入时拒绝，避免保存出无法可靠恢复的
adapter。改变 EP 大小时仍需保持
TP/PP/ETP 不变。旧 routed-expert adapter 若缺少 ETP metadata，需从原训练布局重新
导出，不能直接假设其分片身份正确。

自定义正则只能选择已支持的线性模块，不能自动支持任意算子。TP>1 时不支持
column-parallel 的 `gather_output=True` 或 row-parallel 的 `input_is_parallel=False`；
column 输入宽度必须能被 TP 整除。融合归一化只支持可识别且提供 `eps` 的
LayerNorm/RMSNorm；routed-expert 融合归一化不支持。没有匹配目标或匹配到不支持
模块会报错。显式 `--lora-exclude-modules` 会替换默认排除列表，并非追加。

每个本地 PP/VPP 模型分块都必须至少匹配一个目标。暂不支持让部分分块没有
adapter 的自定义目标选择，此类配置在注入时会报错。

`--save-lora` 和 `--lora-save-merged-hf` 跟随完整 checkpoint 保存流程执行，要求
配置 `--save` 及正数 `--save-interval`，或者配置 `--release-train` 和 `--save`。
它们不会开启独立的 adapter-only 保存流程。恢复完整 checkpoint 时，即使最初的
adapter 目录已删除，也会忽略 `--lora-load`。

底座 config hash 只验证架构/配置兼容性，不验证权重身份。请保留训练 adapter 时
使用的确切底座权重并记录来源/revision；配置相同的其他 SFT checkpoint 不能视为
同一底座。

## 测试与 CI

遵循项目的 [CI 说明](../../en/developer_guide/ci.md)。下列 7 个 CPU 测试文件已注册到
`.github/workflows/pr-test.yml.j2` 的 `cpu-unittest`；`test_lora_parallel_gpu.py`
声明 `NUM_GPUS = 2`，已注册到 Megatron 矩阵（镜像验证也复用该矩阵）。
`_lora_fakes.py` 是辅助文件，不单独注册为测试入口。

CPU 测试需要 CPU workflow 中声明的依赖，包括 PyTorch/Gloo、pytest、safetensors。
请像 CI 一样逐文件、独立进程执行：即使环境已安装 Megatron，LoRA CPU 辅助代码
也使用进程内替身。不要将这些 CPU 测试和真实 Megatron 测试放到同一个 pytest
进程中执行。缺少 safetensors 时 checkpoint 测试会失败，不会以 skip 掩盖未验证。

```bash
# Run from the repository root, with the CI dependencies already available.
python .github/workflows/generate_github_workflows.py
pre-commit run --all-files --show-diff-on-failure

for name in \
  test_megatron_argument_validation \
  test_lora_config \
  test_megatron_lora \
  test_lora_checkpoint \
  test_lora_weight_sync \
  test_lora_lifecycle \
  test_lora_parallel
do
  SLIME_LORA_MULTI_GPU_TEST=0 python "tests/${name}.py" || exit 1
done
```

请在仓库根目录、依赖已齐全的环境执行。若生成结果改变，将模板与生成的 workflow
一起提交，不要手改生成的 YAML。pre-commit 可能修改格式，检查并提交修改后重跑，
直到通过；不要通过放宽 lint 规则绕过检查。

IDC 的训练镜像需安装真实 Megatron/Transformer Engine，并暴露两张 CUDA GPU：

```bash
python tests/test_lora_parallel_gpu.py
```

入口会自行启动两个 worker，不要再套 torchrun。直接执行时显卡不足会失败；普通
pytest 收集在无 GPU 时可能 skip，不能将其作为 GPU 验证通过的证据。GPU worker
必须使用真实 Megatron，无法静默回退到 CPU 替身。

该 GPU 测试覆盖 FP32 线性层、TP=2、SP 开关、local/TE 前反向一致性及分布式
adapter state checkpoint 一致性，不代表已验证 BF16 actor/DDP 优化器、MoE 多卡
checkpoint 或真实 SGLang 同步。同步 mock 测试验证张量内容，不验证多引擎失败恢复。

PR 验证记录应包含 commit、训练镜像 tag/digest、Megatron/TE/SGLang 版本、GPU
数量、并行配置、执行命令和通过/失败/跳过结果。另外执行短程真实 LoRA RL 闭环，
检查 adapter 更新且底座冻结、非零 adapter 的 rollout 一致性、adapter 保存加载、
完整优化器恢复及 `--release-train` 重建，并附日志，明确尚未测试的组合。
CPU job 自动触发；由维护者启用 `run-ci-changed` 可测试改动的测试文件，
`run-ci-megatron` 则运行注册的 Megatron 测试集合。

IDC 除验证 EP>1 外，还应验证 EP=1、TP>ETP 的真实 DDP optimizer step。
