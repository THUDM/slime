# Prism 首轮工程验收

## 单卡共卡训练与 tensor IPC 验收（2026-10-12）

Prism 1.7B v0.4.0 native iteration 100000 在一张 H200 上通过原 `train.py` 完成了
两轮 GRPO，并从完整 checkpoint 恢复后继续第三轮。learner 与完整 Omni pipeline
使用同一个 Ray GPU bundle；每个阶段均核验只有一张可见 GPU，启用 `colocate`、
`offload_train` 和 `offload_rollout`。运行采用明确标注的 ±1 工程奖励，每轮一个
prompt group、四条轨迹，每条八帧，并实际输出 WAV。

| 阶段 | RL rollout | optimizer / scheduler | 学生发布版本 | 梯度范数 | WAV |
| --- | --- | --- | --- | --- | --- |
| 首次初始化与连续训练 | 0、1 | `[0,0]` → `[2,2]`；0 → 8 | 1、2、3 | 11.7136135、11.6547775 | 8 |
| 标准完整恢复 | 2 | `[2,2]` → `[3,3]`；8 → 12 | 3、4 | 11.8521719 | 4 |

每次发布经 `UpdateWeightFromTensor` 发送 68 个 SGLang flattened buckets，五次发布
共确认 340 次 tensor 更新。接收端在确认前同步 CUDA，逐桶保持暂停，所有权重安装完成
且 KV/graph 内存恢复后才开放生成。首次与后续发布在本次短轨迹运行中约需 0.7–0.8 秒。

两秒间隔的 NVML 采样记录中，AR/preprocessing 进程最大观测值为 96,764 MiB，
offload 窗口内为 950–1,048 MiB；vocoder 最大观测值为 2,866 MiB，offload 时为
610–720 MiB。learner 最大观测值为 58,926 MiB。这些是该测试配置的观测值。

当前共卡使用 eager prefill、eager vocoder 与 AR decode CUDA graph。SGLang 的
breakable prefill graph 不支持 memory saver，因此 adapter 显式禁用该 prefill 后端。
推理进程与 learner 都在可选库导入后选择匹配的 preload allocator；独立 GPU IPC
探针复现了错误 allocator 模式的失败，并验证修复后的跨进程 tensor 逐元素一致。

大文件与证据仍位于本文下述 QB 根目录：

- `colocate-005/fresh/`：两轮训练、`iter_0000000` / `iter_0000001`、debug dump 和音频。
- `colocate-005/memory-observations.json`：进程显存采样摘要。
- `colocate-006/resume/`：恢复后的训练、`iter_0000002`、debug dump 和音频。
- `colocate-006/verification.json`：联合通过报告，分别记录两个阶段的源码与产物来源。

fresh 的 slime source 为 `deb731cae89296b5d857f1039fe4d0631d304ea2`，Omni source 为
`032eda473bf6089bfcf0562fa19089e1217b5217`。resume 使用同一 slime source，Omni source
为 `bb73d42c16f246ff493e676b0d9242b61f9d89f3`。首次同容器立即恢复时，旧 HTTP 服务已
正常退出，但端口探测因 `TIME_WAIT` 误拒绝。修复探测与 listener 的复用语义后，从原
已保存 checkpoint 重新验证恢复；原失败日志保留，联合报告明确区分两次作业。

该验收覆盖共卡资源、真实 IPC、音频生成、原生训练、完整保存与恢复。未评估真实
WER/SIM/JudgeCLAP 收益、多 learner DP、长语音或严格跨引擎概率一致性。

验收使用用户指定的 `CQ-科研驾驶舱` 工作空间、H200 GPU 和 `voxrl:editable-v1-slim` 镜像。学生资产是 Prism 1.7B v0.4.0 native iteration 100000；固定路径及训练/Omni 解释器合同见 [INSPIRE.md](../../INSPIRE.md)。实际 learner 为单卡，TP=PP=CP=DP=1。结果保留在共享 QB 盘，完整 checkpoint、原始响应和 debug dump 没有写入 Git。

验证目标是原 `train.py` 的训练、保存、恢复与权重发布闭环。严格跨引擎概率一致性、语音质量提升和真实奖励模型联合验收独立处理。

当前 Prism attention 直接使用 Megatron 原生 TE layer spec 与默认 `auto` 后端选择。下表前三场验收来自各自固定的历史源码，使用先前的自定义 FA4 wrapper；其余验收使用原生 TE。旧报告继续保留为历史证据，各次实现和接口以固定 source SHA 为准。

## 已完成的原生训练

| 场景 | 实际执行与结果 | 验收报告 |
| --- | --- | --- |
| 单学生副本 GRPO 工程闭环 | 4 条轨迹各 8 帧，24 RVQ 和 sampled stop；显式工程 reward ±1；原组优势 ±0.8660247；梯度范数 11.4607334；完整 model/optimizer/scheduler checkpoint `iter_0000000`；学生发布版本 1、2 | `ray-attempt-005/verification.json` |
| 两 domain MOPD 工程闭环 | 按 `ch` / `en` 分别直连两独立冻结教师的 `/score_actions`；每条轨迹 192 个 RVQ 动作、8 个 sampled continue；task reward 0；原生 OPD reverse KL 0.00248493，梯度范数 0.7008069；完整 checkpoint `iter_0000000`；学生发布版本 1、2；训练后教师实际 tensor checksum 与启动 pin 一致 | `mopd-smoke/audit-v2-attempt-20261012T113833Z/verification.json` |
| 完整断点恢复与两个学生副本 | 原标准 loader 恢复单副本 checkpoint；训练前 optimizer steps `[1,1]`、scheduler 4；rollout 1 的 4 条轨迹各 8 帧，数据游标继续到样本 4–7；梯度范数 11.5324316；完整保存 `iter_0000001` 后 optimizer steps `[2,2]`、scheduler 8；两个独立 AR 接收 rank 均完成版本 2、3 安装 | `ray-resume-dual-007/verification.json` |
| 原生 TE attention 与完整断点续训 | 实际 28 个 core 均为 `TEDotProductAttention`；默认 `auto`，运行时 TE 选择 cuDNN `FusedAttention` sub-backend 1；标准 loader 恢复旧 FA4 路径保存的完整 RL checkpoint；optimizer steps `[1,1]` → `[2,2]`，scheduler 4 → 8；梯度范数 11.6026993；完整保存 `iter_0000001`；单 Omni 副本完成版本 2、3 安装 | `native-te-resume-008/verification.json` |
| 通用 initializer 首次模型初始化 | 插件从 native iteration 100000 只导入模型，RL 从 rollout 0 开始；通用 hook、`pretrained_checkpoint`，无显式 `load`；更新前 optimizer `[0,0]`、scheduler 0，保存后 `[1,1]`、scheduler 4；梯度范数 12.0528440；完整保存 `iter_0000000`；学生版本 1、2 | `initializer-refactor-010/fresh/verification.json` |
| 通用加载边界的完整恢复 | 只通过标准 `--load` 恢复上一行的新 checkpoint，initializer、预训练来源和 step 都为空；optimizer `[1,1]` → `[2,2]`，scheduler 4 → 8；数据游标继续到样本 4–7；梯度范数 11.1523933；完整保存 `iter_0000001`；学生版本 2、3 | `initializer-refactor-010/resume/verification.json` |

报告目录根为：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/
```

单副本 source 为 `f06e7e0c81641baa2cdfdc2ca966509efb2c58b6`；MOPD source 为 `dcd77d4e91382d1ae0e5cdc2117d27be66f299f6`；双副本恢复 source 为 `a31d82f299cfcb2bc32a1af4267f549a8255df51`。每次作业从已提交的 Git SHA 导出独立源码 snapshot，运行时不修改。各自的 source manifest 和完整日志保留在结果目录。

原生 TE 验收 source 为 `676d90ebf781f85d430aa2e2372d4e836bfab8d6`，使用一张 learner GPU 与一张完整 Omni pipeline GPU。源码已删除自定义 FA4 wrapper 和 core attention 替换函数，recipe 与 publication smoke 也沿用 Megatron 默认后端选择。运行时先检查真实 module 类，再记录 TE 的实际 forward backend 日志；底层 cuDNN 后端由 TE 自动选择。旧 checkpoint 的空 attention 辅助状态通过原标准加载路径兼容处理，未重置 optimizer 或进行 finetune。

通用 initializer 两阶段验收 source 为 `e231fa349cdc754822cd915022f0bd409cec2167`，联合报告为 `initializer-refactor-010/verification.json`。两阶段都调用原 `train.py`，使用一张 learner GPU、一张 Omni GPU，各自创建和清理独立 Ray head。真实 28 个 attention core 都是 `TEDotProductAttention`，TE 自动选择 `FusedAttention` sub-backend 1；两份 checkpoint 的 optimizer `param`、`exp_avg`、`exp_avg_sq` 各 302 项。来源的预训练 iteration 100000 与 RL rollout 0/1 分开记录，source pin 没有污染恢复进度。

初始化主包不再识别 Prism 模型类型、mossLite 目录或 native manifest，旧的 Prism checkpoint flags 已移除。源格式处理留在模型插件，主包仅在无显式 `--load` 且配置通用 initializer 时调用它，统一同步低精度 optimizer master 参数。示例先生成普通 Megatron 架构 CLI，再进入原 HF 校验；错误的显式恢复路径报错，不回退成重新初始化。

旧 `initializer-refactor-009` 在首次发布版本 1 后，由复制自恢复用例的 prompt 中静态期望版本 2 误拒绝，尚未执行更新。其失败源码与日志保留；010 使用新独立目录和分阶段的版本 1/2 prompt，并在预检中核验阶段对应关系，生产版本校验未放宽。

双副本恢复使用一张 learner GPU 与两张完整 Omni pipeline GPU，申请四 GPU 配额，其中一张空闲。生成共用一个 Python router，NCCL world size 为 3，两个 AR worker 分别使用独立接收 rank。标准 resume 未设置 native 预训练初始化、finetune 或跳过 optimizer 加载；Megatron 的 `no_load_optim` 默认值为 `None`，检查器按其 falsey 语义判断，同时保留原值。首次恢复测试发现 group 提前传入的版本 0 阻止了 actor 根据实际加载进度推导版本；修复 actor/group 的共同起点后，第二次运行实际验证了首发版本 2 和更新后版本 3。原失败收据保留。

MOPD 使用同一份基座制品启动两个独立教师服务，验证真实 domain 路由、冻结约束和原生 OPD 接口；它没有证明两个不同 domain 专家模型的训练收益。教师训练前后 checksum 比较是一次工程验收，正常训练不增加每步全模型 hash。

MOPD 原作业完成训练、保存和版本 2 发布后，其首版检查器误把原框架从 `iteration 0` 保存视为未完成，并假设了不存在的 model key 前缀。原失败日志保留。第二版检查器只读同一份结果，核验实际保存日志、一次 optimizer 更新、scheduler sample steps 2、302 个模型 tensor 与 302 组 optimizer 参数及两类 moments 的名称/形状，以及完整 DCP storage ranges；没有修改模型、算法、checkpoint 或原始日志，也没有补跑 GPU。

## 权重与数值监控

native 302 个 tensor 到 HF 390 个 tensor 的完整映射使用真实 DCP 与转换制品逐项 `torch.equal` 验证。GPU native 模型转换后的全部 HF tensor 也与实际制品一致。一次完整 NCCL 安装覆盖 390 个 tensor、3,959,756,800 字节。

在先前 FA4 wrapper 的一条短轨迹上，BF16 Megatron 与 Omni 的 RVQ selected-action logprob 平均绝对误差约 0.02727、最大约 0.20549，stop 最大误差约 3.58e-7。该结果没有达到独立诊断脚本选用的 0.05 最大误差门限。训练以更新前 Megatron 重放分数作为 PPO 的旧学生分数，Omni 分数持续用于诊断；这些历史误差数值不作为其他 attention 后端的保证或验收门限。

正常训练记录逐动作平均绝对误差和 `mean_sample_train_rollout_logprob_max_abs_diff`。后者先计算每条轨迹的真实动作最大误差，再沿原 microbatch/DP 报表平均，不能解释为全 DP global max。逐 codebook 与 stop 原始分数可通过 debug dump 检查。

原生 TE 续训 step 1 实际记录平均绝对误差 0.0235308129、轨迹最大误差的平均 0.1307324171。该验收报告仍明确记录 `policy_score_alignment_verified: false`，不将工程闭环通过当作严格概率对齐证明。

## CPU 检查与实际范围

模型结构/replay、Omni 生成与教师协议、三个奖励客户端、多动作 GRPO/OPD 归约、native checkpoint dispatch、完整权重映射、Ray/Omni 生命周期和发布故障边界的聚焦测试均已通过。标准 checkpoint 恢复的版本修复另通过 45 项 factory、disk delta、NCCL lifecycle 与 native dispatch 回归，并覆盖所有 rank 恢复边界一致性。

原生 TE 改动另通过 22 项模型回归及 29 项配置、checkpoint dispatch、数值诊断回归。构造边界测试覆盖 `auto`、`flash`、`fused` 都保留原生 TE spec，以及 MLP-only 层不会改变其他层的 attention。原生 attention 接线完成后，当前 TP/PP/CP 与 SP 的首版限制继续保留；Prism 输入和动作 loss 的 CP 切分仍需后续适配。

通用 initializer 重构通过 118 项加载、CLI 和架构配置组合用例，27 项模型用例及 124 项原 checkpoint/recovery/version 回归。测试覆盖 initializer 的来源与返回契约、master 参数同步、标准恢复优先、reference/HF 加载、无效显式路径、step 0、普通模型行为，以及无 CUDA driver 时配置生成可用而实际训练明确失败。配置失败记录与修复前 source snapshot 也保留，实际 CPU fresh/resume 命令预检通过后才提交 GPU 运行。

GPU GRPO 使用明确标注的工程 reward，未调用真实 ASR、speaker embedding 或 JudgeCLAP 三服务联合评分，不能据此判断 WER/SIM/指令遵循收益。首轮还没有验收多 learner DP、长语音、全部 GPU 故障注入、Local 模型、colocate/offload 或局部片段指令奖励。
