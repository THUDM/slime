# Prism 首轮工程验收

验收使用用户指定的 `CQ-科研驾驶舱` 工作空间、H200 GPU 和 `voxrl:editable-v1-slim` 镜像。学生资产是 Prism 1.7B v0.4.0 native iteration 100000；固定路径及训练/Omni 解释器合同见 [INSPIRE.md](../../INSPIRE.md)。实际 learner 为单卡，TP=PP=CP=DP=1。结果保留在共享 QB 盘，完整 checkpoint、原始响应和 debug dump 没有写入 Git。

验证目标是原 `train.py` 的训练、保存、恢复与权重发布闭环。严格跨引擎概率一致性、语音质量提升和真实奖励模型联合验收独立处理。

## 已完成的原生训练

| 场景 | 实际执行与结果 | 验收报告 |
| --- | --- | --- |
| 单学生副本 GRPO 工程闭环 | 4 条轨迹各 8 帧，24 RVQ 和 sampled stop；显式工程 reward ±1；原组优势 ±0.8660247；梯度范数 11.4607334；完整 model/optimizer/scheduler checkpoint `iter_0000000`；学生发布版本 1、2 | `ray-attempt-005/verification.json` |
| 两 domain MOPD 工程闭环 | 按 `ch` / `en` 分别直连两独立冻结教师的 `/score_actions`；每条轨迹 192 个 RVQ 动作、8 个 sampled continue；task reward 0；原生 OPD reverse KL 0.00248493，梯度范数 0.7008069；完整 checkpoint `iter_0000000`；学生发布版本 1、2；训练后教师实际 tensor checksum 与启动 pin 一致 | `mopd-smoke/audit-v2-attempt-20261012T113833Z/verification.json` |
| 完整断点恢复与两个学生副本 | 原标准 loader 恢复单副本 checkpoint；训练前 optimizer steps `[1,1]`、scheduler 4；rollout 1 的 4 条轨迹各 8 帧，数据游标继续到样本 4–7；梯度范数 11.5324316；完整保存 `iter_0000001` 后 optimizer steps `[2,2]`、scheduler 8；两个独立 AR 接收 rank 均完成版本 2、3 安装 | `ray-resume-dual-007/verification.json` |

报告目录根为：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/
```

单副本 source 为 `f06e7e0c81641baa2cdfdc2ca966509efb2c58b6`；MOPD source 为 `dcd77d4e91382d1ae0e5cdc2117d27be66f299f6`；双副本恢复 source 为 `a31d82f299cfcb2bc32a1af4267f549a8255df51`。每次作业从已提交的 Git SHA 导出独立源码 snapshot，运行时不修改。各自的 source manifest 和完整日志保留在结果目录。

双副本恢复使用一张 learner GPU 与两张完整 Omni pipeline GPU，申请四 GPU 配额，其中一张空闲。生成共用一个 Python router，NCCL world size 为 3，两个 AR worker 分别使用独立接收 rank。标准 resume 未设置 native 预训练初始化、finetune 或跳过 optimizer 加载；Megatron 的 `no_load_optim` 默认值为 `None`，检查器按其 falsey 语义判断，同时保留原值。首次恢复测试发现 group 提前传入的版本 0 阻止了 actor 根据实际加载进度推导版本；修复 actor/group 的共同起点后，第二次运行实际验证了首发版本 2 和更新后版本 3。原失败收据保留。

MOPD 使用同一份基座制品启动两个独立教师服务，验证真实 domain 路由、冻结约束和原生 OPD 接口；它没有证明两个不同 domain 专家模型的训练收益。教师训练前后 checksum 比较是一次工程验收，正常训练不增加每步全模型 hash。

MOPD 原作业完成训练、保存和版本 2 发布后，其首版检查器误把原框架从 `iteration 0` 保存视为未完成，并假设了不存在的 model key 前缀。原失败日志保留。第二版检查器只读同一份结果，核验实际保存日志、一次 optimizer 更新、scheduler sample steps 2、302 个模型 tensor 与 302 组 optimizer 参数及两类 moments 的名称/形状，以及完整 DCP storage ranges；没有修改模型、算法、checkpoint 或原始日志，也没有补跑 GPU。

## 权重与数值监控

native 302 个 tensor 到 HF 390 个 tensor 的完整映射使用真实 DCP 与转换制品逐项 `torch.equal` 验证。GPU native 模型转换后的全部 HF tensor 也与实际制品一致。一次完整 NCCL 安装覆盖 390 个 tensor、3,959,756,800 字节。

在一条短轨迹上，BF16 Megatron 与 Omni 的 RVQ selected-action logprob 平均绝对误差约 0.02727、最大约 0.20549，stop 最大误差约 3.58e-7。该结果没有达到独立诊断脚本选用的 0.05 最大误差门限。根据首版精度目标，保留原 Megatron 计算核，以更新前 Megatron 重放分数作为 PPO 的旧学生分数，Omni 分数持续用于诊断。

正常训练记录逐动作平均绝对误差和 `mean_sample_train_rollout_logprob_max_abs_diff`。后者先计算每条轨迹的真实动作最大误差，再沿原 microbatch/DP 报表平均，不能解释为全 DP global max。逐 codebook 与 stop 原始分数可通过 debug dump 检查。

## CPU 检查与实际范围

模型结构/replay、Omni 生成与教师协议、三个奖励客户端、多动作 GRPO/OPD 归约、native checkpoint dispatch、完整权重映射、Ray/Omni 生命周期和发布故障边界的聚焦测试均已通过。标准 checkpoint 恢复的版本修复另通过 45 项 factory、disk delta、NCCL lifecycle 与 native dispatch 回归，并覆盖所有 rank 恢复边界一致性。

GPU GRPO 使用明确标注的工程 reward，未调用真实 ASR、speaker embedding 或 JudgeCLAP 三服务联合评分，不能据此判断 WER/SIM/指令遵循收益。首轮还没有验收多 learner DP、长语音、全部 GPU 故障注入、Local 模型、colocate/offload 或局部片段指令奖励。
