# Prism RL 资产

适用范围：账号 `kayson`，用户指定的 `CQ-科研驾驶舱` 工作空间，项目 `多模态自进化科学模型`。调度配额、可用 GPU 和平台状态每次从 Inspire Live 查询；本文件只记录共享资产及环境合同。

| 资产 | 固定位置 / 身份 |
| --- | --- |
| Native Prism 预训练 RUN | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite/RUNs/delivery/pretrain/prism_pretrain_1.7B_v0.4.0` |
| 本次导出 iteration | `checkpoints/iter_0100000`，模型组件为其 `model/` DCP |
| 对应 HF/Omni 制品 | `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/vox/artifacts/prism17b-v04/hf-iter-0100000` |
| Codec | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-Audio-Tokenizer-v2` |
| 转换命令与检查记录 | `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/vox/artifacts/prism17b-v04/conversion-command.sh`、`conversion-report.json` |
| GPU 容器镜像 | `voxrl:editable-v1-slim`；固定镜像 digest `sha256:f98aa62c7a9926dd7ab42925953459dc33f759f78654692cbea8293abf2dc484` |
| 隔离训练 Python | `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/vox/artifacts/prism17b-v04/gpu-validation/training-env/bin/python` |
| Omni Python | 镜像内 `/opt/venvs/core/bin/python` |

HF 制品由 mossLite 的官方 native 转换工具在 CPU 上生成。结构为 24 RVQ、32 物理层、H2048、MLP-only 28–31、BF16；执行中的重复层保留物理参数共享。模型初始化使用 native DCP 模型组件，RL 续训使用 slime 自己的完整 checkpoint。

Prism learner 使用 Megatron 原生 TE layer spec 与 `TEDotProductAttention`，后端沿用 Megatron 默认 `auto` 选择。模型包不再直接依赖 `flash_attn.cute` wrapper；TE 根据实际环境选择底层 attention kernel。Omni 的推理后端由其独立配置和运行时选择。

训练 Python 的 NumPy 1.26.4、Transformers 5.0.0、SciPy 1.14.1、PyArrow 18.1.0 安装在独立 venv；通过其 `.pth` 和 NVIDIA runtime 符号链接引用镜像内的 Torch/TE。PyArrow 版本需要同时兼容 NumPy 1 与 Ray 的序列化，单独导入 Ray 不足以验证这一项。Omni 使用原 core 的 NumPy 2.3.5、Transformers 5.12.1。两者引用相同的 Torch 2.13.0+cu130 和 TE 2.17.0 二进制，Megatron Core 为 0.19.0。这份训练 venv 依赖镜像内 `/opt/` 的库位置，应在上述 GPU 镜像中使用。

启动 Ray head/node 时也使用训练 Python，保证其默认 worker 解释器具备相同的训练依赖。Omni 的独立 Python 由 `--omni-python` 指定。不要通过继承到 Omni 子进程的 `PYTHONPATH` 切换 NumPy/Transformers 版本。

GPU 验收源码使用 `git archive <完整 SHA>` 写入各验收入口独立的 `source/<SHA>/` 子目录，真实绝对路径由 source manifest 记录。每次 workload 使用新的结果目录，保留完整响应、逐动作分数、发布版本和清理结果；运行中的 snapshot 不修改。临时 workload 名称与结果属于验收记录，不作为永久服务地址。

本项目后续 GPU 验收的大文件输出放在 `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/`。其中 `ray-attempt-005/checkpoint/` 是使用明确标注的工程奖励产生的完整 RL checkpoint，作为 optimizer 与断点恢复的验证 fixture；它不代表语音质量训练结果。CPU 工具、HF 制品与代码入口的原共享路径保留。
