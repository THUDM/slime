slime 文档
===================

slime 是一个面向大规模强化学习的 LLM 后训练框架，将 Megatron 训练与 SGLang rollout 连接起来，并提供自定义数据生成和奖励接口。

训练、rollout、数据缓冲区和环境反馈共享同一条数据流，可以接入数学、代码、工具调用、沙盒和长程智能体任务。

设计与生产实践
-------------------

- slime 是 `GLM-5.3-Flash <https://z.ai/blog/glm-5.3-flash>`_、`GLM-5.3 <https://z.ai/blog/glm-5.3>`_、`GLM-5.2 <https://z.ai/blog/glm-5.2>`_、`GLM-5.1 <https://z.ai/blog/glm-5.1>`_、`GLM-5 <https://z.ai/blog/glm-5>`_、`GLM-4.7 <https://z.ai/blog/glm-4.7>`_、`GLM-4.6 <https://z.ai/blog/glm-4.6>`_、`GLM-4.5 <https://z.ai/blog/glm-4.5>`_ 背后的 RL 训练框架。
- 直接使用 Megatron 参数，通过 ``--sglang-`` 前缀传入当前安装版本的 SGLang 参数。
- 通过自定义接口接入生成函数、奖励函数、验证器和交互环境。
- CPU 正确性测试和 GPU 端到端测试覆盖训练、rollout、checkpoint、精度、异步生成与调试回放。详见 :doc:`developer_guide/ci`。
- 大规模 MoE 配方结合 BF16 训练与 FP8 rollout；``--sglang-kv-cache-dtype fp8_e4m3`` 可提升有效 KV cache 容量。

除 GLM 外，支持的模型系列还包括 Qwen（Qwen3.6、Qwen3.5、Qwen3-Next、Qwen3 MoE、Qwen3、Qwen2.5）、DeepSeek（V3、V3.1、R1）和 Llama 3。可以从下方训练示例和 ``scripts/models/`` 中的模型配置开始。

按使用场景开始
--------------

- 第一次使用 slime：:doc:`get_started/quick_start`
- 配置训练与 rollout 参数：:doc:`get_started/usage`
- 添加自定义生成、奖励或 rollout 函数：:doc:`get_started/customization`
- 构建智能体 RL 工作流：:doc:`get_started/agent`
- 配置 SGLang rollout 部署拓扑：:doc:`advanced/sglang-config`
- 接入外部 rollout 引擎：:doc:`advanced/external-rollout-engines`
- 使用 straw 持久化分布式 rollout 与共享张量：:doc:`advanced/straw`
- 以字节增量同步权重：:doc:`advanced/delta-weight-sync`
- 使用 PD disaggregation：:doc:`advanced/pd-disaggregation`
- 使用 BF16 训练 + FP8 rollout 或 FP8 KV cache：:doc:`advanced/low-precision`
- 了解 CI 和可靠性覆盖：:doc:`developer_guide/ci`
- 调试、追踪和分析长时间运行的任务：:doc:`developer_guide/debug`、:doc:`developer_guide/trace`、:doc:`developer_guide/profiling`

.. toctree::
   :maxdepth: 1
   :caption: 开始使用

   get_started/quick_start.md
   get_started/usage.md
   get_started/customization.md
   get_started/agent.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: Dense

   examples/qwen3-4B.md
   examples/glm4-9B.md

.. toctree::
   :maxdepth: 1
   :caption: MoE

   examples/glm4.7-30B-A3B.md
   examples/qwen3-30B-A3B.md
   examples/qwen3-next-80B-A3B.md
   examples/glm5.2-744B-A40B.md
   examples/glm4.7-355B-A32B.md
   examples/deepseek-r1.md

.. toctree::
   :maxdepth: 1
   :caption: 高级特性

   advanced/on-policy-distillation.md
   advanced/speculative-decoding.md
   advanced/low-precision.md
   advanced/reproducibility.md
   advanced/fault-tolerance.md
   advanced/straw.md
   advanced/observability.md
   advanced/pd-disaggregation.md
   advanced/external-rollout-engines.md
   advanced/delta-weight-sync.md
   advanced/sglang-config.md
   advanced/megatron-config.md
   advanced/arch-support-beyond-megatron.md

.. toctree::
   :maxdepth: 1
   :caption: 其他用法

   examples/qwen3-4b-base-openhermes.md
   _examples_synced/search-r1/README.md
   _examples_synced/fully_async/README.md
   _examples_synced/retool/README.md
   _examples_synced/multi_agent/README.md
   _examples_synced/coding_agent_rl/README.md
   _examples_synced/delta_weight_sync/README.md
   _examples_synced/eval_multi_task/README.md
   _examples_synced/geo3k_vlm/README.md
   _examples_synced/geo3k_vlm_multi_turn/README.md
   _examples_synced/on_policy_distillation/README.md
   _examples_synced/strands_sglang/README.md
   _examples_synced/tau-bench/README.md
   _examples_synced/train_infer_mismatch_helper/README.md

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   developer_guide/ci.md
   developer_guide/debug.md
   developer_guide/trace.md
   developer_guide/profiling.md
   developer_guide/install_flashqla.md

.. toctree::
   :maxdepth: 1
   :caption: 硬件平台

   platform_support/amd_tutorial.md

.. toctree::
   :maxdepth: 1
   :caption: 博客

   blogs/release_v0.1.0.md
   blogs/introducing_slime.md
