slime 文档
====================

slime 是一个面向 RL Scaling 的 LLM 后训练框架，提供两大核心能力：

- 高性能训练：通过连接 Megatron 与 SGLang，支持多种模式下的高效训练；
- 灵活的数据生成：通过自定义数据生成接口与基于服务器的引擎，实现任意训练数据生成流程。

.. toctree::
   :maxdepth: 1
   :caption: 开始使用

   get_started/quick_start.md
   get_started/usage.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: Dense

   examples/qwen3-4B.md
   examples/glm4-9B.md

.. toctree::
   :maxdepth: 1
   :caption: MoE

   examples/qwen3-30B-A3B.md
   examples/glm4.5-355B-A32B.md
   examples/deepseek-r1.md

.. toctree::
   :maxdepth: 1
   :caption: 其他用法

   examples/qwen3-4b-base-openhermes.md
   _examples_synced/search-r1/README.md
   _examples_synced/fully_async/README.md
   _examples_synced/retool/README.md
   _examples_synced/multi_agent/README.md

.. toctree::
   :maxdepth: 1
   :caption: 高级特性

   advanced/speculative-decoding.md

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   developer_guide/debug.md

.. toctree::
   :maxdepth: 1
   :caption: 博客

   blogs/release_v0.1.0.md
   blogs/introducing_slime.md
