slime Documentation
====================

slime is an LLM post-training framework for RL scaling, providing two core capabilities:

- High-Performance Training: Supports efficient training in various modes by connecting Megatron with SGLang;
- Flexible Data Generation: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

slime is the RL-framework behind GLM-4.7, GLM-4.6 and GLM-4.5. Apart from models from Z.ai, we also supports the following models:

- Qwen3 series (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 series;
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/quick_start.md
   get_started/usage.md
   get_started/customization.md
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
   :caption: Advanced Features

   advanced/slime-router.md
   advanced/speculative-decoding.md
   advanced/low-precision.md
   advanced/reproducibility.md
   advanced/fault-tolerance.md
   advanced/pd-disaggregation.md
   advanced/arch-support-beyond-megatron.md

.. toctree::
   :maxdepth: 1
   :caption: Other Usage

   examples/qwen3-4b-base-openhermes.md
   _examples_synced/search-r1/README.md
   _examples_synced/fully_async/README.md
   _examples_synced/retool/README.md
   _examples_synced/multi_agent/README.md
   _examples_synced/on_policy_distillation/README.md   

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/debug.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platform_support/amd_tutorial.md

.. toctree::
   :maxdepth: 1
   :caption: Blogs

   blogs/release_v0.1.0.md
   blogs/introducing_slime.md
