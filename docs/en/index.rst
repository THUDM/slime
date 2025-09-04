slime Documentation
====================

.. raw:: html

   <a href="/zh/index.html" class="btn btn-outline-primary mb-2" target="_blank">切换至中文版</a>


slime is an LLM post-training framework for RL scaling, providing two core capabilities:

- High-Performance Training: Supports efficient training in various modes by connecting Megatron with SGLang;
- Flexible Data Generation: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/quick_start.md
   get_started/usage.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/qwen3-4b-base-openhermes.md
   examples/deepseek-r1.md
   examples/glm4.5-355B-A32B.md
   examples/glm4-9B.md
   examples/qwen3-4B.md
   examples/qwen3-30B-A3B.md


.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced/speculative-decoding.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/debug.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platform_support/amd_tutorial.md
