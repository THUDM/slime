slime Documentation
===================

slime is an LLM post-training framework for RL scaling. It combines Megatron training with SGLang rollout and provides custom interfaces for data generation and rewards.

Training, rollout, the data buffer, and environment feedback share one dataflow, supporting math, code, tools, sandboxes, and long-horizon agent workflows.

Design and Production Experience
--------------------------------

- slime is the RL framework behind `GLM-5.3-Flash <https://z.ai/blog/glm-5.3-flash>`_, `GLM-5.3 <https://z.ai/blog/glm-5.3>`_, `GLM-5.2 <https://z.ai/blog/glm-5.2>`_, `GLM-5.1 <https://z.ai/blog/glm-5.1>`_, `GLM-5 <https://z.ai/blog/glm-5>`_, `GLM-4.7 <https://z.ai/blog/glm-4.7>`_, `GLM-4.6 <https://z.ai/blog/glm-4.6>`_, `GLM-4.5 <https://z.ai/blog/glm-4.5>`_.
- Megatron arguments are available directly; installed SGLang arguments use the ``--sglang-`` prefix.
- Generation functions, reward functions, verifiers, and environments connect through documented customization interfaces.
- CPU correctness tests and GPU end-to-end tests cover training, rollout, checkpointing, precision, asynchronous generation, and debug replay. See :doc:`developer_guide/ci`.
- Large MoE recipes combine BF16 training with FP8 rollout; ``--sglang-kv-cache-dtype fp8_e4m3`` can increase effective KV cache capacity.

Alongside GLM, supported model families include Qwen (Qwen3.6, Qwen3.5, Qwen3-Next, Qwen3 MoE, Qwen3, Qwen2.5), DeepSeek (V3, V3.1, R1), and Llama 3. Start with the recipes below and the model configurations in ``scripts/models/``.

Start by Use Case
-----------------

- New to slime: :doc:`get_started/quick_start`
- Configure training and rollout arguments: :doc:`get_started/usage`
- Add custom generation, reward, or rollout functions: :doc:`get_started/customization`
- Build agentic RL workflows: :doc:`get_started/agent`
- Configure production SGLang rollout topology: :doc:`advanced/sglang-config`
- Connect external rollout engines: :doc:`advanced/external-rollout-engines`
- Persist distributed rollout and shared tensors with straw: :doc:`advanced/straw`
- Sync weights as byte-level deltas: :doc:`advanced/delta-weight-sync`
- Use PD disaggregation: :doc:`advanced/pd-disaggregation`
- Use BF16 training with FP8 rollout or FP8 KV cache: :doc:`advanced/low-precision`
- Understand CI and reliability coverage: :doc:`developer_guide/ci`
- Debug, trace, and profile long-running jobs: :doc:`developer_guide/debug`, :doc:`developer_guide/trace`, :doc:`developer_guide/profiling`

.. toctree::
   :maxdepth: 1
   :caption: Get Started

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
   :caption: Advanced Features

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
   :caption: Other Usage

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
   :caption: Developer Guide

   developer_guide/ci.md
   developer_guide/debug.md
   developer_guide/trace.md
   developer_guide/profiling.md
   developer_guide/install_flashqla.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platform_support/amd_tutorial.md

.. toctree::
   :maxdepth: 1
   :caption: Blogs

   blogs/release_v0.1.0.md
   blogs/introducing_slime.md
