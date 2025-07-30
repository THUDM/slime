# Rollout Buffer

## Overview

Rollout Buffer is an independent component for asynchronous agent trajectory generation, with the main function of using the LLM OpenAI Server launched by slime training to generate agent trajectories.

### Workflow

```
slime Training Process ←─── HTTP API ───→ Rollout Buffer
        ↓                                      ↓
   LLM Server ←─────── HTTP Requests ─────── Agent Framework
        ↓                                      ↓
   Model Response ──────────────────────→ Trajectory Generation
```

For each different Agent task, there should be a corresponding independent Generator class, responsible for generating trajectories for that type of task. Rollout Buffer automatically reads and loads different types of Generators.

## Quick Start

### Basic Usage Process

1. **Copy Template**: Copy `base_generator.py` as a template
2. **Modify Task Type**: Change `TASK_TYPE` to your task name (cannot duplicate with other Generators)
3. **Implement Core Function**: Implement the `run_rollout()` function
4. **Optional Customization**: Rewrite five optional functions as needed


Generator files must end with `_generator.py` and be placed in the `generator/` directory:

```
generator/
├── base_generator.py      # Math task implementation (default template)
└── your_task_generator.py # Your custom task
```

Each Generator file must define `TASK_TYPE` and `run_rollout()`.

In addition, Rollout Buffer also provides some customizable functions to meet special needs of different tasks. If no custom implementation is provided, the system will use default implementations (located in `slime_plugins/rollout_buffer/default_func.py`).

### Example Script

First, you need to follow [Example: Qwen3-4B Model](../../docs/en/models/qwen3-4B.md) to configure the environment, download data and convert model checkpoints. And then run the following scripts:
```bash
cd slime_plugins/rollout_buffer
bash rollout_buffer_example.sh

# In a different terminal
python buffer.py
```
