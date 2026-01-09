# Single-Round Formal Math with Lean 4 and RL

This directory contains an example of training a model to solve formal math problems using Lean 4. It leverages Reinforcement Learning (GRPO) with a "verifier-in-the-loop" approach, where generated proofs are verified for correctness using the [Kimina](https://github.com/project-numina/kimina-lean-server) verifier.

## Overview

-   **Task**: Given a formal math statement in Lean 4, generate a valid proof.
-   **Method**: Single-turn reinforcement learning (GRPO). The model generates a full proof (including thoughts/plans), and the reward is determined by whether the proof compiles and is valid.
-   **Verifier**: Uses `kimina-lean-server` running in a Docker container to verify the generated Lean code.

## Prerequisites

### Docker Setup
You need Docker installed and a specific network for communication between the training process and the Kimina verifier:

```bash
# Create a docker network for kimina and slime to communicate
docker network create formal_math
```

**Note**: The training script will launch a `kimina-lean-server` container. It requires mounting the host Docker socket (`/var/run/docker.sock`) so the script can manage sibling containers. Connect slime container to the same docker network.

### Install Dependencies

```bash
apt update && apt install -y docker-cli
pip install kimina-client polars
```

## Quick Start: Minimal Demo

This minimal demo (`run_minimal.py`) runs a self-contained training loop on a small dataset.

### Prepare Data
Download and process the data (e.g., FineLeanCorpus, MiniF2F).

```bash
python examples/formal_math/single_round/prepare_data.py --output-name minimal_demo
```

### Prepare Models & Environment
Use `run.py` to download the base model (e.g., Qwen3-8B) and set up the environment. We skip the actual training submission here (`SLIME_SCRIPT_ENABLE_RAY_SUBMIT=0`) as we will use the minimal runner next.

```bash
SLIME_SCRIPT_ENABLE_RAY_SUBMIT=0 python examples/formal_math/single_round/run.py
```

### Run Training
Launch the minimal training script.

```bash
python examples/formal_math/single_round/run_minimal.py
```

## Advanced Usage

For full-scale training or standard runs, use `run.py`. This script leverages `slime.utils.external_utils.command_utils` to handle cluster setup and execution.

```bash
python examples/formal_math/single_round/run.py
```

The code also support more complicated cases, e.g.:

* SFT + RL
* Data filter + RL
