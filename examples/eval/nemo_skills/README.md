# Evaluation with Nemo Skills

This directory contains configuration and utilities for offloading complex evaluation benchmarks to a separate environment using the `eval_delegate` mechanism. It is designed to integrate with [Nemo Skills](https://github.com/NVIDIA/NeMo-Skills) for running benchmarks like AIME25, Arena-Hard, and HLE, which may require specific environments distinct from the main training setup.

## Overview

The setup allows slime to delegate evaluation tasks to a dedicated "Skills" server. This creates a clear separation of concerns:

1.  **slime Container**: Runs the main training loop and hosts the model using SGLang.
2.  **Skills Container**: Hosts the `nemo_skills` environment, runs the evaluation logic, and queries the model running in the slime container.

## Prerequisites

-   A writable host directory for cached data (e.g., `/data/.cache`).
-   Docker installed with NVIDIA GPU support.

## Setup Instructions

### Prepare Host Network

Create a Docker network to allow communication between the slime and Skills containers.

```bash
docker network create skills-net
```

### Launch the slime Container

Start the main container where slime and the model will run. Replace `<slime container name>` with your desired name (e.g., `slime_main`).

```bash
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/.cache:/root/.cache \
  -v /dev/shm:/shm \
  --ipc=host \
  --privileged \
  --network skills-net \
  --name <slime container name> \
  slimerl/slime:latest \
  /bin/bash
```

### Launch the Skills Container

Start the container that will run the evaluation benchmarks. Replace `<env container name>` with your desired name (e.g., `skills_env`).

```bash
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/.cache:/root/.cache \
  -v /dev/shm:/shm \
  --ipc=host \
  --privileged \
  --network skills-net \
  --name <env container name> \
  --network-alias skills_server \
  guapisolo/nemoskills:0.7.1 \
  /bin/bash
```

### Configure the Skills Container

Enter the **Skills container** and set up the environment.

**a) Install Dependencies**

```bash
# Clone repositories
git clone -b slime_skills https://github.com/guapisolo/slime.git /opt/slime
git clone -b slime https://github.com/guapisolo/Skills.git /opt/Skills

# Install Skills package
cd /opt/Skills
pip install -e . --no-deps
```

**b) Prepare Datasets**

Download and prepare the datasets you intend to use.

```bash
cd /opt/Skills/nemo_skills/dataset
python3 aime25/prepare.py
python3 hle/prepare.py
python3 arena-hard/prepare.py
```

**c) Start the Evaluation Server**

Start the server that listens for evaluation requests from slime.

```bash
cd /opt/slime
python examples/eval/nemo_skills/skills_server.py \
  --host 0.0.0.0 \
  --port 9050 \
  --output-root /opt/skills-eval \
  --config-dir examples/eval/nemo_skills/config \
  --cluster local_cluster \
  --max-concurrent-requests 512 \
  --openai-model-name slime-openai-model
```
*Note: You can now connect to the server at `skills_server:9050` from within the `skills-net` Docker network. The server always proxies evaluation traffic to an OpenAI-compatible sglang router (slime starts and manage the router), so adjust `--openai-model-name` and `--max-concurrent-requests` as needed for your deployment.

## Running Evaluation

The example scripts are located in `examples/eval/scripts`. Here is an example workflow for training Qwen3-4B with delegated evaluation.

### Prepare slime Container

Enter the **slime container** and install the package.

```bash
cd /root/slime
git pull
pip install -e . --no-deps
```

### Download Model and Data

```bash
# Download model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

### Convert Model to Megatron-LM Format

You need to convert the HF model to the format required by Megatron-LM. Ensure you load the correct model arguments first.

```bash
# Source model arguments
source scripts/models/qwen3-4B.sh

# Convert model
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

### Run the Training Script

Run the training script.

```bash
bash examples/eval/scripts/run-qwen3-4B.sh
```

## Configuration

The evaluation configuration is defined in `examples/eval/scripts/multi_tasks.yaml`. It specifies:
-   `delegate`: Configurations for the external skills server (URL, timeouts).
-   `datasets`: List of datasets to evaluate on (e.g., `aime25`, `arena-hard`).
