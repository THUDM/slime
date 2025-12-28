# Training Multi-Turn Tool-Use Agents: SFT → RFT → GRPO

We report a 4B parameter model achieving **57.1% Pass@4** on tau2-bench (test split): **4× better than the base model** and competitive with models 6-60× larger. The model is faster, cheaper to run, and demonstrates that progressive training (SFT → rejection sampling → GRPO) works for complex, multi-turn tool-use tasks.

This cookbook shows you how we did it. Everything is public and open source: [training data](https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-seed-v3), [checkpoints](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1), and [code](https://github.com/THUDM/slime/tree/main/examples/tau-bench).

![Tau2 pipeline overview](public/slime-pipeline-tau2.jpeg)

*Figure 1: Three-stage training pipeline (SFT → rejection sampling/RFT → GRPO) for multi-turn tool-use agents.*

## TLDR

**Setup** (inside `slimerl/slime:latest` container):
```bash
export SLIME_ROOT="$(pwd)" TAU_BENCH_OUT_DIR="${SLIME_ROOT}/examples/tau-bench/outputs"
git clone https://github.com/sierra-research/tau2-bench.git "${TAU_BENCH_OUT_DIR}/_external/tau2-bench"
cd "${TAU_BENCH_OUT_DIR}/_external/tau2-bench" && git checkout 337326e && pip install -e . --no-deps && cd "${SLIME_ROOT}"
export TAU2_DATA_DIR="${TAU_BENCH_OUT_DIR}/_external/tau2-bench/data"
pip install gymnasium addict deepdiff fs langfuse plotly pydantic-argparse redis ruff scikit-learn seaborn tenacity watchdog "litellm==1.65.0"
cp examples/tau-bench/tau2/.env.template examples/tau-bench/tau2/.env  # ADD OPENAI_API_KEY
set -a && source examples/tau-bench/tau2/.env && set +a
```

**Evaluate** (uses published checkpoint, ~2h on 2xH100):
```bash
# Terminal 1: Policy server
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path Jarrodbarnes/Qwen3-4B-tau2-grpo-v1 \
  --host 0.0.0.0 --port 30000 --tp 2 --mem-fraction-static 0.70

# Terminal 2: Run evaluation (uses GPT-4.1-mini as user simulator)
python3 examples/tau-bench/tau2/eval.py \
  --hf-checkpoint Jarrodbarnes/Qwen3-4B-tau2-grpo-v1 \
  --sglang-url http://127.0.0.1:30000/generate \
  --domains airline,retail,telecom --task-split test --num-samples 4 \
  --temperature 0.8 --top-p 1.0 --top-k 20 \
  --output "${TAU_BENCH_OUT_DIR}/tau2/eval/eval_pass4.json"
```

To train from scratch, see [Train from Scratch](#train-from-scratch-optional).

---

## Contents

- [TLDR](#tldr)
- [Why Tau2-Bench?](#why-tau2-bench)
- [Performance snapshot](#performance-snapshot)
- [Before You Start](#before-you-start)
- [Setup (Tau2)](#setup-tau2)
- [Resources](#resources)
- [Methodology (why this works)](#methodology-why-this-works)
- [Implementation Details](#implementation-details)
- [Quickstart: Reproduce Pass@4](#quickstart-reproduce-pass4)
- [Train from Scratch (Optional)](#train-from-scratch-optional)
- [Smoke tests (documented)](#smoke-tests-documented)
- [Troubleshooting](#troubleshooting)

## Why Tau2-Bench?

Tau2-bench ([paper](https://arxiv.org/pdf/2506.07982), [repo](https://github.com/sierra-research/tau2-bench)) tests multi-turn agents in realistic scenarios: airline bookings, retail purchases, and telecom troubleshooting. Unlike simpler benchmarks, it requires agents to maintain protocol correctness across dozens of turns while managing complex tool schemas.

The telecom domain is particularly challenging. It uses **dual-control**, meaning diagnostic actions are user-only. The agent must *instruct* the user to perform diagnostics rather than executing them directly. This mirrors real customer support workflows and pushes difficulty into communication strategy.

## Performance snapshot

Complete performance comparison (test split; Pass@4 is the headline metric):

| Stage | Overall | Airline | Retail | Telecom |
|------------------------------|---------|---------|--------|---------|
| Baseline (Qwen3-4B-Instruct) | 14.3% | 5.0% | 16.0% | 20.0% |
| SFT | 8.57% | 5.0% | 20.0% | 0.0% |
| SFT1 (RFT) | 27.0% | 20.0% | 50.0% | 7.5% |
| GRPO (Pass@1, greedy) | 32.9% | 15.0% | 76.0% | 4.0% |
| GRPO (Pass@4, temp=0.8, **reported**) | 57.1% | 50.0% | 76.0% | 44.0% |
| Delta (Pass@4 vs Baseline) | +42.8% | +45.0% | +60.0% | +24.0% |

**What worked:**
- **Progressive training compounds**: Baseline → SFT+RFT (27%) → GRPO (32.9%) → Pass@4 (57.1%, reported). Each stage builds on the last.
- **Pass@K matters for RL**: Multi-sampling at inference (Pass@4) gains +24.2 percentage points over greedy decoding. RL models benefit more from exploration than prompted baselines.
- **Domain-specific gains**: Retail (76%) and airline (50%) saw massive improvements. Telecom (44%), constrained by dual-control complexity, still improved 2.2× over baseline.

[WandB runs (public): SFT + GRPO v1 →](https://wandb.ai/jbarnes850-near-protocol/tau2-cookbook)

![Tau2 performance comparison](public/performance-chart.jpeg)

*Figure 2: Qwen3-4B with progressive training (57.1% Pass@4, reported) achieves competitive performance against models 6-60× larger. Stacked bar shows contribution from SFT+RFT and GRPO stages.*

**Local reproduction (Dec 28, 2025)** using the eval command below and full policies (no compressed prompt), with reported sampling settings (`top_p=1.0`):

| Metric | Overall | Airline | Retail | Telecom |
|--------|---------|---------|--------|---------|
| Pass@1 | 36.0% | 20.0% | 50.0% | 30.0% |
| Pass@4 | 57.0% | 30.0% | 82.5% | 45.0% |

Config: `Jarrodbarnes/Qwen3-4B-tau2-grpo-v1`, `tau2-bench` commit `337326e62d8e0ca74c353b004a9c5d748e0ba914`, `TAU2_USE_COMPRESSED_PROMPTS=0`, `TAU2_MAX_STEPS=100`, `TAU2_USER_MODEL=gpt-4.1-mini`, `TAU2_USER_TEMPERATURE=0.7`, `temperature=0.8`, `top_p=1.0`, `top_k=20`, `num_samples=4`.

Reported Pass@4 settings: `TAU2_MAX_STEPS=100`, `TAU2_USER_TEMPERATURE=0.7`, `temperature=0.8`, `top_p=1.0`, `top_k=20`, `num_samples=4`.

## Before You Start

All scripts use `slimerl/slime:latest` and assume you're in the repo root. If you're not already inside the container, start it first:

```bash
docker run --gpus all --rm -it -v "$(pwd)":/workspace/slime -w /workspace/slime slimerl/slime:latest
```

Everything outputs to `TAU_BENCH_OUT_DIR` (defaults to `examples/tau-bench/outputs`):

```bash
export SLIME_ROOT="$(pwd)"
export TAU_BENCH_OUT_DIR="${SLIME_ROOT}/examples/tau-bench/outputs"
```

Training is stochastic. You'll get comparable results, not identical ones. Checkpoints and datasets live on Hugging Face; local runs write to gitignored `outputs/`.

## Setup (Tau2)

This assumes you are running inside `slimerl/slime:latest` and are in the slime repo root.

### 0) Install tau2-bench (official)

```bash
mkdir -p "${TAU_BENCH_OUT_DIR}/_external"
git clone https://github.com/sierra-research/tau2-bench.git "${TAU_BENCH_OUT_DIR}/_external/tau2-bench"
cd "${TAU_BENCH_OUT_DIR}/_external/tau2-bench"
git checkout 337326e62d8e0ca74c353b004a9c5d748e0ba914
# Avoid dependency conflicts with sglang inside slimerl/slime:latest.
pip install -e . --no-deps
export TAU2_DATA_DIR="${TAU_BENCH_OUT_DIR}/_external/tau2-bench/data"
cd "${SLIME_ROOT}"
```

### 1) Python deps (minimal)

Install the tau2-bench runtime deps explicitly (pin `litellm` to avoid upgrading `openai`/`grpcio`):

```bash
pip install gymnasium addict deepdiff fs langfuse plotly pydantic-argparse redis ruff \
  scikit-learn seaborn tenacity watchdog "litellm==1.65.0"
```

Do not run `pip install -e .` without `--no-deps`; it will downgrade `grpcio` and upgrade `openai`, breaking `sglang` in the base image.

Optional (recommended for experiment logging): `wandb`, `weave`.

### 2) API keys and environment

Create `examples/tau-bench/tau2/.env` from the template and source it:

```bash
cp examples/tau-bench/tau2/.env.template examples/tau-bench/tau2/.env
set -a && source examples/tau-bench/tau2/.env && set +a
```

## Resources

**Models** (public on Hugging Face):
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) - Base model
- [Qwen3-4B-tau2-sft1](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-sft1) - After SFT+RFT
- [Qwen3-4B-tau2-grpo-v1](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1) - Final GRPO checkpoint

**Dataset** (public): [tau2-sft-seed-v3](https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-seed-v3) - Filtered trajectories from rejection sampling

**Training logs**: [WandB project](https://wandb.ai/jbarnes850-near-protocol/tau2-cookbook) - Public SFT + GRPO v1 runs and metrics

## Methodology (why this works)

### The Problem: Credit Assignment in Multi-Turn Tool-Use

Consider a telecom troubleshooting task where the agent must guide a user through 20+ turns of diagnostics before solving their MMS issue. At step 15, the agent asks the user to grant app permissions (a critical action that enables MMS). But the final reward (success/failure) only arrives at step 20.

**How does the model know step 15 mattered?**

This is the credit assignment problem. Standard outcome-based rewards (0 for failure, 1 for success) provide essentially zero gradient across the 19 intermediate steps. The model sees:
- Steps 1-19: no signal
- Step 20: success or failure

For a prompted model, this is catastrophic. Without seeing thousands of examples of this exact interaction pattern, it cannot learn which intermediate actions lead to success. Early SFT attempts on tau2-bench showed this clearly: models achieved 8.57% on the test split (*worse* than the unprompted baseline).

### Stage 1: SFT Warm-Start (Teaching Protocol)

Supervised fine-tuning addresses the protocol learning problem. Before a model can *optimize* tool-use, it must first understand:

1. **Turn structure**: One action per turn, wait for environment response
2. **Tool schemas**: 30+ tools across domains with complex argument structures
3. **Dual-control coordination**: In telecom, the agent coaches users through diagnostics rather than executing them

Example trajectory from SFT data:
```
Agent: get_customer_by_phone(phone_number="555-123-2002")
Env: {customer_id: "C1001", ...}
Agent: get_details_by_id(id="L1002")
Env: {line_id: "L1002", status: "Active", ...}
Agent: "Please toggle airplane mode ON, wait 10 seconds, then OFF..."
User: "Done. Still no data."
Agent: "Now open Settings > Apps > Messaging and check permissions..."
```

Without SFT, RL training thrashes. The model doesn't know the rules of the game. With SFT, we achieve 27% on test (after rejection filtering), establishing a foundation for exploration.

### Stage 2: Rejection Sampling (RFT) - Concentrating Success Patterns

After SFT, the model can complete tasks but inconsistently. A critical insight from recent research ([Statistical Rejection Sampling Improves Preference Optimization](https://openreview.net/forum?id=xbjSwwrQOe)): sampling multiple on-policy rollouts and keeping only successes concentrates the training distribution on viable strategies.

Our rejection sampling process:
1. Sample 4-8 attempts per training task at temperature 0.8
2. Keep trajectories where `reward >= 1.0` (true successes only)
3. For tasks with no successes, keep the highest `partial_score` trajectory if ≥ 0.6
4. Retrain SFT on this filtered dataset

This serves two purposes:
- **Exploration**: High temperature discovers diverse solution paths
- **Quality gates**: Hard filters prevent training on broken strategies

The published [tau2-sft-seed-v3](https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-seed-v3) dataset is the result of this filtering with a 25% success rate during RFT.

⚠️ **Limitation**: Rejection sampling requires an SFT policy that can occasionally succeed. On new domains where SFT achieves <5%, you may need teacher demonstrations or curriculum learning first.

### Stage 3: GRPO + Turn-Level Reward Shaping

GRPO solves the credit assignment problem through two mechanisms:

**1. Group-based advantage estimation**

For each prompt, GRPO samples K trajectories (K=4 in our setup), scores them, and trains the model to increase probability of high-reward actions relative to the group average:

```
advantage_k = (reward_k - mean(rewards)) / std(rewards)
loss = -mean(log_prob * advantage)
```

This is *relative* optimization. The model learns "this action was better than my other attempts" rather than "this action is objectively good." For multi-turn tasks with many valid paths, this is exactly what we want.

**2. Dense reward shaping from partial scores**

Tau2-bench provides `reward_info` with turn-level evaluation:
- `action_checks`: Did the agent call expected tools?
- `communicate_checks`: Did the agent mention required information?
- `env_assertions`: Are environment states correct?

We extract a `partial_score` from these signals:

```python
partial_score = 0.7 * (correct_actions / total_expected) +
                0.2 * (env_assertions_met / total_assertions) +
                0.1 * (communication_checks / total_checks)
```

The final shaped reward becomes:
```python
shaped_reward = task_reward + 0.3 * partial_score
```

This provides gradient at every turn, not just at task completion. Research on [turn-level credit assignment](https://arxiv.org/html/2505.11821v1) shows this is critical for multi-turn learning. Trajectory-level rewards fail to distinguish which *turns* contributed to success.

⚠️ **Watch for reward hacking**: We observed that adding partial credit for "taking more turns" caused the model to repeat tool calls indefinitely. Dense rewards must align with true task objectives.

### Why This Pipeline Works: Empirical Evidence

Recent research validates the SFT→RFT→GRPO progression:

1. **SFT establishes foundation**: Models learn reasoning patterns and task structure ([On the Generalization of SFT](https://arxiv.org/pdf/2508.05629))
2. **RFT enables exploration**: Statistical rejection sampling improves policy estimation over pure SFT ([Statistical Rejection Sampling](https://openreview.net/forum?id=xbjSwwrQOe))
3. **GRPO optimizes efficiency**: Group comparisons stabilize learning without critic overhead ([Two-Stage SFT+GRPO Pipeline](https://www.emergentmind.com/topics/two-stage-sft-grpo-training-pipeline))

Hybrid RL branching (SFT→RFT→GRPO) reaches maximum SFT performance with only 55% of the compute while pushing the Pareto frontier on both accuracy and efficiency.

For tau2-bench specifically, the progression shows:
- Baseline: 14.3% (no task understanding)
- SFT+RFT: 27.0% (protocol learned, inconsistent execution)
- GRPO Pass@1: 32.9% (optimized for single best path)
- GRPO Pass@4: 57.1% (reported; robust across multiple sampling attempts)

The 24.2 percentage point gain from Pass@1 to Pass@4 demonstrates that RL-trained models benefit significantly from inference-time exploration. They've learned multiple viable strategies rather than overfitting to a single path.

## Implementation Details

**Dual-control (telecom)**: Diagnostic actions are user-only. The agent instructs rather than executes:
```
Agent: "Please toggle airplane mode ON, wait 10 seconds, then OFF. Tell me what changes."
User: "Done. Still no data."
```

**Function calling**: Qwen3 uses native format `<tool_call>{...}</tool_call>`. Include `</tool_call>` in stop sequences.

**Chat templates**: Training on multi-turn conversations requires `--apply-chat-template` flag.

**User simulator**: Training uses a local instruct model on port 30001 (`TAU2_USER_API_BASE=http://127.0.0.1:30001/v1`). Evaluation defaults to GPT-4.1-mini for cleaner signal (fewer function calling errors).

## Quickstart: Reproduce Pass@4

Download the [GRPO checkpoint](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1), start the policy server, and run evaluation:

**1. Policy model (port 30000)**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path Jarrodbarnes/Qwen3-4B-tau2-grpo-v1 \
  --host 0.0.0.0 --port 30000 --tp 2 --mem-fraction-static 0.70
```

**2. Run evaluation** (uses GPT-4.1-mini as user simulator):
```bash
python3 examples/tau-bench/tau2/eval.py \
  --hf-checkpoint Jarrodbarnes/Qwen3-4B-tau2-grpo-v1 \
  --sglang-url http://127.0.0.1:30000/generate \
  --domains airline,retail,telecom --task-split test --num-samples 4 \
  --temperature 0.8 --top-p 1.0 --top-k 20 \
  --output "${TAU_BENCH_OUT_DIR}/tau2/eval/eval_pass4.json"
```

This takes ~2 hours on 2×H100. Results: Pass@1 and Pass@4 metrics across all domains.

The script outputs both Pass@1 and Pass@4. Results are stochastic; see the local reproduction table above for a concrete run and config.

To run without external API keys, start the local user simulator and set:
```bash
export TAU2_USER_API_BASE=http://127.0.0.1:30001/v1
export TAU2_USER_MODEL=openai/Qwen/Qwen3-4B-Instruct-2507
```

To run without external API keys, start the local user simulator and set:
```bash
export TAU2_USER_API_BASE=http://127.0.0.1:30001/v1
export TAU2_USER_MODEL=openai/Qwen/Qwen3-4B-Instruct-2507
```

## Train from Scratch (Optional)

We publish the [SFT checkpoint](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-sft1) and [GRPO checkpoint](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1) (public; login only needed for uploads). To train from scratch:

### Prerequisites

**1. Download base model and SFT training data**:
```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir "${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507"
mkdir -p "${TAU_BENCH_OUT_DIR}/tau2/data/sft1"
huggingface-cli download Jarrodbarnes/tau2-sft-seed-v3 --local-dir "${TAU_BENCH_OUT_DIR}/tau2/data/sft1" --repo-type dataset
export TAU2_SFT_DATA_DIR="${TAU_BENCH_OUT_DIR}/tau2/data/sft1"
export SFT_DATA_JSONL="${TAU2_SFT_DATA_DIR}/tau2_sft_merged_v3_rft.jsonl"
```

**2. Convert to Megatron format**:
```bash
source scripts/models/qwen3-4B-Instruct-2507.sh
python3 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint "${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507" \
  --save "${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507_torch_dist" \
  ${MODEL_ARGS[@]}
```

### Stage 1: SFT

Run supervised fine-tuning on the filtered RFT trajectories:
```bash
bash examples/tau-bench/tau2/run_sft.sh
```
For a smaller debug run, set `SFT_DATA_JSONL="${TAU2_SFT_DATA_DIR}/seed_sft_v3.jsonl"`.

### Stage 2: GRPO

**Generate task indices**:
```bash
python3 examples/tau-bench/tau2/tasks.py \
  --local_dir "${TAU_BENCH_OUT_DIR}/tau2/tasks" \
  --domains airline,retail,telecom --splits train
```

**Start user simulator** (separate terminal, keep GPUs distinct from training):
```bash
GPUS=2,3 bash examples/tau-bench/tau2/start_user_sim_server.sh
```

**Run GRPO training** (example for 4 GPUs total):
```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash examples/tau-bench/tau2/run_grpo.sh
```

Adjust `GPUS`, `CUDA_VISIBLE_DEVICES`, and `NUM_GPUS` for your machine to avoid overlap.

Training takes ~2 hours on 8×H100s. [Reference logs (SFT + GRPO v1)](https://wandb.ai/jbarnes850-near-protocol/tau2-cookbook).

## Smoke tests (documented)

- Import: `python3 -c "from tau2.gym.gym_agent import AgentGymEnv; print('ok')"`
- Task index: run `examples/tau-bench/tau2/tasks.py` with `--limit 1` on `train,test`
- Prompt formatting: ensure `--apply-chat-template` is passed for multi-turn training
- Tiny eval sanity: run `eval.py` with `--max-tasks-per-domain 1`

## Troubleshooting

- **SGLang abort/OOM**: reduce `--mem-fraction-static`, reduce `--max-tokens-per-gpu`, reduce `--rollout-batch-size`.
- **Ray working directory issues**: the provided scripts submit Ray jobs with `working_dir` set to the slime repo root and `PYTHONPATH` set explicitly; avoid running from random directories.
- **Ray dashboard exposure**: `run_grpo.sh` binds the dashboard to `127.0.0.1` by default. If you override `RAY_DASHBOARD_HOST`, avoid exposing it on shared networks.
- **Telecom is slow / low Pass@K**: dual-control pushes difficulty into communication. Inspect failures for (a) tool ownership violations, (b) premature `done`, (c) missing follow-up questions after user diagnostics.
