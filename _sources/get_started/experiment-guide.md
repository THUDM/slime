# Tutorial: build and understand an RL experiment

{ref}`Open the interactive Quick Start <lab>`. Choose a model, watch the GPU layout change, try alternatives, and download the resulting shell. The same tutorial applies across GLM, Qwen3.8-27B, and DeepSeek-R1; model dimensions come from the repository's model configurations.

## The loop you are building

```{mermaid}
flowchart LR
    A[Prompts / environment] --> B[SGLang: sample responses]
    B --> C[Reward / verifier]
    C --> D[Batch or persistent queue]
    D --> E[Megatron: policy update]
    E -->|updated weights| B
    E --> F[Checkpoint + held-out evaluation]
```

A training example is more than a prompt and an answer: it carries tokens, rewards, masks, and the probabilities or replay data required by your objective. SGLang generates experience; Megatron computes gradients. Changing precision, kernels, or weight versions can make their distributions differ. Scaling RL means managing that difference while using compute effectively.

The lab's **Play a round** animation illustrates this loop. It is not a simulation of GPU speed or a training run. Every choice is reversible: use the previous step, any step label, undo/redo, or reset. Choices survive reload locally. A share link carries architecture choices but excludes machine paths, engine addresses, and custom hook paths. `experiment.json` preserves the complete local configuration.

## 1. Choose the task and model

The initial experiment uses verifiable math, JSONL keys `prompt` and `label`, a chat template, and the built-in `deepscaler` reward. It samples eight responses per prompt, uses GRPO, and runs three rounds. This is a correctness run, not a claim of capability improvement.

For an agent or a different reward, choose **My agent / reward**, then provide importable generation and reward functions. Those hooks must exist in the same environment on workers. The lab does not invent a reward function for your task. See [agent workflows](agent.md) and [hook contracts](customization.md).

Model selection changes architecture, TP/PP/CP/EP, conversion strategy, and serving-engine size together. These are starting configurations, not a memory-fit calculator. Default training node counts are inherited from repository recipes: GLM-4.7-Flash and GLM-4-9B use one 8-GPU node; GLM-4.7 uses 8; GLM-5.3 uses 32; DeepSeek-R1 uses 16. Qwen3.8-27B uses the four-node 27B hybrid-attention configuration.

[Qwen3.8-27B's official config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) identifies `qwen3_5` with dimensions matching `scripts/models/qwen3.5-27B.sh`. The text-RL path reuses this model spec and its HF conversion support. This architecture check is not GPU validation of the new checkpoint or a multimodal RL recipe. FLA/Gated Delta Net dependencies still apply; see [architecture support](../advanced/arch-support-beyond-megatron.md).

[GLM-5.3’s official config](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) matches the 78-layer, 256-expert DSA architecture and cross-layer index-sharing schedule in the existing `glm5.2-744B-A40B` model specification. The lab reuses that architecture source with the GLM-5.3 checkpoint; the older six-layer deterministic regression is not a validation of this new checkpoint.

## 2. Choose GPU ownership, then scheduling

The lab shows starting GPU counts, total allocation, and clickable TP/PP/EP/CP/ZeRO diagrams. CPU Adam and activation recomputation switches change the exported shell. See [memory accounting and parallelism](../advanced/parallelism-memory.md), including allgather/zigzag CP.

| Choice | What happens | What you must provide |
|---|---|---|
| Colocated | Training and rollout alternate on one GPU pool | Enough memory for each phase and offload |
| Disaggregated | slime manages separate training and serving pools | Ray capacity for both pools |
| External SGLang | slime connects to engines owned by another deployment | Engine addresses and compatible weight transport |
| Synchronous | Finish a rollout batch, train, sync weights | A straightforward ordering baseline |
| Fully async | Keep generation in flight across updates | Separate pools and monitoring of sample age |

Selecting fully async from colocation moves the layout to separate GPUs in the same undoable action. Selecting colocation restores synchronous execution. Other incompatibilities remain visible until fixed, so the lab never silently exports a different algorithm.

External engines are not a generic text-completion API: slime needs SGLang server information, weight-update endpoints, and sampler metadata. The generated `serving-reference.sh` shows the required settings. Deployment-specific hosts, ports, multi-node ranks, and RDMA configuration belong to that external launcher. A trainer CLI cannot reconfigure an already-running engine.

Partial rollout keeps unfinished prefixes for later continuation, including under synchronous training. straw persists payloads and queue state; combined with fully async it selects distributed producer processes across Ray nodes. The lab exposes both choices independently. Compare their timelines, mixed-version probabilities, and checkpoint boundaries in [rollout scheduling](../advanced/rollout-scheduling.md).

## 3. Choose numerical behavior

Start with BF16 on both sides to establish a baseline, or the maintained BF16-training / FP8-rollout path for large MoE. The latter keeps a BF16 training checkpoint and uses a separately quantized HF checkpoint for serving. KV cache precision is another independent choice.

FP8 training is experimental; INT4 rollout is beta. Every lower-precision path needs a compatible GPU/kernel stack. See [precision and memory derivation](../advanced/rl-systems.md#precision).

Then choose corrections deliberately. TIS clips importance weights; IcePop masks out-of-range weights. R3 replays MoE routes. SC centers weighted scores and uses REINFORCE. Deterministic execution addresses reproducibility but does not automatically make two engines identical. The **why & math** buttons lead to [full derivations](../advanced/policy-mismatch.md) and the original papers.

## 4. Size serving for the workload

Keep ordinary serving as a baseline. Enable PD when measurements show prefill and decode need independent resources. The lab requires both groups to contain whole engines and uses Mooncake/RDMA; fill in actual device names. `sglang.yaml` is embedded inside `experiment.sh`, so a separate manual file copy is unnecessary.

Enable HiCache for reused prefixes after budgeting host RAM and I/O. Under PD this recipe enables it on prefill only. Tune concurrency and response-token limits against actual memory and latency. See [the serving cost models](../advanced/rl-systems.md#pd).

## 5. Prepare once, convert, then run

First [install the environment](quick_start.md). Keep the same checkout, dependencies, model/data paths, and configuration paths on every participating host. Multi-node runs need a shared filesystem or equivalent identical local copies; a path inside only the head container is insufficient. Make the generated `.slime-lab.*` configuration directory visible to Ray workers through the same repository path.

Download `experiment.sh` from the lab into the slime repository root. Review its paths and notes, then run:

```bash
# Once, on the shared filesystem: download source weights and data,
# and prepare selected FP8/INT4 rollout weights.
bash experiment.sh prepare

# Convert the BF16 source into a reshardable Megatron checkpoint.
bash experiment.sh convert
```

Small/medium recipes convert on one node with 8 GPUs. For GLM-4.7, GLM-5.3, and DeepSeek-R1, the generated conversion is a four-node job with 8 GPUs per node. Run `convert` on **all four nodes** with the same `CONVERT_MASTER_ADDR` and a different `CONVERT_NODE_RANK` in 0–3. Conversion parallelism can differ from training because `torch_dist` is reshardable. Conversion itself needs sufficient GPU and host memory.

GLM-5.3 and DeepSeek-R1 publish FP8 weights, so `prepare` downloads them into a separate source directory, converts them to BF16, and then prepares the selected serving format. An existing BF16 HF checkpoint and `torch_dist` checkpoint can be used directly by editing paths and skipping preparation.

Start the Ray head and join workers using the number of GPUs reported by the lab:

```bash
# Head node
ray start --head --node-ip-address HEAD_IP --num-gpus 8 \
  --dashboard-host 0.0.0.0 --dashboard-port 8265

# Each worker node
ray start --address HEAD_IP:6379 --num-gpus 8
```

With external engines, deploy and verify those engines first; their GPUs are outside the trainer's Ray capacity. For disk transport, mount the weight directory at the same path on both sides. Delta additionally needs the patched serving endpoints and local checkpoint directories.

On the head, from the repository root:

```bash
bash experiment.sh check
bash experiment.sh train
```

`check` verifies the presence of input files; it does not load the model or certify capacity. `train` submits the job with explicit runtime environment settings. Set `RAY_DASHBOARD` if using another Ray head. The shell does not kill processes or start a cluster implicitly.

The shell defaults to a short training run. Use a fresh output directory and straw run ID for a new experiment; existing save directories follow the backend’s recovery rules. To resume a later run, add `--load` for the saved checkpoint and follow the [checkpoint/recovery requirements](../advanced/fault-tolerance.md). Decide a save interval appropriate for your experiment; the short default may finish before the interval is reached.

## 6. Decide whether the experiment worked

Read generated responses first. Check reward and the reference/trainer log-probs; then inspect training/rollout mismatch and correction clipping/masking. With fully async, inspect staleness. A fast run with meaningless rewards is not progress.

Add a held-out dataset before a long run. A synchronous example after downloading AIME data is:

```bash
--eval-interval 20 \
--eval-prompt-data aime /data/aime-2024/aime-2024.jsonl \
--n-samples-per-eval-prompt 8
```

The fully async queue does not implement evaluation itself; schedule a separate supported evaluation path or job. See [evaluation configuration](usage.md) and the [fully async guide](../_examples_synced/fully_async/README.md).

If output is wrong, use [debug and replay](../developer_guide/debug.md). If an otherwise correct run is slow, use [trace](../developer_guide/trace.md), [profiling](../developer_guide/profiling.md), and [observability](../advanced/observability.md) to identify the next change. The homepage's expandable advanced sections follow the same diagnostic order.
