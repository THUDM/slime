# GPUs, memory, and parallelism

Open **Placement** in the {ref}`interactive lab <lab>` to explore the diagrams. First establish that each phase fits and learns correctly, then optimize useful samples per second.

## Starting GPU allocations

The lab assumes eight GPUs per training node and starts from repository recipes for H100/H200-class servers with fast interconnects. These are recipe-based starting points, not measured minimums or guarantees for every GPU in that class. The short experiment uses a 4,096-token response limit, eight prompts and eight responses per prompt. Prompt length still counts toward memory. MoE and Qwen profiles enable CPU Adam and recomputation.

| Model | Training GPUs | TP / PP / CP / EP | GPUs per serving engine | Source |
|---|---:|---|---:|---|
| GLM-4-9B | 8 | 2 / 1 / 2 / 1 | 2 | [9B](https://github.com/THUDM/slime/blob/main/scripts/run-glm4-9B.sh) |
| GLM-4.7-Flash | 8 | 2 / 2 / 2 / 4 | 8 | [Flash](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B.sh) |
| GLM-4.7 | 64 | 8 / 4 / 2 / 16 | 32 | [355B](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-355B-A32B.sh) |
| GLM-5.3 | 256 | 4 / 8 / 8 / 32 | 64 | [744B](https://github.com/THUDM/slime/blob/main/scripts/run-glm5.2-744B-A40B.sh) |
| DeepSeek-R1 | 128 | 8 / 4 / 4 / 32 | 64 | [R1](https://github.com/THUDM/slime/blob/main/scripts/run-deepseek-r1.sh) |
| Qwen3.8-27B | 32 | 4 / 2 / 4 / 1 | 2 | [Matching architecture](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3.5-27B.sh) |

The Flash lab profile uses EP4 so EP × PP fits eight GPUs. Qwen3.8 reuses a matching architecture configuration; this documentation change has not GPU-validated its checkpoint. Start with 9B or Flash to learn the loop, then measure peak memory on your target model before reserving a long job.

Colocation reserves one pool; separate placement reserves training plus rollout GPUs. PD divides the rollout pool into whole prefill/decode engines. External serving capacity is outside the trainer's Ray allocation. Adding GPUs without changing model parallelism can simply add replicas rather than reducing per-GPU model memory.

## The memory ledger

For an illustrative mixed-precision Adam with $P$ parameters, BF16 weights take $2P$ bytes, FP32 gradients $4P$, FP32 master weights $4P$, and two FP32 moments $8P$:

$$M_{\rm states}=(2+4+4+8)P=18P\ \text{bytes}.$$

Actual precision-aware optimizer state dtypes can differ. Activations, buffers, temporary casts, allocator reserve, reference-policy copies and KV cache are additional. A 30B model has about 60 GB of BF16 weights and 240 GB of FP32 moments before sharding. For MoE storage use total parameters, not only active parameters.

```{mermaid}
flowchart TB
    T[Training memory] --> W[Weights + gradients]
    T --> O[Optimizer states]
    T --> A[Saved activations]
    T --> B[Buffers + temporaries]
    O --> C[CPU Adam: host memory and compute]
    A --> R[Recomputation: save less, compute again]
    S[Serving memory] --> SW[Weights]
    S --> KV[KV cache + workspace]
```

With ideal phase offload, colocated GPU demand approaches $\max(M_{\rm train},M_{\rm rollout})$; resident buffers and transitions add overhead. Separate pools have independent capacity constraints.

## TP: split one layer

Split a matrix by columns:

$$W=[W_0\;W_1],\qquad Y=XW=[XW_0\;XW_1].$$

Each GPU computes a slice. A following row-partitioned layer needs $Z=Y_0V_0+Y_1V_1$, a reduction of partial outputs. This explains both memory savings and frequent communication. Use fast links for TP. Sequence parallelism partitions additional activation operations; it does not replace CP. See the [Megatron paper](https://arxiv.org/abs/2104.04473) and [parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

## PP: split model layers

```{mermaid}
flowchart LR
    A[Early layers] -->|activations| B[Middle layers]
    B -->|activations| C[Late layers]
    C -->|gradients| B
    B -->|gradients| A
```

Each stage stores its layers. For $p$ equal-cost stages, $m$ microbatches and stage time $\tau$, a simple forward pipeline takes $(m+p-1)\tau$. Utilization is $m/(m+p-1)$ and the fill/drain bubble fraction is $(p-1)/(m+p-1)$. Backward scheduling, communication and unequal layer costs change actual training efficiency. More microbatches amortize bubbles but each must still fit. See the [Megatron paper](https://arxiv.org/abs/2104.04473).

## EP: split MoE experts

For selected experts $K(x)$:

$$y(x)=\sum_{e\in K(x)}g_e(x)E_e(x).$$

GPUs hold different experts. All-to-all dispatch sends tokens to their owners; combine returns weighted outputs. Uneven expert traffic can bottleneck an otherwise balanced weight allocation. See [GShard](https://arxiv.org/abs/2006.16668).

Dense-layer rank accounting commonly uses $N=TP\cdot PP\cdot CP\cdot DP$. Experts use another grouping with expert TP and expert DP: do not multiply EP on top of the dense identity. With expert TP=1, the lab checks divisibility by both $TP\cdot PP\cdot CP$ and $EP\cdot PP$. Model dimensions and Megatron impose further constraints.

## CP: split tokens

CP distributes queries by token position. Attention still needs remote keys/values, introducing communication. CP reduces local sequence activations; it does not itself shard weights.

| Layout: 8 tokens, CP2 | Rank 0 | Rank 1 |
|---|---|---|
| Default zigzag | 0, 1, 6, 7 | 2, 3, 4, 5 |
| slime allgather CP | 0, 1, 2, 3 | 4, 5, 6, 7 |

Zigzag splits each padded sequence into $2c$ chunks. Rank $r$ gets chunks $r$ and $2c-r-1$. Pairing early and late positions balances causal attention: for eight tokens, contiguous ranks process 10 and 26 query-key pairs, while these zigzag ranks each process 18. This counts pairs, not sparse-kernel runtime.

slime `--allgather-cp` packs sequences and assigns contiguous slices of the padded stream. The DSA backend gathers needed remote attention/index data; loss processing restores sample ownership. This is different from zigzag per-sequence splitting. A backend-incompatible flag can scramble token order.

```{mermaid}
flowchart LR
    A[Rank 0 local queries] --> C[Attention over required keys]
    B[Rank 1 keys / values] -->|backend communication| C
    C --> D[Rank 0 local outputs]
```

Zigzag describes placement, not one universal ring algorithm. The attention backend chooses communication. Allgather can materialize large tensors, so CP does not divide every memory allocation equally.

At CP > 1, slime permits allgather CP only for `DeepseekV32ForCausalLM` and `GlmMoeDsaForCausalLM`. The GLM-5.3 lab profile enables it through its model config. Other profiles retain their supported layout; diagram buttons do not change runtime flags. See [validation](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/arguments.py), [packing](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/data.py), [CP indexing](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/cp_utils.py), and the [Megatron guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

## ZeRO and distributed optimizer

Data-parallel replicas process different samples. ZeRO progressively shards optimizer states (stage 1), gradients (stage 2), then parameters (stage 3). For $D$ replicas and the illustrative 18-byte state budget:

$$M_0=18P,\quad M_1=6P+12P/D,\quad M_2=2P+16P/D,\quad M_3=18P/D.$$

Derive these by first dividing master weights plus moments by $D$, then gradients, then BF16 weights. These ideal persistent-state formulas omit transient communication buffers. $P$ may be a local model-parallel shard and $D$ its applicable replica group; expert groups can differ. See [ZeRO](https://arxiv.org/abs/1910.02054).

slime enables Megatron's `use_distributed_optimizer` by default: shard optimizer state/work, then gather updated parameters. This does not imply full ZeRO-3 parameter sharding or a DeepSpeed runtime. See [backend defaults](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/arguments.py).

## CPU Adam and recomputation

Adam maintains moments and uses bias-corrected estimates:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,$$
$$\hat m_t=m_t/(1-\beta_1^t),\quad\hat v_t=v_t/(1-\beta_2^t),\quad\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon).$$

CPU Adam keeps optimizer state and computes updates on the host. The switch exports `--optimizer-cpu-offload`, `--overlap-cpu-optimizer-d2h-h2d` and `--use-precision-aware-optimizer`. Sharding decides which rank owns state; offload decides where that rank stores/updates it. Host RAM, CPU throughput and PCIe bandwidth replace part of the GPU burden; activations and KV cache are unaffected. See [Adam](https://arxiv.org/abs/1412.6980) and [ZeRO-Offload](https://arxiv.org/abs/2101.06840) for background; slime uses Megatron's implementation.

The independent recomputation switch saves boundaries and repeats forward work during backward, trading FLOPs for activation memory. See [activation checkpointing](https://arxiv.org/abs/1604.06174). For training OOM, identify whether states or activations dominate before choosing offload, more model parallelism, or smaller token microbatches. For serving OOM, examine weights, KV dtype, concurrency, response length and engine size. Measure each change with [profiling](../developer_guide/profiling.md).
