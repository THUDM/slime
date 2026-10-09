# Understand the RL system you are building

[Open the experiment lab](../index.rst). These are independent dimensions of an experiment: **placement** owns GPUs, **scheduling** orders work, **precision** changes arithmetic, and **serving topology** divides inference work. The simple cost models below are explanatory estimates, not benchmark predictions.

## Placement

A synchronous round generates responses, computes rewards, trains, and distributes new weights. With colocation, rollout and training share GPUs and alternate residency. With disaggregation they own separate GPU pools, even when scheduling is synchronous. External SGLang adds independent lifecycle ownership; it does not by itself imply asynchronous training.

If training needs $N_T$ GPUs and rollout needs $N_R$, separate pools reserve $N_T+N_R$ GPUs. Colocation reuses a pool, but each phase still needs enough devices and host memory for its own parallelism/offload. The lab assumes 8 GPUs per training node and checks divisibility by TP × PP × CP and expert-TP × EP × PP. Divisibility is necessary, not a memory-capacity proof.

For full weight size $S_W$, transfer bandwidth $B_W$, and control/load overhead $t_0$, a rough sync cost is

$$t_W\approx S_W/B_W+t_0.$$

Delta transfer substitutes changed bytes $S_\Delta$ but adds snapshot, diff, encoding, and reconstruction time. It wins only when the bytes saved outweigh those costs. slime uses CUDA IPC for colocation; disaggregated full updates can use NCCL or disk. Delta requires disk transport, a shared directory, host-local checkpoint directories, and patched SGLang `/pull_weights` endpoints.

Related systems research: [HybridFlow](https://arxiv.org/abs/2409.19256). slime's actual interfaces: [external engines](external-rollout-engines.md), [delta sync](delta-weight-sync.md), and [training topology](megatron-config.md).

## Async

Let $t_R$ be rollout plus reward time, $t_T$ training time, and $t_W$ weight-update time. A serial round takes approximately

$$t_{\rm sync}=t_R+t_T+t_W.$$

If independent resources and a sufficiently full queue overlap generation and training perfectly, the optimistic steady-state lower bound becomes

$$t_{\rm async}\gtrsim\max(t_R,t_T)+t_W,\quad
\text{speedup}\lesssim\frac{t_R+t_T+t_W}{\max(t_R,t_T)+t_W}.
$$

Weight-update stalls, sampling dependencies, reward latency, queue warmup, and GPU contention weaken this bound. It does not predict a measured speedup. Variable-length samples make synchronous barriers especially costly.

The cost of overlap is staleness. If a sample was generated at weight version $v_s$ and consumed at $v_t$, define age $d=v_t-v_s$. At a prefix $h$, mismatch can be split into

$$
\log p_{v_t}(a\mid h)-\log q_{v_s}(a\mid h)
=\underbrace{\log p_{v_t}-\log p_{v_s}}_{\text{parameter staleness}}
+\underbrace{\log p_{v_s}-\log q_{v_s}}_{\text{same-version engine mismatch}}.
$$

Deterministic kernels can reduce the second term, not the first. A long trajectory may contain several versions; a single age statistic is only a diagnostic. TIS/SC can be useful, but do not justify an unlimited stale queue.

slime uses `train.py` with `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`. It preserves in-flight generations across calls and needs separate GPUs for meaningful overlap. Monitor `staleness/mean` and `staleness/max`. The generated recipe omits evaluation on that continuous queue; use a separate evaluation path/job. See the [fully async implementation guide](../_examples_synced/fully_async/README.md). Related work: [PipelineRL](https://arxiv.org/abs/2509.19128); this citation is conceptual context, not a claim that slime implements the paper's entire algorithm.

## Precision

Suppose serving logits are $z+\delta$ while trainer logits are $z$. For a fixed token, differentiate log-softmax:

$$
\frac{\partial\log p_a}{\partial z_j}=\mathbf1_{a=j}-p_j.
$$

A first-order expansion gives

$$\log q_a-\log p_a\approx\delta_a-\sum_jp_j\delta_j.$$

A common shift cancels, while token-dependent perturbations change probabilities. Quantization, differing kernels, and accumulation order all contribute; this approximation can fail near discontinuities such as expert selection.

A tensor with $P$ elements needs approximately $Pb/8$ bytes at $b$ bits, excluding scales, metadata, padding, and replicated copies. Halving the bit width halves the raw tensor storage, not necessarily total GPU memory. Optimizer state, activations, and KV cache remain separate budgets.

The lab offers BF16 training + BF16 or FP8 rollout, experimental FP8 training, and beta INT4 rollout. BF16 training uses a converted `torch_dist` checkpoint. FP8 rollout uses a separate block-quantized HF checkpoint whose `quantization_config` controls online weight conversion. FP8 KV cache is independent of weight precision. FP8 training uses blockwise TE settings; it omits `fp8-param-gather` because of CPU Adam incompatibility.

Format reference: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433). See [slime precision support and caveats](low-precision.md). The lab's INT4 options require model/kernel validation; no GLM INT4 performance claim is made.

## Deterministic

Floating-point addition is not associative. Changing a batch shape or reduction tree can change a result even without any random sampling. Reproducibility asks whether repeated execution of one configuration produces the same output; alignment asks whether two **different** implementations produce the same probabilities:

$$\text{repeatable}(p)\land\text{repeatable}(q)\not\Rightarrow p=q.$$

The lab's deterministic switch enables Megatron deterministic mode, deterministic SGLang inference, and the documented NCCL/TE/cuBLAS environment settings. It is a reproducibility configuration, not a proof of cross-engine equality. The dependency stack must support the selected model and kernels; follow the [reproducibility guide](reproducibility.md), including its FlashAttention 3 preparation.

Exact GLM-5 alignment additionally uses aligned DSA attention, batch-invariant DeepGEMM forward, matching router and head precision, and the patched Megatron/DeepEP path. The maintained regression is **six-layer GLM-5.2, EP8**, with `train_rollout_logprob_abs_diff < 1e-6`; a separate layerwise gate checks exact equality. This evidence does not establish full 744B, arbitrary topology, async, or PD alignment.

Original technical work: [SGLang deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/). Run the documented gate before expanding the claim.

## PD

Prefill consumes the prompt; decode produces tokens iteratively. PD disaggregation splits these **serving** phases, independently of whether the trainer shares GPUs with serving. See [DistServe](https://arxiv.org/abs/2401.09670) for the research motivation, and [slime's PD guide](pd-disaggregation.md) for its SGLang/Mooncake implementation.

Let $n_P,n_D$ be counts of prefill/decode engines and $\mu_P,\mu_D$ their measured requests/second under the target workload. With KV payload $S_{KV}$ per request and network bandwidth $B_{KV}$, flow conservation gives the bottleneck bound

$$\lambda\le\min(n_P\mu_P,n_D\mu_D,B_{KV}/S_{KV}).$$

Ignoring transfer, balancing $n_P\mu_P\approx n_D\mu_D$ prevents a growing interstage queue. The correct split follows measurements of your prompt/response distribution, not a universal 1:1 ratio.

For a conventional GQA/MHA cache, a rough unsharded payload for a prompt of length $L$ is $S_{KV}\approx2L N_{\rm layers}N_{\rm KVheads}d_{\rm head}b/8$. The factor 2 accounts for K and V. MLA/DSA and compressed or paged layouts need their actual representation instead of this formula.

The lab generates `sglang.yaml` with prefill and decode groups. Group GPU counts must be multiples of the model's GPUs-per-engine and sum to the rollout pool. The selected Mooncake recipe needs working RDMA devices and corresponding dependencies. External deployments apply PD in their own launcher; `--sglang-config` must not be combined with `--rollout-external-engine-addrs`.

## HiCache

The [official HiCache design and usage guide](https://docs.sglang.io/docs/advanced_features/hicache_best_practices) describes a cache hierarchy extending prefix reuse beyond GPU memory. A simple per-request model compares reused computation with I/O:

$$\Delta t\approx h\,t_{\rm avoided\ prefill}-t_{\rm cache\ IO}-t_{\rm bookkeeping},$$

where $h$ is the reusable-prefix hit rate. A positive hit rate alone does not prove speedup; large transfer cost can dominate. Useful metrics include prefix hit rate, host RAM, cache I/O time, and TTFT.

The lab uses `--sglang-enable-hierarchical-cache --sglang-hicache-ratio 2`, configuring host cache without an L3 storage service. Under managed PD, it enables HiCache on prefill only, matching the engine builder's behavior. It does not configure decode offload or distributed cache storage. KV represents activations under specific weights; reuse across actor weight updates needs correct invalidation. Measure useful hits within that validity window.

## Debug before optimizing

Start with a three-round run and held-out evaluation. If generations are wrong, check the input template, checkpoint conversion, and weight update first. [Debug replay](../developer_guide/debug.md) freezes the sampled batch. [Tracing](../developer_guide/trace.md) identifies which phase waits. [Profiling](../developer_guide/profiling.md) then inspects the expensive kernels or engine stage. Optimize the measured bottleneck while comparing reward and mismatch; throughput alone is not the RL objective.
