# P2P Weight Synchronization

Use `--update-weight-transport p2p` for full weight updates between separate Megatron training GPUs and SGLang serving GPUs. Training ranks expose their converted HF tensors directly to serving workers. There is no NCCL process group spanning the two runtimes and no shared checkpoint directory.

The current data-plane implementation uses Mooncake Transfer Engine. The user-facing backend, control protocol, and SGLang loader are named `p2p`; the provider can be replaced without changing the training configuration. This option is separate from P2P expert redistribution inside the colocated CUDA IPC updater.

## Setup

Install the same slime checkout in both environments. Install a matching Mooncake wheel in each environment, preferably at the same version. For example, for CUDA 12 environments:

```bash
conda run -n train python -m pip install mooncake-transfer-engine==0.3.11.post1
conda run -n serve python -m pip install mooncake-transfer-engine==0.3.11.post1
```

For CUDA 13, use `mooncake-transfer-engine-cuda13` in that environment instead. Install only one Mooncake wheel per environment. See the [official installation instructions](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html) for CUDA compatibility and required system libraries, including RDMA providers. CUDA GPUs, GPUDirect RDMA support, and a reachable RDMA network are required by this provider. Both sides must also be able to reach the dynamically allocated TCP control and Mooncake RPC ports.

Each side keeps its own PyTorch, CUDA, and NCCL libraries; training-side TP collectives still use the training runtime. Environment separation does not remove the native library and driver requirements of either side. Mooncake is optional for other weight-update modes.

If automatic NIC selection needs restricting, set `SLIME_MOONCAKE_IB_DEVICES` to a comma-separated list of eligible devices on each node. The provider selects eligible NICs near the worker's CUDA device. Configure `MC_GID_INDEX` for the network where necessary.

## Usage

Add the following to a disaggregated Megatron training job:

```bash
--update-weight-mode full \
--update-weight-transport p2p \
--update-weight-buffer-size 536870912 \
--update-weight-p2p-timeout 120
```

slime-managed engines automatically enable the loader. For [external rollout engines](external-rollout-engines.md), install slime in the serving environment and launch SGLang with:

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --custom-weight-loader slime.backends.sglang_utils.p2p.load_weights \
  ...
```

The timeout bounds transfer/loading and each engine-control phase. Increase it for large models or slow networks. The current adapter supports serving PP=1; serving DP must be 1 unless DP attention is enabled. Training PP/VPP, dense TP, expert TP, and EP use the existing weight naming and conversion code. Training and serving TP/EP layouts can differ. Model and quantization support depend on the existing Megatron-to-HF conversion and SGLang native loader; not every combination has been validated. Draft-model weights are not updated by this loader.

`p2p` currently supports full updates with Megatron on separate training/serving GPUs. Delta updates continue to use disk transport. Colocated jobs retain the CUDA IPC updater.

## SGLang Prefill/Decode Disaggregation

PD can use Mooncake for KV transfer while `p2p` uses Mooncake for weight updates. They use separate Transfer Engine instances, RPC sessions, and registered buffers. Weight updates never borrow or reset SGLang's shared KV engine. Each instance still consumes NIC resources, and Mooncake's process-wide `MC_*` settings affect both uses; independent engines do not provide bandwidth or configuration isolation. Allow enough RPC ports for both, and keep their range separate from HTTP/bootstrap ports.

For PD, this backend requires `--flush-cache-interval 1`. All prefill and decode engines are paused with request abortion, then cache flushing must succeed before loading starts. All engines must finish loading and commit their version before any resumes. Retaining old KV across an update with `in_place` pause is not supported for PD. For external PD deployments, all prefill and decode engines for the updated model must be attached to slime's rollout service.

With `--offload-rollout`, SGLang must deregister KV memory before releasing its backing allocation and register it again after resume. Reusing the same virtual address does not preserve an RDMA registration across remapping. Use the repository's patched SGLang lifecycle; do not replace it with resetting the shared KV engine from the weight loader. Weight-transfer buffers are separately released between updates. Abort/drain correctness still depends on the SGLang PD implementation, including completion of native KV transfers.

## Data Flow And Memory

Dense layers are assigned across training DP/CP replicas; experts stay on their EP owners. Training-side TP gathers reconstruct each owned tensor, which is converted once to HF format. Each serving rank pulls those tensors and passes them to its native model loader for local TP/EP slicing. Every serving rank receives the full HF tensor stream, including experts it may not own. This keeps the transport independent of model-specific slicing, at the cost of redundant network traffic. Destination-specific sharding is a possible further optimization.

TCP carries JSON descriptors, sequence numbers, and acknowledgements. Tensor bytes move through registered GPU storage. Two sending buffers per source permit overlap; one receiving buffer per source lets receivers drain independent sources without blocking training collectives. Native model loaders receive owned copies so they can retain paired projections across batches.

The normal per-source bucket capacity is `max(1 MiB, update_weight_buffer_size / training_world_size)`. A tensor is never split, so the buffer setting is **not a hard GPU memory limit**. Oversized tensors, conversion temporaries, receive buffers, and loader-owned copies add to peak memory. Successful updates release idle registered buffers and reset provider connections; they do not retain GPU transfer buffers between updates.

## Completion And Failure

Generation is paused during synchronization. Cache handling follows `--flush-cache-interval`. Each transfer validates version, consumer identity, sequence, and tensor bounds. Source buffers can be overwritten only after all consumers acknowledge their reads. Receive buffers can be reused only after the loader's CUDA reads complete.

All trainers and serving workers must finish loading before weight post-processing and version commit; generation resumes afterward. A partial update is not rolled back. If loading or version commit fails, generation stays paused and the updater rejects retries. Restart affected trainers and engines, then perform a fresh full sync. Training TP groups are aborted when a peer failure would otherwise strand a collective. Buffers that may still be accessed by DMA are retained until process exit.

The implementation is in `slime/utils/p2p/`: `stream.py` owns the protocol, `buffers.py` owns registered storage, and `mooncake.py` implements the data-plane contract in `transport.py`. Megatron ownership/conversion and the SGLang native-loader adapter remain in their respective backends.

## Verification

The CPU tests cover stream ordering, concurrent sources, buffer reuse, read failures, ownership, and commit ordering:

```bash
python tests/test_p2p_stream.py
python tests/test_p2p_mooncake_transport.py
python tests/test_p2p_weight_loader.py
python tests/test_p2p_weight_update.py
```

On a machine with four suitable GPUs and Mooncake installed:

```bash
python tests/test_p2p_qwen3_moe.py
```

This e2e test builds a tiny Qwen3 MoE checkpoint locally, runs Megatron with TP=2/EP=2 and SGLang with TP=2/EP=2, and performs repeated updates through the actual backend. Serving log probabilities are compared to independently saved HF checkpoints, and the test checks that training changed expert weights. It needs no downloaded model or dataset. Multi-node RDMA and mixed-runtime deployments need additional validation on the target cluster.

The PD variants require six GPUs: two each for training, prefill, and decode. Both serving groups use Mooncake KV transfer alongside P2P weight updates:

```bash
python tests/test_p2p_qwen3_moe.py --pd
python tests/test_p2p_qwen3_moe.py --pd --offload-rollout
```

These exercise repeated generations after weight-engine reset, with and without KV memory offload. They run synchronous rollouts; they do not validate updating during an in-flight PD request. A passing CPU suite does not establish RDMA or PD compatibility: run these GPU cases on the target software stack and network before relying on this combination.
