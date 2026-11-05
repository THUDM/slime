# LMCache for frozen models

This example uses SGLang's existing LMCache multiprocess connector with a frozen
reference or teacher model. LMCache keeps KV in a separate process, so retained
prefixes can be reused after restarting SGLang with the **same checkpoint**. This
can avoid repeated prefill during recovery or repeated multi-turn requests to a
frozen model. It does not restore rollout or agent state.

This targets generation with reusable prefixes. Requests that require recomputing
prompt logprobs, such as full reference scoring, may not benefit from prefix reuse.

## Scope

Keep LMCache off the trainable actor in this example. slime flushes the engine's
cache during weight updates, but the SGLang LMCache MP path's local cache reset
does not clear the standalone store. A local flush alone is therefore insufficient
to make external KV safe after changing weights. Do not rely on request cache salts
for external isolation without validating support in your SGLang/LMCache versions.

Use a dedicated LMCache server for this checkpoint. If its weights change, start
with an empty store. Restarting only SGLang is appropriate when the weights, model
identity, precision, and KV layout stay compatible and the needed entries have not
been evicted. This example uses one local inference GPU; it does not validate
multi-node sharing, PD disaggregation, offload/reload, or automatic fault recovery.

## Standalone restart check

Use an environment containing compatible SGLang and LMCache installations. Match
LMCache's native extensions to the installed PyTorch/CUDA build; see the
[LMCache installation guide](https://docs.lmcache.ai/getting_started/installation.html)
and [SGLang connector documentation](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/lmcache).
Run from the slime repository root.

Start a fresh LMCache server in one terminal and leave it running throughout the check:

```bash
lmcache server --host 127.0.0.1 --port 5556 \
  --l1-size-gb 2 --eviction-policy LRU --chunk-size 256
```

In another terminal, launch SGLang:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port 31000 \
  --enable-lmcache --lmcache-config-file examples/lmcache/lmcache.yaml \
  --context-length 8192 --max-total-tokens 8192 --mem-fraction-static 0.1 \
  --disable-cuda-graph --attention-backend triton \
  --enable-cache-report --skip-server-warmup
```

After SGLang is ready, send the cold request from a third terminal:

```bash
python examples/lmcache/check_cache.py http://127.0.0.1:31000 cold
```

Wait for LMCache to log that the prefix was stored. Stop **only SGLang** with Ctrl-C,
wait for its workers to exit, and run the identical launch command again. Keep
the LMCache process running. Once the replacement engine is ready:

```bash
python examples/lmcache/check_cache.py http://127.0.0.1:31000 warm
```

The cold check requires zero cached tokens. The warm check requires an external
host-cache hit and zero device-cache hits, so a local radix-cache hit cannot pass.
These checks use `/generate`, the native endpoint used by slime rollouts. They
verify cache reuse, not numerical parity or an RL throughput improvement. Start
with a fresh LMCache server before repeating the full experiment.

### Validation

The standalone restart check passed with Qwen3-0.6B in BF16, SGLang 0.5.20
(commit `94602c9`), LMCache 0.5.5, PyTorch 2.13.0+cu130, Python 3.12, and one
NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. The cold request reported zero
cached tokens. After stopping and relaunching SGLang while leaving LMCache running,
the same prompt reported 2560/2561 cached tokens, all from the host and none from
the device cache.

The slime model configuration and argument forwarding were checked with SGLang
0.5.15.post1, the version used by slime's Docker setup. The standalone result does
not establish compatibility between slime and SGLang 0.5.20; keep the supported
SGLang version in the training environment. A complete multi-model RL run remains
unvalidated.

## Use a frozen model in slime

[`sglang.yaml`](sglang.yaml) declares one actor GPU and one frozen reference GPU.
The actor inherits `--hf-checkpoint` and receives weight updates. Only the `ref`
model enables LMCache, with `update_weights: false` explicitly set even if its
checkpoint initially matches the actor's.

Add these options to your existing non-colocated, multi-model training command:

```bash
--sglang-config examples/lmcache/sglang.yaml \
--rollout-num-gpus 2 \
--rollout-num-gpus-per-engine 1
```

Set `ref.model_path` to your frozen checkpoint and make the LMCache config file
available at the specified path in the rollout workers. The example's relative
path assumes they start in the repository root; use an absolute path otherwise.
Keep the LMCache daemon on the same host as the reference engine.

Your custom rollout must explicitly call the reference model, using
`get_model_url(args, "ref", "/generate")`; declaring a reference engine does not
automatically add a reference loss or teacher query. See the
[multi-model guide](../../docs/en/advanced/sglang-config.md#3-multi-model-serving).

The YAML overrides are ordinary SGLang `ServerArgs` fields. For a standalone
rollout-only configuration, the equivalent CLI arguments are
`--sglang-enable-lmcache --sglang-lmcache-config-file examples/lmcache/lmcache.yaml`.
Avoid applying these globally to a training job, where they would also enable
LMCache on the actor. No new slime backend or mandatory dependency is needed.
