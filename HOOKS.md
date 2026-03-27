# Training Loop Hooks

Lightweight hook system for observability and extensibility of the SLIME training loop.

**Issue:** https://github.com/THUDM/slime/issues/1728

## Problem

Downstream consumers of SLIME (custom telemetry, logging, profiling) need to wrap
key training loop operations without modifying internal function signatures or
monkey-patching. Changes to `train.py` and `train_async.py` should be minimal and
conflict-resistant when syncing with upstream.

## Design

`slime/hooks.py` provides:

- **`Op` enum** -- names every hookable point in the training pipeline
- **`hook(op, rollout_id)`** -- context manager that fires pre/post callbacks
- **`on_pre(op, fn)` / `on_post(op, fn)`** -- callback registration

When no callbacks are registered, `hook()` is a near-zero-cost no-op.

### Hooked operations

#### Training loop (train.py / train_async.py)

```
for rollout_id in range(...):
    ITERATION
    |-- EVAL                    # pre-train eval (rollout_id == 0)
    |-- GENERATE                # ray.get(rollout_manager.generate.remote())
    |-- OFFLOAD_ROLLOUT         # ray.get(rollout_manager.offload.remote())
    |-- TRAIN                   # ray.get(actor_model.async_train())
    |-- SAVE_MODEL              # actor_model.save_model()
    |-- OFFLOAD_TRAIN           # actor_model.offload() / clear_memory()
    |-- ONLOAD_ROLLOUT_WEIGHTS
    |-- UPDATE_WEIGHTS          # actor_model.update_weights()
    |-- ONLOAD_ROLLOUT_KV
    |-- EVAL                    # periodic eval
    +-- ASYNC_ROLLOUT_SYNC      # train_async.py only
```

All training loop hooks receive `rollout_id` as their first argument.

#### Per-node lifecycle

```
NODE_INIT                       # fires once per Ray actor process during init
```

`NODE_INIT` fires in `TrainRayActor.init()` and `RolloutManager.__init__()`,
ensuring every node in a multi-node run can start background tasks like
hardware metrics collection.  It receives `node_name` (from `SLIME_NODE_NAME`
env var) instead of `rollout_id`.

Set `SLIME_NODE_NAME` in your launcher to a value that identifies the physical
node (e.g. EC2 instance ID, k8s node name, hostname).

### Call-site examples

```python
# Training loop operation (train.py)
from slime.hooks import Op, hook

with hook(Op.GENERATE, rollout_id):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
```

```python
# Per-node lifecycle (slime/ray/train_actor.py)
with hook(Op.NODE_INIT, node_name=os.environ.get("SLIME_NODE_NAME", "unknown")):
    pass
```

## Example: OpenTelemetry tracing

This is how we use hooks in our fork to add OTel span tracing to training runs.
No SLIME source code is modified beyond what's in this PR -- all OTel logic lives
in our downstream package.

### Registering callbacks

```python
from opentelemetry import context as context_api
from opentelemetry import trace

from slime.hooks import Op, on_pre, on_post


def register_otel_hooks():
    """Register OTel span callbacks for all hook operations."""
    for op in Op:
        if op is Op.NODE_INIT:
            # NODE_INIT starts metrics polling only, no span.
            on_pre(op, lambda **_: start_metrics_collection())
            continue

        span_name = op.value

        def on_start(**kwargs):
            tracer = trace.get_tracer(__name__)
            attrs = {}
            rollout_id = kwargs.get("rollout_id")
            if rollout_id is not None:
                attrs["rollout.id"] = rollout_id

            span = tracer.start_span(span_name, attributes=attrs)
            token = context_api.attach(trace.set_span_in_context(span))
            return {"_otel_span": span, "_otel_token": token}

        def on_end(error=None, **kwargs):
            span = kwargs.get("_otel_span")
            token = kwargs.get("_otel_token")
            if span is not None:
                if error is not None:
                    span.set_status(trace.StatusCode.ERROR, str(error))
                    span.record_exception(error)
                span.end()
            if token is not None:
                context_api.detach(token)

        on_pre(op, on_start)
        on_post(op, on_end)
```

### Activating in the entry point

```python
# Called once before training starts (e.g. in your launcher entry point)
register_otel_hooks()
```

### What this produces

Each training iteration produces a trace like:

```
iteration (rollout_id=0)                          102.90s
|-- eval                                           24.82s
|-- generate                                       15.64s
|-- train                                          53.49s
|-- update_weights                                  2.20s
|-- offload_train                                   0.01s
+-- onload_rollout_weights                          0.10s
```

Spans nest automatically -- `generate` and `train` are children of `iteration`
because OTel context propagates within the process.

### Per-node metrics via NODE_INIT

The `NODE_INIT` hook fires once per Ray actor process.  Downstream consumers
use it to start background metric collectors (GPU utilization, memory, CPU)
on every node without requiring any per-iteration overhead.

Since `NODE_INIT` is not a training operation, it typically does not produce
a span -- just a side effect like starting a polling thread.

### Pre-callback state injection

Pre callbacks can return a dict to inject state into post callbacks. This is how
the OTel example passes the span handle from `on_start` to `on_end`:

```python
def on_start(**kwargs):
    span = tracer.start_span(...)
    return {"_otel_span": span}    # injected into on_end's **kwargs

def on_end(**kwargs):
    span = kwargs["_otel_span"]    # received from on_start
    span.end()
```

### Error handling

Post callbacks always fire, even if the wrapped operation raises. The exception
is passed as `error`:

```python
def on_end(error=None, **kwargs):
    if error is not None:
        # operation failed
        span.set_status(trace.StatusCode.ERROR, str(error))
    span.end()
```

Callback exceptions are logged but never propagate -- hooks cannot break training.
