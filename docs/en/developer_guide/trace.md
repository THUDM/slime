# Trace Viewer

slime can attach lightweight execution traces to each rollout sample. These traces capture span-style events such as generation and reward-model calls, and they can be inspected later from a saved rollout debug dump.

![trace timeline viewer](../../_static/image/trace.png)

## Save rollout trace data

To inspect traces later, save rollout debug data during a run:

```bash
python train.py \
    ... \
    --save-debug-rollout-data /path/to/debug/rollout_{rollout_id}.pt
```

Each saved `.pt` file contains the rollout samples together with their `trace` payloads. You can also replay the same dump later with `--load-debug-rollout-data`.

## Open the timeline viewer

Use the trace viewer script on a saved rollout dump:

```bash
python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

The script generates:

- `rollout_0.trace_timeline_cache.json`
- `rollout_0.trace_timeline_viewer.html`

By default it also starts a local static server so you can open the generated HTML immediately. If you only want the files, use `--no-serve`.

## How to read the viewer

- Each row corresponds to one sample.
- Bars represent spans, while point markers represent instant events.
- Span attributes recorded at the start or end of `trace_span(...)` are shown in the details panel.
- When SGLang returns PD disaggregation timings, the viewer adds synthetic `[P]` and `[D]` lanes to break out prefill/decode work.
- When PD is not enabled, those virtual lanes are omitted automatically and the base trace still renders normally.

## Instrument custom code

For custom rollout or reward code, reuse helpers from `slime.utils.trace_utils`:

- `trace_span(target, name, attrs=...)`: record a duration span.
- `trace_event(target, name, attrs=...)`: record an instant event.
- `bind_trace(sample)`: ensure a sample already has a trace carrier before passing it across helpers or tasks.

If you want to record SGLang generation metadata in a consistent way, reuse `build_sglang_meta_trace_attrs`:

```python
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span

with trace_span(sample, "sglang_generate") as span:
    output = await post(url, payload)
    span.update(build_sglang_meta_trace_attrs(output["meta_info"]))
```

## Tips

- Save a small number of rollouts first; the viewer is easiest to read when each dump contains a manageable number of samples.
- The viewer is built from the saved `.pt` dump, so traces can be inspected offline on another machine.
- For GPU/kernel-level SGLang profiling traces, see [Profiling](./profiling.md).

