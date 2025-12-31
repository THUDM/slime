# Tau-Bench: Multi-Turn Tool-Use Training

This folder provides two benchmark entrypoints with parallel conventions. The canonical documentation lives in `examples/tau-bench/training_cookbook.md`; other docs link into it without duplication.

| Benchmark | Repo | Domains | Dual-control | Primary metric | Folder |
|----------|------|---------|--------------|----------------|--------|
| Tau1 | https://github.com/sierra-research/tau-bench | airline, retail | no | pass@1 | `examples/tau-bench/tau1/` |
| Tau2 | https://github.com/sierra-research/tau2-bench | airline, retail, telecom | yes (telecom user-only tools) | pass@4 + pass@1 | `examples/tau-bench/tau2/` |

### Quick Links
- Training cookbook: `examples/tau-bench/training_cookbook.md`.
- Tau1 README: `examples/tau-bench/tau1/README.md`.
- Tau2 implementation: `examples/tau-bench/tau2/README.md`

Note: Tau1 includes a small offline stub for debug/CI without external API keys.

### Outputs

All generated artifacts are written under `TAU_BENCH_OUT_DIR` (default: `examples/tau-bench/outputs`) and are gitignored. The cookbook assumes the `slimerl/slime:latest` container baseline.
