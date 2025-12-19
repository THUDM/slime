# Tau-Bench: Multi-Turn Tool-Use Training

Canonical guide: `examples/tau-bench/training_cookbook.md`.

This folder provides two benchmark entrypoints with parallel conventions:

| Benchmark | Repo | Domains | Dual-control | Primary metric | Entrypoint |
|----------|------|---------|--------------|----------------|------------|
| Tau1 | https://github.com/sierra-research/tau-bench | airline, retail | no | pass@1 | `examples/tau-bench/tau1/README.md` |
| Tau2 | https://github.com/sierra-research/tau2-bench | airline, retail, telecom | yes (telecom user-only tools) | pass@4 (headline) + pass@1 | `examples/tau-bench/tau2/README.md` |

### Quick Links

- Canonical walkthrough: `examples/tau-bench/training_cookbook.md`
- Tau1 entrypoint (kept verbatim): `examples/tau-bench/tau1/README.md`
- Tau2 entrypoint: `examples/tau-bench/tau2/README.md`

### Outputs

All generated artifacts are written under `TAU_BENCH_OUT_DIR` (default: `examples/tau-bench/outputs`) and are gitignored. The cookbook assumes the `slimerl/slime:latest` container baseline.
