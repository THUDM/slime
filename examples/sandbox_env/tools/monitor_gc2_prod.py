#!/usr/bin/env python3
"""Monitor gc2 prod training: parse run.log, write summary, flag anomalies.

Output:
  - data_output/monitor_gc2_prod.md   ← human-readable summary
  - data_output/monitor_gc2_prod.json ← machine-readable state (last_step + alerts)
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jy_workspace")
LOG = ROOT / "slime/examples/sandbox_env/output/qwen3_5_35b_gc2_prod/qwen3.5-35b-a3b-swe-inspire-gc2/logs/run.log"
SUMMARY_MD = ROOT / "slime/examples/sandbox_env/data_output/monitor_gc2_prod.md"
STATE_JSON = ROOT / "slime/examples/sandbox_env/data_output/monitor_gc2_prod.json"

ALARM_LOW_PASS_STREAK = 5  # 5 consecutive perf steps with pass_rate < 5%
ALARM_GRAD_EXPLODE = 50.0  # grad_norm > 50
ALARM_STALE_MIN = 30  # log not advancing for 30 min => probably dead


def parse_log():
    steps = {}
    perfs = {}
    sandbox_201 = sandbox_4xx5xx = burst = 0
    fatal_lines = []
    pat_step = re.compile(r"step (\d+): \{(.+?)\}")
    pat_perf = re.compile(r"perf (\d+): \{(.+?)\}")
    pat_201 = re.compile(r"Response 201")
    pat_45 = re.compile(r"Response (4\d\d|5\d\d)")
    pat_burst = re.compile(r"too many sandboxes starting|ResourceExhausted|No available resources", re.I)
    pat_fatal = re.compile(r"(MemoryError|Server process terminated|ChildFailedError|RuntimeError: CUDA|FATAL|OutOfMemoryError|Out of memory)")
    if not LOG.exists():
        return None
    with LOG.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if pat_201.search(line): sandbox_201 += 1
            if pat_45.search(line): sandbox_4xx5xx += 1
            if pat_burst.search(line): burst += 1
            if pat_fatal.search(line): fatal_lines.append(line.strip()[:200])
            m = pat_step.search(line)
            if m:
                sn = int(m.group(1)); body = m.group(2)
                d = {}
                for k in ("train/pg_loss", "train/grad_norm", "train/entropy_loss",
                         "train/pg_clipfrac", "train/ppo_kl"):
                    mk = re.search(rf"'{re.escape(k)}':\s*([\-\d.eE+]+)", body)
                    if mk: d[k] = float(mk.group(1))
                steps[sn] = d
            m = pat_perf.search(line)
            if m and "'swe/pass_rate'" in m.group(2):  # rollout perf, not train perf
                sn = int(m.group(1)); body = m.group(2)
                d = {}
                for k in ("swe/pass_rate", "swe/success_count", "swe/sample_count",
                         "rollout/zero_std/count_0.0", "rollout/zero_std/count_1.0",
                         "swe/rollout_time_sec", "rollout/repetition_frac"):
                    mk = re.search(rf"'{re.escape(k)}':\s*([\d.]+)", body)
                    if mk: d[k] = float(mk.group(1))
                perfs[sn] = d
    return dict(steps=steps, perfs=perfs,
                sandbox_201=sandbox_201, sandbox_4xx5xx=sandbox_4xx5xx,
                burst=burst, fatal_lines=fatal_lines)


def main():
    if not LOG.exists():
        SUMMARY_MD.write_text(f"# monitor — LOG NOT FOUND\nexpected: {LOG}\n")
        sys.exit(1)
    log_mtime = LOG.stat().st_mtime
    now = time.time()
    stale_min = (now - log_mtime) / 60
    data = parse_log()
    if not data:
        return
    steps = data["steps"]
    perfs = data["perfs"]
    n_train = len(steps)
    n_perf = len(perfs)
    last_perf = max(perfs) if perfs else -1
    last_step = max(steps) if steps else -1

    # Compute recent pass-rate trend
    last_perfs = sorted(perfs.items())[-10:]
    recent_pass = [p[1].get("swe/pass_rate", 0) for p in last_perfs]
    recent_grad = [steps[s].get("train/grad_norm", 0) for s in sorted(steps)[-10:]]

    # Anomaly checks
    alerts = []
    if stale_min > ALARM_STALE_MIN:
        alerts.append(f"⚠️ STALE: run.log not updated for {stale_min:.0f} min (cap {ALARM_STALE_MIN}). Process may be dead.")
    if data["fatal_lines"]:
        alerts.append(f"⚠️ FATAL log lines: {len(data['fatal_lines'])} found. Latest: {data['fatal_lines'][-1]}")
    # 5 consecutive pass_rate < 5%
    low_streak = 0
    for p in recent_pass:
        if p < 0.05: low_streak += 1
        else: low_streak = 0
    if low_streak >= ALARM_LOW_PASS_STREAK:
        alerts.append(f"⚠️ LOW PASS STREAK: {low_streak} consecutive rollouts with pass_rate < 5%")
    if any(g > ALARM_GRAD_EXPLODE for g in recent_grad):
        alerts.append(f"⚠️ GRAD EXPLODE: grad_norm > {ALARM_GRAD_EXPLODE} observed. recent: {[round(g,1) for g in recent_grad]}")

    burst_rate = data["burst"] / max(data["sandbox_201"], 1)
    if burst_rate > 0.10:
        alerts.append(f"⚠️ SANDBOX BURST: {burst_rate:.1%} burst-failure rate ({data['burst']}/{data['sandbox_201']})")

    # Write summary
    lines = []
    lines.append("# gc2 prod training monitor — auto-updated by Claude")
    lines.append("")
    lines.append(f"updated at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"run.log: mtime {datetime.fromtimestamp(log_mtime, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}, stale {stale_min:.1f} min, size {LOG.stat().st_size:,} B")
    lines.append(f"train_steps_seen: {n_train}, perf_rollouts_seen: {n_perf}")
    lines.append("")
    if alerts:
        lines.append("## 🚨 ALERTS")
        for a in alerts:
            lines.append(f"- {a}")
        lines.append("")
    else:
        lines.append("## ✓ no alerts")
        lines.append("")

    lines.append("## last 10 rollouts")
    lines.append("")
    lines.append("| step | pass_rate | success | zero_std(0/1) | grad_norm | entropy | rollout_s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for sn in sorted(set(perfs) | set(steps))[-10:]:
        p = perfs.get(sn, {})
        s = steps.get(sn, {})
        zs0 = int(p.get("rollout/zero_std/count_0.0", 0))
        zs1 = int(p.get("rollout/zero_std/count_1.0", 0))
        pr = p.get("swe/pass_rate")
        sc = int(p.get("swe/success_count", 0))
        sn_count = int(p.get("swe/sample_count", 0))
        gn = s.get("train/grad_norm")
        ent = s.get("train/entropy_loss")
        rt = int(p.get("swe/rollout_time_sec", 0))
        lines.append(f"| {sn} | {pr if pr is not None else '-'} | {sc}/{sn_count or 32} | {zs0}/{zs1} | "
                     f"{f'{gn:.3f}' if gn is not None else '-'} | "
                     f"{f'{ent:.3f}' if ent is not None else '-'} | "
                     f"{rt or '-'} |")

    lines.append("")
    lines.append("## summary stats")
    if recent_pass:
        lines.append(f"- recent_pass_rates (last {len(recent_pass)}): {[round(p,3) for p in recent_pass]}")
        lines.append(f"- mean pass_rate: {sum(recent_pass)/len(recent_pass):.3f}")
    if recent_grad:
        lines.append(f"- recent_grad_norms: {[round(g,2) for g in recent_grad]}")
    lines.append(f"- sandbox: {data['sandbox_201']} OK / {data['sandbox_4xx5xx']} 4xx5xx / {data['burst']} burst")

    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    STATE_JSON.write_text(json.dumps({
        "ts": int(now),
        "last_train_step": last_step,
        "last_perf_step": last_perf,
        "stale_min": stale_min,
        "alerts": alerts,
        "recent_pass_rates": recent_pass,
        "recent_grad_norms": recent_grad,
        "sandbox_burst": data["burst"],
    }, indent=2))
    print(f"[monitor] last_step={last_step} last_perf={last_perf} pass_rate_recent={recent_pass}")
    if alerts:
        for a in alerts: print(f"[ALERT] {a}")


if __name__ == "__main__":
    main()
