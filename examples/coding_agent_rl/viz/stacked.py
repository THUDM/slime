"""Horizontal stacked-bar view: 1 row per sample, segments stacked left to
right. Used to scan 100s of samples at once and spot per-sample mask-rate /
reward / segment-distribution outliers.

CLI:
    python -m examples.coding_agent_rl.viz.stacked <glob> --out <png>
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import os
from pathlib import Path

from . import swimlane as _swim

logger = logging.getLogger(__name__)

_SEGMENT_COLOR = _swim._SEGMENT_COLOR
_DEFAULT_COLOR = _swim._DEFAULT_COLOR


def render(samples: list[dict], out_path: str | Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("matplotlib required for viz/stacked") from e

    groups = _swim._group_segments_by_sample(samples)
    if not groups:
        logger.warning("[viz.stacked] no samples to render")
        return

    rows = len(groups)
    fig, ax = plt.subplots(figsize=(14, max(2, rows * 0.35)))
    for y, (key, segs) in enumerate(groups.items()):
        x = 0
        total_mask = 0
        total_resp = 0
        reward = 0.0
        for s in segs:
            meta = s.get("metadata") or {}
            resp_len = int(s.get("response_length") or 0)
            kind = meta.get("segment_kind") or meta.get("list_trajectory_segment_kind") or "?"
            color = _SEGMENT_COLOR.get(kind, _DEFAULT_COLOR)
            tito = int(meta.get("tito_masked_turns") or 0)
            ax.broken_barh([(x, max(1, resp_len))], (y - 0.4, 0.8),
                           facecolors=color, edgecolors="black", linewidth=0.3,
                           hatch="//" if tito > 0 else None)
            x += resp_len
            total_resp += resp_len
            total_mask += tito
            reward = float(s.get("reward") or 0.0) * len(segs)  # approx total
        ax.text(x + 200, y, f"r={reward:.1f} mask={total_mask}", fontsize=6, va="center")

    ax.set_yticks(range(rows))
    ax.set_yticklabels(list(groups.keys()), fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("response tokens (cumulative)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("[viz.stacked] wrote %s (%d samples)", out_path, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="paths/globs to .pt files")
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    matched: list[str] = []
    for pat in args.inputs:
        matched.extend(sorted(_glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not matched:
        logger.error("[viz.stacked] no input files matched")
        return 2

    all_samples: list[dict] = []
    for pt in matched:
        try:
            all_samples.extend(_swim._load_samples_from_pt(pt))
        except Exception as e:
            logger.warning("[viz.stacked] skip %s: %s", pt, e)
    render(all_samples, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
