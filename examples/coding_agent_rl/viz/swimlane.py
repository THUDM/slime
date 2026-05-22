"""Per-sample swimlane Gantt visualization for list_trajectory segments.

Each input rollout dump file contains one or more Sample dicts; each Sample's
metadata carries the segment fields written by middleware §7 SEGMENTS
(see SPEC §6.1):

    segment_kind         "pre_wipe" | "subagent" | "final"
    completed_turns      int
    finish_reason        str
    num_aborts           int
    tito_masked_turns    int

For each sample we draw a horizontal swimlane: y = segment index, bar length
proportional to len(response_ids), color by segment_kind, red-hatch overlay
when tito_masked_turns > 0, "↻N" annotation when num_aborts > 0.

CLI:
    python -m examples.coding_agent_rl.viz.swimlane <glob> --out-dir <html_dir>
"""

from __future__ import annotations

import argparse
import collections
import glob as _glob
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Color palette per segment_kind (SPEC §9.1).
_SEGMENT_COLOR = {
    "pre_wipe": "#5C6BC0",  # blue
    "subagent": "#FF9800",  # orange
    "final": "#43A047",     # green
}
_DEFAULT_COLOR = "#9E9E9E"


def _group_segments_by_sample(samples: list[dict]) -> "collections.OrderedDict[str, list[dict]]":
    """Group fanned-out sub-samples back by their parent sample (instance_id +
    base session). Each entry's value is the ordered segment list."""
    out: "collections.OrderedDict[str, list[dict]]" = collections.OrderedDict()
    for s in samples:
        meta = (s.get("metadata") if isinstance(s, dict) else None) or {}
        instance = meta.get("instance_id") or "unknown"
        sid_base = (s.get("session_id") or "").rsplit("-", 1)[0] or instance
        key = f"{instance}@{sid_base}"
        out.setdefault(key, []).append(s)
    # Sort segments inside each group by segment_idx for chronological order.
    for k in out:
        out[k].sort(key=lambda x: (x.get("metadata") or {}).get("segment_idx", 0))
    return out


def render(samples: list[dict], out_path: str | Path, *, cols: int = 4) -> None:
    """Render a grid of swimlanes (one subplot per sample), save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - matplotlib optional
        raise RuntimeError("matplotlib required for viz/swimlane") from e

    groups = _group_segments_by_sample(samples)
    if not groups:
        logger.warning("[viz.swimlane] no samples to render")
        return
    n = len(groups)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.5), squeeze=False)

    for idx, (key, segs) in enumerate(groups.items()):
        ax = axes[idx // cols][idx % cols]
        _draw_one(ax, key, segs)
    # Hide remaining axes.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("[viz.swimlane] wrote %s (%d samples)", out_path, n)


def _draw_one(ax, title: str, segs: list[dict]) -> None:
    # X axis = cumulative response tokens; Y axis = segment index (top=earliest).
    bars: list[tuple[int, int]] = []  # (start, length)
    colors: list[str] = []
    labels: list[str] = []
    hatches: list[str | None] = []
    cur = 0
    for s in segs:
        meta = s.get("metadata") or {}
        resp_len = int(s.get("response_length") or 0)
        kind = meta.get("segment_kind") or meta.get("list_trajectory_segment_kind") or "?"
        color = _SEGMENT_COLOR.get(kind, _DEFAULT_COLOR)
        bars.append((cur, max(1, resp_len)))
        colors.append(color)
        tito = int(meta.get("tito_masked_turns") or 0)
        aborts = int(meta.get("num_aborts") or 0)
        finish = meta.get("finish_reason") or ""
        reward = float(s.get("reward") or 0.0)
        annotation = f"{kind} tok={resp_len} r={reward:.2f}"
        if tito:
            annotation += f" m={tito}"
        if aborts:
            annotation += f" ↻{aborts}"
        if finish:
            annotation += f" [{finish}]"
        labels.append(annotation)
        hatches.append("//" if tito > 0 else None)
        cur += resp_len

    for i, ((x, w), c, lbl, hatch) in enumerate(zip(bars, colors, labels, hatches)):
        ax.broken_barh([(x, w)], (i - 0.4, 0.8), facecolors=c, edgecolors="black",
                       linewidth=0.5, hatch=hatch)
        ax.text(x + w * 0.02, i, lbl, fontsize=6, va="center")

    ax.set_yticks(range(len(segs)))
    ax.set_yticklabels([f"seg{i}" for i in range(len(segs))], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(cur, 1) * 1.05)
    ax.set_title(title, fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)


def _load_samples_from_pt(path: str | Path) -> list[dict]:
    """Load a save-debug-rollout-data .pt file. Returns list of sample dicts."""
    import torch  # noqa: WPS433  (torch lazy import)
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "samples" in blob:
        return list(blob["samples"])
    if isinstance(blob, list):
        return list(blob)
    raise ValueError(f"unrecognised dump shape: {type(blob).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="paths or globs to rollout dump .pt files")
    parser.add_argument("--out-dir", required=True, help="output directory for PNG files")
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched: list[str] = []
    for pat in args.inputs:
        matched.extend(sorted(_glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not matched:
        logger.error("[viz.swimlane] no input files matched")
        return 2

    for pt in matched:
        try:
            samples = _load_samples_from_pt(pt)
        except Exception as e:
            logger.warning("[viz.swimlane] skip %s: %s", pt, e)
            continue
        out_png = out_dir / (Path(pt).stem + ".swimlane.png")
        render(samples, out_png, cols=args.cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
