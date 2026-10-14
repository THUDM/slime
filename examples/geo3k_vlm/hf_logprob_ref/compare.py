#!/usr/bin/env python3
"""SGLang vs Megatron vs HF on one rollout (login node, torch only).

    python3 compare.py --hf-score hf_score.pt --train-data <dump>/train_data/0.pt

SG = the rollout dump's rollout_log_probs; MG = the train dump's log_probs (Megatron's forward
before the optimizer step, same weights as the rollout); HF = hf_score.py on rollout_data/0.pt.
SG-MG reproduces that step's train_rollout_logprob_abs_diff. Every mean|d| is slime's
sample mean: |d| averaged within a sample over loss-mask tokens, then over samples.
"""
import argparse

import torch


def sample_mean(ds):
    v = [float(d.abs().mean()) for d in ds if d.numel()]
    return sum(v) / len(v) if v else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-score", required=True)
    p.add_argument("--train-data", required=True)
    args = p.parse_args()

    hf = torch.load(args.hf_score, weights_only=False)
    recs = {r["index"]: r for r in hf["records"] if "lp_hf" in r}
    train = torch.load(args.train_data, weights_only=False)

    rows = []
    for t in train["samples"]:
        r = recs.get(t["rollout_position"])
        if r is None:
            continue
        m = torch.as_tensor(t["loss_masks"]).bool()
        sg, mg, h = (torch.as_tensor(t["rollout_log_probs"]).float(), torch.as_tensor(t["log_probs"]).float(),
                     r["lp_hf"].float())
        if not (len(sg) == len(mg) == len(h)) or not torch.equal(sg, r["lp_sglang"].float()):
            continue  # train dump and HF scores must describe the same rollout
        rows.append({"sg": sg[m], "mg": mg[m], "hf": h[m]})
    print(f"aligned {len(rows)} of {len(train['samples'])} train samples ({len(recs)} HF-scored)")
    if not rows:
        return 1

    pairs = {"SG-MG": ("sg", "mg"), "SG-HF": ("sg", "hf"), "MG-HF": ("mg", "hf")}
    print("\n          mean|d|   mean d")
    for name, (a, b) in pairs.items():
        ds = [r[a] - r[b] for r in rows]
        print(f"  {name}  {sample_mean(ds):.5f}  {float(torch.cat(ds).mean()):+.5f}")

    print("\nmean|d| by relative position in the response (fifths):")
    for q in range(5):
        cells = []
        for name, (a, b) in pairs.items():
            part = torch.cat([(r[a][q * len(r[a]) // 5:(q + 1) * len(r[a]) // 5]
                               - r[b][q * len(r[a]) // 5:(q + 1) * len(r[a]) // 5]).abs() for r in rows])
            cells.append(f"{name} {float(part.mean()):.4f}")
        print(f"  [{q / 5:.1f}, {(q + 1) / 5:.1f})  " + "   ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
