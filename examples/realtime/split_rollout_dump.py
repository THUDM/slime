"""Split a --save-debug-rollout-data .pt dump into one text file per completion.

Each dump is {"rollout_id": int, "samples": [Sample.to_dict(), ...]}. This writes
one human-readable file per sample into a directory, so you can open / diff
individual rollouts. Filenames sort by index and encode outcome at a glance:
    000_grp00_won.txt   001_grp00_lost.txt   ...
(grpNN = the GRPO group, i.e. samples sharing a seed, in encounter order.)

Usage:
    python split_rollout_dump.py PATH.pt [-d OUTDIR] [--prompt]
"""

import argparse
import os


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="rollout_dumps/<rollout_id>.pt file")
    ap.add_argument("-d", "--outdir", default=None, help="output dir (default: <path>_samples/)")
    ap.add_argument("--prompt", action="store_true", help="include the opening prompt in each file")
    args = ap.parse_args()

    blob = torch.load(args.path, weights_only=False)
    samples = blob["samples"]
    outdir = args.outdir or os.path.splitext(args.path)[0] + "_samples"
    os.makedirs(outdir, exist_ok=True)

    # Assign a stable group index per seed in first-seen order (GRPO group).
    group_of = {}
    width = len(str(len(samples) - 1))
    for i, s in enumerate(samples):
        meta = s.get("metadata") or {}
        seed = s.get("label")
        gid = group_of.setdefault(seed, len(group_of))
        outcome = "won" if meta.get("env_won") else (s.get("status") or "?")
        fname = f"{i:0{width}d}_grp{gid:02d}_{outcome}.txt"
        with open(os.path.join(outdir, fname), "w") as f:
            f.write(
                f"sample={i}  group={gid}  seed={seed}\n"
                f"status={s.get('status')}  reward={s.get('reward')}  "
                f"won={meta.get('env_won')}  moves={meta.get('move_count')}  "
                f"resp_len={s.get('response_length')}\n"
            )
            f.write("=" * 100 + "\n")
            if args.prompt:
                f.write("----- PROMPT -----\n" + str(s.get("prompt")).rstrip() + "\n\n")
            f.write("----- RESPONSE -----\n" + str(s.get("response")).rstrip() + "\n")

    print(f"wrote {len(samples)} files to {outdir}/")


if __name__ == "__main__":
    main()
