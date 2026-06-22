"""Pretty-print a --save-debug-rollout-data .pt dump to a human-readable text file.

Each dump is {"rollout_id": int, "samples": [Sample.to_dict(), ...]}; every sample
carries the decoded `response` (the <think>/<tool_call>/<observation> completion),
plus prompt, reward, status and metadata (env_reward, env_won, move_count).

Usage:
    python view_rollout_dump.py PATH.pt [-o OUT.txt] [--limit N] [--won] [--prompt]
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="rollout_dumps/<rollout_id>.pt file")
    ap.add_argument("-o", "--out", default=None, help="output .txt (default: alongside the .pt)")
    ap.add_argument("--limit", type=int, default=None, help="max samples to write (default all)")
    ap.add_argument("--won", action="store_true", help="only episodes that reached the GOAL")
    ap.add_argument("--prompt", action="store_true", help="include the opening prompt per sample")
    args = ap.parse_args()

    blob = torch.load(args.path, weights_only=False)
    samples = blob["samples"]
    out_path = args.out or os.path.splitext(args.path)[0] + ".txt"

    n = len(samples)
    won = sum(1 for s in samples if (s.get("metadata") or {}).get("env_won"))
    rl = [s.get("response_length", 0) for s in samples]

    with open(out_path, "w") as f:
        f.write(f"rollout_id={blob.get('rollout_id')}   samples={n}   won={won}/{n}\n")
        f.write(f"response_length: min={min(rl)} max={max(rl)} mean={sum(rl)//n}\n")
        f.write("=" * 100 + "\n")

        written = 0
        for i, s in enumerate(samples):
            meta = s.get("metadata") or {}
            if args.won and not meta.get("env_won"):
                continue
            if args.limit is not None and written >= args.limit:
                break
            written += 1

            f.write(
                f"\n#### SAMPLE {i}  "
                f"status={s.get('status')}  reward={s.get('reward')}  "
                f"won={meta.get('env_won')}  moves={meta.get('move_count')}  "
                f"resp_len={s.get('response_length')}  seed={s.get('label')}\n"
            )
            f.write("-" * 100 + "\n")
            if args.prompt:
                f.write("----- PROMPT -----\n")
                f.write(str(s.get("prompt")).rstrip() + "\n\n")
            f.write("----- RESPONSE -----\n")
            f.write(str(s.get("response")).rstrip() + "\n")
            f.write("=" * 100 + "\n")

    print(f"wrote {written} samples to {out_path}")


if __name__ == "__main__":
    main()
