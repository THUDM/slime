#!/usr/bin/env python3
"""Teacher-forced HF log-probs for the samples an SGLang rollout already generated.

Input: the .pt files slime writes with --dump-details (rollout_data/{eval_0,0}.pt).
For each sample, one forward over the exact token ids SGLang was conditioned on and
produced, with the exact image slime sent, gives the log-prob HF assigns to every
response token (log_softmax(logits / T), as SGLang and Megatron report them).

Pre-checks per sample: C1 HF's processor reproduces slime's prompt ids, C2 prompt length
matches what SGLang reported, C3 pixel_values equal the training side's, C4 one SGLang
log-prob per response token.
"""
import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


def log(msg=""):
    print(msg, flush=True)


def ssm_is_fused():
    """True if transformers bound a fused mamba-ssm / causal-conv1d kernel instead of its torch reference."""
    from transformers.models.nemotron_h import modeling_nemotron_h as m

    for name in ("mamba2_chunk_scan", "mamba2_selective_state_update", "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(m, name, None)
        try:  # the hub-kernel decorator keeps its choice in a closure variable
            impl = inspect.getclosurevars(fn).nonlocals.get("implementation") or fn
        except TypeError:
            impl = fn
        if (getattr(impl, "__module__", "") or "").startswith(("mamba_ssm", "causal_conv1d")):
            return True
    return False


def apply_checkpoint_dt_limit(model, model_path):
    """transformers clamps Mamba dt to >= time_step_min (0.001); the checkpoint, Megatron and SGLang
    use (0, inf). Some heads run at dt ~ 1e-5, so the clamp makes HF drift from the trained model."""
    mixers = [m for m in model.modules() if hasattr(m, "time_step_limit") and hasattr(m, "A_log")]
    cfg = json.load(open(Path(model_path) / "config.json"))
    raw = (cfg.get("llm_config") or cfg).get("time_step_limit")
    if raw is None:
        return f"checkpoint has no time_step_limit; unchanged on {len(mixers)} mixers"
    lo, hi = (float(v["__float__"]) if isinstance(v, dict) else float(v) for v in raw)
    for m in mixers:
        m.time_step_limit = (lo, hi)
    return f"dt limit ({lo}, {hi}) on {len(mixers)} mixers"


def prompt_checks(sample, processor, n_prompt):
    images = (sample.get("multimodal_inputs") or {}).get("images") or None
    hf_inputs = processor(text=[sample["prompt"]], images=images, return_tensors="pt")
    sglang_prompt = (sample.get("prefix_cache_info") or {}).get("total_prompt_tokens")
    stored = (sample.get("multimodal_train_inputs") or {}).get("pixel_values")
    if images is None:
        c3 = None
    elif stored is None or "pixel_values" not in hf_inputs:
        c3 = False
    else:
        a, b = hf_inputs["pixel_values"].float(), torch.as_tensor(stored).float()
        c3 = a.shape == b.shape and torch.equal(a, b)
    lp = sample.get("rollout_log_probs") or []
    checks = {
        "c1": hf_inputs["input_ids"][0].tolist() == sample["tokens"][:n_prompt],
        "c2": sglang_prompt == n_prompt if sglang_prompt else None,
        "c3": c3,
        "c4": len(lp) == sample.get("response_length", -1) and all(v is not None for v in lp),
    }
    return checks, hf_inputs


def score_once(model, tokens, hf_inputs, n_prompt, temperature, dtype):
    input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if "pixel_values" in hf_inputs:
        kwargs["pixel_values"] = hf_inputs["pixel_values"].to(model.device, dtype)
    with torch.inference_mode():
        logits = model(**kwargs).logits[0]  # every position, fp32
        rows = logits[n_prompt - 1 : len(tokens) - 1].float()  # row t predicts token t+1
        target = input_ids[0, n_prompt:].to(rows.device)
        return torch.log_softmax(rows / temperature, -1).gather(-1, target[:, None])[:, 0].cpu()


def sample_mean_abs(deltas):
    """slime's sum_of_sample_mean: mean |d| within each sample, then over samples."""
    v = [float(d.abs().mean()) for d in deltas if d.numel()]
    return sum(v) / len(v) if v else math.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-data", action="append", required=True, help="dump .pt, repeatable")
    p.add_argument("--model-path", default=os.environ.get("HF_MODEL_PATH"))
    p.add_argument("--temperature", type=float, default=0.8, help="the run's --rollout-temperature")
    p.add_argument("--limit", type=int, default=0, help="first N samples per file; 0 = all")
    p.add_argument("--repeat", type=int, default=1, help=">= 2 also reports the HF-vs-HF noise floor")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log(f"transformers {transformers.__version__}  torch {torch.__version__}  gpus {torch.cuda.device_count()}")
    if ssm_is_fused():
        log("ERROR: a fused SSM kernel is bound; the reference side must use transformers' torch path.")
        return 2

    work = []  # read every dump before the 235 GB load, so a bad path fails fast
    for path in args.rollout_data:
        blob = torch.load(path, map_location="cpu", weights_only=False)  # holds PIL images
        samples = blob["samples"][: args.limit or None]
        log(f"{path}: rollout_id={blob.get('rollout_id')} scoring {len(samples)}")
        work += [(path, i, s) for i, s in enumerate(samples)]

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    ).eval()
    dt_note = apply_checkpoint_dt_limit(model, args.model_path)
    log(dt_note)

    records = []
    for k, (path, index, sample) in enumerate(work):
        tokens, n_resp = sample["tokens"], sample.get("response_length", 0)
        n_prompt = len(tokens) - n_resp
        rec = {
            "file": str(path), "index": index, "response_length": n_resp, "reward": sample.get("reward"),
            "lp_sglang": torch.tensor([float("nan") if v is None else v for v in (sample.get("rollout_log_probs") or [])]),
        }
        records.append(rec)
        if n_resp <= 0:
            rec["skipped"] = "empty response"
            continue
        checks, hf_inputs = prompt_checks(sample, processor, n_prompt)
        rec.update(checks)
        t0 = time.time()
        try:
            runs = [score_once(model, tokens, hf_inputs, n_prompt, args.temperature, dtype) for _ in range(args.repeat)]
        except torch.OutOfMemoryError as exc:  # the torch SSM scan is memory-hungry on the longest samples
            rec["skipped"] = f"OOM at {len(tokens)} tokens"
            torch.cuda.empty_cache()
            log(f"[{k + 1}/{len(work)}] {Path(path).name}#{index} SKIPPED ({exc.args[0][:60]})")
            continue
        rec["lp_hf"], rec["lp_hf_repeats"] = runs[0], runs[1:]
        d = rec["lp_sglang"] - rec["lp_hf"] if rec["lp_sglang"].numel() == n_resp else None
        log(f"[{k + 1}/{len(work)}] {Path(path).name}#{index} resp={n_resp} {time.time() - t0:.1f}s "
            f"checks={''.join('.' if v is not False else 'X' for v in checks.values())}"
            + (f" mean|d|={float(d.abs().mean()):.4f}" if d is not None else ""))

    scored = [r for r in records if "lp_hf" in r]
    log(f"\nscored {len(scored)}  skipped {len(records) - len(scored)}")
    for key in ("c1", "c2", "c3", "c4"):
        vals = [r.get(key) for r in scored]
        log(f"  {key}: pass {sum(v is True for v in vals)}  fail {sum(v is False for v in vals)}  n/a {sum(v is None for v in vals)}")
    ok = [r for r in scored if r["lp_sglang"].numel() == r["response_length"] and not r["lp_sglang"].isnan().any()]
    if ok:
        deltas = [r["lp_sglang"] - r["lp_hf"] for r in ok]
        flat = torch.cat(deltas)
        log(f"SGLang vs HF: {len(ok)} samples, {flat.numel()} tokens: "
            f"mean|d| {sample_mean_abs(deltas):.5f}  mean d {float(flat.mean()):+.5f}")
    noise = [r["lp_hf"] - rep for r in scored for rep in r["lp_hf_repeats"]]
    if noise:
        log(f"HF vs HF (noise floor): mean|d| {sample_mean_abs(noise):.6f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": {"model_path": args.model_path, "temperature": args.temperature, "dt": dt_note,
                         "rollout_data": args.rollout_data}, "records": records}, args.output)
    log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
