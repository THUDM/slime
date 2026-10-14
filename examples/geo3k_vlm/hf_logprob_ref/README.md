# HF reference log-prob parity

Scores the tokens SGLang sampled with plain `transformers` and compares them with the Megatron
forward. HF is a third implementation: it reads the original checkpoint and shares no code or
weight conversion with slime, so it separates "SGLang is wrong" from "slime's trainer and engine
are wrong in the same way". Expected values are in `../README_nemotron35.md` §4.2.

| File | Role |
| --- | --- |
| `hf_score.py` | Teacher-forced HF log-probs for a rollout dump; pre-checks C1-C4; SGLang-vs-HF summary |
| `compare.py` | SG / MG / HF three-way table and per-fifth drift |
| `run_hf_score.sh` | Runs `hf_score.py` on the dumps of a run, on one node |

Notes that matter for the numbers:
- Temperature is the run's `--rollout-temperature` (0.8); every response position is scored.
- transformers clamps Mamba `dt >= 0.001`; the checkpoint, Megatron and SGLang do not.
  `hf_score.py` restores the checkpoint's `time_step_limit`, otherwise HF drifts from the trained model.
- Eager attention and the torch SSM path, on purpose: the script exits if a fused kernel is bound.
- Only rollout 0 and eval 0 are comparable: their weights are still the original checkpoint.

## Reproduce

```bash
# 1. In the training environment: cold-start run that dumps rollout 0, eval 0 and the train-side log-probs.
SLIME_SCRIPT_NUM_ROLLOUT=1 SLIME_SCRIPT_EXTRA_ARGS="--dump-details $DUMP" \
    bash examples/geo3k_vlm/run_geo3k_nemotron35.sh

# 2. In an environment with transformers >= 5.8 and without mamba-ssm / causal-conv1d.
#    Smoke test first: append --limit 8 --repeat 2 (the repeat also reports the HF-vs-HF noise floor).
HF_MODEL_PATH=/path/to/NVIDIA-Nemotron-3.5-Super-EA-09112026 \
    bash examples/geo3k_vlm/hf_logprob_ref/run_hf_score.sh $DUMP

# 3. Compare (torch only).
python3 examples/geo3k_vlm/hf_logprob_ref/compare.py \
    --hf-score $DUMP/hf_score.pt --train-data $DUMP/train_data/0.pt
```

`SG-MG` reproduces the run's `train_rollout_logprob_abs_diff` for that step; the SG-HF and MG-HF
gaps should be similar to each other, and the per-fifth table should show no growth along the response.
