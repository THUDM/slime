#!/usr/bin/env bash
# 96-GPU (12 node × 8 H200) production launcher for Qwen3.5-35B-A3B + SWE-Rebench
# on G_C2 sandbox spec. Point the CQ-科研驾驶舱 job submission "command" field
# at this file (full absolute path).
#
# Topology:
#   - Actor : 8 nodes × 8 GPU = 64 GPU  (TP=1, CP=32, EP=32, DP=2)
#   - Rollout: 4 nodes × 8 GPU = 32 GPU (4 SGLang engines × 8 GPU)
#   - SWE_SAMPLE_CONCURRENCY=32 (sandbox burst-safe ceiling per benchmark 2026-05-11)
set -euo pipefail
cd /inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jy_workspace

NUM_NODES=12 \
  ACTOR_NUM_NODES=8 \
  ROLLOUT_GPUS_TOTAL=32 \
  SWE_SAMPLE_CONCURRENCY=32 \
  bash slime/examples/sandbox_env/scripts/train/run_qwen3_5_35b_a3b_swe_inspire_gc2_prod.sh
