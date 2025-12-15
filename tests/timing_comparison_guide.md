# Qwen32B P2P vs Distributed Transfer Performance Comparison

## Overview
This document describes the timing measurements for comparing P2P Transfer Engine performance against torch.distributed baseline.

## Test Methods

### 1. P2P Transfer Test: `test_qwen32b_model_transfer`
**Command**:
```bash
python -m pytest tests/test_p2p_engine_model_qwen2.py::TestQwen32BP2PTransfer::test_qwen32b_model_transfer -v -s --tb=short --capture=no
```

**Results**: Saved to `/root/slime/tests/p2p_timing_results.json`

### 2. Distributed Baseline Test: `test_qwen32b_model_transfer_baseline`
**Command**:
```bash
python -m pytest tests/test_p2p_engine_model_qwen2.py::TestQwen32BP2PTransfer::test_qwen32b_model_transfer_baseline -v -s --tb=short --capture=no
```

**Results**: Saved to `/root/slime/tests/distributed_timing_results.json`

## Timing Metrics

### P2P Transfer Engine Metrics

#### Training Side:
- `register_and_start_time`: Time to register weights and start the P2P training engine
- `update_weights_time`: Time to update/re-register weights
- `stop_time`: Time to stop the training engine


#### Rollout Side:
- `submit_tasks_time`: Time to submit all transfer tasks
- `wait_transfers_time`: Time waiting for all transfers to complete
- `sync_time`: CUDA synchronization time
- `total_transfer_time`: Pure transfer time (submit to sync)

### Distributed Baseline Metrics

#### Training Side:
- `init_process_group_time`: Time to initialize NCCL process group
- `broadcast_time`: Time to broadcast all weights to other processes
- `destroy_group_time`: Time to destroy the process group


#### Rollout Side:
- `init_process_group_time`: Time to initialize NCCL process group
- `broadcast_time`: Time to receive weights via broadcast
- `sync_time`: CUDA synchronization time
- `destroy_group_time`: Time to destroy the process group
- `total_transfer_time`: Pure transfer time (broadcast + sync)


## Key Performance Comparisons

1. **Core Transfer Time**:
   - P2P: `wait_transfers_time` vs Distributed: `broadcast_time`

2. **Setup/Teardown Overhead**:
   - P2P: `register_and_start_time` + `stop_time` vs Distributed: `init_process_group_time` + `destroy_group_time`

3. **Total Transfer Pipeline**:
   - P2P: `total_transfer_time` vs Distributed: `total_transfer_time`
