# Knowledge Distillation Example

This example shows how to run **knowledge distillation (KD)** using slime. A student model learns from a teacher model by minimizing KL divergence on teacher-generated trajectories.

## Key Features

- **Online KD**: Teacher generates trajectories via external SGLang server (`--rm-url`)
- **Offline KD**: Load pre-saved teacher data from JSONL files (no teacher server needed)
- **Top-K KL**: Forward KL on teacher's top-K tokens (configurable via `KD_TOP_K`)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KD_TOP_K` | `8` | Top-K tokens for KL (0 = sampled KL on generated tokens) |
| `KD_TEMPERATURE` | `1.0` | Temperature for top-K KL loss |
| `KD_SAVE_PATH` | - | Save teacher data (supports `{rollout_id}` placeholder) |
| `KD_LOAD_PATH` | - | Load teacher data for offline KD |

## Components

- `knowledge_distillation.py`: Online rollout function that calls teacher server and optionally saves data
- `offline_kd.py`: Offline rollout function that loads pre-saved teacher data
- `kd_loss.py`: KD loss functions (top-K KL and sampled KL)

## Data Format

Teacher data is saved as JSONL with metadata header:

```json
{"__metadata__": true, "distillation_type": "top_k", "top_k": 8}
{"prompt": "...", "tokens": [...], "response": "...", "response_length": 1024, "teacher_log_probs": [...], "teacher_top_k_ids": [[...]], "teacher_top_k_logprobs": [[...]]}
```

| Field | Description |
|-------|-------------|
| `tokens` | Full sequence (prompt + response token IDs) |
| `teacher_log_probs` | Teacher's log-prob for each response token |
| `teacher_top_k_ids` | Top-K token IDs at each position |
| `teacher_top_k_logprobs` | Top-K log-probs at each position |

## Running the Example

### Online KD

1. Start teacher SGLang server:
```bash
python -m sglang.launch_server --model-path /path/to/teacher --port 13141
```

2. Run training:
```bash
bash examples/knowledge_distillation/run-qwen3-1.7B-kd.sh
```

### Offline KD

```bash
# First generate teacher data with online KD (set KD_SAVE_PATH)
# Then train from saved data:
bash examples/knowledge_distillation/run-qwen3-1.7B-offline-kd.sh
```
