# RLVE Integration

This example shows slime training with [RLVE](https://github.com/Zhiyuan-Zeng/RLVE) verifiable environments (math/logic problems with deterministic verification).

## Scope

This integration includes 7 environments via the `rlve-gym` pip shim. The full RLVE repository contains 400+ environments.

| Included | How to extend |
|----------|---------------|
| Integration infrastructure (generate, reward, provider) | N/A |
| 7 environments via `pip install rlve-gym` | Add to YAML config |
| Curriculum learning with difficulty controllers | Set `use_controllers: true` |

The `rlve-gym` shim packages only RLVE's `Gym/` directory, avoiding the vendored slime dependency conflict.

To use all 400+ environments, clone the full [RLVE repo](https://github.com/Zhiyuan-Zeng/RLVE), add to `PYTHONPATH`, and add entries to the YAML config.

## Environment Setup

Inside `slimerl/slime` container (e.g., `slimerl/slime:v0.5.0rc0-cu126`):

```bash
pip install -e /root/slime
pip install rlve-gym

export RLVE_CONFIG_PATH=/root/slime/examples/rlve/configs/starter_pack.yaml
```

Initialize the model:

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B

cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```

Use an instruct model to ensure the model follows the `<answer>...</answer>` format.

## Running the Script

```bash
cd /root/slime
bash examples/rlve/run_qwen3_8B_instruct.sh
```

The script generates a dummy JSONL at runtime to satisfy slime's `--prompt-data` requirement. Actual prompts are generated on-the-fly in `rlve_generate.py`.

## Code Structure

```
slime rollout loop
      |
      v
rlve_generate.py    # Samples env, generates prompt
      |
      v
rlve_reward.py      # Restores env state, calls verifier
      |
      v
--reward-key accuracy   # Binary signal (0/1)
```

| File | Purpose |
|------|---------|
| `rlve_prompt_provider.py` | Samples environments by weight, generates problems, tracks accuracy |
| `rlve_generate.py` | Custom generate function; populates `sample.prompt` and `sample.metadata` |
| `rlve_reward.py` | Restores env from metadata, runs verifier, returns `{reward, accuracy, format_score}` |
| `configs/starter_pack.yaml` | Environment weights and difficulty parameters |

## Configuration

Edit `configs/starter_pack.yaml`:

```yaml
environments:
  Multiplication:
    weight: 1.0
    kwargs:
      digit_num: 3

format_coef: 0.0  # Set > 0 to reward correct formatting

# Curriculum
use_controllers: false  # Set true to enable adaptive difficulty
initial_difficulty: 0
difficulty_sliding_window_size: 1
min_metric_to_increase_difficulty: 0.5
min_prompts_before_difficulty_check: 8
```

### Environments

| Environment | Weight | Key Parameter |
|-------------|--------|---------------|
| Multiplication | 1.0 | `digit_num` |
| Division | 1.0 | `divisor_digit_num`, `answer_digit_num` |
| Sorting | 1.0 | `N` (array length) |
| EuclidGame | 1.0 | `MAX_X_Y` |
| ShortestPath | 1.0 | `N`, `edge_density` |
| SpiralMatrix | 1.0 | `MAX_M_N` |
| LightUpPuzzle | 1.0 | `MAX_N_M`, `density` |

### Adding Environments

```python
from Gym.environments import identifier2environment
print(list(identifier2environment.keys()))
```

Add to YAML with weight and kwargs matching the environment's generator signature.

### Answer Format

Model must wrap answers:

```
<answer>42</answer>
```

The prompt template instructs this format. The verifier extracts and validates.

## Checkpoint/Resume

```python
from examples.rlve import get_provider

provider = get_provider()

# Save state (difficulty levels, accuracy counters, seed)
provider.save_state("/path/to/rlve_state.json")

# Load state to resume
provider.load_state("/path/to/rlve_state.json")
```

## Troubleshooting

**Gym not found**: `pip install rlve-gym`. For full RLVE, clone and set `PYTHONPATH=/path/to/RLVE:$PYTHONPATH`.

**Low success rate**: Reduce `kwargs` difficulty, use instruct model, check `<answer>` tags.

**Environment not found**: Check `identifier2environment.keys()` for valid IDs.
