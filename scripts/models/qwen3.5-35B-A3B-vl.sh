source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen3.5-35B-A3B.sh"

MODEL_ARGS[1]="slime_plugins.models.qwen3_5_vl"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"
