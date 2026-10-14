# NVIDIA Nemotron 3.5 Super VL (nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026).
#
# The decoder is the file this sources, unchanged, because the two models'
# language configs are field-for-field identical: 88 layers, hidden 4096, 32
# heads over 2 KV groups, head_dim 128, vocab 131072, 512 routed experts at
# top-k 22, moe_latent_size 1024, one shared expert, one MTP layer. Their layer
# patterns match too -- Nemotron 3.5 spells it as `layers_block_type`, which
# derives to exactly the NEMOTRON_H_PATTERN below.
#
# So the only thing 3.5 adds is the vision half, and the only thing that changes
# here is --spec: the provider builds the same MCore HybridModel and hangs a
# RADIO tower and projector off it. Same idiom as qwen3.5-35B-A3B-vl.sh.
#
# Sourcing rather than copying is what keeps the shapes from drifting apart.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/nemotron3-super-120b-a12b.sh"

MODEL_ARGS[1]="slime_plugins.models.nemotron_h_vl"
MODEL_ARGS[2]="get_nemotron_h_vl_model_provider"
