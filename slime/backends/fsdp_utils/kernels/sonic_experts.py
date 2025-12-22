import importlib
import sys
from pathlib import Path

import torch


def _import_sonicmoe_functional():
    """Import sonicmoe.functional with a best-effort sys.path fallback.

    This repo vendors SonicMoE under `sonic-moe/`. If users didn't `pip install -e sonic-moe`,
    we try to add that folder to sys.path to make `import sonicmoe` work.
    """

    try:
        return importlib.import_module("sonicmoe.functional")
    except ModuleNotFoundError:
        # Try to locate `sonic-moe` directory from current file location.
        cur = Path(__file__).resolve()
        for parent in cur.parents:
            candidate = parent / "sonic-moe"
            if candidate.is_dir():
                sys.path.insert(0, str(candidate))
                break
        return importlib.import_module("sonicmoe.functional")


def sonic_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """SonicMoE experts implementation driven by externally-provided routing results.

    Args:
        hidden_states: (T, H)
        w1: (E, 2I, H) where 2I = concat(gate_proj, up_proj) along dim 1
        w2: (E, H, I)
        topk_weights: (T, K)
        topk_ids: (T, K)

    Returns:
        (T, H)
    """

    assert hidden_states.is_cuda, "SonicMoE requires CUDA tensors."
    assert hidden_states.dim() == 2, "hidden_states must be 2D (T, H)."
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1.dim() == 3 and w2.dim() == 3, "w1/w2 must be 3D tensors"

    T, _H = hidden_states.shape
    _E, _twoI, _H2 = w1.shape
    _E2, _H3, _I = w2.shape
    assert _H == _H2 == _H3, "Hidden size mismatch"
    assert _E == _E2, "Expert count mismatch"

    # SonicMoE expects:
    # - w1: (2I, H, E)
    # - w2: (H, I, E)
    w1_sonic = w1.permute(1, 2, 0).contiguous()
    w2_sonic = w2.permute(1, 2, 0).contiguous()

    # Build "general routing" inputs.
    # - selected_E: (T*K,)
    # - router_scores_selected: (T*K,)
    # - sorted_selected_T: (T*K,) sorted ascendingly
    K = topk_ids.shape[1]
    selected_E = topk_ids.reshape(-1).to(torch.int32)
    router_scores_selected = topk_weights.reshape(-1).to(torch.float32)
    sorted_selected_T = torch.arange(T, device=hidden_states.device, dtype=torch.int32).repeat_interleave(K)

    sonicmoe_functional = _import_sonicmoe_functional()
    moe_general_routing_inputs = getattr(sonicmoe_functional, "moe_general_routing_inputs")

    stream_id = int(torch.cuda.current_stream().cuda_stream)
    is_inference_mode_enabled = (not torch.is_grad_enabled()) or torch.is_inference_mode_enabled()

    out, _expert_freq = moe_general_routing_inputs(
        hidden_states,
        router_scores_selected,
        sorted_selected_T,
        selected_E,
        w1_sonic,
        None,
        w2_sonic,
        None,
        _E,
        stream_id,
        is_inference_mode_enabled,
    )
    return out


