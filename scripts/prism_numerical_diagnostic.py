"""A read-only numerical oracle hook for the TP1 Prism GPU smoke.

Run ``diagnose`` as --before-trajectory-hook with a saved --trajectory-json.
The hook compares the native model, functional projections of its captured
head inputs, and the user's converted Hugging Face model at the same BF16
dtype. It never updates weights, optimizer state, action masks or tolerances.
Only this user-provided artifact is loaded with trust_remote_code=True.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F


def canonical_rows(value: torch.Tensor, *, native: bool) -> torch.Tensor:
    """Convert [T,1,H] native state or [1,T,H] HF state to [1,T,H]."""
    if value.ndim != 3 or value.shape[1 if native else 0] != 1:
        raise ValueError("the numerical diagnostic requires one packed TP1 sample")
    return value.transpose(0, 1) if native else value


def selected_scores(rvq_logits, stop_logits, codes, stops):
    """Return the native Q-code-plus-stop layout from either model's logits."""
    if rvq_logits.ndim != 4 or tuple(rvq_logits.shape[:-1]) != tuple(codes.shape):
        raise ValueError("RVQ logits and successor codes must have shape [1,T,Q,V] / [1,T,Q]")
    if tuple(stop_logits.shape) != (*stops.shape, 2) or tuple(stops.shape) != tuple(codes.shape[:2]):
        raise ValueError("stop logits/actions must align with the chronological rows")
    code_scores = rvq_logits.float().log_softmax(-1).gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    stop_scores = stop_logits.float().log_softmax(-1).gather(-1, stops.unsqueeze(-1)).squeeze(-1)
    return torch.cat((code_scores, stop_scores.unsqueeze(-1)), dim=-1)


def differences(actual, expected, mask=None):
    if actual.shape != expected.shape:
        raise ValueError(f"diagnostic shapes differ: {actual.shape} versus {expected.shape}")
    delta = (actual.detach().float() - expected.detach().float()).abs()
    if mask is not None:
        delta = delta[mask]
    if not delta.numel():
        return {"count": 0, "max_abs_diff": None, "mean_abs_diff": None}
    return {
        "count": delta.numel(),
        "max_abs_diff": delta.max().item(),
        "mean_abs_diff": delta.mean().item(),
    }


def rms_rounding_references(raw, gamma, eps):
    """Pure tensor references for one versus two BF16 rounding boundaries."""
    if raw.shape[-1] != gamma.numel() or gamma.ndim != 1:
        raise ValueError("RMSNorm gamma must match the captured hidden width")
    normalized = raw.float() * torch.rsqrt(raw.float().square().mean(-1, keepdim=True) + eps)
    return {
        "single": (normalized * gamma.float()).to(raw.dtype),
        "double": normalized.to(raw.dtype) * gamma.to(raw.dtype),
    }


def _sglang_prediction_norms(model, captures, rows, eps):
    """Call the real SGLang operator; missing runtime context is explicit."""
    try:
        from sglang.srt.layers.layernorm import RMSNorm
    except (ImportError, RuntimeError) as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}, {}
    report, tensors = {"status": "measured", "sites": {}}, {}
    for index, native_norm in enumerate(model.prediction_norms):
        key = str(index)
        raw, actual = captures["norms"][key]["raw"], captures["norms"][key]["normalized"]
        gamma = native_norm.weight.detach()
        references = rms_rounding_references(raw, gamma, eps)
        try:
            norm = RMSNorm(raw.shape[-1], eps=eps).to(device=raw.device, dtype=raw.dtype).eval()
            norm.weight.copy_(gamma)
            result = norm(raw.squeeze(0).contiguous(), None)
            if not isinstance(result, torch.Tensor) or result.shape != raw.shape[1:]:
                raise TypeError("SGLang RMSNorm with residual=None did not return a [T,H] tensor")
            result = result.unsqueeze(0)
            tensors[key] = result.detach().clone()
            report["sites"][key] = {
                "status": "measured",
                "input_dtype": str(raw.dtype),
                "output_dtype": str(result.dtype),
                "vs_native_same_raw": differences(result, actual, rows),
                "vs_torch_single": differences(result, references["single"], rows),
                "vs_torch_double": differences(result, references["double"], rows),
            }
        except Exception as error:
            report["status"] = "partially_unavailable"
            report["sites"][key] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    return report, tensors


def score_differences(actual, expected, prepared):
    q = prepared["code_mask"].shape[-1]
    if actual.shape[-1] != q + 1:
        raise ValueError("unexpected native selected-score layout")
    code_mask, stop_mask = prepared["code_mask"], prepared["stop_mask"]
    report = {
        "code": differences(actual[..., :q], expected[..., :q], code_mask),
        "stop": differences(actual[..., q], expected[..., q], stop_mask),
        "per_channel": [
            differences(actual[..., channel], expected[..., channel], code_mask[..., channel]) for channel in range(q)
        ],
    }
    return report


def oracle_inputs(batch, spec):
    """Use exactly the native padded time dimension with padding attention hidden."""
    from slime_plugins.models.prism import prepare_policy_inputs

    if len(batch["total_lengths"]) != 1:
        raise ValueError("HF oracle comparisons currently require one trajectory")
    mm = batch["multimodal_train_inputs"]
    prepared = prepare_policy_inputs(
        batch["tokens"],
        audio_codes=mm["audio_codes"],
        item_kind=mm["item_kind"],
        audio_role=mm["audio_role"],
        successor_audio_codes=mm["successor_audio_codes"],
        successor_audio_mask=mm["successor_audio_mask"],
        target_history_retention_mask=mm.get("target_history_retention_mask"),
        policy_code_mask=mm["policy_code_mask"],
        policy_stop_mask=mm["policy_stop_mask"],
        policy_stop_actions=mm["policy_stop_actions"],
        n_vq=spec.topology.n_vq,
        speech_vocab_size=spec.speech_vocab_size,
        text_vocab_size=spec.text_vocab_size,
        text_pad_idx=spec.text_pad_idx,
        audio_start_token_id=spec.audio_start_token_id,
        cu_seqlens=batch["packed_seq_params"].cu_seqlens_q,
    )
    time = batch["tokens"].shape[1]
    length = int(batch["total_lengths"][0])
    attention = torch.arange(time, device=batch["tokens"].device).unsqueeze(0) < length
    inputs = {
        "input_ids": prepared["tokens"],
        "attention_mask": attention,
        "position_ids": torch.arange(time, device=attention.device).unsqueeze(0),
        "item_kind": prepared["item_kind"],
        "audio_role": prepared["audio_role"],
        "target_history_retention_mask": prepared["retention"],
        "successor_audio_mask": prepared["successor_mask"],
        "successor_audio_codes": prepared["successor_codes"],
        "use_cache": False,
        "return_dict": True,
    }
    if "loop_block_prefix_counts" in mm:
        inputs["loop_block_prefix_counts"] = mm["loop_block_prefix_counts"]
    return inputs, prepared


def _capture(model, *, native):
    captures, handles = {"norms": {}, "heads": {}, "stop": {}}, []

    def norm(index):
        def hook(_module, inputs, output):
            captures["norms"][str(index)] = {
                "raw": canonical_rows(inputs[0].detach().clone(), native=native),
                "normalized": canonical_rows(output.detach().clone(), native=native),
            }

        return hook

    def head(index, *, stop=False):
        def hook(_module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            value = {
                "input": canonical_rows(inputs[0].detach().clone(), native=native),
                "logits": canonical_rows(logits.detach().clone(), native=native),
            }
            if stop:
                captures["stop"] = value
            else:
                captures["heads"][str(index)] = value

        return hook

    for index, module in enumerate(model.prediction_norms):
        handles.append(module.register_forward_hook(norm(index)))
    for index, module in enumerate(model.audio_lm_heads):
        handles.append(module.register_forward_hook(head(index)))
    handles.append(model.stop_head.register_forward_hook(head(0, stop=True)))
    return captures, handles


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


@contextmanager
def activation_fusion(model, enabled):
    """Temporarily select MCore's own SwiGLU path and restore shared configs."""
    configs = {id(layer.mlp.config): layer.mlp.config for layer in model.decoder.layers}
    previous = [(config, config.bias_activation_fusion) for config in configs.values()]
    try:
        for config, _ in previous:
            config.bias_activation_fusion = enabled
        yield
    finally:
        for config, value in previous:
            config.bias_activation_fusion = value


def _activation_capture(model):
    """Record actual FC1 output and FC2 input; no replacement activation."""
    values, handles = {}, []
    indices = (0, *model.config.prism_mlp_only_layers[:1])

    def before(index):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            values.setdefault(str(index), {}).setdefault("pre_activation", tensor.detach().clone())

        return hook

    def after(index):
        def hook(_module, inputs):
            values.setdefault(str(index), {}).setdefault("post_activation", inputs[0].detach().clone())

        return hook

    for index in indices:
        mlp = model.decoder.layers[index].mlp
        handles.append(mlp.linear_fc1.register_forward_hook(before(index)))
        handles.append(mlp.linear_fc2.register_forward_pre_hook(after(index)))
    return values, handles


@torch.no_grad()
def diagnose(context):
    """Smoke before-hook; GPU execution and the HF artifact are explicit inputs."""
    from scripts.test_prism_gpu import forward, input_samples, make_batch
    from transformers import AutoModel

    from slime_plugins.models.prism import read_prism_spec

    args, wrapper = context["args"], context["model"]
    if not args.trajectory_json or args.omni_url:
        raise ValueError("the oracle hook requires an immutable saved Omni trajectory")
    model = wrapper
    while hasattr(model, "module"):
        model = model.module
    spec = read_prism_spec(args.hf_checkpoint)
    if model.config.tensor_model_parallel_size != 1:
        raise ValueError("the functional head oracle requires TP=1")
    output = Path(context["artifact_dir"])
    samples, _ = input_samples(args, spec, output)
    _, batch = make_batch(samples)
    inputs, prepared = oracle_inputs(batch, spec)
    rows = prepared["code_mask"].any(-1)
    device = batch["tokens"].device
    previous_training = wrapper.training
    native, native_handles = _capture(model, native=True)
    hf_handles, variant_handles, oracle = [], [], None
    try:
        wrapper.eval()
        with torch.no_grad():
            native_scores = forward(wrapper, batch)
        for handle in native_handles:
            handle.remove()
        native_handles.clear()
        sglang_norms, sglang_norm_tensors = _sglang_prediction_norms(model, native, rows, spec.norm_epsilon)

        # Same stored weights and same captured hidden input isolate the
        # projection kernel/rounding from all backbone and feedback effects.
        torch_bf16 = torch.stack(
            [F.linear(native["heads"][str(q)]["input"], head.weight) for q, head in enumerate(model.audio_lm_heads)],
            dim=2,
        )
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            torch_fp32 = torch.stack(
                [
                    F.linear(native["heads"][str(q)]["input"].float(), head.weight.float())
                    for q, head in enumerate(model.audio_lm_heads)
                ],
                dim=2,
            )
        finally:
            torch.set_float32_matmul_precision(precision)
        stop_logits = native["stop"]["logits"]
        bf16_scores = selected_scores(torch_bf16, stop_logits, prepared["successor_codes"], prepared["stop_actions"])
        fp32_scores = selected_scores(torch_fp32, stop_logits, prepared["successor_codes"], prepared["stop_actions"])

        # Trust is scoped to the converted mossLite artifact supplied by the
        # user. This hook never fetches another model or substitutes weights.
        oracle = (
            AutoModel.from_pretrained(
                args.hf_checkpoint,
                trust_remote_code=True,
                dtype=getattr(torch, spec.dtype),
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        hf, hf_handles = _capture(oracle, native=False)
        with torch.no_grad():
            hf_output = oracle(**inputs)
        hf_scores = selected_scores(
            hf_output.rvq_logits,
            hf_output.stop_logits,
            prepared["successor_codes"],
            prepared["stop_actions"],
        )
        omni_codes = torch.zeros_like(prepared["successor_codes"], dtype=torch.float32)
        omni_stops = torch.zeros_like(prepared["stop_actions"], dtype=torch.float32)
        sample_mm = samples[0].multimodal_train_inputs
        real_length = int(batch["total_lengths"][0])
        omni_codes[:, :real_length] = sample_mm["policy_rollout_code_logprobs"].to(device)
        omni_stops[:, :real_length] = sample_mm["policy_rollout_stop_logprobs"].to(device)
        omni_scores = torch.cat((omni_codes, omni_stops.unsqueeze(-1)), dim=-1)
        variants, variant_scores, variant_activations = {}, {}, {}
        for enabled in (True, False):
            key = "fused" if enabled else "unfused"
            activations, variant_handles = _activation_capture(model)
            try:
                with activation_fusion(model, enabled):
                    scores = forward(wrapper, batch)
            finally:
                for handle in variant_handles:
                    handle.remove()
                variant_handles.clear()
            variant_scores[key] = scores
            variant_activations[key] = activations
            variants[key] = {
                "bias_activation_fusion": enabled,
                "vs_native_baseline": score_differences(scores, native_scores, prepared),
                "vs_omni": score_differences(scores, omni_scores, prepared),
                "vs_hf_bf16": score_differences(scores, hf_scores, prepared),
                "activation_dtypes": {
                    layer: {stage: str(tensor.dtype) for stage, tensor in values.items()}
                    for layer, values in activations.items()
                },
            }
        activation_deltas = {
            layer: {
                stage: differences(
                    canonical_rows(variant_activations["fused"][layer][stage], native=True),
                    canonical_rows(variant_activations["unfused"][layer][stage], native=True),
                    rows,
                )
                for stage in ("pre_activation", "post_activation")
            }
            for layer in variant_activations["fused"]
        }
        native_logits = torch.stack([native["heads"][str(q)]["logits"] for q in range(spec.topology.n_vq)], dim=2)
        report = {
            "dtype": spec.dtype,
            "time_rows": inputs["input_ids"].shape[1],
            "real_rows": real_length,
            "hf_attention_implementation": oracle.config.language_config._attn_implementation,
            "native_vs_omni": score_differences(native_scores, omni_scores, prepared),
            "torch_bf16_head_vs_native": score_differences(bf16_scores, native_scores, prepared),
            "torch_fp32_head_vs_native": score_differences(fp32_scores, native_scores, prepared),
            "torch_fp32_head_vs_omni": score_differences(fp32_scores, omni_scores, prepared),
            "torch_bf16_logits_vs_native": differences(torch_bf16, native_logits, rows),
            "hf_bf16_vs_native": score_differences(hf_scores, native_scores, prepared),
            "hf_bf16_vs_omni": score_differences(hf_scores, omni_scores, prepared),
            "sglang_prediction_norms_same_raw": sglang_norms,
            "activation_fusion_variants": variants,
            "activation_fusion_deltas": activation_deltas,
            "sites": {
                key: {
                    name: differences(native["norms"][key][name], hf["norms"][key][name], rows)
                    for name in ("raw", "normalized")
                }
                for key in native["norms"]
            },
        }
        (output / "numerical-diagnostic.json").write_text(json.dumps(report, indent=2))
        torch.save(
            _cpu(
                {
                    "native": native,
                    "hf": hf,
                    "native_scores": native_scores,
                    "hf_scores": hf_scores,
                    "torch_bf16_head_scores": bf16_scores,
                    "torch_fp32_head_scores": fp32_scores,
                    "prepared": prepared,
                    "sglang_prediction_norm_tensors": sglang_norm_tensors,
                    "activation_variant_scores": variant_scores,
                    "activation_variant_tensors": variant_activations,
                }
            ),
            output / "numerical-diagnostic.pt",
        )
        return report
    finally:
        for handle in [*native_handles, *hf_handles, *variant_handles]:
            handle.remove()
        wrapper.train(previous_training)
        del oracle
