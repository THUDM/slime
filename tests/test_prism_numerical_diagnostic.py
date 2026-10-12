"""CPU shape/probability contracts for the GPU-only reference oracle hook."""

from types import SimpleNamespace

import pytest
import torch
from scripts.prism_numerical_diagnostic import (
    _activation_capture,
    activation_fusion,
    canonical_rows,
    differences,
    oracle_inputs,
    score_differences,
    selected_scores,
)

NUM_GPUS = 0


def _batch():
    return {
        "tokens": torch.tensor([[151652, 151643, 151643, 151643]]),
        "total_lengths": [2],
        "packed_seq_params": SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2, 4])),
        "multimodal_train_inputs": {
            "audio_codes": torch.tensor([[0, 0], [1, 2]]),
            "item_kind": torch.tensor([0, 1]),
            "audio_role": torch.tensor([0, 2]),
            "successor_audio_codes": torch.tensor([[1, 2], [0, 0]]),
            "successor_audio_mask": torch.tensor([True, False]),
            "target_history_retention_mask": torch.ones(2, 2, dtype=torch.bool),
            "policy_code_mask": torch.tensor([[True, True], [False, False]]),
            "policy_stop_mask": torch.tensor([True, True]),
            "policy_stop_actions": torch.tensor([0, 1]),
        },
    }


def _spec():
    return SimpleNamespace(
        topology=SimpleNamespace(n_vq=2),
        speech_vocab_size=4,
        text_vocab_size=151700,
        text_pad_idx=151643,
        audio_start_token_id=151652,
    )


def test_oracle_matches_native_padded_time_and_uses_current_successor_codes():
    inputs, prepared = oracle_inputs(_batch(), _spec())
    assert inputs["input_ids"].shape == (1, 4, 3)
    assert inputs["attention_mask"].tolist() == [[True, True, False, False]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert inputs["input_ids"][0, 0].tolist() == [151652, 0, 0]
    assert inputs["input_ids"][0, 1].tolist() == [151643, 1, 2]
    assert inputs["successor_audio_codes"][0, 0].tolist() == [1, 2]
    assert inputs["successor_audio_codes"][0, 1:].eq(0).all()
    assert prepared["stop_actions"].tolist() == [[0, 1, 0, 0]]
    assert not inputs["use_cache"]


def test_oracle_rejects_multiple_packed_documents_without_attention_isolation():
    batch = _batch()
    batch["total_lengths"] = [1, 1]
    with pytest.raises(ValueError, match="one trajectory"):
        oracle_inputs(batch, _spec())


def test_site_axis_conversion_preserves_native_row_order():
    native = torch.arange(12).reshape(4, 1, 3)
    expected = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
    torch.testing.assert_close(canonical_rows(native, native=True), expected)
    torch.testing.assert_close(canonical_rows(expected, native=False), expected)
    with pytest.raises(ValueError, match="one packed"):
        canonical_rows(torch.ones(4, 2, 3), native=True)


def test_selected_code_and_stop_scores_match_independent_categorical_distributions():
    code_logits = torch.tensor([[[[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]]])
    stop_logits = torch.tensor([[[0.0, 2.0]]])
    codes, stops = torch.tensor([[[1, 2]]]), torch.tensor([[1]])
    actual = selected_scores(code_logits, stop_logits, codes, stops)
    expected_codes = torch.distributions.Categorical(logits=code_logits).log_prob(codes)
    expected_stop = torch.distributions.Categorical(logits=stop_logits).log_prob(stops)
    torch.testing.assert_close(actual, torch.cat((expected_codes, expected_stop.unsqueeze(-1)), -1))


def test_diagnostic_masks_exclude_prompt_padding_and_terminal_rvq_but_keep_stop():
    _, prepared = oracle_inputs(_batch(), _spec())
    actual, expected = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3)
    actual[0, 0, 0] = 0.2
    actual[0, 1, 2] = 0.3
    actual[0, 1, :2] = 99
    actual[0, 2:] = 99
    report = score_differences(actual, expected, prepared)
    assert report["code"]["count"] == 2
    assert report["code"]["max_abs_diff"] == pytest.approx(0.2)
    assert report["stop"]["count"] == 2
    assert report["stop"]["max_abs_diff"] == pytest.approx(0.3)
    assert report["per_channel"][0]["mean_abs_diff"] == pytest.approx(0.2)
    assert differences(torch.zeros(2), torch.zeros(2), torch.tensor([False, False]))["count"] == 0


def test_selected_scores_reject_misaligned_successor_head_layout():
    with pytest.raises(ValueError, match="successor codes"):
        selected_scores(
            torch.zeros(1, 2, 3, 4),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.long),
        )


def test_activation_variant_restores_shared_and_independent_configs_on_error():
    shared, independent = SimpleNamespace(bias_activation_fusion=True), SimpleNamespace(bias_activation_fusion=False)
    model = SimpleNamespace(
        decoder=SimpleNamespace(
            layers=[
                SimpleNamespace(mlp=SimpleNamespace(config=shared)),
                SimpleNamespace(mlp=SimpleNamespace(config=shared)),
                SimpleNamespace(mlp=SimpleNamespace(config=independent)),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="diagnostic failure"), activation_fusion(model, False):
        assert not shared.bias_activation_fusion
        assert not independent.bias_activation_fusion
        raise RuntimeError("diagnostic failure")
    assert shared.bias_activation_fusion
    assert not independent.bias_activation_fusion


def test_activation_capture_reads_actual_linear_boundaries_and_keeps_first_loop_call():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = torch.nn.Linear(4, 8, bias=False).to(torch.bfloat16)
            self.linear_fc2 = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)

        def forward(self, value):
            gate, up = self.linear_fc1(value).chunk(2, -1)
            return self.linear_fc2(torch.nn.functional.silu(gate) * up)

    mlp = MLP()
    model = SimpleNamespace(
        config=SimpleNamespace(prism_mlp_only_layers=()), decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
    )
    captures, handles = _activation_capture(model)
    try:
        mlp(torch.ones(3, 1, 4, dtype=torch.bfloat16))
        first = captures["0"]["pre_activation"].clone()
        mlp(torch.zeros(3, 1, 4, dtype=torch.bfloat16))
        torch.testing.assert_close(captures["0"]["pre_activation"], first)
        assert captures["0"]["pre_activation"].shape == (3, 1, 8)
        assert captures["0"]["post_activation"].shape == (3, 1, 4)
        assert captures["0"]["pre_activation"].dtype == torch.bfloat16
        assert captures["0"]["post_activation"].dtype == torch.bfloat16
    finally:
        for handle in handles:
            handle.remove()
