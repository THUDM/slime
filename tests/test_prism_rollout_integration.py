"""Exercise the public Omni Sample -> native packing -> Prism replay seam.

Only MCore's process-global state and accelerator placement are replaced for
CPU execution. BatchBuilder, tensor transport conversion, get_batch and the
Prism carrier validator are the production implementations. This test does
not construct a TE model or assert numerical agreement between GPU engines.
"""

import math
import os
import sys
import types
from argparse import Namespace

import _cp_dist_helpers  # noqa: F401: shared CPU MCore global-state stub
import pytest
import torch

from slime.data.batch_builder import BatchBuilder
from slime.observability.rollout_data_utils import tensorize_rollout_data_for_training
from slime.utils.types import Sample
from slime_plugins import omni_rollout
from slime_plugins.models import prism

NUM_GPUS = 0


@pytest.fixture
def config():
    """Optionally bind these wire/carrier checks to a real inference artifact."""
    checkpoint = os.environ.get("SLIME_TEST_PRISM_HF_CHECKPOINT")
    if checkpoint:
        result = omni_rollout.PrismConfig.from_path(checkpoint)
        spec = prism.read_prism_spec(checkpoint)
        assert result.n_vq == spec.topology.n_vq
        assert result.speech_vocab_size == spec.speech_vocab_size
        assert result.text_pad_idx == spec.text_pad_idx
        assert result.input_rvq_channels == spec.input_rvq_channels
        return result
    return omni_rollout.PrismConfig(
        n_vq=3,
        speech_vocab_size=8,
        text_vocab_size=151700,
        text_pad_idx=151643,
        pad_token_id=0,
        audio_start_token_id=151652,
        sample_rate=24000,
        context_length=128,
        input_rvq_channels=(1, 3),
        identity={
            "policy_family": "moss_tts_prism",
            "architecture": "MOSS-TTS-Prism",
            "config_sha256": "config",
            "modeling_sha256": "model",
            "processor_sha256": "processor",
            "prompt_protocol_sha256": "prompt",
            "n_vq": 3,
            "audio_vocab_size": 8,
            "sample_rate": 24000,
            "stop_semantics": "pre_frame_threshold_v1",
        },
    )


@pytest.fixture
def native_data(monkeypatch):
    # get_batch only needs PackedSeqParams' data attributes. Constructing real
    # model/attention kernels is outside this CPU protocol/packing check.
    try:
        import megatron.core.packed_seq_params  # noqa: F401
    except ImportError:
        packed_module = types.ModuleType("megatron.core.packed_seq_params")
        packed_module.PackedSeqParams = types.SimpleNamespace
        monkeypatch.setitem(sys.modules, packed_module.__name__, packed_module)

    from slime.backends.megatron_utils import data

    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data.accelerator, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data.accelerator, "current_device", lambda: torch.device("cpu"))
    return data


def trace(config, frames, finish, index):
    """The same typed enums/geometry that Omni emits, including reference edges."""
    q = config.n_vq
    zero = [0] * q
    reference = [(channel + 2) % config.speech_vocab_size for channel in range(q)]
    rows = [
        [7, *zero],
        [config.audio_start_token_id, *zero],
        [config.text_pad_idx, *reference],
        [11, *zero],
        [config.audio_start_token_id, *zero],
    ]
    codes = [[(frame + channel + 1) % config.speech_vocab_size for channel in range(q)] for frame in range(frames)]
    code_scores = [[-0.4 - 0.1 * channel - 0.05 * frame for channel in range(q)] for frame in range(frames)]
    stops = [0] * frames + ([1] if finish == "stop" else [])
    probabilities = [0.2] * frames + ([0.8] if finish == "stop" else [])
    stop_scores = [math.log(p if action else 1 - p) for action, p in zip(stops, probabilities, strict=True)]
    result = {
        "version": 2,
        "model_family": "moss_tts_prism",
        "stages": ["tts_engine"],
        "request_id": f"request-{index}",
        "model_identity": config.identity,
        "admission_weight_version": "step_7",
        "logprob_semantics": omni_rollout.LOGPROB_SEMANTICS,
        "stop_semantics": omni_rollout.STOP_SEMANTICS,
        "sampling": {
            "audio_temperature": 1.0,
            "audio_top_p": 1.0,
            "audio_top_k": -1,
            "audio_repetition_penalty": 1.0,
            "stop_sampling": True,
        },
        "replay_inputs": {
            "layout": omni_rollout.LAYOUT,
            "prompt_rows": rows,
            "item_kind": [0, 0, 1, 0, 0],
            "audio_role": [0, 0, 1, 0, 0],
            "successor_audio_mask": [False, True, False, False, False],
            "successor_audio_codes": [zero, reference, zero, zero, zero],
            "target_history_retention_mask": [[True] * q for _ in rows],
        },
        "action_streams": [
            {
                "name": "codes",
                "stage": "tts_engine",
                "modality": "audio",
                "action_type": "discrete",
                "layout": "time_depth_autoregressive",
                "flatten_order": "time_major",
                "shape": [frames, q],
                "vocab_size": config.speech_vocab_size,
                "actions": codes,
                "logprobs": code_scores,
                "action_mask": [[1] * q for _ in codes],
                "deterministic_mask": None,
                "channel_ids": list(range(q)),
                "channel_roles": [f"rvq_{channel}" for channel in range(q)],
            },
            {
                "name": "stop_decisions",
                "stage": "tts_engine",
                "modality": "control",
                "action_type": "discrete",
                "layout": "pre_frame_autoregressive",
                "flatten_order": "time_major",
                "shape": [len(stops)],
                "vocab_size": 2,
                "actions": stops,
                "logprobs": stop_scores,
                "action_mask": [1] * len(stops),
                "deterministic_mask": None,
            },
        ],
        "non_action_outputs": [{"name": "stop_probabilities", "stage": "tts_engine", "values": probabilities}],
        "finish_reason": finish,
        "total_action_count": frames * q + len(stops),
    }
    return {
        "meta_info": {
            "omni_rollout": result,
            "weight_version": "step_7",
            "finish_reason": {"type": finish},
            "prompt_tokens": len(rows),
            "completion_tokens": frames,
        },
        "audio": None,
    }


def accepted_samples(config, remove_zero_frame):
    samples = []
    for index, (frames, finish, reward) in enumerate(((2, "stop", 1.0), (1, "length", 3.0), (0, "stop", 0.0))):
        sample = Sample(
            index=index,
            group_index=0,
            prompt="你好",
            reward=reward,
            metadata={
                "task_type": "Full VoiceClone",
                "show_language": False,
                "references": [{"id": "audio1", "uri": "/shared/reference.wav"}],
            },
        )
        omni_rollout.parse_rollout(
            trace(config, frames, finish, index), sample, config, expected_weight_version="step_7"
        )
        sample.remove_sample = remove_zero_frame and frames == 0
        # Existing Sample serialization preserves typed tensors and flags.
        samples.append(Sample.from_dict(sample.to_dict()))
    return samples


@pytest.mark.parametrize("remove_zero_frame", [False, True])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_omni_samples_native_conversion_packing_and_stop_gradients(config, native_data, remove_zero_frame, order):
    from slime.backends.megatron_utils.policy_actions import build_policy_action_view

    samples = accepted_samples(config, remove_zero_frame)
    args = Namespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=3,
        rollout_batch_size=1,
        reward_key=None,
        use_rollout_logprobs=True,
        use_opd=False,
        entropy_coef=0.0,
        policy_stop_loss_weight=1.0,
    )
    converted = BatchBuilder(args).convert(samples)
    converted["total_lengths"] = [len(tokens) for tokens in converted["tokens"]]
    tensorize_rollout_data_for_training(converted)
    assert converted["response_lengths"] == [2, 1, 0]
    assert converted["rewards"] == pytest.approx([-1 / 3, 5 / 3, -4 / 3])
    assert [mm["policy_sample_mask"].item() for mm in converted["multimodal_train_inputs"]] == [
        True,
        True,
        not remove_zero_frame,
    ]
    # Frame masks remain empty for both retained and removed immediate stops.
    assert converted["loss_masks"][2].numel() == 0

    batch = native_data.get_batch(
        native_data.DataIterator(converted, [list(order)]),
        ["tokens", "loss_masks", "multimodal_train_inputs", "total_lengths", "response_lengths"],
        pad_multiplier=16,
    )
    total_rows = sum(converted["total_lengths"])
    padded_rows = batch["tokens"].shape[1]
    assert 0 < padded_rows - total_rows < 16
    expected_boundaries = [0]
    for index in order:
        expected_boundaries.append(expected_boundaries[-1] + converted["total_lengths"][index])
    expected_boundaries.append(padded_rows)
    assert batch["packed_seq_params"].cu_seqlens_q.tolist() == expected_boundaries

    mm = batch["multimodal_train_inputs"]
    prepared = prism.prepare_policy_inputs(
        input_ids=batch["tokens"],
        **{
            key: mm[key]
            for key in (
                "audio_codes",
                "item_kind",
                "audio_role",
                "successor_audio_codes",
                "successor_audio_mask",
                "target_history_retention_mask",
                "policy_code_mask",
                "policy_stop_mask",
                "policy_stop_actions",
            )
        },
        n_vq=config.n_vq,
        speech_vocab_size=config.speech_vocab_size,
        text_vocab_size=config.text_vocab_size,
        text_pad_idx=config.text_pad_idx,
        cu_seqlens=batch["packed_seq_params"].cu_seqlens_q,
    )
    assert prepared["tokens"].shape == (1, padded_rows, config.n_vq + 1)
    assert prepared["code_mask"].sum().item() == 3 * config.n_vq
    assert prepared["stop_mask"].sum().item() == 5
    assert not prepared["code_mask"][:, total_rows:].any()
    assert not prepared["stop_mask"][:, total_rows:].any()
    assert prepared["retention"][:, total_rows:].all()

    offset = 0
    zero_stop_position = None
    for index in order:
        frames = samples[index].response_length
        section = slice(offset, offset + converted["total_lengths"][index])
        # Reference successor survives packing but never becomes a loss action.
        assert prepared["successor_mask"][0, offset + 1]
        assert not prepared["code_mask"][0, offset + 1].any()
        assert not prepared["stop_mask"][0, offset + 1]
        assert prepared["retention"][0, offset : offset + 5].all()
        assert prepared["code_mask"][0, section].sum().item() == frames * config.n_vq
        assert prepared["stop_mask"][0, section].sum().item() == frames + int(index != 1)
        assert not prepared["successor_mask"][0, section.stop - 1]
        if frames == 0:
            zero_stop_position = section.stop - 1
            assert prepared["stop_actions"][0, zero_stop_position] == 1
        offset = section.stop

    # Feed known selected scores through the actual model extractor/action
    # reducer. This verifies removed zero-frame stop gradients without a GPU.
    output = torch.cat((mm["policy_rollout_code_logprobs"], mm["policy_rollout_stop_logprobs"][:, None]), dim=-1)
    output = torch.cat((output, torch.zeros(padded_rows - total_rows, config.n_vq + 1))).unsqueeze(0)
    output.requires_grad_()
    scores = prism.extract_policy_log_probs(args, batch, output)
    view = build_policy_action_view(args, batch, scores)
    (-view.reduce(torch.cat(view.current))).backward()
    assert output.grad[0, zero_stop_position, config.n_vq].item() == (0 if remove_zero_frame else -1)
    assert not output.grad[0, total_rows:].any()
    assert not output.grad[0, zero_stop_position, : config.n_vq].any()
