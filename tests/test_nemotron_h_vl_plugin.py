"""Tests for the Nemotron-H model providers in slime_plugins/models.

These exercise provider validation and the vision-embedding scatter without
building a model, but the plugin modules import MCore's hybrid model at the top,
so they need a Megatron install with `megatron.core.models.hybrid` (MCore >= 0.20).
"""

import types

import pytest
import torch

pytest.importorskip("megatron.core.models.hybrid.hybrid_model")

from slime_plugins.models import nemotron_h as nemotron_h_plugin  # noqa: E402
from slime_plugins.models import nemotron_h_vl as vl_plugin  # noqa: E402

NUM_GPUS = 0


def _args(**overrides):
    fields = dict(hybrid_override_pattern=None, num_layers=3)
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


# ------------------------------------------------------ nemotron_h model provider


@pytest.mark.unit
def test_text_provider_requires_a_hybrid_pattern():
    with pytest.raises(ValueError, match="--hybrid-override-pattern"):
        nemotron_h_plugin.get_nemotron_h_model_provider(_args(), config=None)


@pytest.mark.unit
def test_text_provider_requires_a_symbol_for_every_layer():
    with pytest.raises(ValueError, match="2 characters but --num-layers is 3"):
        nemotron_h_plugin.get_nemotron_h_model_provider(_args(hybrid_override_pattern="ME"), config=None)


# ------------------------------------------------------ hybrid pattern resolution


@pytest.mark.unit
def test_cli_pattern_wins_over_the_checkpoint_config():
    hf_config = types.SimpleNamespace(llm_config=types.SimpleNamespace(hybrid_override_pattern="EEE"))
    assert vl_plugin.get_hybrid_override_pattern(_args(hybrid_override_pattern="M*E"), hf_config) == "M*E"


@pytest.mark.unit
def test_pattern_is_read_from_the_nested_llm_config():
    hf_config = types.SimpleNamespace(llm_config=types.SimpleNamespace(hybrid_override_pattern="M*E"))
    assert vl_plugin.get_hybrid_override_pattern(_args(), hf_config) == "M*E"


@pytest.mark.unit
def test_pattern_is_derived_from_layers_block_type():
    hf_config = types.SimpleNamespace(
        llm_config=types.SimpleNamespace(layers_block_type=["mamba", "attention", "moe"])
    )
    assert vl_plugin.get_hybrid_override_pattern(_args(), hf_config) == "M*E"


@pytest.mark.unit
def test_unknown_block_type_is_named_in_the_error():
    hf_config = types.SimpleNamespace(llm_config=types.SimpleNamespace(layers_block_type=["mamba", "rnn", "moe"]))
    with pytest.raises(ValueError, match="'rnn'"):
        vl_plugin.get_hybrid_override_pattern(_args(), hf_config)


@pytest.mark.unit
def test_missing_pattern_everywhere_is_an_error():
    with pytest.raises(ValueError, match="needs a hybrid layer pattern"):
        vl_plugin.get_hybrid_override_pattern(_args(), types.SimpleNamespace(llm_config=types.SimpleNamespace()))


@pytest.mark.unit
def test_only_the_main_stack_counts_towards_num_layers():
    # '<main>/*E' appends one MTP depth; the whole string still goes to HybridModel.
    pattern = "M*E/*E"
    assert vl_plugin.get_hybrid_override_pattern(_args(hybrid_override_pattern=pattern), None) == pattern


@pytest.mark.unit
def test_main_stack_length_must_match_num_layers():
    with pytest.raises(ValueError, match="4 main-stack characters but --num-layers is 3"):
        vl_plugin.get_hybrid_override_pattern(_args(hybrid_override_pattern="M*EM/*E"), None)


# ----------------------------------------------------------- packed position ids


@pytest.mark.unit
def test_packed_position_ids_restart_at_every_sequence_boundary():
    cu_seqlens = torch.tensor([0, 3, 5, 9])
    input_ids = torch.zeros(1, 9, dtype=torch.long)
    positions = vl_plugin._packed_position_ids(cu_seqlens, input_ids, None)
    assert positions.tolist() == [[0, 1, 2, 0, 1, 0, 1, 2, 3]]


@pytest.mark.unit
def test_packed_position_ids_must_line_up_with_the_local_tokens():
    with pytest.raises(ValueError, match="MTP position_ids"):
        vl_plugin._packed_position_ids(torch.tensor([0, 3]), torch.zeros(1, 4, dtype=torch.long), None)


# --------------------------------------------------------- vision-embedding scatter

IMAGE_TOKEN = 7
HIDDEN = 4


class _FakeProjector(torch.nn.Module):
    """Stands in for the checkpoint's projector: image [n, ...] -> [n, tokens, hidden].

    The token count is read off the image, so differently sized images produce
    differently sized outputs -- the case a plain torch.cat over the list cannot do.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, image, vision_model):
        tokens = image.shape[-1]
        values = image.reshape(image.shape[0], -1)[:, :1].float()  # one value per tile
        return (values.unsqueeze(-1) * torch.ones(image.shape[0], tokens, HIDDEN)) * self.scale


def _inject(pixel_values, input_ids):
    embedding = lambda input_ids, position_ids: torch.zeros(input_ids.shape[1], 1, HIDDEN)  # noqa: E731
    fake_self = types.SimpleNamespace(
        language_model=types.SimpleNamespace(embedding=embedding),
        vision_projector=_FakeProjector(),
        vision_model=None,
        img_context_token_id=IMAGE_TOKEN,
        config=types.SimpleNamespace(sequence_parallel=False),
    )
    cu_seqlens = torch.tensor([0, input_ids.shape[1]])
    return vl_plugin.NemotronHVLModel._inject_vision_embeddings(
        fake_self, input_ids, input_ids, cu_seqlens, None, pixel_values
    )


@pytest.mark.unit
def test_images_of_different_size_scatter_into_their_own_token_runs():
    # Two images producing 3 and 2 tokens, in one packed sequence.
    input_ids = torch.tensor([[1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2, IMAGE_TOKEN, IMAGE_TOKEN, 3]])
    first = torch.full((1, 3, 3), 10.0)  # 3 tokens, value 10
    second = torch.full((1, 3, 2), 20.0)  # 2 tokens, value 20

    embeddings = _inject([first, second], input_ids)  # [seq, 1, hidden]

    assert embeddings.shape == (8, 1, HIDDEN)
    assert embeddings[[1, 2, 3], 0].eq(10).all()
    assert embeddings[[5, 6], 0].eq(20).all()
    assert embeddings[[0, 4, 7], 0].eq(0).all()


@pytest.mark.unit
def test_a_single_tensor_of_images_takes_the_same_path():
    input_ids = torch.tensor([[IMAGE_TOKEN, IMAGE_TOKEN, 1]])
    embeddings = _inject(torch.full((1, 3, 2), 5.0), input_ids)
    assert embeddings[[0, 1], 0].eq(5).all() and embeddings[2, 0].eq(0).all()


@pytest.mark.unit
def test_token_and_feature_counts_must_agree():
    input_ids = torch.tensor([[IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 1]])
    with pytest.raises(ValueError, match="3 <image> tokens, 2 projected features"):
        _inject([torch.full((1, 3, 2), 1.0)], input_ids)
