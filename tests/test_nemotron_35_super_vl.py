"""CPU-only checks for the Nemotron 3.5 Super VL port.

None of these needs a GPU, and none reads a safetensors *shard* -- the weight
checks go against `model.safetensors.index.json`, which is 4.7 MB. That is the
point: every name-map hole this port can have is findable before a node is
allocated, and certainly before the hour that `04_convert_checkpoint.sh` costs.

Set `NEMOTRON_35_VL_CHECKPOINT` to the checkpoint directory to enable the tests
that need it; the rest run anywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slime.backends.megatron_utils.hf_to_megatron import _LOADERS  # noqa: E402
from slime.backends.megatron_utils.hf_to_megatron.nemotron_h import (  # noqa: E402
    _hf_prefix,
    _language_config,
    _RADIO_PREFIX,
    _VISION_LAYER_RE,
    _VISION_LAYER_SCALE_RE,
    _VISION_PER_LAYER,
    _VISION_QKV_CHUNK,
    _VISION_QKV_RE,
    _VISION_TOP_LEVEL,
    _vision_hf_tensor,
)

CHECKPOINT = os.environ.get(
    "NEMOTRON_35_VL_CHECKPOINT",
    "/lustre/fsw/coreai_devtech_all/shifangx/download_hf/models/nvidia/"
    "NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth",
)

needs_checkpoint = pytest.mark.skipif(
    not (Path(CHECKPOINT) / "config.json").is_file(),
    reason=f"Nemotron 3.5 Super VL checkpoint not at {CHECKPOINT}",
)


# ---------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def raw_config() -> dict:
    return json.loads((Path(CHECKPOINT) / "config.json").read_text())


@pytest.fixture(scope="module")
def weight_index() -> set[str]:
    path = Path(CHECKPOINT) / "model.safetensors.index.json"
    return set(json.loads(path.read_text())["weight_map"])


def expected_vision_parameter_names(vision_config) -> list[str]:
    """The vision half's module tree, derived from the config.

    This mirrors `modeling_radio.py` and `modeling_nemotron_h_omni.py` by hand,
    which is exactly the kind of transcription that rots. It does not have to be
    trusted: `test_analytic_tree_matches_the_real_modules` builds the real
    modules and asserts they produce this list and nothing else.
    """
    names = ["vision_model.summary_idxs"]
    names += [
        "vision_model.embeddings.patch_projection.weight",
        "vision_model.embeddings.position_embedding",
        "vision_model.embeddings.cls_register_token",
    ]
    if getattr(vision_config, "video_temporal_patch_size", None) is not None:
        names.append("vision_model.embeddings.video_patch_projection.weight")

    for i in range(vision_config.num_hidden_layers):
        block = f"vision_model.encoder.layer.{i}"
        for norm in ("norm1", "norm2"):
            names += [f"{block}.{norm}.weight", f"{block}.{norm}.bias"]
        for projection in ("query", "key", "value"):
            names += [
                f"{block}.attention.attention.{projection}.weight",
                f"{block}.attention.attention.{projection}.bias",
            ]
        names += [
            f"{block}.attention.output.dense.weight",
            f"{block}.attention.output.dense.bias",
            f"{block}.mlp.fc1.weight",
            f"{block}.mlp.fc1.bias",
            f"{block}.mlp.fc2.weight",
            f"{block}.mlp.fc2.bias",
            f"{block}.layer_scale1.lambda1",
            f"{block}.layer_scale2.lambda1",
        ]

    names += [
        "vision_projector.mlp1.norm.weight",
        "vision_projector.mlp1.linear1.weight",
        "vision_projector.mlp1.linear2.weight",
        "vision_projector.vision_final_layernorm.weight",
        "vision_projector.vision_final_layernorm.bias",
    ]
    return names


# ------------------------------------------------------- the prefix switch --
class _FakeConfig:
    def __init__(self, model_type, llm_config=None):
        self.model_type = model_type
        if llm_config is not None:
            self.llm_config = llm_config


def test_hf_prefix_switches_on_model_type():
    assert _hf_prefix(_FakeConfig("nemotron_h_omni")) == "language_model."
    # Nemotron-3 must keep working through this file unchanged: an accidental
    # prefix here is 42,683 KeyErrors in the sibling directory.
    assert _hf_prefix(_FakeConfig("nemotron_h")) == ""


def test_language_config_is_where_merge_qkv_reads_its_shapes():
    llm = object()
    assert _language_config(_FakeConfig("nemotron_h_omni", llm)) is llm
    text_only = _FakeConfig("nemotron_h")
    assert _language_config(text_only) is text_only


def test_both_model_types_are_registered():
    assert _LOADERS["nemotron_h"] is _LOADERS["nemotron_h_omni"]


# ----------------------------------------------------------- the vision map --
@needs_checkpoint
def test_every_vision_name_resolves_to_a_real_tensor(raw_config, weight_index):
    """The check this file exists for.

    Every parameter the vision half will ask for must map to a key that is
    actually in the released index -- or be one of the two families that are
    deliberately synthesized from the config because the checkpoint has no
    tensor for them.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
    synthesized = []
    resolved = 0
    keys_used: set[str] = set()

    for name in expected_vision_parameter_names(hf_config.vision_config):
        layer = _VISION_LAYER_RE.fullmatch(name)
        rest = layer.group(2) if layer else ""

        if name == "vision_model.summary_idxs" or _VISION_LAYER_SCALE_RE.fullmatch(rest):
            synthesized.append(name)
            # Synthesized: assert it really is absent, so that the day the
            # checkpoint starts shipping it, this test says so instead of
            # silently preferring the config.
            assert name not in weight_index
            _vision_hf_tensor(name, reader=None, hf_config=hf_config)
            continue

        if name in _VISION_TOP_LEVEL:
            target = _VISION_TOP_LEVEL[name]
        else:
            assert layer, name
            hf_block = f"{_RADIO_PREFIX}.blocks.{layer.group(1)}"
            qkv = _VISION_QKV_RE.fullmatch(rest)
            if qkv:
                target = f"{hf_block}.attn.qkv.{qkv.group(2)}"
            else:
                assert rest in _VISION_PER_LAYER, name
                target = f"{hf_block}.{_VISION_PER_LAYER[rest]}"

        assert target in weight_index, f"{name} -> {target}"
        keys_used.add(target)
        resolved += 1

    # 521 module parameters, because q/k/v/{weight,bias} are six names reading
    # two fused checkpoint tensors per block.
    assert resolved == 521, resolved
    assert len(synthesized) == 1 + 2 * hf_config.vision_config.num_hidden_layers

    # The real assertion: every vision tensor in the checkpoint is accounted
    # for, either read or explicitly not read. Nothing is silently dropped.
    vision_keys = {name for name in weight_index if not name.startswith("language_model.")}
    assert len(vision_keys) == 395
    assert vision_keys - keys_used == {
        "vision_model.radio_model.input_conditioner.norm_mean",
        "vision_model.radio_model.input_conditioner.norm_std",
    }


@needs_checkpoint
def test_deliberately_unloaded_tensors_are_named(weight_index):
    """The input conditioner is in the checkpoint and has no module.

    `make_preprocessor_external()` replaces it with an Identity because the
    image processor already normalizes. Recording that here is what stops it
    being rediscovered as a mystery later.
    """
    assert "vision_model.radio_model.input_conditioner.norm_mean" in weight_index
    assert "vision_model.radio_model.input_conditioner.norm_std" in weight_index
    for name in _VISION_TOP_LEVEL.values():
        assert "input_conditioner" not in name


@needs_checkpoint
def test_the_language_half_is_nemotron3s_under_one_prefix(weight_index):
    """§0.3 of the development plan, as a test rather than a transcript."""
    language = {n for n in weight_index if n.startswith("language_model.")}
    vision = weight_index - language
    assert len(language) == 42683
    assert len(vision) == 395
    # And the pattern, which is what makes the model bundle a copy.
    config = json.loads((Path(CHECKPOINT) / "config.json").read_text())
    symbol = {"mamba": "M", "moe": "E", "attention": "*"}
    pattern = "".join(symbol[block] for block in config["llm_config"]["layers_block_type"])
    assert pattern == (
        "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
    )
    assert len(pattern) == 88
    assert [i for i, c in enumerate(pattern) if c == "*"] == [7, 16, 25, 36, 47, 58, 69, 78]


def test_vision_qkv_is_split_into_equal_thirds():
    """RADIO is plain MHA, so the fused qkv splits evenly -- no GQA asymmetry."""

    class _Reader:
        def get_tensor(self, name):
            assert name == "vision_model.radio_model.model.blocks.0.attn.qkv.weight"
            # [3 * 1280, 1280], with each third filled with its own marker.
            return torch.cat([torch.full((1280, 1280), float(i)) for i in range(3)], dim=0)

    class _Config:
        class vision_config:
            hidden_size = 1280
            layerscale_value = 1.0
            summary_idxs = [0, 1]

    for projection, expected in _VISION_QKV_CHUNK.items():
        tensor = _vision_hf_tensor(
            f"vision_model.encoder.layer.0.attention.attention.{projection}.weight",
            _Reader(),
            _Config(),
        )
        assert tensor.shape == (1280, 1280)
        assert torch.all(tensor == float(expected))


def test_synthesized_families_come_from_the_config():
    class _Config:
        class vision_config:
            hidden_size = 1280
            layerscale_value = 1.0
            summary_idxs = [0, 1]

    summary = _vision_hf_tensor("vision_model.summary_idxs", None, _Config())
    assert summary.tolist() == [0, 1]
    assert summary.dtype == torch.long

    # layerscale_value is 1.0, so LayerScale is an identity op -- synthesizing
    # ones reproduces exactly what `from_pretrained` would leave in place.
    scale = _vision_hf_tensor("vision_model.encoder.layer.7.layer_scale2.lambda1", None, _Config())
    assert scale.shape == (1280,)
    assert torch.all(scale == 1.0)


def test_an_unknown_vision_name_raises_rather_than_guessing():
    class _Config:
        class vision_config:
            hidden_size = 1280
            layerscale_value = 1.0
            summary_idxs = [0, 1]

    with pytest.raises(KeyError):
        _vision_hf_tensor("vision_model.encoder.layer.0.mlp.fc3.weight", None, _Config())
    with pytest.raises(KeyError):
        _vision_hf_tensor("vision_model.something_new", None, _Config())


# ------------------------------------- the analytic tree vs the real modules --
@needs_checkpoint
def test_analytic_tree_matches_the_real_modules():
    """Build the two vision modules and compare their parameter names.

    This is what makes `expected_vision_parameter_names` trustworthy: without
    it, a rename inside `modeling_radio.py` would leave the map above passing
    and the model failing. Needs no weights -- the modules are built on the meta
    device -- but it does need `transformers` new enough to import the
    checkpoint's remote code, which is itself worth knowing before a launch.
    """
    from transformers import AutoConfig

    from slime_plugins.models.nemotron_35_super_vl import _load_remote_class

    hf_config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
    radio_cls = _load_remote_class(CHECKPOINT, "modeling_radio.RadioModel")
    projector_cls = _load_remote_class(
        CHECKPOINT, "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector"
    )

    with torch.device("meta"):
        vision_model = radio_cls(hf_config.vision_config)
        vision_model.make_preprocessor_external()
        vision_projector = projector_cls(hf_config)

    actual = {f"vision_model.{n}" for n, _ in vision_model.named_parameters()}
    actual |= {f"vision_model.{n}" for n, _ in vision_model.named_buffers()}
    actual |= {f"vision_projector.{n}" for n, _ in vision_projector.named_parameters()}
    actual |= {f"vision_projector.{n}" for n, _ in vision_projector.named_buffers()}

    expected = set(expected_vision_parameter_names(hf_config.vision_config))
    assert actual == expected, {
        "missing from the analytic list": sorted(actual - expected),
        "in the analytic list but not built": sorted(expected - actual),
    }


# ------------------------------------------------------- the loss-mask claim --
@needs_checkpoint
def test_the_mask_alignment_survives_the_image_expansion():
    """§4.3 of the development plan.

    `get_loss_mask_with_multimodal_alignment` left-pads the text-only mask by
    `len(input_ids) - len(text_mask)` zeros, which is correct only when every
    token the processor added sits strictly before the first trainable token.
    That is a claim about *this* chat template, so it is measured here rather
    than inherited from the Qwen sibling.
    """
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    messages = [
        {"role": "user", "content": "<image>\nFind x in the figure."},
        {"role": "assistant", "content": "Answer: \\boxed{48}"},
    ]
    token_ids, mask = generator.get_loss_mask(messages)
    assert len(token_ids) == len(mask)
    assert any(mask), "an all-zero mask trains on nothing while reporting a plausible loss"
    text_span = generator.get_text_from_loss_mask(token_ids, mask)

    # Now simulate what the processor does: one "<image>" becomes
    # "<img>" + "<image>" * n + "</img>". Only the count matters to the aligner.
    image_token_id = 18
    position = token_ids.index(image_token_id)
    expanded = token_ids[:position] + [image_token_id] * 256 + token_ids[position + 1 :]

    aligned_ids, aligned_mask = generator.get_loss_mask_with_multimodal_alignment(messages, expanded)
    assert len(aligned_ids) == len(aligned_mask) == len(token_ids) + 255
    assert sum(aligned_mask) == sum(mask)
    # The load-bearing assertion: the trainable span is the assistant turn,
    # unchanged by the expansion. A few-token drift here is silent otherwise.
    assert generator.get_text_from_loss_mask(aligned_ids, aligned_mask) == text_span
    assert text_span == ["<think></think>Answer: \\boxed{48}<|im_end|>\n"]
