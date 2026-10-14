"""NVIDIA Nemotron 3.5 Super VL (`nemotron_h_omni`) for slime's Megatron backend.

The language half needs nothing new. Nemotron 3.5 Super's ``llm_config`` is
field-for-field identical to Nemotron 3 Super 120B-A12B -- 88 layers, hidden
4096, 512 experts at top-k 22, ``moe_latent_size`` 1024, one MTP layer -- and
its ``layers_block_type`` derives to exactly the ``hybrid_override_pattern``
that ``scripts/models/nemotron3-super-120b-a12b.sh`` already carries. So the
decoder here is MCore's ``HybridModel``, built the same way
``slime_plugins/models/nemotron_h.py`` builds it.

What this module adds is the vision half:

    RADIO ViT-H tower  ->  LayerNorm  ->  pixel shuffle  ->  projector MLP
                                             |
                                    scattered into the decoder
                                    embedding at <image> positions

Both vision modules are instantiated from the *checkpoint's own remote code*
rather than reimplemented:

    modeling_radio.RadioModel
    modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector

That is the same idiom ``qwen3_5_vl.py`` uses for its Transformers ViT, and it
is deliberate: the projector owns a LayerNorm whose presence is keyed on the
language model having MTP layers, a pixel-shuffle whose version flag matters,
and an ``mlp1`` whose activation is squared-ReLU. Reimplementing any of that
here would be transcribing semantics that the checkpoint already ships. The
weight names then match the checkpoint too, which is what lets
``hf_to_megatron/nemotron_h_vl.py`` stay a thin dispatcher.

Unlike Qwen3.5-VL this model does *not* use mrope, and it does not use rope
either: the 8 attention layers in the hybrid stack are NoPE. SGLang's
``nemotron_h.py`` has no rotary embedding at all -- ``NemotronHAttention.forward``
is ``qkv_proj -> RadixAttention -> o_proj`` and takes no ``positions`` argument
-- and neither does the checkpoint's ``modeling_nemotron_h.py``. ``config.json``
still carries ``rope_theta`` and ``partial_rotary_factor``; nothing reads them,
and configuring ``--position-embedding-type rope`` off the back of them is what
put ``train_rollout_logprob_abs_diff`` at 2.3 against slime's 0.1 bound. So
there is no position-id rebuild here -- ``position_ids`` passes straight
through, into an embedding that ignores it under ``none``.

Usage (see scripts/models/nemotron3.5-super-vl.sh):

    --spec slime_plugins.models.nemotron_h_vl get_nemotron_h_vl_model_provider
"""

from __future__ import annotations

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig

from slime.utils.accelerator import current_device

from .qwen3_5_vl_utils import gather_packed_input_ids, get_packed_cp_local_indices


def _load_vision_modules(hf_checkpoint: str, hf_config, dtype: torch.dtype, use_cpu_initialization: bool):
    """Build the RADIO tower and the projector from the checkpoint's remote code."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    radio_cls = get_class_from_dynamic_module("modeling_radio.RadioModel", hf_checkpoint)
    projector_cls = get_class_from_dynamic_module(
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector", hf_checkpoint
    )

    device = torch.device("cpu") if use_cpu_initialization else current_device()
    with torch.device(device):
        vision_model = radio_cls(hf_config.vision_config)
        vision_projector = projector_cls(hf_config)

    # The reference model calls this right after construction: it moves image
    # normalization out of the tower, because the processor has already done it.
    # Leaving it in would normalize twice.
    if hasattr(vision_model, "make_preprocessor_external"):
        vision_model.make_preprocessor_external()

    # C-RADIO ships no LayerScale, but RadioLayer builds one unconditionally
    # (modeling_radio.py:372,381) as `layerscale_value * ones(hidden_size)`.
    # layerscale_value is 1.0 for this checkpoint -- configuration_radio.py:67
    # says so in as many words ("C-RADIO has no layerscale; 1.0 makes the
    # (inherited) layerscale an identity op") -- so `hidden_state * lambda1` is
    # the identity, the released index carries no key for it, and the HF loader
    # has to excuse 64 missing keys to load at all (_keys_to_ignore_on_load_missing,
    # modeling_radio.py:412).
    #
    # Materialising an identity as an nn.Parameter is not free here. It puts 64
    # names in named_parameters() that no checkpoint writes and no loader owns,
    # it holds device memory inside the pool a colocated engine releases, and --
    # because this tower is not frozen in the GRPO recipe -- the optimizer moves
    # it, so trainer and engine have to stay in sync on a constant. Dropping the
    # module is exact rather than an approximation, and it is the encoding
    # RadioLayer already uses one line further down for drop_path
    # (modeling_radio.py:373). sglang's side of this is internvl.py:249-250.
    if getattr(hf_config.vision_config, "layerscale_value", None) == 1.0:
        for layer in vision_model.encoder.layer:
            layer.layer_scale1 = torch.nn.Identity()
            layer.layer_scale2 = torch.nn.Identity()

    vision_model.to(dtype=dtype)
    vision_projector.to(dtype=dtype)

    # HF modules are replicated across TP ranks. Mark them explicitly so slime's
    # direct weight exporter does not try to tensor-parallel all-gather them.
    for module in (vision_model, vision_projector):
        for parameter in module.parameters():
            parameter.tensor_model_parallel = False
            parameter.partition_dim = -1
            parameter.partition_stride = 1
    return vision_model, vision_projector


class NemotronHVLModel(MegatronModule):
    """MCore HybridModel decoder with a replicated RADIO tower and projector."""

    def __init__(
        self,
        config,
        hf_config,
        args,
        *,
        pre_process: bool,
        post_process: bool,
        vp_stage: int | None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process

        # <image> in this checkpoint; the processor expands it to one token per
        # projected patch, so this id marks every position a vision feature goes.
        self.img_context_token_id = hf_config.img_context_token_id
        if self.img_context_token_id is None:
            raise ValueError("Nemotron 3.5 VL needs img_context_token_id in the HF config")

        hybrid_override_pattern = get_hybrid_override_pattern(args, hf_config)

        self.language_model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            hybrid_override_pattern=hybrid_override_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            # Only the 8 attention layers use positions; HybridModel defaults to
            # 'none' for a pure-Mamba stack, so a hybrid that has attention has
            # to say so. Same reasoning as slime_plugins/models/nemotron_h.py.
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            # The embedding must NOT shard the sequence itself. It does by
            # default under --sequence-parallel (LanguageModelEmbedding:143-145),
            # and _inject_vision_embeddings() builds its <image> mask from the
            # full packed sequence -- so with TP2 the mask is twice the length of
            # what it indexes:
            #
            #   IndexError: The shape of the mask [2304] at index 0 does not
            #               match the shape of the indexed tensor [1152, 4096]
            #
            # Vision injection has to happen on the whole
            # sequence, so the scatter is deferred to the end of that method.
            # HybridModel re-scatters for a standalone LM forward
            # (hybrid_model.py:501), but only on the branch that builds
            # decoder_input from the embedding; this wrapper passes its own, so
            # that branch is skipped and the sequence is scattered exactly once.
            # ../qwen3_5_vl.py passes this flag for the same reason.
            scatter_embedding_sequence_parallel=False,
            vp_stage=vp_stage,
        )

        # Only the first pipeline stage embeds tokens, so only it needs a tower.
        self.vision_model = None
        self.vision_projector = None
        if pre_process:
            self.vision_model, self.vision_projector = _load_vision_modules(
                args.hf_checkpoint,
                hf_config,
                dtype=config.params_dtype,
                use_cpu_initialization=args.use_cpu_initialization,
            )

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def _inject_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        full_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_group,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None).clone()
        embeddings_bsh = embeddings.transpose(0, 1).contiguous()

        # Without CP the local token stream *is* the full packed stream, so the
        # mapping is the identity. Skipping the lookup is not just an
        # optimization: get_packed_cp_local_indices() requires every packed
        # sequence to divide by 2 * cp_size, which at cp_size=1 degenerates into
        # "must be even" and rejects any odd-length packed sequence.
        if cp_group is None or cp_group.size() == 1:
            local_indices = None
        else:
            local_indices = get_packed_cp_local_indices(
                cu_seqlens,
                cp_group.size(),
                cp_group.rank(),
                input_ids.device,
            )

        # The projector owns the whole pipeline -- tower, LayerNorm, pixel
        # shuffle, mlp1 -- and takes the tower as an argument, exactly as the
        # reference model calls it.
        #
        # pixel_values arrives as a list when the micro-batch holds images of
        # different resolutions, because this tower keeps them as pictures
        # rather than ragged-packed patches and they do not concatenate
        # (backends/megatron_utils/data.py).
        #
        # The list is iterated HERE rather than handed to the projector whole,
        # and that is the whole point of this block. The reference projector
        # does recurse on list/tuple, but it combines the per-image results with
        #
        #     torch.cat([self.forward(pv, vision_model) for pv in pixel_values], dim=0)
        #
        # (modeling_nemotron_h_omni.py:231) over tensors still shaped
        # [n, tokens_i, hidden]. A cat on dim 0 requires every other dimension to
        # agree, so that only works when every image in the list has the same
        # resolution. This processor is aspect-ratio-preserving and
        # variable-resolution (preprocessor_config.json: min_num_patches 1024,
        # max_num_patches 13312), so tokens_i differs per image and the cat
        # raises:
        #
        #   RuntimeError: Sizes of tensors must match except in dimension 0.
        #                 Expected size 286 but got size 270 for tensor number 1
        #
        # (e.g. 286 vs 270 tokens for two images in one micro-batch). The
        # reference branch is fine for its own use, which is generate() on one
        # image at a time.
        #
        # Flattening each image to [tokens_i, hidden] first makes the
        # concatenation well defined for any mix of resolutions, and it is what
        # sglang does for this same model on the same checkpoints
        # (srt/models/nano_nemotron_vl.py:extract_feature_dynamic, which ends in
        # `img_feats.view(-1, hidden)` per image and then one cat).
        #
        # For a list of equal-resolution images this is bit-identical to the old
        # path -- cat-then-flatten and flatten-then-cat produce the same row
        # order -- so it is a strict generalisation, not a behaviour change.
        # Calling the projector per single tensor also keeps its own _project,
        # and with it the vision_final_layernorm that only Super checkpoints
        # carry.
        projector_dtype = next(self.vision_projector.parameters()).dtype
        if isinstance(pixel_values, (list, tuple)):
            per_image = []
            for image in pixel_values:
                features = self.vision_projector(image.to(dtype=projector_dtype), self.vision_model)
                per_image.append(features.reshape(-1, features.shape[-1]))
            vision_output = torch.cat(per_image, dim=0)
        else:
            vision_output = self.vision_projector(
                pixel_values.to(dtype=projector_dtype), self.vision_model
            )
        vision_embeddings = vision_output.reshape(-1, vision_output.shape[-1]).to(
            device=embeddings.device, dtype=embeddings.dtype
        )

        full_vision_positions = (
            (full_input_ids[0] == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        )
        if full_vision_positions.numel() != vision_embeddings.shape[0]:
            raise ValueError(
                f"Nemotron 3.5 VL token/feature mismatch: {full_vision_positions.numel()} "
                f"<image> tokens, {vision_embeddings.shape[0]} projected features"
            )

        feature_indices = torch.full(
            (full_input_ids.shape[1],),
            -1,
            dtype=torch.long,
            device=input_ids.device,
        )
        feature_indices[full_vision_positions] = torch.arange(
            vision_embeddings.shape[0], device=input_ids.device
        )
        local_feature_indices = feature_indices if local_indices is None else feature_indices[local_indices]
        local_vision_mask = local_feature_indices >= 0
        if not torch.equal(local_vision_mask, input_ids[0] == self.img_context_token_id):
            raise ValueError("Nemotron 3.5 VL CP token layout does not match its full packed sequence")
        embeddings_bsh[0, local_vision_mask] = vision_embeddings[local_feature_indices[local_vision_mask]]

        embeddings = embeddings_bsh.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings).contiguous()
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        loss_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        # The processor also emits num_patches / num_tokens / imgs_sizes, which
        # slime forwards verbatim from multimodal_train_inputs. The projector
        # derives the same facts from pixel_values itself, so they are accepted
        # and ignored rather than left to land in **kwargs and reach HybridModel.
        num_patches: torch.Tensor | None = None,
        num_tokens: torch.Tensor | None = None,
        imgs_sizes: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if packed_seq_params is None:
            raise ValueError("Nemotron 3.5 VL native training currently requires packed sequences")

        # slime asks for the MTP loss GPTModel's way, with
        # mtp_kwargs={"mtp_labels": tokens} (backends/megatron_utils/model.py, only
        # under --enable-mtp-training). HybridModel takes no such argument: it
        # derives the MTP labels from input_ids itself -- the same tensor as
        # mtp_labels here -- and gates the branch on compute_mtp_loss, which
        # defaults to True. So translate rather than forward: without this the
        # call would raise on the unknown kwarg, and with MTP layers built but
        # training off, dropping the kwarg alone would still train them.
        mtp_kwargs = kwargs.pop("mtp_kwargs", None)
        kwargs["compute_mtp_loss"] = mtp_kwargs is not None

        # The MTP branch rolls position_ids together with input_ids and asserts
        # they exist (hybrid_model.py:606), but slime never passes any: the main
        # stack is NoPE and GPTModel's MTP path does not insist. Number the tokens
        # within each packed sequence, as MCore's own get_batch would. Under
        # position_embedding_type 'none' the values are not read, so this only has
        # to be well-formed -- but it is also the correct value if that changes.
        if kwargs["compute_mtp_loss"] and position_ids is None:
            position_ids = _packed_position_ids(
                packed_seq_params.cu_seqlens_q, input_ids, mpu.get_context_parallel_group()
            )

        decoder_input = None
        if self.pre_process:
            if pixel_values is None:
                raise ValueError(
                    "Nemotron 3.5 VL received no pixel_values. A text-only batch would need the "
                    "vision branch skipped; this recipe's dataset carries an image per sample."
                )
            cp_group = mpu.get_context_parallel_group()
            full_input_ids = gather_packed_input_ids(input_ids, packed_seq_params.cu_seqlens_q, cp_group)
            decoder_input = self._inject_vision_embeddings(
                input_ids,
                full_input_ids,
                packed_seq_params.cu_seqlens_q,
                cp_group,
                pixel_values,
            )

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            **kwargs,
        )


def _packed_position_ids(cu_seqlens: torch.Tensor, input_ids: torch.Tensor, cp_group) -> torch.Tensor:
    """[1, local tokens] positions, restarting at 0 at every packed-sequence boundary.

    Built over the full packed stream (cu_seqlens describes all of it) and, under
    CP, cut down to this rank's two-chunk slice with the same index map the
    vision injection uses, so it lines up with input_ids token for token.
    """
    cu_seqlens = cu_seqlens.to(device=input_ids.device, dtype=torch.long)
    total = int(cu_seqlens[-1])
    starts = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1])
    positions = torch.arange(total, device=input_ids.device) - starts
    if cp_group is not None and cp_group.size() > 1:
        positions = positions[
            get_packed_cp_local_indices(cu_seqlens, cp_group.size(), cp_group.rank(), input_ids.device)
        ]
    if positions.numel() != input_ids.shape[-1]:
        raise ValueError(
            f"MTP position_ids: {positions.numel()} positions from cu_seqlens, {input_ids.shape[-1]} local tokens"
        )
    return positions.view(1, -1)


# 'M' Mamba2, 'E' MoE, '*' attention -- MCore's Symbols, and the three block
# types this family's layers_block_type uses.
_BLOCK_TYPE_SYMBOLS = {"mamba": "M", "moe": "E", "attention": "*"}


def get_hybrid_override_pattern(args, hf_config) -> str:
    """The M/E/* string MCore needs, from --hybrid-override-pattern or the config.

    Nemotron 3's HF config spells this out as ``hybrid_override_pattern``; the
    3.5 Omni config nests the language model and spells it as a
    ``layers_block_type`` list instead. Both derive to the same 88-character
    pattern for the Super models, so accept either rather than making the model
    config carry a literal that can drift from the checkpoint.
    """
    pattern = getattr(args, "hybrid_override_pattern", None)
    if not pattern:
        llm_config = getattr(hf_config, "llm_config", hf_config)
        pattern = getattr(llm_config, "hybrid_override_pattern", None)
    if not pattern:
        llm_config = getattr(hf_config, "llm_config", hf_config)
        block_types = getattr(llm_config, "layers_block_type", None)
        if block_types:
            try:
                pattern = "".join(_BLOCK_TYPE_SYMBOLS[block] for block in block_types)
            except KeyError as exc:
                raise ValueError(f"unknown layers_block_type entry {exc.args[0]!r}") from exc
    if not pattern:
        raise ValueError(
            "Nemotron-H needs a hybrid layer pattern: pass --hybrid-override-pattern, or use a "
            "checkpoint whose config carries hybrid_override_pattern or layers_block_type."
        )

    # MCore's unified spelling may append MTP depths after '/' -- '<main>/*E/*E'
    # is two depths of the checkpoint's attention+MoE MTP layer. Only the main
    # stack counts towards --num-layers (hybrid_layer_allocation.py:
    # get_hybrid_total_layer_count), and the whole string goes to HybridModel,
    # which builds the MTP block from the segments after the separator.
    main_pattern = pattern.split("/")[0]
    if len(main_pattern) != args.num_layers:
        raise ValueError(
            f"hybrid layer pattern has {len(main_pattern)} main-stack characters but --num-layers is "
            f"{args.num_layers}; every layer needs a symbol."
        )
    return pattern


def get_nemotron_h_vl_model_provider(args, config, vp_stage=None):
    """Return the native Nemotron 3.5 Super VL model provider."""

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if not hasattr(hf_config, "vision_config") or not hasattr(hf_config, "llm_config"):
        raise ValueError(
            f"{args.hf_checkpoint} is not a Nemotron omni checkpoint "
            "(expected both vision_config and llm_config)"
        )
    # Fail here rather than inside the first model_provider call, which on a
    # pipeline-parallel run happens once per stage.
    get_hybrid_override_pattern(args, hf_config)

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> NemotronHVLModel:
        return NemotronHVLModel(
            config,
            hf_config,
            args,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    return model_provider
