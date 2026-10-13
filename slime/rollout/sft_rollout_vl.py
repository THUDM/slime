"""SFT rollout for vision-language models.

`sft_rollout.py` is text only: it loads a processor at module level and never
uses it, so tokens come from `MultiTurnLossMaskGenerator.get_loss_mask`, nothing
populates `Sample.multimodal_inputs`/`multimodal_train_inputs`, and a VL model
pointed at it trains on text-only batches while looking perfectly healthy. That
is why no SFT script in this repo sources a `-vl` model config.

This module is that file with the processor actually used. The pieces it needs
already existed and were simply unwired:

* `Dataset` builds `Sample.multimodal_inputs` for every row whenever
  `--multimodal-keys` is set (`slime/utils/data.py`), the same way the RL path
  gets its images -- nothing there is specific to generation;
* `MultiTurnLossMaskGenerator.get_loss_mask_with_multimodal_alignment`
  (`slime/utils/mask_utils.py`) takes `input_ids` as an *input* rather than
  producing them, i.e. it was written for a caller that has already run the
  processor over text + images. Before this module there was no such caller
  anywhere in the tree;
* `RolloutManager` forwards `Sample.multimodal_train_inputs` to the trainer
  whenever any sample carries it (`slime/ray/rollout.py`), the megatron backend
  concatenates the per-sample tensors and passes them as forward kwargs, and
  `slime_plugins/models/qwen3_5_vl.py` consumes them.

So the only new code is the encode step in the middle, and it is deliberately
kept in one pure function -- `encode_multimodal_sample` -- so it can be tested
without Ray, a data buffer or a GPU.

Use it exactly like the text one, with the multimodal flags the RL path already
passes:

    --rollout-function-path slime.rollout.sft_rollout_vl.generate_rollout \
    --input-key messages \
    --multimodal-keys '{"image": "images"}' \
    --loss-type sft_loss --loss-mask-type qwen3_5 \
    --calculate-per-token-loss --disable-compute-advantages-and-returns \
    --debug-train-only

Do **not** pass `--apply-chat-template`: it renders the prompt to a string at
dataset construction, and both this module and `sft_rollout.py` need the message
list. `--multimodal-keys` alone is enough to get the conversation form.

Rows without images fall back to the text path, so this module is a superset of
`sft_rollout.py` and a mixed dataset trains correctly.
"""

import logging

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import build_processor_kwargs, load_processor, load_tokenizer

__all__ = ["encode_multimodal_sample", "generate_rollout", "has_multimodal_content"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def has_multimodal_content(multimodal_inputs) -> bool:
    """Whether a sample carries anything the processor has to look at.

    `Dataset` sets `multimodal_inputs` to `{"images": None, "videos": None}` for
    a text-only row rather than to None, so a plain truth test is not enough.
    This is the same predicate `filter_long_prompt` uses in `slime/utils/data.py`.
    """
    return bool(multimodal_inputs) and any(value is not None for value in multimodal_inputs.values())


def encode_multimodal_sample(messages, multimodal_inputs, processor, mask_generator, tools=None):
    """Tokenize one multimodal SFT row and align its loss mask to the result.

    Returns ``(token_ids, loss_mask, multimodal_train_inputs)``.

    Two details carry the correctness of the whole module:

    ``add_generation_prompt=False``
        SFT trains *on* the assistant turn, so it has to be inside the rendered
        text. `get_loss_mask` renders the same way for its text projection, so
        the two sequences agree everywhere outside the vision span.

    the alignment is a left-pad
        `get_loss_mask_with_multimodal_alignment` computes the mask on the
        text-only projection of `messages` and prepends
        ``len(input_ids) - len(text_mask)`` zeros. That is correct exactly when
        every token the processor added sits before the first trainable token --
        true for the usual shape of an image in a user turn followed by an
        assistant turn, and not a general guarantee. Verify it by decoding the
        mask back with `MultiTurnLossMaskGenerator.get_text_from_loss_mask`: it
        must return the assistant turn and nothing else. `prepare_sft_dataset.py`
        in the sft_vlm_geo3k_qwen3.5 scripts does exactly that, before a node is
        allocated.
    """
    text = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    processor_output = processor(text=text, **build_processor_kwargs(multimodal_inputs))
    input_ids = processor_output["input_ids"][0]

    token_ids, loss_mask = mask_generator.get_loss_mask_with_multimodal_alignment(messages, input_ids, tools=tools)

    # Everything the processor produced except the token stream itself. These
    # are the tensors the model provider takes as forward kwargs (pixel_values,
    # image_grid_thw, ...); attention_mask is dropped because slime packs
    # sequences and rebuilds it from cu_seqlens.
    multimodal_train_inputs = {
        key: value for key, value in processor_output.items() if key not in ("input_ids", "attention_mask")
    } or None

    return token_ids, loss_mask, multimodal_train_inputs


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Build one SFT training batch from a multimodal dataset.

    Same contract as `slime.rollout.sft_rollout.generate_rollout`: no engine is
    queried, no reward is computed, and the "rollout" is a read of the dataset.

    `evaluation=True` is rejected for the same reason it is there: with
    `--debug-train-only` no inference engine exists, and slime defaults
    `--eval-function-path` to `--rollout-function-path`, so setting
    `--eval-interval` on an SFT run reaches this assert at rollout 0 rather than
    producing an eval. Leave `--eval-interval` unset.
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    samples = data_buffer.get_samples(args.rollout_batch_size)

    num_multimodal = 0
    num_all_zero = 0
    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None)

        if PROCESSOR is not None and has_multimodal_content(sample.multimodal_inputs):
            token_ids, loss_mask, multimodal_train_inputs = encode_multimodal_sample(
                messages,
                sample.multimodal_inputs,
                PROCESSOR,
                MASK_GENERATOR,
                tools=tools,
            )
            sample.multimodal_train_inputs = multimodal_train_inputs
            num_multimodal += 1
        else:
            # A text-only row, or a checkpoint whose AutoProcessor is really a
            # tokenizer (load_processor returns None for those). Identical to
            # sft_rollout.py from here on.
            token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)

        if len(token_ids) != len(loss_mask):
            raise ValueError(
                f"VL SFT rollout produced mismatched token_ids/loss_mask lengths: "
                f"{len(token_ids)=}, {len(loss_mask)=}"
            )
        if not any(loss_mask):
            # Nothing downstream objects to this: the step runs, reports a
            # plausible loss, and learns nothing from the row. A single one can
            # be deliberate (`step_loss_mask: 0` on the only assistant turn), so
            # it is counted rather than raised -- but a whole batch of them is a
            # chat-template / --loss-mask-type mismatch and is fatal below.
            num_all_zero += 1
            if num_all_zero <= 3:
                logger.warning(f"sft_rollout_vl: sample {i} has an all-zero loss mask; it trains on nothing")

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        # `loss_mask[-0:]` is the whole list, not the empty one, so the zero
        # case has to be spelled out.
        sample.loss_mask = loss_mask[-response_length:] if response_length else []

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"sft_rollout_vl::generate_rollout example data: {sample=} (raw){messages=} "
                f"(raw){token_ids=} (raw){loss_mask=} {response_length=}"
            )
            SAMPLE_PRINTED = True

    if num_all_zero == len(samples):
        raise ValueError(
            f"VL SFT rollout {rollout_id}: every one of the {len(samples)} samples has an all-zero loss mask, "
            "so this step would train on nothing. Check that --loss-mask-type matches the checkpoint's chat "
            "template and that each row's last message is an assistant turn."
        )

    logger.info(
        f"sft_rollout_vl::generate_rollout {rollout_id=}: {num_multimodal}/{len(samples)} samples with images"
        + (f", {num_all_zero} with an all-zero loss mask" if num_all_zero else "")
    )

    return samples
