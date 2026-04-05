import logging

from slime.utils.processing_utils import load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

TOKENIZER = None
SAMPLE_PRINTED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Rollout function for Continued Pre-Training (CPT).

    Unlike SFT rollout which only computes loss on assistant responses,
    CPT computes loss on all tokens (full causal LM objective).

    Expects plain-text data (a single text field per sample).
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    max_length = args.seq_length
    samples = data_buffer.get_samples(args.rollout_batch_size)

    for i, sample in enumerate(samples):
        (sample,) = sample
        text = sample.prompt

        token_ids = TOKENIZER(text, add_special_tokens=False)["input_ids"]

        # Truncate to max sequence length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # CPT: compute loss on all tokens except the first.
        # The first token acts as "prompt" (prompt_length=1) so that
        # logit slicing in loss.py works correctly:
        #   logits[0 : n-1] predicts tokens[1 : n]
        loss_mask = [1] * (len(token_ids) - 1)

        sample.tokens = token_ids
        sample.response_length = len(token_ids) - 1
        sample.reward = 0
        sample.loss_mask = loss_mask

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"cpt_rollout::generate_rollout example: len={len(token_ids)}, "
                f"text_preview={text[:200]!r}"
            )
            SAMPLE_PRINTED = True

    return samples
