import json
import os
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.on_policy_distillation import (
    _align_common_tokens_1to1,
    _render_teacher_prompt,
    post_process_rewards_cross_vocab,
    reward_func_cross_vocab,
)
from slime.utils.data import Dataset
from slime.utils.types import Sample


class FakeTokenizer:
    def __init__(self, decoded=None):
        self.decoded = decoded or {}
        self.rendered_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.rendered_messages = messages
        rendered = "<teacher>" + "|".join(message["content"] for message in messages) + "<assistant>"
        if tokenize:
            return list(range(len(messages) + 1))
        return rendered

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.decoded[token_id] for token_id in token_ids)


def test_cross_vocab_render_prefers_metadata_messages():
    tokenizer = FakeTokenizer()
    args = SimpleNamespace(opd_prompt_messages_key="opd_messages")
    messages = [{"role": "user", "content": "solve"}]
    sample = Sample(prompt="<student>solve<assistant>", response="42", metadata={"opd_messages": messages})

    prompt_text, prompt_ids = _render_teacher_prompt(args, sample, tokenizer)

    assert prompt_text == "<teacher>solve<assistant>"
    assert prompt_ids == [0, 1]
    assert tokenizer.rendered_messages == messages


def test_dataset_preserves_raw_messages_for_cross_vocab_opd(tmp_path):
    data_path = tmp_path / "data.jsonl"
    messages = [{"role": "user", "content": "solve"}]
    data_path.write_text(json.dumps({"messages": messages, "metadata": {"source": "unit"}}) + "\n")

    dataset = Dataset(
        str(data_path),
        tokenizer=FakeTokenizer(),
        processor=None,
        max_length=None,
        prompt_key="messages",
        metadata_key="metadata",
        apply_chat_template=True,
        prompt_messages_key="opd_messages",
    )

    sample = dataset[0]
    assert sample.prompt == "<teacher>solve<assistant>"
    assert sample.metadata["source"] == "unit"
    assert sample.metadata["opd_messages"] == messages


def test_align_common_tokens_uses_student_logprob_for_tokenizer_mismatch():
    aligned, matched = _align_common_tokens_1to1(
        teacher_texts=["A", "BC"],
        teacher_lps=[-1.0, -2.0],
        student_texts=["A", "B", "C"],
        student_rollout_lps=[-0.1, -0.2, -0.3],
    )

    assert matched == 1
    assert aligned == [-1.0, -0.2, -0.3]


def test_cross_vocab_post_process_aligns_teacher_response_logprobs():
    args = SimpleNamespace(
        reward_key=None,
        teacher_tokenizer_path="unused",
        opd_mask_teacher_logprob_tokens=None,
        _cross_vocab_teacher_tok=FakeTokenizer({1: "A", 2: "B"}),
        _cross_vocab_student_tok=FakeTokenizer({3: "A", 4: "B"}),
    )
    sample = Sample(
        tokens=[99, 3, 4],
        response_length=2,
        rollout_log_probs=[-0.3, -0.4],
        reward={
            "_cross_vocab_meta": {"teacher_prompt_len": 2},
            "meta_info": {
                "input_token_logprobs": [
                    [None, 10],
                    [-0.1, 11],
                    [-1.0, 1],
                    [-2.0, 2],
                ]
            },
        },
    )

    rewards, raw_rewards = post_process_rewards_cross_vocab(args, [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert torch.equal(sample.teacher_log_probs, torch.tensor([-1.0, -2.0]))
    assert sample.metadata["cross_vocab_token_overlap"] == 1.0


def test_cross_vocab_fallback_does_not_require_tokenizers():
    args = SimpleNamespace(reward_key=None)
    sample = Sample(
        response_length=2,
        rollout_log_probs=[-4.0, -5.0],
        reward={"_opd_teacher_fallback": True, "_opd_teacher_fallback_reason": "TimeoutError"},
    )

    rewards, raw_rewards = post_process_rewards_cross_vocab(args, [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert torch.equal(sample.teacher_log_probs, torch.tensor([-4.0, -5.0]))
    assert sample.metadata["opd_teacher_fallback"] is True


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.getenv("SLIME_OPD_TEACHER_RM_URL")
        and os.getenv("SLIME_OPD_STUDENT_TOKENIZER_PATH")
        and os.getenv("SLIME_OPD_TEACHER_TOKENIZER_PATH")
    ),
    reason=(
        "set SLIME_OPD_TEACHER_RM_URL, SLIME_OPD_STUDENT_TOKENIZER_PATH, "
        "and SLIME_OPD_TEACHER_TOKENIZER_PATH to run live cross-vocab OPD teacher API test"
    ),
)
@pytest.mark.asyncio
async def test_cross_vocab_reward_func_with_live_teacher_api():
    """Optional live check for a running SGLang teacher /generate endpoint.

    Example:
        SLIME_OPD_TEACHER_RM_URL=http://127.0.0.1:30000/generate \
        SLIME_OPD_STUDENT_TOKENIZER_PATH=/path/to/student \
        SLIME_OPD_TEACHER_TOKENIZER_PATH=/path/to/teacher \
        pytest tests/test_on_policy_distillation.py::test_cross_vocab_reward_func_with_live_teacher_api
    """
    from transformers import AutoTokenizer

    messages = [{"role": "user", "content": os.getenv("SLIME_OPD_LIVE_PROMPT", "What is 2+2?")}]
    response = os.getenv("SLIME_OPD_LIVE_RESPONSE", "4")
    student_tokenizer_path = os.environ["SLIME_OPD_STUDENT_TOKENIZER_PATH"]
    teacher_tokenizer_path = os.environ["SLIME_OPD_TEACHER_TOKENIZER_PATH"]
    rm_url = os.environ["SLIME_OPD_TEACHER_RM_URL"]

    student_tok = AutoTokenizer.from_pretrained(student_tokenizer_path, trust_remote_code=True)
    student_prompt = student_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    student_prompt_ids = student_tok.encode(student_prompt, add_special_tokens=False)
    response_ids = student_tok.encode(response, add_special_tokens=False)
    response_length = len(response_ids)
    assert response_length > 0

    sample = Sample(
        prompt=student_prompt,
        tokens=student_prompt_ids + response_ids,
        response=response,
        response_length=response_length,
        rollout_log_probs=[-1.0] * response_length,
        metadata={"opd_messages": messages},
    )
    args = SimpleNamespace(
        hf_checkpoint=student_tokenizer_path,
        teacher_tokenizer_path=teacher_tokenizer_path,
        rm_url=rm_url,
        reward_key=None,
        custom_rm_path=None,
        opd_prompt_messages_key="opd_messages",
        opd_mask_teacher_logprob_tokens=None,
        opd_teacher_timeout=float(os.getenv("SLIME_OPD_TEACHER_TIMEOUT", "300")),
        opd_teacher_retries=int(os.getenv("SLIME_OPD_TEACHER_RETRIES", "0")),
        opd_teacher_concurrency=0,
    )

    sample.reward = await reward_func_cross_vocab(args, sample)
    if isinstance(sample.reward, dict) and sample.reward.get("_opd_teacher_fallback"):
        pytest.fail(f"teacher request fell back: {sample.reward.get('_opd_teacher_fallback_reason')}")

    rewards, raw_rewards = post_process_rewards_cross_vocab(args, [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert sample.teacher_log_probs is not None
    assert len(sample.teacher_log_probs) == response_length
    assert torch.isfinite(sample.teacher_log_probs).all()
    assert 0.0 <= sample.metadata["cross_vocab_token_overlap"] <= 1.0
