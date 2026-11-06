"""A multimodal partial-rollout continuation is conditioned on its generated response."""

import asyncio
from types import SimpleNamespace

import pytest
from test_score_centering import args, meta

from slime.utils.types import Sample

NUM_GPUS = 0


def _image_sample(rollout_args, *, generated: bool) -> Sample:
    sample = Sample(index=0, prompt="describe the picture", tokens=[9])
    sample.multimodal_inputs = {"images": ["picture"]}
    if generated:
        sample.append_response_tokens(rollout_args, tokens=[3], log_probs=[-0.5], meta_info=meta(), text="A")
        sample.status = Sample.Status.ABORTED
    return sample


@pytest.mark.parametrize("generated", [False, True])
def test_generate_sends_the_generated_response_with_the_images(monkeypatch, generated):
    from slime.rollout import sglang_rollout as rollout

    a = args(
        sglang_router_ip="localhost",
        sglang_router_port=1234,
        use_rollout_routing_replay=False,
        ci_test=False,
        use_score_centering=False,
    )
    monkeypatch.setattr(rollout, "GenerateState", lambda _: SimpleNamespace(tokenizer=None, processor=None))
    monkeypatch.setattr(rollout, "_prepare_prompt_ids", lambda sample, *_: [9])
    monkeypatch.setattr(rollout, "encode_image_for_rollout_engine", lambda image: f"encoded-{image}")
    captured = []

    async def post(url, payload, **kwargs):
        captured.append(payload)
        info = meta()
        info["output_token_logprobs"] = [[-0.25, 4, None]]
        info["finish_reason"] = {"type": "stop"}
        return {"text": "B", "meta_info": info}

    monkeypatch.setattr(rollout, "post", post)
    sample = asyncio.run(rollout.generate(a, _image_sample(a, generated=generated), {"max_new_tokens": 8}))

    payload = captured[0]
    assert payload["image_data"] == ["encoded-picture"]
    if generated:
        # The continuation resumes after token 3 instead of restarting at the prompt.
        assert payload["input_ids"] == [9, 3]
        assert "text" not in payload
        assert payload["sampling_params"]["max_new_tokens"] == 7
        assert sample.tokens == [9, 3, 4]
    else:
        assert payload["text"] == "describe the picture"
        assert "input_ids" not in payload
        assert sample.tokens == [9, 4]
