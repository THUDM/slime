"""SGLang top-p metadata must survive overlapped result copies and the next forward."""

import pytest
import torch

NUM_GPUS = 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "field,dtype,expected,replacement",
    [
        ("next_token_top_p_token_ids", torch.int32, 17, 99),
        ("next_token_top_p_log_probs", torch.float32, -0.5, 0.75),
    ],
)
def test_top_p_copy_preserves_values_after_source_release(field, dtype, expected, replacement):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult

    forward = torch.cuda.Stream()
    copy = torch.cuda.Stream()
    for _ in range(32):
        with torch.cuda.stream(forward):
            rows = [torch.full((128,), expected, dtype=dtype, device="cuda") for _ in range(16)]
            logits = LogitsProcessorOutput(next_token_logits=None)
            setattr(logits, field, rows)
            result = GenerationBatchResult(
                logits_output=logits,
                next_token_ids=torch.zeros(1, dtype=torch.long, device="cuda"),
                copy_done=torch.cuda.Event(),
            )
        copy.wait_stream(forward)
        with torch.cuda.stream(copy):
            # Keep the copy pending while the next forward reuses allocations.
            torch.cuda._sleep(3_000_000)
            result.copy_to_cpu(return_logprob=True)
        del rows
        with torch.cuda.stream(forward):
            next_forward = [torch.full((128,), replacement, dtype=dtype, device="cuda") for _ in range(32)]
        try:
            result.copy_done.synchronize()
            for row in getattr(result.logits_output, field):
                assert row.device.type == "cpu"
                torch.testing.assert_close(row, torch.full_like(row, expected), rtol=0, atol=0)
        finally:
            forward.synchronize()
            del next_forward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
