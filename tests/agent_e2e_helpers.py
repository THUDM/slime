"""Assertions executed inside real coding-agent rollout and Megatron workers."""

import json
import math
import os
from dataclasses import asdict
from pathlib import Path


def audit_token_records(samples, turns):
    """Match every trained span to its exact model input, output IDs and logprobs."""
    import torch

    turns = [turn for turn in turns if turn["output_ids"]]
    used = set()
    replay_fields = set()
    for sample in samples:
        prompt_length = len(sample.tokens) - sample.response_length
        position = 0
        while position < sample.response_length:
            if not sample.loss_mask[position]:
                position += 1
                continue
            offset = prompt_length + position
            candidates = [
                (index, turn)
                for index, turn in enumerate(turns)
                if index not in used
                and turn["prompt_ids"] == sample.tokens[:offset]
                and turn["output_ids"] == sample.tokens[offset : offset + len(turn["output_ids"])]
                and turn["output_log_probs"] == sample.rollout_log_probs[position : position + len(turn["output_ids"])]
            ]
            assert candidates, "Training tokens/context/logprobs differ from the original model turn"
            index, turn = candidates[0]
            count = len(turn["output_ids"])
            assert sample.loss_mask[position : position + count] == [1] * count
            for key, expected in (turn.get("replay") or {}).items():
                actual = getattr(sample, key)
                assert actual is not None, f"Missing sampler replay metadata: {key}"
                assert torch.equal(torch.as_tensor(actual), torch.as_tensor(expected)), key
                replay_fields.add(key)
            used.add(index)
            position += count
    assert len(used) == len(turns), "A sampled turn was dropped or counted more than once"
    sampled_tokens = sum(len(turn["output_ids"]) for turn in turns)
    assert sum(sum(sample.loss_mask) for sample in samples) == sampled_tokens
    return {
        "model_turns": len(turns),
        "sampled_tokens": sampled_tokens,
        "training_segments": len(samples),
        "exact_input_output_ids_and_logprobs": True,
        "every_sampled_token_retained_once": True,
        "replay_metadata_fields_verified": sorted(replay_fields),
    }


async def generate(args, sample, sampling_params, evaluation=False):
    import torch
    from examples.coding_agent_rl import generate as agent

    assert args.rollout_top_k == -1 and sampling_params.get("top_k", -1) == -1, "Top-k replay is unsupported"
    assert sampling_params.get("top_p") == 0.95 and args.use_score_centering
    adapter = agent._AdapterService(args).adapter
    assert adapter.debug_callback is None, "This CI runs one instrumented agent at a time"
    turns = []
    adapter.debug_callback = lambda sid, messages, tools, response, turn: turns.append({"sid": sid, **asdict(turn)})
    run_dir = Path(os.environ["SLIME_AGENT_TEST_RUN_DIR"])
    try:
        samples = await agent.generate(args, sample, sampling_params, evaluation=evaluation)
    finally:
        adapter.debug_callback = None
        torch.save({"turns": turns}, run_dir / "model-turns.pt")
        summaries = [
            {
                **turn,
                "replay": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in (turn.get("replay") or {}).items()
                },
            }
            for turn in turns
        ]
        (run_dir / "model-turns.jsonl").write_text("".join(json.dumps(turn) + "\n" for turn in summaries))

    # Keep the real trajectory even when a later CI assertion fails.
    torch.save({"samples": [branch.to_dict() for branch in samples]}, run_dir / "agent-full.pt")
    assert samples, "Agent returned no training segments"
    assert any(branch.reward == 1 for branch in samples), "The CI task was not solved by the agent"
    for branch in samples:
        assert branch.group_index == sample.group_index and branch.index == sample.index
        assert branch.rollout_id == (sample.rollout_id if sample.rollout_id is not None else sample.index)
        assert not branch.remove_sample, branch.metadata
        assert branch.metadata["agent_exit_code"] == 0, branch.metadata
        assert branch.response_length == len(branch.loss_mask) == len(branch.rollout_log_probs)
        assert sum(branch.loss_mask) > 0, "Agent segment contains no sampled training tokens"
        assert all(math.isfinite(lp) for lp in branch.rollout_log_probs)
    assert len(samples) >= 2, "The fixture must exercise multiple agent turns"
    audit = audit_token_records(samples, turns)
    audit["sampling_params"] = dict(sampling_params)
    (run_dir / "token-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Keep the entire real trajectory as evidence. Replayed context makes each
    # turn a separate training segment; training all of them makes a tiny CI
    # repair unexpectedly expensive. The first action and final response cover
    # both ends of the token/logprob path with a bounded optimizer workload.
    return [samples[0], samples[-1]]


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Check the actual optimizer call and observe model parameter changes.

    This wrapper neither modifies rewards nor updates parameters itself. Small
    deterministic slices avoid saving a 27B model just to prove that it changed.
    """
    import torch
    import torch.distributed as dist

    def snapshot():
        values = []
        for chunk in model:
            for parameter in chunk.parameters():
                if parameter.requires_grad and parameter.numel():
                    flat = parameter.detach().view(-1)
                    values.append(flat[:: max(1, flat.numel() // 64)][:64].float().cpu())
        return torch.cat(values)

    original_step = optimizer.step

    def checked_step(*positional, **keywords):
        before = snapshot()
        try:
            result = original_step(*positional, **keywords)
            after = snapshot()
            delta = (after - before).abs()
            assert result[0], "Optimizer skipped the training step"
            assert math.isfinite(float(result[1])) and float(result[1]) > 0, result
            assert torch.isfinite(after).all() and torch.count_nonzero(delta) > 0, "Model parameters did not change"
            evidence = {
                "rank": dist.get_rank(),
                "rollout_id": rollout_id,
                "step_id": step_id,
                "grad_norm": float(result[1]),
                "sampled_parameters": before.numel(),
                "changed_parameters": torch.count_nonzero(delta).item(),
                "max_parameter_delta": delta.max().item(),
            }
            path = Path(os.environ["SLIME_AGENT_TEST_RUN_DIR"]) / f"optimizer-rank-{dist.get_rank()}.json"
            path.write_text(json.dumps(evidence, indent=2) + "\n")
            return result
        finally:
            optimizer.step = original_step

    optimizer.step = checked_step
