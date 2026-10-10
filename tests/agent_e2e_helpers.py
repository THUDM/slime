"""Assertions executed inside real coding-agent rollout and Megatron workers."""

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

_CAPTURED_TURNS = {}


def _capture_turn(sid, messages, tools, response, turn):
    _CAPTURED_TURNS[sid].append({"sid": sid, **asdict(turn)})


def audit_token_records(samples, turns):
    """Match every trained span to its exact model input, output IDs and logprobs."""
    import torch

    from slime.utils.score_centering import validate_sampler_top_p

    turns = [turn for turn in turns if turn["output_ids"]]
    used = set()
    replay_fields = set()
    for sample in samples:
        prompt_length = len(sample.tokens) - sample.response_length
        if sample.rollout_top_p_log_probs is not None:
            # Check masked shared prefixes too: retained replay distributions
            # must still agree with their original sampled-token logprobs.
            validate_sampler_top_p(
                sample.rollout_top_p_token_ids,
                sample.rollout_top_p_token_offsets,
                sample.rollout_top_p_log_probs,
                sample.response_length,
                sample.loss_mask,
                sample.tokens[prompt_length:],
                sample.rollout_log_probs,
            )
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
                if key.startswith("rollout_top_p_"):
                    offsets = torch.as_tensor(sample.rollout_top_p_token_offsets)
                    begin, end = int(offsets[position]), int(offsets[position + count])
                    actual = (
                        offsets[position : position + count + 1] - begin
                        if key.endswith("offsets")
                        else actual[begin:end]
                    )
                elif key.startswith("rollout_topk_"):
                    actual = actual[position : position + count]
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


def audit_grpo_training(samples, trained_samples):
    """Check the trainer's actual rewards and token advantages against outcomes."""
    import torch

    outcomes = {}
    for sample in samples:
        rid = sample["rollout_id"]
        outcome = (sample["group_index"], sample["reward"])
        assert outcomes.setdefault(rid, outcome) == outcome
    groups, expected = [], {}
    for group in sorted({group for group, _ in outcomes.values()}):
        ids = [rid for rid, (gid, _) in outcomes.items() if gid == group]
        assert len(ids) == 4, "Each prompt must have four independent agent outcomes"
        raw = torch.tensor([outcomes[rid][1] for rid in ids], dtype=torch.float32)
        assert set(raw.tolist()) <= {0.0, 1.0}, "Rewards must come from the real binary grader"
        normalized = (raw - raw.mean()) / (raw.std() + 1e-6)
        expected.update(zip(ids, normalized.tolist(), strict=True))
        groups.append(
            {
                "group_index": group,
                "rollout_ids": ids,
                "raw_rewards": raw.tolist(),
                "reward_mean": raw.mean().item(),
                "reward_std": raw.std().item(),
                "normalized_advantages": normalized.tolist(),
                "has_learning_signal": bool(torch.count_nonzero(normalized)),
            }
        )
    assert groups, "No prompt groups entered training"
    seen = set()
    for trained in trained_samples:
        rid = trained["rollout_ids"]
        assert math.isclose(float(trained["rewards"]), expected[rid], rel_tol=1e-5, abs_tol=1e-6)
        mask = trained["loss_masks"].bool()
        advantages = trained["advantages"][mask].float()
        assert advantages.numel() > 0
        assert torch.allclose(
            advantages, torch.full_like(advantages, expected[rid]), rtol=1e-5, atol=1e-6
        ), "Training token advantages differ from group-normalized real rewards"
        seen.add(rid)
    assert seen == set(outcomes), "Both successful and failed rollouts must enter training"
    return {
        "groups": groups,
        "has_learning_signal": any(group["has_learning_signal"] for group in groups),
        "training_token_advantages_verified": True,
    }


async def generate(args, sample, sampling_params, evaluation=False):
    import torch
    from examples.coding_agent_rl import generate as agent

    assert args.rollout_top_k == -1 and sampling_params.get("top_k", -1) == -1, "Top-k replay is unsupported"
    assert sampling_params.get("top_p") == 0.95 and args.use_score_centering
    assert args.n_samples_per_prompt == 4 and args.rewards_normalization and args.grpo_std_normalization
    assert args.sglang_speculative_algorithm == "EAGLE", "Inference must exercise the model's MTP head"
    step = sample.index // (args.rollout_batch_size * args.n_samples_per_prompt)
    assert step in (0, 1), "The E2E has exactly two sampling/training steps"
    name = ("codex", "claude_code")[step]
    sample.metadata = {**(sample.metadata or {}), "agent": name}
    adapter, _, _ = agent._AdapterService(args).endpoint(name)
    assert adapter.debug_callback in (None, _capture_turn)
    adapter.debug_callback = _capture_turn
    assert sample.session_id is not None and sample.session_id not in _CAPTURED_TURNS
    turns = _CAPTURED_TURNS[sample.session_id] = []
    run_dir = Path(os.environ["SLIME_AGENT_TEST_RUN_DIR"]) / "agents" / str(sample.index)
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        samples = await agent.generate(args, sample, sampling_params, evaluation=evaluation)
    finally:
        _CAPTURED_TURNS.pop(sample.session_id)
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
    assert {branch.reward for branch in samples} in ({0.0}, {1.0}), "One trajectory must have one real outcome"
    for branch in samples:
        assert branch.group_index == sample.group_index and branch.index == sample.index
        assert branch.rollout_id == (sample.rollout_id if sample.rollout_id is not None else sample.index)
        assert not branch.remove_sample, branch.metadata
        assert branch.metadata["agent_exit_code"] == 0, branch.metadata
        assert branch.response_length == len(branch.loss_mask) == len(branch.rollout_log_probs)
        assert sum(branch.loss_mask) > 0, "Agent segment contains no sampled training tokens"
        assert all(math.isfinite(lp) for lp in branch.rollout_log_probs)
    audit = audit_token_records(samples, turns)
    assert audit["model_turns"] >= 2, "The fixture must exercise multiple agent turns"
    assert len(samples) < audit["model_turns"], "Continuous model turns were not merged"
    audit["sampling_params"] = dict(sampling_params)
    audit.update(index=sample.index, group_index=sample.group_index, agent=name, reward=samples[0].reward)
    (run_dir / "token-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return samples


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
            assert math.isfinite(float(result[1])) and float(result[1]) >= 0, result
            assert torch.isfinite(after).all(), "Model parameters became nonfinite"
            if float(result[1]) > 0:
                assert torch.count_nonzero(delta) > 0, "Nonzero gradients did not update model parameters"
            evidence = {
                "rank": dist.get_rank(),
                "rollout_id": rollout_id,
                "step_id": step_id,
                "grad_norm": float(result[1]),
                "sampled_parameters": before.numel(),
                "changed_parameters": torch.count_nonzero(delta).item(),
                "max_parameter_delta": delta.max().item(),
            }
            path = (
                Path(os.environ["SLIME_AGENT_TEST_RUN_DIR"])
                / f"optimizer-rollout-{rollout_id}-rank-{dist.get_rank()}.json"
            )
            path.write_text(json.dumps(evidence, indent=2) + "\n")
            return result
        finally:
            optimizer.step = original_step

    optimizer.step = checked_step
