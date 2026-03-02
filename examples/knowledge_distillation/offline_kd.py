import json
import logging
import os

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

KD_LOAD_PATH = os.environ.get("KD_LOAD_PATH")
KD_TOP_K = int(os.environ.get("KD_TOP_K", "8"))


def _load_from_jsonl(load_path, rollout_id, batch_size, num_rollouts_per_prompt):
    load_path = load_path.format(rollout_id=rollout_id)
    assert os.path.exists(load_path), f"KD data file not found: {load_path}"

    samples, metadata = [], None
    with open(load_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("__metadata__"):
                metadata = record
                continue

            resp_len = record["response_length"]
            assert len(record["teacher_log_probs"]) == resp_len, f"Sample {len(samples)}: log_probs length mismatch"

            sample = Sample(
                prompt=record["prompt"],
                tokens=record["tokens"],
                response=record["response"],
                response_length=resp_len,
                teacher_log_probs=record["teacher_log_probs"],
                reward=0.0,
                status=Sample.Status.COMPLETED,
            )

            if "teacher_top_k_ids" in record and "teacher_top_k_logprobs" in record:
                assert (
                    len(record["teacher_top_k_ids"]) == resp_len
                ), f"Sample {len(samples)}: top_k_ids length mismatch"
                assert (
                    len(record["teacher_top_k_logprobs"]) == resp_len
                ), f"Sample {len(samples)}: top_k_logprobs length mismatch"
                sample.train_metadata = {
                    "teacher_top_k_ids": record["teacher_top_k_ids"],
                    "teacher_top_k_logprobs": record["teacher_top_k_logprobs"],
                }

            samples.append(sample)

    # Validate metadata
    assert metadata, f"Missing metadata in {load_path}"
    file_type = metadata.get("distillation_type")
    expected_type = "top_k" if KD_TOP_K > 0 else "sampled_kl"
    assert file_type == expected_type, f"Type mismatch: file={file_type}, expected={expected_type}"
    if file_type == "top_k":
        assert metadata.get("top_k") == KD_TOP_K, f"Top-K mismatch: file={metadata.get('top_k')}, KD_TOP_K={KD_TOP_K}"

    total = batch_size * num_rollouts_per_prompt
    assert len(samples) >= total, f"Not enough samples: got {len(samples)}, need {total}"

    grouped = []
    for i in range(batch_size):
        group = samples[i * num_rollouts_per_prompt : (i + 1) * num_rollouts_per_prompt]
        for j, s in enumerate(group):
            s.group_index, s.index = i, j
        grouped.append(group)

    logger.info(f"Loaded {len(samples)} samples from {load_path} ({file_type}, top_k={metadata.get('top_k')})")
    return grouped


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        return RolloutFnEvalOutput(data={})
    assert KD_LOAD_PATH, "KD_LOAD_PATH must be set for offline KD."

    samples = _load_from_jsonl(KD_LOAD_PATH, rollout_id, args.rollout_batch_size, args.n_samples_per_prompt)

    # Store top-k data for loss function access
    if KD_TOP_K > 0:
        from examples.knowledge_distillation.kd_loss import store_topk_data

        store_topk_data(samples)

    first = samples[0][0]
    logger.info(
        f"Offline KD: prompt={first.prompt[:80]!r}, response={first.response[:80]!r}, len={first.response_length}"
    )

    token_count = sum(s.response_length for g in samples for s in g)
    return RolloutFnTrainOutput(samples=samples, metrics={"kd/token_count": token_count})
