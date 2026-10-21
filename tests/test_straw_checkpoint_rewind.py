"""Train three async steps, rewind in place, and replay debug archives.

Uses four GPUs: one trains Qwen2.5-0.5B and three serve rollouts. Checkpoint
IDs are zero based: save 0/1/2, load 1 in the same directory, and train 2 again.
The rollout hook checks restored state before any worker can advance it.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from shlex import quote

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def load_source_checkpoint(directory, step):
    from straw.protocol import RecordSetRef

    from slime.data.transport import DiskPayloadRef

    index = json.loads((Path(directory) / "rollout" / f"queue_state_{step}.json").read_text())
    return DiskPayloadRef(RecordSetRef.from_dict(index["manifest"]), index["root"]).load()


def checked_rollout(args, rollout_id, data_source, evaluation=False):
    """Observe the real load boundary, then run the production rollout function."""
    import ray

    from slime.data.transport import DiskPayloadRef, rollout_store
    from slime.rollout.fully_async_rollout import generate_rollout_fully_async

    if data_source.restore_plan.mode == "rewind" and rollout_id == args.start_rollout_id:
        assert rollout_id == args.ckpt_step + 1
        assert data_source.restored_source.queue_reset
        assert not data_source.consumers, "Check the cursor before generation starts"
        saved = load_source_checkpoint(args.load, args.ckpt_step)
        expected = saved["reader"].load()["pending"].load()
        ref = ray.get(data_source.controller.pending_snapshot.remote())
        try:
            actual = DiskPayloadRef(ref, args.rollout_data_dir).load()
        finally:
            store, _, lock = rollout_store(args)
            with lock:
                store.release_publications([ref])
        # Check all cursor fields, not just offset (which wraps at epoch end).
        assert actual["producer_cursor"] == expected["producer_cursor"]
        assert not actual["inflight"] and not expected["inflight"]
        from slime.data.transport import load_rollout_samples
        from slime.rollout.base_types import iter_samples
        from slime.utils.types import Sample

        expected_ids = {
            sample.index for task in expected["tasks"] for sample in iter_samples(task["input_ref"].load())
        }
        ready = saved["consumers"]["fully_async"]["scheduler"]["ready"]
        for group, _ in ready:
            if group is not None:
                expected_ids.update(sample.index for sample in iter_samples(load_rollout_samples([group])[0]))
        samples = [sample for task in actual["tasks"] for sample in iter_samples(task["input_ref"].load())]
        assert {sample.index for sample in samples} == expected_ids
        assert all(
            sample.status == Sample.Status.PENDING and not sample.tokens and sample.reward is None
            for sample in samples
        )
        assert not data_source._restored_consumers
        assert data_source.data_config["queue_id"] == saved["configuration"]["queue_id"]
        assert not (Path(args.save) / "rollout" / f"committed_{rollout_id}.json").exists()
        report = {
            "expected_cursor": expected["producer_cursor"],
            "restored_cursor": actual["producer_cursor"],
            "pending_groups": len(actual["tasks"]),
            "regenerated_ready_groups": len(ready),
        }
        Path(args.save).mkdir(parents=True, exist_ok=True)
        Path(args.save, "restored_source.json").write_text(json.dumps(report, indent=2))
    if data_source.restore_plan.mode == "empty" and rollout_id == args.start_rollout_id:
        import torch

        assert not data_source.consumers
        ref = ray.get(data_source.controller.pending_snapshot.remote())
        try:
            actual = DiskPayloadRef(ref, args.rollout_data_dir).load()
        finally:
            store, _, lock = rollout_store(args)
            with lock:
                store.release_publications([ref])
        expected = torch.load(data_source.restore_plan.dataset_cursor, weights_only=False)
        assert actual["producer_cursor"] == expected
        assert actual["tasks"] == [] and not actual["inflight"]
        assert not getattr(data_source, "_restored_consumers", {})
        Path(args.save, "restored_source.json").write_text(json.dumps(actual, indent=2))
    return generate_rollout_fully_async(args, rollout_id, data_source, evaluation=evaluation)


def sample_values(samples):
    """Compare archived training input values."""
    from slime.data.tensor import materialize_tensor_refs

    keys = (
        "index",
        "group_index",
        "prompt",
        "tokens",
        "response_length",
        "reward",
        "status",
        "rollout_log_probs",
        "loss_mask",
    )
    values = []
    for sample in sorted(samples, key=lambda sample: sample.index):
        row = materialize_tensor_refs({key: getattr(sample, key) for key in keys})
        values.append({key: value.tolist() if hasattr(value, "tolist") else value for key, value in row.items()})
    return values


def execute(*, model_path=None, prompt_data=None, work_dir=None, num_gpus_per_node=NUM_GPUS):
    import torch

    from slime.data.archive import RolloutArchive

    model_path = model_path or f"/root/models/{MODEL_NAME}"
    prompt_data = prompt_data or "/root/datasets/gsm8k/train.parquet"
    # work_dir must be shared when the four GPUs live on different hosts.
    with tempfile.TemporaryDirectory(prefix="slime_straw_rewind_", dir=work_dir) as directory:
        root = Path(directory)
        common = (
            f"--hf-checkpoint {quote(str(model_path))} --ref-load {quote(str(model_path))} "
            f"--prompt-data {quote(str(prompt_data))} "
            "--input-key messages --label-key label --apply-chat-template --rollout-shuffle --rm-type math "
            "--rollout-data-transport straw --rollout-queue-online-gc "
            "--rollout-function-path test_straw_checkpoint_rewind.checked_rollout "
            "--rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size 8 "
            "--rollout-max-response-len 128 --rollout-temperature 0.8 "
            "--pg-loss-type reinforce --advantage-estimator grpo --disable-grpo-std-normalization "
            # Short responses can all receive zero reward. Entropy makes the
            # positive gradient check independent of sampled math accuracy.
            "--calculate-per-token-loss --entropy-coef 0.01 --kl-coef 0 "
            "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 "
            "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
            "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size --max-tokens-per-gpu 2048 --log-probs-chunk-size 128 "
            "--rollout-num-gpus-per-engine 1 --sglang-server-concurrency 8 "
            "--sglang-mem-fraction-static 0.4 --sglang-cuda-graph-max-bs 8 "
            "--attention-dropout 0 --hidden-dropout 0 --attention-backend flash "
            "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
            f"--actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node {num_gpus_per_node} "
            "--rollout-num-gpus 3 --save-interval 1 "
        )
        parent = root / "run"
        child = parent
        outputs = root / "outputs"
        legacy = root / "legacy-model"
        phases = [
            ("parent", parent, f"--num-rollout 3 --rollout-data-dir {quote(str(root / 'pool'))}", [0, 1, 2]),
            # No queue lifecycle flags: use normal model checkpoint arguments.
            ("rewind", child, f"--num-rollout 3 --load {quote(str(parent))} --ckpt-step 1", [2]),
            # Select the same boundary again through an explicit marker.
            ("repeat", child, f"--num-rollout 3 --load {quote(str(parent / 'rollout/committed_1.json'))}", [2]),
            ("tracker", child, f"--num-rollout 3 --load {quote(str(child))}", [2]),
            # Extending num-rollout changes Megatron's default LR/WD horizon;
            # keep the saved optimizer schedule while extending the run.
            (
                "resume",
                child,
                f"--num-rollout 4 --load {quote(str(child))} --use-checkpoint-opt-param-scheduler",
                [3],
            ),
            ("empty", root / "empty", f"--num-rollout 3 --load {quote(str(legacy))} --ckpt-step 1", [2]),
            (
                "replay_straw",
                root / "replay_straw",
                f"--num-rollout 1 --load-debug-rollout-data {quote(str(outputs / 'parent/debug_2.straw.json'))}",
                [0],
            ),
            (
                "replay_pt",
                root / "replay_pt",
                f"--num-rollout 1 --load-debug-rollout-data {quote(str(root / 'legacy.pt'))}",
                [0],
            ),
        ]
        checkpoint_hash = None
        for name, destination, options, steps in phases:
            output = outputs / name
            output.mkdir(parents=True)
            if name == "tracker":
                # Edit the logical tracker without passing --ckpt-step.
                (child / "latest_checkpointed_iteration.txt").write_text("1")
            U.execute_train(
                train_args=(
                    f"{common} {options} --save {quote(str(destination))} "
                    f"--save-debug-rollout-data {quote(str(output / 'debug_{rollout_id}.straw.json'))} "
                    f"--ci-save-grad-norm {quote(str(output / 'grad_{rollout_id}_{step_id}.pt'))} "
                ),
                num_gpus_per_node=num_gpus_per_node,
                megatron_model_type=MODEL_TYPE,
                extra_env_vars={
                    "PYTHONPATH": f"{Path(__file__).resolve().parent}:{U.repo_base_dir}:/root/Megatron-LM/"
                },
            )
            physical = destination
            assert not (destination / "branches").exists()
            assert not (destination / "rollout/current.json").exists()
            gradients = sorted(output.glob("grad_*.pt"))
            assert len(gradients) == len(steps)
            norms = []
            for path in gradients:
                norm = torch.as_tensor(torch.load(path, weights_only=False))
                assert torch.isfinite(norm).all() and (norm > 0).all(), path
                norms.append(norm.item())
            for step in steps:
                assert (physical / f"iter_{step:07d}").is_dir()
                if name not in {"replay_straw", "replay_pt"}:
                    marker = json.loads((physical / "rollout" / f"committed_{step}.json").read_text())
                    assert marker["rollout_id"] == step and marker["resumable"]
            if name == "parent":
                assert (parent / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
                checkpoint_hash = hashlib.sha256((parent / "rollout/committed_1.json").read_bytes()).hexdigest()
                # Present a real model checkpoint with only the legacy dataset
                # cursor: missing queue state must select an empty namespace.
                legacy.mkdir()
                (legacy / "iter_0000001").symlink_to(parent / "iter_0000001", target_is_directory=True)
                (legacy / "latest_checkpointed_iteration.txt").write_text("1")
                (legacy / "rollout").mkdir()
                saved = load_source_checkpoint(parent, 1)
                cursor = saved["reader"].load()["pending"].load()["producer_cursor"]
                torch.save(cursor, legacy / "rollout/global_dataset_state_dict_1.pt")
                with RolloutArchive(output / "debug_2.straw.json") as archive:
                    sample_key, task_key = archive.keys()[0]
                    assert len(archive.load_samples(sample_key=sample_key)) == 1
                    assert len(archive.load_samples(task_key=task_key)) == 4
                    archive.export_pt(root / "legacy.pt")
            elif name in {"rewind", "repeat", "tracker", "resume"}:
                assert physical == parent
                report = json.loads((physical / "restored_source.json").read_text())
                assert report["expected_cursor"] == report["restored_cursor"]
                assert report["regenerated_ready_groups"] >= 2
                assert (
                    hashlib.sha256((parent / "rollout/committed_1.json").read_bytes()).hexdigest() == checkpoint_hash
                )
                print(f"{name}: regenerated checkpoint inputs: {report}", flush=True)
            elif name == "empty":
                report = json.loads((physical / "restored_source.json").read_text())
                assert report["tasks"] == []
                assert report["producer_cursor"] == torch.load(
                    legacy / "rollout/global_dataset_state_dict_1.pt", weights_only=False
                )
            else:
                with RolloutArchive(outputs / "parent/debug_2.straw.json") as old, RolloutArchive(
                    output / "debug_0.straw.json"
                ) as new:
                    assert sample_values(old.load_samples()) == sample_values(new.load_samples())
            print(f"{name}: steps {steps}, gradient norms {norms}, archive checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--prompt-data")
    parser.add_argument("--work-dir", help="Shared temporary directory for multi-host runs")
    parser.add_argument("--num-gpus-per-node", type=int, default=NUM_GPUS)
    options = parser.parse_args()
    if not (options.model_path and options.prompt_data):
        prepare()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    execute(**vars(options))
