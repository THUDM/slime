"""Unit test for StalenessDataSource.

Usage::

    python examples/fully_async/search_agent/test_stale_data_source.py
"""

import asyncio
import sys
import threading
import time
from argparse import Namespace
from unittest import mock

from examples.fully_async.search_agent.stale_data_source import StalenessDataSource


def _make_args(**overrides):
    defaults = dict(
        rollout_global_dataset=False,
        n_samples_per_prompt=2,
        rollout_batch_size=4,
        staleness_threshold=1,
        buffer_filter_path=None,
        rollout_shuffle=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ============================================================
# Test 1: StalenessDataSource 纯逻辑
# ============================================================

def test_get_samples_basic():
    ds = StalenessDataSource(_make_args())
    groups = ds.get_samples(3)
    assert len(groups) == 3
    for g in groups:
        assert len(g) == 2
    print("  PASS test_get_samples_basic")


def test_blocks_when_threshold_exceeded():
    ds = StalenessDataSource(_make_args())
    for _ in range(4):
        ds.get_samples(1)

    result = []

    def take():
        result.append("done")

    t = threading.Thread(target=lambda: (take(), ds.get_samples(1)), daemon=True)
    t.start()
    time.sleep(0.3)
    assert t.is_alive(), "should be blocked"
    assert result == [], "should not have returned"
    ds.reset_staleness()
    t.join(timeout=2.0)
    print("  PASS test_blocks_when_threshold_exceeded")


def test_reset_unblocks():
    ds = StalenessDataSource(_make_args())
    for _ in range(4):
        ds.get_samples(1)

    result = []

    def take():
        result.append(ds.get_samples(1))

    t = threading.Thread(target=take, daemon=True)
    t.start()
    time.sleep(0.3)
    assert t.is_alive()

    ds.reset_staleness()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(result) == 1
    print("  PASS test_reset_unblocks")


def test_counter_resets_after_unblock():
    ds = StalenessDataSource(_make_args())
    for _ in range(4):
        ds.get_samples(1)

    result = []

    def take():
        result.append(ds.get_samples(1))

    t = threading.Thread(target=take, daemon=True)
    t.start()
    time.sleep(0.3)
    ds.reset_staleness()
    t.join(timeout=2.0)
    result.clear()

    for _ in range(4):
        ds.get_samples(1)

    t2 = threading.Thread(target=lambda: result.append(ds.get_samples(1)), daemon=True)
    t2.start()
    time.sleep(0.3)
    assert t2.is_alive(), "should block again after reset + re-consume"

    ds.reset_staleness()
    t2.join(timeout=2.0)
    assert not t2.is_alive()
    print("  PASS test_counter_resets_after_unblock")


def test_add_samples_does_not_affect_staleness():
    ds = StalenessDataSource(_make_args())
    fake_group = ds.get_samples(1)
    ds.add_samples(fake_group)

    for _ in range(3):
        ds.get_samples(1)

    result = []

    def take():
        result.append(ds.get_samples(1))

    t = threading.Thread(target=take, daemon=True)
    t.start()
    time.sleep(0.3)
    assert t.is_alive(), "should block at 5th call (add_samples doesn't count)"

    ds.reset_staleness()
    t.join(timeout=2.0)
    print("  PASS test_add_samples_does_not_affect_staleness")


def test_buffer_drained_first():
    ds = StalenessDataSource(_make_args())
    original = ds.get_samples(1)
    ds.add_samples(original)

    groups = ds.get_samples(1)
    assert groups == original, "buffer should be drained first"
    print("  PASS test_buffer_drained_first")


# ============================================================
# Test 2: Worker 集成（mock generate，无需 SGLang）
# ============================================================

def test_worker_pauses_and_resumes():
    import slime.rollout.fully_async_rollout as _fr

    args = _make_args(
        sglang_server_concurrency=2,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_response_len=128,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=50,
        rollout_stop=None,
        rollout_stop_token_ids=[],
        rollout_skip_special_tokens=False,
        rollout_seed=42,
        hf_checkpoint="mock",
        sglang_dp_size=1,
        ci_test=False,
        use_rollout_routing_replay=False,
        sglang_enable_deterministic_inference=False,
        group_rm=False,
        custom_generate_function_path=None,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
        router_policy=None,
        sglang_model_routers=None,
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
    )

    ds = StalenessDataSource(args)

    async def fake_generate_and_rm_group(args_, group, sampling_params, evaluation=False):
        await asyncio.sleep(0.01)
        return group

    # Patch the worker's generate function with our mock
    with (
        mock.patch.object(_fr, "generate_and_rm_group", fake_generate_and_rm_group),
        mock.patch.object(_fr, "get_rollout_num_engines", lambda _a: 1),
    ):
        worker = _fr.AsyncRolloutWorker(args, ds, concurrency=2)
        worker.start()

        time.sleep(1.0)

        completed = worker.get_completed_groups()
        assert len(completed) > 0, f"expected >0 completed groups, got {len(completed)}"

        time.sleep(0.5)
        completed2 = worker.get_completed_groups()
        assert len(completed2) == 0, (
            f"Worker should be paused, but got {len(completed2)} new groups"
        )

        ds.reset_staleness()
        time.sleep(1.0)
        completed3 = worker.get_completed_groups()
        assert len(completed3) > 0, f"Worker should resume after reset, got {len(completed3)}"

        worker.stop()

    print("  PASS test_worker_pauses_and_resumes")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    print("=== Test 1: StalenessDataSource 纯逻辑 ===")
    tests = [
        test_get_samples_basic,
        test_blocks_when_threshold_exceeded,
        test_reset_unblocks,
        test_counter_resets_after_unblock,
        test_add_samples_does_not_affect_staleness,
        test_buffer_drained_first,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            sys.exit(1)

    print("\n=== Test 2: Worker 集成 ===")
    try:
        test_worker_pauses_and_resumes()
    except Exception as e:
        print(f"  FAIL test_worker_pauses_and_resumes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nAll tests passed.")
