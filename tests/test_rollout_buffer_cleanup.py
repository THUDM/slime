import pytest
from fastapi.testclient import TestClient

from slime_plugins.rollout_buffer import buffer as module

NUM_GPUS = 0


def test_http_reads_release_consumed_group_timestamps(monkeypatch):
    rollout_buffer = module.RolloutBuffer(group_size=1)
    monkeypatch.setattr(module, "buffer", rollout_buffer)

    with TestClient(module.app) as client:
        for index in range(32):
            item = {"instance_id": str(index), "reward": 1}
            written = client.post("/buffer/write", json=item)
            assert written.status_code == 200
            consumed = client.post("/get_rollout_data", json={})
            assert consumed.status_code == 200
            assert consumed.json()["data"]["data"] == [item]

    assert rollout_buffer.total_written == rollout_buffer.total_read == 32
    assert rollout_buffer.buffer.data == {}
    assert rollout_buffer.buffer.temp_data == {}
    assert rollout_buffer.buffer.group_timestamps == {}


def test_read_preserves_timestamp_for_incomplete_group():
    rollout_buffer = module.RolloutBuffer(group_size=2)
    complete = {"instance_id": "complete", "reward": 1}
    waiting = {"instance_id": "waiting", "reward": 0}
    rollout_buffer.write(complete)
    rollout_buffer.write(waiting)
    rollout_buffer.write(complete)
    waiting_timestamp = rollout_buffer.buffer.group_timestamps["waiting"]

    assert rollout_buffer.read()["data"] == [complete, complete]
    assert rollout_buffer.buffer.data == {"waiting": [waiting]}
    assert rollout_buffer.buffer.group_timestamps == {"waiting": waiting_timestamp}


def test_consumed_instance_id_can_be_used_again():
    rollout_buffer = module.RolloutBuffer(group_size=1)
    item = {"instance_id": "reused", "reward": 1}

    for _ in range(2):
        rollout_buffer.write(item)
        assert set(rollout_buffer.buffer.group_timestamps) == {"reused"}
        assert rollout_buffer.read()["data"] == [item]
        assert rollout_buffer.buffer.group_timestamps == {}


def test_transform_failure_preserves_that_groups_data_and_timestamp():
    def fail_transform(group, task_type):
        raise ValueError("conversion failed")

    rollout_buffer = module.RolloutBuffer(group_size=1, transform_group_func=fail_transform)
    item = {"instance_id": "retry", "reward": 1}
    rollout_buffer.write(item)
    timestamp = rollout_buffer.buffer.group_timestamps["retry"]

    with pytest.raises(ValueError, match="conversion failed"):
        rollout_buffer.read()

    assert rollout_buffer.buffer.data == {"retry": [item]}
    assert rollout_buffer.buffer.group_timestamps == {"retry": timestamp}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
