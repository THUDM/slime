"""Publication visibility belongs to Straw, including plain synchronous readers."""

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest
from straw.errors import CorruptData
from slime.data import transport

NUM_GPUS = 0


@pytest.mark.parametrize("loader", ["plain", "async", "disk_ref"])
def test_result_and_shelve_payloads_recover_at_the_storage_layer(tmp_path, loader, capfd):
    args = SimpleNamespace(rollout_data_transport="straw", rollout_data_dir=str(tmp_path))
    value = {"session_id": "test-result", "status": "shelved", "tokens": [1, 2, 3]}
    transport.pack_rollout_payload({"prefix": True}, args, 1)
    ref = transport.pack_rollout_payload(value, args, 1)
    segment = ref.manifest.manifest.segment
    path = tmp_path / segment.path
    with path.open("r+b") as f:
        f.seek(segment.offset)
        header = f.read(12)
        f.seek(segment.offset)
        f.write(bytes(12))
        f.flush()
        os.fsync(f.fileno())

    def restore():
        with path.open("r+b") as f:
            f.seek(segment.offset)
            f.write(header)
            f.flush()
            os.fsync(f.fileno())

    timer = threading.Timer(0.05, restore)
    timer.start()
    try:
        if loader == "plain":
            result = transport.unpack_rollout_payload(ref)
        elif loader == "async":
            result = asyncio.run(transport.unpack_published_payload(ref))
        else:
            result = ref.load()
        assert result == value
    finally:
        timer.join()
    assert "Straw extent visibility retry" in capfd.readouterr().err


@pytest.mark.parametrize("message", ["Invalid segment header", "Record payload checksum mismatch"])
@pytest.mark.parametrize("legacy_exception", [False, True])
def test_native_failure_is_not_retried_again_by_slime(monkeypatch, message, legacy_exception):
    if legacy_exception:
        monkeypatch.setattr(CorruptData, "add_note", None, raising=False)
    ref = transport.DiskPayloadRef(
        SimpleNamespace(manifest=SimpleNamespace(segment=SimpleNamespace(path="raw/test.pack", offset=123))), "/test"
    )
    calls = []
    failure = CorruptData(message)

    def load(value):
        calls.append(value)
        raise failure

    monkeypatch.setattr(transport, "unpack_rollout_payload", load)
    with pytest.raises(CorruptData) as error:
        asyncio.run(transport.unpack_published_payload(ref))
    assert len(calls) == 1
    assert error.value is failure
    assert message in str(error.value)
    if getattr(error.value, "add_note", None) is not None:
        assert "offset=123" in error.value.__notes__[0]
    else:
        assert "offset=123" in str(error.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
