"""Download interruption, proxy Range handling, and incomplete model caches."""

import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from ci.oci_cache import download, pull

spec = importlib.util.spec_from_file_location("agent_e2e_test", REPO_ROOT / "tests/test_agent_sunabako_codex_e2e.py")
e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e)
NUM_GPUS = 0


@pytest.fixture
def blob_server(monkeypatch):
    payload = b"cache test\n" * 300000
    state = {"ranges": [], "interrupt": False, "ignore_range": False, "bad_range": False, "omit_length": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range")
            state["ranges"].append(requested)
            offset = int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
            if state["ignore_range"]:
                offset = 0
            self.send_response(206 if offset else 200)
            if offset:
                start = offset + 1 if state["bad_range"] else offset
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            if not state["omit_length"]:
                self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            if state["interrupt"]:
                state["interrupt"] = False
                self.wfile.write(payload[offset : offset + 1024**2])
                self.close_connection = True
            else:
                self.wfile.write(payload[offset:])

        def log_message(self, *_):
            pass

    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/blob", payload, state
        finally:
            server.shutdown()
            thread.join()


def test_interrupted_download_resumes_without_losing_completed_bytes(blob_server, tmp_path, monkeypatch):
    url, payload, state = blob_server
    state["interrupt"] = True
    monkeypatch.setattr("ci.oci_cache.time.sleep", lambda _: None)
    destination = tmp_path / "blob"
    download(lambda: urllib.request.Request(url), destination, hashlib.sha256(payload).hexdigest(), len(payload))
    assert state["ranges"] == [None, f"bytes={1024**2}-"]
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".partial").exists()


def test_proxy_ignoring_range_restarts_instead_of_appending(blob_server, tmp_path):
    url, payload, state = blob_server
    state["ignore_range"] = True
    destination = tmp_path / "blob"
    destination.with_suffix(".partial").write_bytes(payload[:100])
    download(lambda: urllib.request.Request(url), destination, hashlib.sha256(payload).hexdigest(), len(payload))
    assert state["ranges"] == ["bytes=100-"]
    assert destination.read_bytes() == payload


def test_wrong_range_keeps_partial_cache_and_rejects_response(blob_server, tmp_path):
    url, payload, state = blob_server
    state["bad_range"] = True
    destination = tmp_path / "blob"
    partial = destination.with_suffix(".partial")
    partial.write_bytes(payload[:100])
    with pytest.raises(ValueError, match="Content-Range"):
        download(lambda: urllib.request.Request(url), destination, hashlib.sha256(payload).hexdigest(), len(payload))
    assert partial.read_bytes() == payload[:100]
    assert not destination.exists()


def test_cached_download_is_verified_without_network(tmp_path):
    destination = tmp_path / "blob"
    destination.write_bytes(b"cached")
    download(
        lambda: pytest.fail("Cache hit must not access the network"),
        destination,
        hashlib.sha256(b"cached").hexdigest(),
        6,
    )
    destination.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Corrupt"):
        download(
            lambda: pytest.fail("Corrupt cache must be detected"),
            destination,
            hashlib.sha256(b"cached").hexdigest(),
            6,
        )


def test_completed_partial_is_promoted_without_network(tmp_path):
    destination = tmp_path / "blob"
    destination.with_suffix(".partial").write_bytes(b"completed")
    download(
        lambda: pytest.fail("Completed partial must not access the network"),
        destination,
        hashlib.sha256(b"completed").hexdigest(),
        9,
    )
    assert destination.read_bytes() == b"completed"
    assert not destination.with_suffix(".partial").exists()


def test_integrity_mismatch_does_not_publish_cache(blob_server, tmp_path):
    url, payload, _ = blob_server
    destination = tmp_path / "blob"
    with pytest.raises(ValueError, match="mismatch"):
        download(lambda: urllib.request.Request(url), destination, "0" * 64, len(payload))
    assert not destination.exists()
    assert not destination.with_suffix(".partial").exists()


@pytest.mark.parametrize("omit_length", [False, True])
def test_npm_integrity_algorithm_is_supported(blob_server, tmp_path, omit_length):
    url, payload, state = blob_server
    state["omit_length"] = omit_length
    destination = tmp_path / "archive.tgz"
    download(
        lambda: urllib.request.Request(url),
        destination,
        hashlib.sha512(payload).hexdigest(),
        None if omit_length else len(payload),
        algorithm="sha512",
    )
    assert destination.read_bytes() == payload


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("umoci") is None, reason="Requires the OCI unpacker installed by agent CI setup")
def test_cached_oci_layout_unpacks_and_reuses_rootfs(tmp_path, monkeypatch):
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        info = tarfile.TarInfo("marker")
        info.size = 6
        tar.addfile(info, io.BytesIO(b"cached"))
    tar_bytes = tar_stream.getvalue()
    layer = gzip.compress(tar_bytes)
    config = json.dumps(
        {
            "architecture": "amd64",
            "os": "linux",
            "config": {"Env": ["PATH=/bin"], "WorkingDir": "/", "Cmd": ["/bin/sh"]},
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(tar_bytes).hexdigest()]},
        }
    ).encode()

    def descriptor(content, media_type):
        return {
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "mediaType": media_type,
        }

    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": descriptor(config, "application/vnd.oci.image.config.v1+json"),
            "layers": [descriptor(layer, "application/vnd.oci.image.layer.v1.tar+gzip")],
        }
    ).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    blobs = tmp_path / "oci" / digest / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    for content in (manifest, config, layer):
        (blobs / hashlib.sha256(content).hexdigest()).write_bytes(content)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("Cached OCI layout must work offline")
    )
    destination = tmp_path / "bundle"
    image = "docker.io/example/test@sha256:" + digest
    pull(image, destination, tmp_path / "oci")
    assert (destination / "rootfs" / "marker").read_bytes() == b"cached"
    metadata = json.loads((destination / "image.json").read_text())
    assert metadata["manifest_digest"] == "sha256:" + digest
    assert metadata["env"]["PATH"] == "/bin"
    pull(image, destination, tmp_path / "oci")


def test_config_file_does_not_mean_checkpoint_is_complete(tmp_path):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "shard1", "b": "shard2"}}))
    (tmp_path / "shard1").write_bytes(b"weights")
    assert not e2e.checkpoint_ready(tmp_path)
    (tmp_path / "shard2").write_bytes(b"weights")
    assert e2e.checkpoint_ready(tmp_path)


def test_parallel_preparation_serializes_cache_writers(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_TEST_CACHE", str(tmp_path))
    entered = threading.Event()
    waiting = threading.Event()
    release = threading.Event()
    calls = []

    def prepare(cache):
        calls.append(cache)
        entered.set()
        assert release.wait(5)
        return "ready"

    def wait(_):
        waiting.set()
        assert release.wait(5)

    monkeypatch.setattr(e2e, "_prepare_assets", prepare)
    monkeypatch.setattr(e2e.time, "sleep", wait)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(e2e.prepare_assets)
        try:
            assert entered.wait(5)
            second = executor.submit(e2e.prepare_assets)
            assert waiting.wait(5)
            assert calls == [tmp_path]
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == "ready"
    assert calls == [tmp_path, tmp_path]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
