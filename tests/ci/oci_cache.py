"""Persistent, resumable Docker Hub downloads for the pinned agent E2E image.

Keep compressed blobs across failed runs, verify their SHA256 digests, and let
umoci unpack the OCI layout. Both downloads and unpacked bundles are cached.
"""

import fcntl
import hashlib
import http.client
import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(request, path, digest, size, attempts=8, algorithm="sha256"):
    """Request factories refresh authentication on each resumed attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if (size is None or path.stat().st_size == size) and _digest(path, algorithm) == digest:
            print(f"[agent-assets] Cached blob {digest[:12]} ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
            return
        raise ValueError(f"Corrupt cached blob: {path}")
    partial = path.with_suffix(".partial")
    if partial.exists() and (size is None or partial.stat().st_size >= size):
        if (size is None or partial.stat().st_size == size) and _digest(partial, algorithm) == digest:
            partial.replace(path)
            return
        if size is not None:
            partial.unlink()
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        total = f"{size / 1024**2:.1f}" if size is not None else "unknown"
        print(f"[agent-assets] Download {digest[:12]}: {offset / 1024**2:.1f}/{total} MiB", flush=True)
        try:
            req = request()
            if offset:
                req.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(req, timeout=45) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(rf"bytes {offset}-\d+/(\d+)", content_range)
                    if not match or (size is not None and int(match[1]) != size):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                    size = int(match[1])
                elif response.status == 200:
                    # Some proxies ignore Range. Never append a full response.
                    offset = 0
                    if size is None and response.headers.get("Content-Length"):
                        size = int(response.headers["Content-Length"])
                else:
                    raise OSError(f"Unexpected HTTP status {response.status}")
                last_report = time.monotonic()
                last_offset = offset
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 20:
                            speed = (offset - last_offset) / (now - last_report) / 1024**2
                            progress = f"{offset / size:.1%}, " if size else ""
                            total = f"{size / 1024**2:.1f}" if size is not None else "unknown"
                            print(
                                f"[agent-assets] {digest[:12]}: {progress}{offset / 1024**2:.1f}/{total} MiB, {speed:.2f} MiB/s",
                                flush=True,
                            )
                            last_report, last_offset = now, offset
            if size is not None and offset != size:
                raise OSError("Download ended before the expected size")
            if _digest(partial, algorithm) != digest:
                partial.unlink()
                raise ValueError(f"{algorithm} mismatch for {digest}")
            partial.replace(path)
            return
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            if attempt + 1 == attempts:
                raise
            # Signed redirect URLs and proxy credentials must stay out of logs.
            print(
                f"[agent-assets] {type(exc).__name__}; retry {attempt + 1}/{attempts - 1}, retaining partial download",
                flush=True,
            )
            time.sleep(min(2**attempt, 20))


def pull(image, destination, cache):
    repository, manifest_digest = image.removeprefix("docker.io/").split("@sha256:")
    if not re.fullmatch(r"[a-z0-9._/-]+", repository) or not re.fullmatch(r"[a-f0-9]{64}", manifest_digest):
        raise ValueError("Expected a Docker Hub image pinned by SHA256")
    destination = Path(destination).resolve()
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / f"{manifest_digest}.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print("[agent-assets] Waiting for another image cache writer", flush=True)
                time.sleep(20)
        if (destination / "image.json").exists():
            info = json.loads((destination / "image.json").read_text())
            if info["manifest_digest"] != "sha256:" + manifest_digest or not (destination / "rootfs").is_dir():
                raise ValueError(f"Image cache does not match {image}: {destination}")
            print(f"[agent-assets] Cached rootfs: {destination}", flush=True)
            return destination
        layout = cache / manifest_digest
        blobs = layout / "blobs" / "sha256"
        blobs.mkdir(parents=True, exist_ok=True)
        manifest_path = blobs / manifest_digest
        if not manifest_path.exists():
            raw = subprocess.check_output(
                ["skopeo", "--command-timeout", "60s", "inspect", "--raw", f"docker://{image}"], timeout=65
            )
            if hashlib.sha256(raw).hexdigest() != manifest_digest:
                raise ValueError("Pinned image manifest digest mismatch")
            temporary = manifest_path.with_suffix(".partial")
            temporary.write_bytes(raw)
            temporary.replace(manifest_path)
        if _digest(manifest_path) != manifest_digest:
            raise ValueError("Cached image manifest digest mismatch")
        manifest = json.loads(manifest_path.read_text())
        descriptors = [manifest["config"], *manifest["layers"]]
        print(
            f"[agent-assets] One task image, {len(manifest['layers'])} layer(s), {sum(d['size'] for d in descriptors) / 1024**3:.2f} GiB compressed",
            flush=True,
        )
        for descriptor in descriptors:
            algorithm, digest = descriptor["digest"].split(":")
            if algorithm != "sha256" or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("Unsupported blob digest")

            def request(blob_digest=descriptor["digest"]):
                auth = (
                    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repository + ":pull"
                )
                with urllib.request.urlopen(auth, timeout=45) as response:
                    token = json.load(response)["token"]
                return urllib.request.Request(
                    f"https://registry-1.docker.io/v2/{repository}/blobs/{blob_digest}",
                    headers={"Authorization": "Bearer " + token},
                )

            download(request, blobs / digest, digest, descriptor["size"])
        (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
        (layout / "index.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "manifests": [
                        {
                            "mediaType": manifest.get("mediaType", "application/vnd.oci.image.manifest.v1+json"),
                            "digest": "sha256:" + manifest_digest,
                            "size": manifest_path.stat().st_size,
                            "annotations": {"org.opencontainers.image.ref.name": "image"},
                        }
                    ],
                }
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[agent-assets] Unpacking cached OCI image to {destination}", flush=True)
        with tempfile.TemporaryDirectory(prefix=".unpack-", dir=destination.parent) as temporary:
            bundle = Path(temporary) / "bundle"
            subprocess.run(
                ["umoci", "unpack", "--rootless", "--image", f"{layout}:image", str(bundle)], check=True, timeout=600
            )
            config = json.loads((bundle / "config.json").read_text())["process"]
            (bundle / "image.json").write_text(
                json.dumps(
                    {
                        "image": image,
                        "manifest_digest": "sha256:" + manifest_digest,
                        "env": dict(v.split("=", 1) for v in config.get("env", [])),
                        "cwd": config.get("cwd", "/"),
                        "command": config.get("args", []),
                    },
                    indent=2,
                )
            )
            bundle.rename(destination)
        return destination
