"""Import the prepared MiMo dataset's OCI images and write a sunabako image map.

Use --format oci to download compressed layers once onto shared storage.
Directory mode imports on each sandbox node using the same absolute --root.
Existing bundles are reused after checking their image metadata.
"""

import argparse
import fcntl
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def prepare(data: Path, root: Path, workers: int = 1, image_format: str = "directory") -> dict:
    from sunabako.images import pull

    if image_format not in {"directory", "oci"}:
        raise ValueError(f"Unknown image format: {image_format}")

    root = root.resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = {}
    tags = {}
    for line in data.read_text().splitlines():
        if not line.strip():
            continue
        metadata = json.loads(line)["metadata"]
        image = metadata["image"]
        workdir = metadata["workdir"]
        if image in tasks:
            if tasks[image][1] != workdir:
                raise ValueError(f"Conflicting workdirs for image {image}")
            continue
        # The official mapping uses one flat tag per task.
        tag = image.rsplit(":", 1)[-1]
        if not tag or tag in {".", ".."} or "/" in tag:
            raise ValueError(f"Expected a tagged MiMo image, got {image!r}")
        if tag in tags:
            raise ValueError(f"Images {tags[tag]} and {image} have the same bundle tag {tag}")
        tags[tag] = image
        tasks[image] = (tag, workdir)
    if not tasks:
        raise ValueError("No task images found in the prepared dataset")
    root.mkdir(parents=True, exist_ok=True)

    def import_image(image):
        tag, workdir = tasks[image]
        bundle = root / tag
        # Concurrent preparation commands can share a cache without racing the
        # final bundle rename. sunabako itself publishes completed bundles only.
        with (root / f".{tag}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if image_format == "oci":
                from sunabako.images import download

                download(image, bundle)
            elif not bundle.exists():
                pull(image, bundle)
            info = json.loads((bundle / "image.json").read_text())
            if info["image"] != image or (image_format == "directory" and not (bundle / "rootfs").is_dir()):
                raise ValueError(f"Existing bundle does not match {image}: {bundle}")
        return {
            "rootfs": str(bundle if image_format == "oci" else bundle / "rootfs"),
            "workdir": workdir,
            "env": info["env"],
        }

    images = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(import_image, image): image for image in tasks}
        try:
            for future in as_completed(pending):
                image = pending[future]
                images[image] = future.result()
                print(f"Prepared {len(images)}/{len(tasks)}: {image}", flush=True)
        except BaseException:
            # Finish the active imports, retaining their completed bundles, but
            # do not download the rest of a large dataset after a failed import.
            for future in pending:
                future.cancel()
            raise
    return {image: images[image] for image in tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL produced by prepare_mimo.py")
    parser.add_argument(
        "--root", type=Path, required=True, help="Image bundle directory, identical on every sandbox node"
    )
    parser.add_argument("--output", type=Path, required=True, help="Write the SUNABAKO_IMAGES JSON mapping here")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent image imports")
    parser.add_argument(
        "--format",
        choices=("directory", "oci"),
        default="directory",
        help="oci keeps compressed layers on shared storage; sunabako caches extracted rootfs on local disk",
    )
    args = parser.parse_args()
    images = prepare(args.data, args.root, args.workers, args.format)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(images, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "images": len(images)}))


if __name__ == "__main__":
    main()
