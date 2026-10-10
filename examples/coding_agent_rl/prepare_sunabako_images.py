"""Import the prepared MiMo dataset's OCI images and write a sunabako image map.

Run on each sandbox node with the same absolute --root. Existing bundles are
reused after checking their image metadata. Requires sunabako, skopeo and umoci.
"""

import argparse
import json
from pathlib import Path


def prepare(data: Path, root: Path) -> dict:
    from sunabako.images import pull

    root = root.resolve()
    images = {}
    for line in data.read_text().splitlines():
        if not line.strip():
            continue
        metadata = json.loads(line)["metadata"]
        image = metadata["image"]
        workdir = metadata["workdir"]
        if image in images:
            if images[image]["workdir"] != workdir:
                raise ValueError(f"Conflicting workdirs for image {image}")
            continue
        # The official mapping uses one flat tag per task.
        tag = image.rsplit(":", 1)[-1]
        if not tag or tag in {".", ".."} or "/" in tag:
            raise ValueError(f"Expected a tagged MiMo image, got {image!r}")
        bundle = root / tag
        if not bundle.exists():
            pull(image, bundle)
        info = json.loads((bundle / "image.json").read_text())
        if info["image"] != image or not (bundle / "rootfs").is_dir():
            raise ValueError(f"Existing bundle does not match {image}: {bundle}")
        images[image] = {
            "rootfs": str(bundle / "rootfs"),
            "workdir": workdir,
            "env": info["env"],
        }
    if not images:
        raise ValueError("No task images found in the prepared dataset")
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL produced by prepare_mimo.py")
    parser.add_argument(
        "--root", type=Path, required=True, help="Image bundle directory, identical on every sandbox node"
    )
    parser.add_argument("--output", type=Path, required=True, help="Write the SUNABAKO_IMAGES JSON mapping here")
    args = parser.parse_args()
    images = prepare(args.data, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(images, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "images": len(images)}))


if __name__ == "__main__":
    main()
