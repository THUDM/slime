"""Tests for the native-resize path in ``slime.utils.processing_utils.process_vision_info``.

For image processors listed in ``_NATIVE_RESIZE_IMAGE_PROCESSORS`` (Nemotron 3.5
VL), each image is resized once to the size the processor itself picks, instead
of going through qwen_vl_utils. The point is that the training side (the HF
processor) and the rollout side (SGLang's own processor) then agree on every
image's patch grid and image-token count -- they use different sizing rules, and
that size is a fixed point of both.

The CPU unit tests below use fake processors and always run. The geo3k test runs
slime's real path over the whole dataset with the real HF processor and SGLang's
own sizing code; it needs the checkpoint, the dataset and sglang, so it is meant
for the slime container and skips elsewhere.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slime.utils import processing_utils
from slime.utils.processing_utils import process_vision_info, uses_native_resize


NUM_GPUS = 0

NATIVE_CLASS_NAME = "NemotronH_Omni_Reasoning_V3ImageProcessor"


def _data_uri(width: int, height: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _messages(*uris: str) -> list[dict]:
    content = [{"type": "image", "image": uri} for uri in uris] + [{"type": "text", "text": "Find x."}]
    return [{"role": "user", "content": content}]


def _fake_processor(class_name: str, target_of):
    """A processor whose image_processor reports target sizes via `target_of(w, h) -> (w, h)`."""
    calls = []

    def __call__(self, images, return_tensors=None):
        calls.append([image.size for image in images])
        return {"imgs_sizes": [tuple(reversed(target_of(*image.size))) for image in images]}

    image_processor = type(class_name, (), {"__call__": __call__})()
    return types.SimpleNamespace(image_processor=image_processor), calls


@pytest.fixture
def qwen_vl_utils_spy(monkeypatch):
    """Stand-in for qwen_vl_utils that records whether the qwen path ran."""
    calls = []

    def fake_process_vision_info(prompt, image_patch_size=None):
        calls.append(image_patch_size)
        return ["qwen-image"], None

    monkeypatch.setitem(
        sys.modules, "qwen_vl_utils", types.SimpleNamespace(process_vision_info=fake_process_vision_info)
    )
    return calls


def test_native_path_resizes_to_the_processor_target(qwen_vl_utils_spy):
    processor, calls = _fake_processor(NATIVE_CLASS_NAME, lambda w, h: (512, 544))

    out = process_vision_info(_messages(_data_uri(250, 258)), processor)

    assert [image.size for image in out["images"]] == [(512, 544)]
    assert out["images"][0].mode == "RGB"
    assert out["videos"] is None
    assert qwen_vl_utils_spy == [], "the native path must not go through qwen_vl_utils"


def test_native_path_asks_once_per_prompt(qwen_vl_utils_spy):
    """All of a prompt's images go to the processor together, so its per-prompt budget applies."""
    processor, calls = _fake_processor(NATIVE_CLASS_NAME, lambda w, h: (2 * w, 2 * h))

    out = process_vision_info(_messages(_data_uri(100, 60, seed=1), _data_uri(40, 80, seed=2)), processor)

    assert calls == [[(100, 60), (40, 80)]]
    assert [image.size for image in out["images"]] == [(200, 120), (80, 160)]


def test_native_path_leaves_target_sized_images_untouched(qwen_vl_utils_spy):
    uri = _data_uri(512, 512, seed=3)
    processor, _ = _fake_processor(NATIVE_CLASS_NAME, lambda w, h: (w, h))

    out = process_vision_info(_messages(uri), processor)

    original = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
    assert np.array_equal(np.asarray(out["images"][0]), np.asarray(original))


def test_other_processors_keep_the_qwen_path(qwen_vl_utils_spy):
    processor, calls = _fake_processor("Qwen2VLImageProcessorFast", lambda w, h: (w, h))
    processor.image_processor.patch_size = 16

    out = process_vision_info(_messages(_data_uri(250, 258)), processor)

    assert out == {"images": ["qwen-image"], "videos": None}
    assert qwen_vl_utils_spy == [16]
    assert calls == [], "a non-native processor must not be asked for target sizes"


def test_uses_native_resize_is_by_class_name():
    assert uses_native_resize(_fake_processor(NATIVE_CLASS_NAME, None)[0])
    assert not uses_native_resize(_fake_processor("Qwen2VLImageProcessorFast", None)[0])
    assert not uses_native_resize(types.SimpleNamespace())


# --------------------------------------------------------------------------- geo3k


GEO3K_DIR = Path(os.environ.get("SLIME_TEST_GEO3K_DIR", "/root/datasets/geo3k_imgurl"))
NEMOTRON_VL_CKPT = Path(
    os.environ.get("SLIME_TEST_NEMOTRON_VL_CKPT", "/root/models/NVIDIA-Nemotron-3.5-Super-EA-09112026")
)


@pytest.mark.integration
def test_geo3k_train_and_rollout_sides_agree_on_every_image():
    """Over all of geo3k, slime's images give the same grid and token count on both sides.

    Training side: the checkpoint's HF image processor, as slime calls it to build
    the training prompt. Rollout side: SGLang's own sizing code
    (compute_budgeted_image_sizes), as its Nemotron VL processor calls it on the
    image slime sends. Without the native resize these disagree on 1570 of 2702
    images; with qwen_vl_utils they agree only because its resize hides the gap.
    """
    try:
        inputs_present = (GEO3K_DIR / "train.parquet").is_file() and (NEMOTRON_VL_CKPT / "config.json").is_file()
    except OSError:  # e.g. an unreadable default path on a machine without such a mount
        inputs_present = False
    if not inputs_present:
        pytest.skip(
            f"needs {GEO3K_DIR} and {NEMOTRON_VL_CKPT} (set SLIME_TEST_GEO3K_DIR / SLIME_TEST_NEMOTRON_VL_CKPT)"
        )
    pq = pytest.importorskip("pyarrow.parquet")
    internvl_utils = pytest.importorskip("sglang.srt.multimodal.internvl_utils")

    from slime.utils.data import _build_messages

    processor = processing_utils.load_processor(str(NEMOTRON_VL_CKPT), trust_remote_code=True)
    assert uses_native_resize(processor), type(getattr(processor, "image_processor", None)).__name__
    image_processor = processor.image_processor

    # The rollout side's parameters, read the way SGLang reads them -- from
    # config.json, not from the HF processor's preprocessor_config.json -- so the
    # two sides really are checked against independent sources
    # (sglang/srt/configs/nano_nemotron_vl.py:122-123,
    #  sglang/srt/multimodal/processors/nano_nemotron_vl.py:137-153).
    hf_config = json.loads((NEMOTRON_VL_CKPT / "config.json").read_text())
    sglang_params = dict(
        patch_size=hf_config["patch_size"],
        downsample_ratio=hf_config["downsample_ratio"],
        min_num_patches=hf_config["vision_config"]["min_num_patches"],
        max_num_patches=hf_config["vision_config"]["max_num_patches"],
    )
    # SGLang's budget is context_length (unset by slime) or 8192, minus the text
    # tokens. geo3k prompts carry one small image, far below it either way.
    sglang_token_budget = 8192

    checked, mismatches = 0, []
    for split in ("train", "test"):
        for row_index, row in enumerate(pq.read_table(GEO3K_DIR / f"{split}.parquet").to_pylist()):
            # Exactly as _train.sh configures it: --input-key problem, --apply-chat-template,
            # --multimodal-keys '{"image": "images"}'.
            messages = _build_messages(row, "problem", True, {"image": "images"})
            images = process_vision_info(messages, processor)["images"]

            hf = image_processor(images=images, return_tensors=None)
            sglang = internvl_utils.compute_budgeted_image_sizes(
                [image.size for image in images], sglang_token_budget, **sglang_params
            )
            for image, (hf_h, hf_w), hf_tokens, (sg_w, sg_h, sg_tokens) in zip(
                images, hf["imgs_sizes"], hf["num_tokens"], sglang, strict=True
            ):
                checked += 1
                # Same grid and token count on both sides, and slime's image already at
                # that size, so neither side resizes it again.
                if (hf_w, hf_h, hf_tokens) != (sg_w, sg_h, sg_tokens) or image.size != (hf_w, hf_h):
                    mismatches.append(
                        f"{split}[{row_index}] slime {image.size}: HF {hf_w}x{hf_h}/{hf_tokens} tok, "
                        f"SGLang {sg_w}x{sg_h}/{sg_tokens} tok"
                    )

    assert checked == 2702, f"geo3k_imgurl should have 2702 images (2101 train + 601 test), saw {checked}"
    assert not mismatches, f"{len(mismatches)}/{checked} images disagree, e.g.\n" + "\n".join(mismatches[:10])
