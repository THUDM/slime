import base64
import json
from io import BytesIO

from datasets import load_dataset
from PIL import Image

# -------------------------
# 1. Load dataset
# -------------------------
dataset = load_dataset("parquet", data_files="/root/datasets/opencua/train-00000-of-00162.parquet")["train"]

# -------------------------
# 2. Helper functions
# -------------------------
NEW_EASYR1_PREFIX = """You are an expert UI element locator.

Given a GUI image and a user's element description, identify the specified element
and output its location as a normalized coordinate.

The coordinate system is defined as:
- x and y are normalized to the range [0, 1]
- (0, 0) corresponds to the top-left corner of the image
- (1, 1) corresponds to the bottom-right corner of the image

For elements with area, return the center point.

Output the final answer in \\boxed{(x,y)}.
Do not include any explanation outside the box.
"""


def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    while len(img_base64) % 4 != 0:
        img_base64 += "="
    return f"data:image/png;base64,{img_base64}"


# -------------------------
# 3. Processing function (batched)
# -------------------------
def process_batch(batch):
    new_easyr1_prompts = []
    relative_answers = []
    relative_bboxes = []
    images_base64 = []

    for i in range(len(batch["easyr1_prompt"])):
        ex = {k: batch[k][i] for k in batch.keys()}

        # -------- easyr1_prompt --------
        prompt = ex["easyr1_prompt"]
        if "<image>" in prompt:
            _, after = prompt.split("<image>", 1)
            prompt = NEW_EASYR1_PREFIX + "\n\n<image>" + after
        new_easyr1_prompts.append(prompt)

        # -------- relative answer --------
        x_px, y_px = ex["click_point"]
        w, h = ex["image_width"], ex["image_height"]
        rx = x_px / w
        ry = y_px / h
        relative_answers.append(f"\\boxed{{({rx:.4f},{ry:.4f})}}")

        # -------- relative bbox --------
        x1, y1, x2, y2 = ex["bbox"]
        rel_bbox = [x1 / w, y1 / h, x2 / w, y2 / h]
        relative_bboxes.append(json.dumps(rel_bbox))

        # -------- images -> base64 list --------
        img_entry = ex["images"]  #
        if isinstance(img_entry, list):
            base64_list = []
            for im in img_entry:
                if isinstance(im, Image.Image):
                    base64_list.append(pil_to_base64(im))
                elif isinstance(im, str):
                    base64_list.append(im)
                else:
                    raise ValueError(f"Unsupported image type in list: {type(im)}")
            images_base64.append(base64_list)
        elif isinstance(img_entry, Image.Image):
            images_base64.append([pil_to_base64(img_entry)])
        else:
            raise ValueError(f"Unsupported image type: {type(img_entry)}")

    return {
        "easyr1_prompt": new_easyr1_prompts,
        "relative_answer": relative_answers,
        "relative_bbox": relative_bboxes,
        "images_base64": images_base64,
    }


# -------------------------
# 4. Map dataset
# -------------------------

# if dataset is too large, you can select a small subset to test
dataset = dataset.select(range(500))


dataset = dataset.map(
    process_batch,
    batched=True,
    batch_size=16,
    num_proc=1,
)


# -------------------------
# 6. Save
# -------------------------
print(dataset[0])
dataset.to_parquet("/root/datasets/opencua/train_relative_base64.parquet")
print("Saved train_relative_base64.parquet")
