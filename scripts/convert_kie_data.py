"""
将 KIE 数据集转换为 SLIME 训练格式

支持的输入格式：
- conversations 格式的 JSONL 文件

输出格式：
- SLIME 需要的 parquet/jsonl 格式
"""
import json
import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def convert_conversation_to_slime(data: dict) -> dict | None:
    """
    将 conversation 格式转换为 SLIME 格式

    输入格式:
    {
        "id": "...",
        "image": "/path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\n问题..."},
            {"from": "gpt", "value": "{\"key\": \"value\"}"}
        ],
        "metadata": {
            "rm_type": "kie"
        }
    }

    输出格式:
    {
        "problem": "<image>\n问题...",
        "answer": "{\"key\": \"value\"}",
        "images": ["/path/to/image.jpg"],
        "rm_type": "kie"
    }
    """
    conversations = data.get("conversations", [])
    if len(conversations) < 2:
        return None

    # 提取 human 和 gpt 的内容
    human_msg = None
    gpt_msg = None

    for conv in conversations:
        if conv.get("from") == "human":
            human_msg = conv.get("value", "")
        elif conv.get("from") == "gpt":
            gpt_msg = conv.get("value", "")

    if not human_msg or not gpt_msg:
        return None

    # 获取图像路径
    image_path = data.get("image", "")
    images = [image_path] if image_path else []

    # 构建 SLIME 格式
    result = {
        "problem": human_msg,
        "answer": gpt_msg,
        "images": images,
    }

    # 保留一些有用的元数据
    if "id" in data:
        result["id"] = data["id"]
    if "type" in data:
        result["task_type"] = data["type"]
    if "image_type" in data:
        result["image_type"] = data["image_type"]
    if "language" in data:
        result["language"] = data["language"]

    # 保留 metadata 中的 rm_type（用于指定 reward 函数）
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict) and "rm_type" in metadata:
        result["rm_type"] = metadata["rm_type"]

    return result


def convert_file(input_path: str, output_path: str, output_format: str = "parquet"):
    """
    转换单个文件

    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出文件路径
        output_format: 输出格式 ("parquet" 或 "jsonl")
    """
    print(f"Converting: {input_path}")

    converted_data = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                converted = convert_conversation_to_slime(data)
                if converted:
                    converted_data.append(converted)
                else:
                    skipped += 1
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                skipped += 1
                continue

    print(f"Converted: {len(converted_data)}, Skipped: {skipped}")

    # 保存
    if output_format == "parquet":
        df = pd.DataFrame(converted_data)
        df.to_parquet(output_path, index=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")
    return len(converted_data)


def convert_multiple_files(input_paths: list[str], output_path: str, output_format: str = "parquet"):
    """
    合并转换多个文件

    Args:
        input_paths: 输入文件路径列表
        output_path: 输出文件路径
        output_format: 输出格式
    """
    all_data = []

    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {Path(input_path).name}"):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    converted = convert_conversation_to_slime(data)
                    if converted:
                        # 添加来源信息
                        converted["source_file"] = Path(input_path).name
                        all_data.append(converted)
                except json.JSONDecodeError:
                    continue

    print(f"Total converted: {len(all_data)}")

    # 保存
    if output_format == "parquet":
        df = pd.DataFrame(all_data)
        df.to_parquet(output_path, index=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")
    return len(all_data)


def validate_converted_data(path: str, num_samples: int = 5):
    """验证转换后的数据"""
    print(f"\n=== Validating: {path} ===")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        data = df.to_dict("records")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

    print(f"Total samples: {len(data)}")
    print(f"Columns: {list(data[0].keys()) if data else 'N/A'}")

    # 检查必要字段
    required_fields = ["problem", "answer", "images"]
    for field in required_fields:
        has_field = all(field in item for item in data)
        print(f"  - {field}: {'✓' if has_field else '✗'}")

    # 显示样例
    print(f"\n=== Sample Data (first {num_samples}) ===")
    for i, item in enumerate(data[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Problem: {item['problem'][:100]}...")
        print(f"Answer: {item['answer'][:100]}...")
        print(f"Images: {item['images']}")

    # 检查图片是否存在
    print(f"\n=== Image Path Check ===")
    missing_images = 0
    for item in data[:100]:  # 只检查前100个
        for img_path in item.get("images", []):
            if img_path and not os.path.exists(img_path):
                missing_images += 1
                if missing_images <= 3:
                    print(f"  Missing: {img_path}")

    if missing_images > 0:
        print(f"  Total missing (in first 100): {missing_images}")
    else:
        print("  All images exist ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KIE dataset to SLIME format")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input JSONL file(s)")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--format", "-f", choices=["parquet", "jsonl"], default="parquet", help="Output format")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate after conversion")

    args = parser.parse_args()

    if len(args.input) == 1:
        convert_file(args.input[0], args.output, args.format)
    else:
        convert_multiple_files(args.input, args.output, args.format)

    if args.validate:
        validate_converted_data(args.output)
