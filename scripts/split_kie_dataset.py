#!/usr/bin/env python3
"""
KIE 数据集拆分脚本

功能：
1. 将数据集按 7:3 比例拆分为 SFT 和 RL
2. 自动检测答案格式（JSON Dict / JSON 嵌套 List / 纯文本）
3. 根据格式自动添加 metadata.rm_type

用法:
    python scripts/split_kie_dataset.py \
        --input-list data_list.txt \
        --output-dir /path/to/output \
        --sft-ratio 0.7 \
        --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def detect_answer_format(answer: str) -> Tuple[str, str]:
    """
    检测答案格式

    Returns:
        (format_type, rm_type)
        - format_type: "json_dict" | "json_nested_list" | "json_list" | "plain_text" | "empty"
        - rm_type: 推荐的 reward 类型
    """
    if not answer or not answer.strip():
        return "empty", "kie"

    answer = answer.strip()

    try:
        parsed = json.loads(answer)

        if isinstance(parsed, dict):
            # 检查是否有嵌套 list
            has_list = any(isinstance(v, list) and len(v) > 0 for v in parsed.values())
            # 检查是否所有值都为空
            all_empty = all(
                v == "" or v == [] or v is None
                for v in parsed.values()
            )

            if all_empty:
                return "empty", "kie"
            elif has_list:
                # 嵌套 List，用标准版（需要列表匹配）
                return "json_nested_list", "kie"
            else:
                # 简单 Dict，可以用严格版或标准版
                return "json_dict", "kie"

        elif isinstance(parsed, list):
            return "json_list", "kie"
        else:
            return "plain_text", "kie"

    except json.JSONDecodeError:
        # 无法解析为 JSON，是纯文本
        return "plain_text", "kie"


def extract_answer_from_sample(sample: Dict[str, Any]) -> str:
    """从样本中提取答案（gpt 的回复）"""
    conversations = sample.get("conversations", [])
    for conv in conversations:
        if conv.get("from") == "gpt":
            return conv.get("value", "")
    return ""


def convert_to_rl_format(sample: Dict[str, Any], rm_type: str) -> Dict[str, Any]:
    """
    将 SFT 格式转换为 RL 格式

    SFT 格式:
    {
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }

    RL 格式:
    {
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ],
        "metadata": {
            "rm_type": "kie"
        }
    }
    """
    rl_sample = sample.copy()

    # 确保 metadata 存在
    if "metadata" not in rl_sample:
        rl_sample["metadata"] = {}
    elif not isinstance(rl_sample["metadata"], dict):
        rl_sample["metadata"] = {}

    # 添加 rm_type
    rl_sample["metadata"]["rm_type"] = rm_type

    return rl_sample


def process_single_file(
    input_path: str,
    sft_output_path: str,
    rl_output_path: str,
    sft_ratio: float = 0.7,
    seed: int = 42
) -> Dict[str, Any]:
    """处理单个 JSONL 文件"""

    # 读取所有数据
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  警告: 第 {line_num} 行 JSON 解析失败: {e}")

    if not samples:
        print(f"  警告: 文件为空或无有效数据")
        return {
            "total": 0, "sft": 0, "rl": 0,
            "formats": defaultdict(int)
        }

    # 统计格式分布
    format_stats = defaultdict(int)
    sample_formats = []

    for sample in samples:
        answer = extract_answer_from_sample(sample)
        fmt_type, rm_type = detect_answer_format(answer)
        format_stats[fmt_type] += 1
        sample_formats.append((fmt_type, rm_type))

    # 随机打乱
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)

    # 计算拆分点
    split_idx = int(len(samples) * sft_ratio)
    sft_indices = set(indices[:split_idx])
    rl_indices = set(indices[split_idx:])

    # 写入 SFT 数据（保持原格式）
    os.makedirs(os.path.dirname(sft_output_path), exist_ok=True)
    with open(sft_output_path, 'w', encoding='utf-8') as f:
        for idx in indices[:split_idx]:
            f.write(json.dumps(samples[idx], ensure_ascii=False) + '\n')

    # 写入 RL 数据（添加 metadata.rm_type）
    os.makedirs(os.path.dirname(rl_output_path), exist_ok=True)
    rl_format_stats = defaultdict(int)

    with open(rl_output_path, 'w', encoding='utf-8') as f:
        for idx in indices[split_idx:]:
            fmt_type, rm_type = sample_formats[idx]
            rl_sample = convert_to_rl_format(samples[idx], rm_type)
            rl_format_stats[fmt_type] += 1
            f.write(json.dumps(rl_sample, ensure_ascii=False) + '\n')

    return {
        "total": len(samples),
        "sft": len(sft_indices),
        "rl": len(rl_indices),
        "formats": dict(format_stats),
        "rl_formats": dict(rl_format_stats)
    }


def main():
    parser = argparse.ArgumentParser(description="KIE 数据集拆分脚本")
    parser.add_argument(
        "--input-list",
        type=str,
        required=True,
        help="包含数据集路径列表的文件，每行一个路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--sft-ratio",
        type=float,
        default=0.7,
        help="SFT 数据比例 (默认: 0.7)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只分析不写入文件"
    )
    args = parser.parse_args()

    # 读取数据集列表
    with open(args.input_list, 'r', encoding='utf-8') as f:
        dataset_paths = [line.strip() for line in f if line.strip()]

    print("=" * 70)
    print("KIE 数据集拆分工具")
    print("=" * 70)
    print(f"数据集数量: {len(dataset_paths)}")
    print(f"拆分比例: SFT {args.sft_ratio*100:.0f}% / RL {(1-args.sft_ratio)*100:.0f}%")
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 70)

    # 汇总统计
    total_stats = {
        "total": 0, "sft": 0, "rl": 0,
        "formats": defaultdict(int),
        "rl_formats": defaultdict(int)
    }

    results = []

    for input_path in dataset_paths:
        if not os.path.exists(input_path):
            print(f"⚠️  文件不存在，跳过: {input_path}")
            continue

        filename = os.path.basename(input_path)
        sft_output = os.path.join(args.output_dir, "sft", filename)
        rl_output = os.path.join(args.output_dir, "rl", filename)

        print(f"\n📁 处理: {filename}")

        if args.dry_run:
            # 只分析不写入
            samples = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except:
                            pass

            format_stats = defaultdict(int)
            for sample in samples:
                answer = extract_answer_from_sample(sample)
                fmt_type, _ = detect_answer_format(answer)
                format_stats[fmt_type] += 1

            stats = {
                "total": len(samples),
                "sft": int(len(samples) * args.sft_ratio),
                "rl": len(samples) - int(len(samples) * args.sft_ratio),
                "formats": dict(format_stats)
            }
        else:
            stats = process_single_file(
                input_path, sft_output, rl_output,
                args.sft_ratio, args.seed
            )

        # 打印统计
        print(f"   总计: {stats['total']:,} | SFT: {stats['sft']:,} | RL: {stats['rl']:,}")
        print(f"   格式分布: ", end="")
        fmt_parts = []
        for fmt, cnt in sorted(stats['formats'].items()):
            fmt_parts.append(f"{fmt}={cnt}")
        print(", ".join(fmt_parts))

        # 累加统计
        total_stats["total"] += stats["total"]
        total_stats["sft"] += stats["sft"]
        total_stats["rl"] += stats["rl"]
        for fmt, cnt in stats["formats"].items():
            total_stats["formats"][fmt] += cnt

        results.append({"file": filename, **stats})

    # 打印汇总
    print("\n" + "=" * 70)
    print("📊 汇总统计")
    print("=" * 70)
    print(f"总样本数: {total_stats['total']:,}")
    print(f"SFT 样本: {total_stats['sft']:,} ({total_stats['sft']/max(total_stats['total'],1)*100:.1f}%)")
    print(f"RL 样本:  {total_stats['rl']:,} ({total_stats['rl']/max(total_stats['total'],1)*100:.1f}%)")
    print()
    print("格式分布:")
    for fmt, cnt in sorted(total_stats["formats"].items(), key=lambda x: -x[1]):
        pct = cnt / max(total_stats["total"], 1) * 100
        rm_type = "kie"  # 所有格式都用 kie
        print(f"  {fmt:20s}: {cnt:>8,} ({pct:5.1f}%) → rm_type={rm_type}")

    if not args.dry_run:
        print()
        print(f"✅ 输出目录:")
        print(f"   SFT: {os.path.join(args.output_dir, 'sft')}")
        print(f"   RL:  {os.path.join(args.output_dir, 'rl')}")

        # 生成合并后的文件列表
        sft_list_path = os.path.join(args.output_dir, "sft_data_list.txt")
        rl_list_path = os.path.join(args.output_dir, "rl_data_list.txt")

        with open(sft_list_path, 'w') as f:
            for r in results:
                f.write(os.path.join(args.output_dir, "sft", r["file"]) + "\n")

        with open(rl_list_path, 'w') as f:
            for r in results:
                f.write(os.path.join(args.output_dir, "rl", r["file"]) + "\n")

        print()
        print(f"📝 数据列表文件:")
        print(f"   SFT: {sft_list_path}")
        print(f"   RL:  {rl_list_path}")


if __name__ == "__main__":
    main()
