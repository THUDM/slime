#!/usr/bin/env python3
"""
根据 dump 的请求数据测试 SGLang 服务

使用方法:
    python scripts/test_sglang_request.py --url http://localhost:30000/generate
    python scripts/test_sglang_request.py --url http://localhost:30000/generate --dump outputs/sglang_request_dump.json
"""

import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(description="测试 SGLang 请求")
    parser.add_argument("--url", type=str, default="http://localhost:30000/generate",
                        help="SGLang 服务地址")
    # parser.add_argument("--url", type=str, default="http://mingming-slime-dev-1h-b56d82-master-0:15012/generate", #http://mingming-slime-dev-1h-b56d82-master-0:15012
    #                     help="SGLang 服务地址")
    parser.add_argument("--dump", type=str,
                        default="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/sglang_request_dump.json",
                        help="dump 文件路径")
    parser.add_argument("--timeout", type=int, default=120, help="请求超时时间(秒)")
    args = parser.parse_args()

    # 读取 dump 的请求
    print(f"读取 dump 文件: {args.dump}")
    with open(args.dump, "r") as f:
        payload = json.load(f)

    print(f"\n=== 请求信息 ===")
    print(f"URL: {args.url}")
    print(f"Payload keys: {list(payload.keys())}")

    if "text" in payload:
        print(f"text 长度: {len(payload['text'])}")
        print(f"text 内容 (前500字符):\n{payload['text'][:500]}")

    if "input_ids" in payload:
        print(f"input_ids 长度: {len(payload['input_ids'])}")
        print(f"input_ids 前20个: {payload['input_ids'][:20]}")

    if "image_data" in payload:
        print(f"image_data 数量: {len(payload['image_data'])}")
        print(f"image_data[0] 前100字符: {payload['image_data'][0][:100]}...")

    if "sampling_params" in payload:
        print(f"sampling_params: {payload['sampling_params']}")

    # 发送请求
    print(f"\n=== 发送请求 ===")
    try:
        response = requests.post(args.url, json=payload, timeout=args.timeout)
        print(f"响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n=== 响应结果 ===")
            print(f"生成的文本:\n{result.get('text', 'N/A')}")

            if 'meta_info' in result:
                meta = result['meta_info']
                print(f"\nmeta_info keys: {list(meta.keys())}")
                print(f"finish_reason: {meta.get('finish_reason', 'N/A')}")
                print(f"prompt_tokens: {meta.get('prompt_tokens', 'N/A')}")
                print(f"completion_tokens: {meta.get('completion_tokens', 'N/A')}")
        else:
            print(f"错误响应: {response.text}")
    except requests.exceptions.Timeout:
        print(f"请求超时 ({args.timeout}秒)")
    except requests.exceptions.ConnectionError as e:
        print(f"连接失败: {e}")
    except Exception as e:
        print(f"请求失败: {e}")


if __name__ == "__main__":
    main()
