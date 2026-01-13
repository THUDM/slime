#!/usr/bin/env python
import argparse
import os
import random

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="local BF16 path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True, help="dataset path")
    parser.add_argument("--quant-type", type=str, choices=["W4A16", "W8A16"], default="W4A16")
    parser.add_argument("--num-calibration-samples", type=int, default=256, help="sample nums")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quant-group-size", type=int, default=32, help="GPTQ Group Size")
    return parser.parse_args()


def get_calibration_dataset(tokenizer, num_samples, seq_len, local_data_path):

    train_file = os.path.join(local_data_path, "train-00000-of-00001.parquet")

    if not os.path.exists(train_file):
        print(f"can't find the localpath: {train_file}")
        exit(1)

    try:
        ds_raw = load_dataset("parquet", data_files={"train": train_file}, split="train")
    except Exception as e:
        print(f"load Parquet file failed: {e}")
        exit(1)

    text_stream = "".join(ds_raw["text"])
    encoded = tokenizer(text_stream, return_tensors="pt").input_ids[0]

    data_list = []
    for _ in range(num_samples):
        i = random.randint(0, encoded.shape[0] - seq_len - 1)
        chunk = encoded[i : i + seq_len]

        data_list.append({"input_ids": chunk.tolist(), "attention_mask": torch.ones_like(chunk).tolist()})

    ds_hf = Dataset.from_list(data_list)
    return ds_hf


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_hf = get_calibration_dataset(
        tokenizer, args.num_calibration_samples, args.max_sequence_length, args.local_data_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    ignore_patterns = [
        "re:.*lm_head.*",
        "re:.*norm.*",
        "re:.*embed.*",
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
    ]

    recipe = GPTQModifier(
        targets="Linear",
        scheme=args.quant_type,
        ignore=ignore_patterns,
        dampening_frac=args.dampening_frac,
        block_size=32,
    )

    oneshot(
        model=model,
        dataset=ds_hf,  # dataset
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
    )


if __name__ == "__main__":
    main()
