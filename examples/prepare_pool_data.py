#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
AVALANCHE_ROOT = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche")

# 引入 multidomain_v1 中的数据处理模块来复用他们的 parser
sys.path.append(str(SCRIPT_DIR / "multidomain_v1"))
import prepare_multidomain_v1_data as mv1
import importlib.util

V0_SCRIPT = SCRIPT_DIR / "multidomain_v0" / "prepare_mixed_domain_data.py"
spec = importlib.util.spec_from_file_location("prepare_mixed_domain_data_v0", V0_SCRIPT)
mv0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mv0
spec.loader.exec_module(mv0)

def dump_pool(samples, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(samples)} samples to {out_path}")

def process_dataset(path_str, format_name):
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"⚠️ Source missing: {path}")
        return
    
    domain = mv1.dataset_domain(format_name)
    spec = mv0.SourceSpec(source=path, dataset_format=format_name, ratio=1.0, domain=domain)
    
    # 动态执行转换
    print(f"Processing {format_name} ({domain})...")
    
    samples_iter = mv1.iter_converted_samples(spec.source, spec.dataset_format)
    samples = list(mv0.iter_selected_samples(samples_iter, 0, None))
    
    # Tool call dataset 特别处理（如果需要转换 parser type）
    if domain == "tool":
        samples = mv1._rewrite_parser_type(samples, "qwen3")
        
    out_path = AVALANCHE_ROOT / "data" / "pool" / domain / f"{format_name}_{path.stem}.jsonl"
    dump_pool(samples, out_path)

def process_ifrl(path_str):
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"⚠️ IF-RL Source missing: {path}")
        return
    
    print("Processing IF-RL...")
    out_path = AVALANCHE_ROOT / "data" / "pool" / "ifrl" / f"ifrl_{path.stem}.jsonl"
    
    sys.path.append(str(SCRIPT_DIR / "if_rl"))
    import prepare_ifrl_data as ifrl
    
    samples = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip(): continue
            row = json.loads(line)
            prompt_text = ifrl.normalize_prompt(row.get("prompt", ""))
            
            # Extract metadata exactly as prepare_ifrl_data.py does
            metadata = {
                "record_id": row.get("id", ""),
                "prompt_text": prompt_text,
                "dataset": row.get("dataset", ""),
                "agent_ref": row.get("agent_ref", ""),
            }
            if "kwargs" in row: metadata["kwargs"] = row["kwargs"]
            if "instruction_id_list" in row: metadata["instruction_id_list"] = row["instruction_id_list"]
            
            samples.append({
                "prompt": prompt_text,
                "label": "",
                "metadata": metadata
            })
    dump_pool(samples, out_path)

# ====== Dataset List derived from context ======

DATASETS = [
    # Tool (Train)
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/agent_function_calling_open_dataset/deepnlp_agent_function_call_202510.json", "agent_function_calling_open_dataset"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/agent_function_calling_open_dataset/deepnlp_agent_function_call_202601.json", "agent_function_calling_open_dataset"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apigen_mt_5k/apigen-mt_5k.json", "apigen_mt_5k"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/xlam_function_calling_60k/xlam-function-calling-60k.parquet", "xlam_function_calling_60k"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apibench/huggingface_train.json", "apibench"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apibench/tensorflow_train.json", "apibench"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apibench/torchhub_train.json", "apibench"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00000-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00001-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00002-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00003-of-00004.parquet", "toolbench_v1"),

    # Tool (Eval)
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/bfcl_v3/data/train-00000-of-00001.parquet", "bfcl_v3"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/bfcl_v3_multi_turn_base/data/train-00000-of-00001.parquet", "bfcl_v3_multi_turn_base"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g1_tool-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g1_category-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g1_instruction-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g2_category-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g2_instruction-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g3_instruction-00000-of-00001.parquet", "toolbench_v1_benchmark"),

    # Structured
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/nemotron_structured_outputs/structured_outputs_251027_nano_v3_sdg_json_train.jsonl", "nemotron_structured_outputs"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/ifeval/ifeval_input_data.jsonl", "ifeval"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/jsonschemabench/data/train-00000-of-00001.parquet", "jsonschemabench"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/jsonschemabench/data/test-00000-of-00001.parquet", "jsonschemabench"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/ifbench_test/data/train-00000-of-00001.parquet", "ifbench_test"),

    # Stem
    (f"{AVALANCHE_ROOT}/data/open_data/stem/nemotron_knowledge_mcqa/data/train-00000-of-00004.parquet", "nemotron_knowledge_mcqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet", "ai2_arc"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet", "scienceqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/openbookqa/main/train-00000-of-00001.parquet", "openbookqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/sciq/data/train-00000-of-00001.parquet", "sciq"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/medmcqa/data/train-00000-of-00001.parquet", "medmcqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/mmlu_pro/data/test-00000-of-00001.parquet", "mmlu_pro"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/gpqa/gpqa_main.csv", "gpqa")
]

if __name__ == "__main__":
    print("🚀 Building pool data from open_data...")
    for path, fmt in DATASETS:
        process_dataset(path, fmt)
        
    process_ifrl(f"{AVALANCHE_ROOT}/data/raw_data/Nemotron-Cascade-2-RL-data/IF-RL/train.jsonl")
    print("🎉 Done collecting pool data!")

