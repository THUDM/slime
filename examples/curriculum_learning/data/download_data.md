# Data Preparation
## Download
Download all four JSONL files to the current directory:

```bash
# You should be under examples/curriculum_learning/data
huggingface-cli download zhangzx369/curriculum-learning-minimal-example \
  --include "data/*.jsonl" \
  --local-dir ../
```
After downloading, the data file structure should be:
```
examples/
└── curriculum_learning/
    └── data/
        ├── aime-2024.jsonl
        ├── dapo-math-17k.jsonl
        ├── IFBench_eval.jsonl
        └── verinstruct.jsonl
```
## Data Details

### Training Data

**DAPO-Math-17k** — 17,398 lines

Math reasoning training data from the [DAPO paper](https://arxiv.org/abs/2503.14476) (ByteDance Seed). Sourced from [`zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k). Each line includes a `data_source` field set to `dapo-math-17k`.

**VerInstruct** — 19,756 lines

Instruction-following training data from [`THU-KEG/VerInstruct`](https://huggingface.co/datasets/THU-KEG/VerInstruct) ([paper](https://arxiv.org/pdf/2506.09942)). The original dataset provides both *hard* (function-verifiable) and *soft* (LLM-judge rubric-based) reward signals. For simplicity, only items with hard constraints are included here.

### Evaluation Data

**AIME 2024** — 32 lines

Math evaluation benchmark from [`zhuzilin/aime-2024`](https://huggingface.co/datasets/zhuzilin/aime-2024).

**IFBench** — 300 lines

Instruction-following evaluation benchmark from [`zyzshishui0627/IFBench`](https://huggingface.co/datasets/zyzshishui0627/IFBench).

