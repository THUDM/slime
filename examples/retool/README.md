# Retool: from SFT to RL

This example demonstrates how to use the retool functionality for tool-enabled language model generation.

## Overview

The retool example provides:
- Safe Python code execution in a sandbox environment
- Tool registry for managing available tools
- Integration with language model generation
- Reward calculation for tool usage

## Files

- `generate_with_retool.py`: Main generation function with tool support
- `tool_sandbox.py`: Tool execution and safety management
- `sft_data_processing.py`: Process SFT dataset

## Usage

1. Setup and download datasets:
```bash
cd slime
pip install -e .
# For SFT part, you can use later model to RL directly and skip SFT. 
hf download --repo-type dataset JoeYing/ReTool-SFT  --local-dir /root/JoeYing/ReTool-SFT
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen/Qwen3-4B-Instruct-2507

# For RL part
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /root/aime-2024
# download our SFT model if you want to skip SFT
hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir /root/font-info/qwen3-4b-sft
```

2. Create torch dict
For SFT 
```bash
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen/Qwen3-4B-Instruct-2507 \
    --rotary-base 5000000 \
    --save /root/Qwen/Qwen3-4B-Instruct-2507_torch_dist
```

Or RL only
```bash
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/font-info/qwen3-4b-sft \
    --rotary-base 5000000 \
    --save /root/font-info/qwen3-4b-sft_torch_dist

```

3. SFT:
```bash
python examples/retool/sft_data_processing.py
bash examples/retool/retool_qwen3_4b_sft.sh
```

4. RL:
```bash
bash examples/retool/retool_qwen3_4b_rl.sh
```

5. Use in your training scripts by importing the generate function:
```python
from generate_with_retool import generate, reward_func
```

## Tool Format

The system uses the following tool format:

```
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute."}}, "required": ["code"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

## Safety Features

- Code execution in isolated sandbox
- Memory and time limits
- Dangerous operation detection
- Allowed module restrictions 