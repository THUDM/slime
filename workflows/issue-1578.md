# Issue #1578 Workflow

1) Issue 链接与摘要
- URL: https://github.com/THUDM/slime/issues/1578
- 现象：HF -> Megatron 转换时，传入模型参数会发生黏连，导致 flag 名异常（如 `--disable-bias-linear...`）。
- 期望：命令行参数应按独立 token 透传，不应被拼接。

2) 根因分析
- `slime/utils/external_utils/command_utils.py::convert_checkpoint()` 拼接命令时，`--save {path_dst}` 与 `extra_args` 之间没有保证空格分隔。
- 当调用方传入不带前导空格的 `extra_args` 时，会与前一个参数黏连。

3) 修改清单
- `slime/utils/external_utils/command_utils.py`
  - 新增 `normalized_extra_args = f" {extra_args.strip()}" if extra_args.strip() else ""`
  - 命令拼接改用 `normalized_extra_args`，保证参数边界。
- `tests/utils/test_command_utils.py`
  - 新增轻量单测，验证 `--save ...` 后的 `extra_args` 有空格分隔。

4) 如何验证
- 本机最小验证（CPU）：
  - `python3.12` 注入最小 fake module 后调用 `convert_checkpoint()`，断言生成命令包含：
    `--save /tmp/slime-test/demo-model_torch_dist --disable-bias-linear --untie-embeddings-and-output-weights`
  - 结果：通过（输出 `ok`）。
- 额外建议（有完整 Python/pytest 环境时）：
  - `PYTHONPATH=. pytest -q tests/utils/test_command_utils.py`

5) 可选 PR 草稿
- 标题：Fix command arg concatenation in convert_checkpoint
- Body：Normalize and prepend whitespace for `extra_args` in `convert_checkpoint()` to avoid merged CLI flags during HF->Megatron conversion.
