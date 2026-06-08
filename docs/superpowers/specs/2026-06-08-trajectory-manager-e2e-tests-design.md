# TrajectoryManager 端到端测试脚本 — 设计文档

日期：2026-06-08
状态：已批准设计，待实现

## 目标

写一个独立的端到端测试脚本，通过 `TrajectoryManager` 的两个公共接口
`append_turn` 和 `get_trajectory`，从**数据结构角度**全面覆盖各种分叉情形：
prompt 分叉、assistant 分叉、token-ID 分叉，以及它们的组合。测试数据要
**方便人类阅读**（语义化 token ID + 反查表），运行时既做严格 assertion，
又能打印可读的 tree / 线性化结果供人眼审查。

## 背景

`TrajectoryManager`（`slime/agent/trajectory_manager.py`）维护一棵 per-sid 的
逐 message 路由树，并在 `get_trajectory` 时把每个 leaf 链线性化成 slime
`Sample`。它有两个正交的层：

- **路由树层**（`append_turn`）：按 `(role, node_match_key)` 匹配，**只看
  message 身份**，与 token ID 无关。相同 message 前缀总落在同一路径。
- **线性化层**（`get_trajectory`）：按 **token-ID 前缀**匹配累积 token，按
  漂移位置走 case A / B1 / B2 路由（fork / replace），并做 cross-leaf dedup
  和 reward 均分。

现有 `tests/test_agent/test_trajectory_manager.py` 已有 28 个测试覆盖这些机制，
但用 `ord(char)` 映射 token、断言不易一眼看懂分叉点，且「两层同时发生」的
组合格子覆盖较薄。本脚本是**独立新增**，与现有文件并存、互不依赖。

## 文件

`tests/test_agent/test_trajectory_manager_e2e.py`

不改动 `trajectory_manager.py`；不依赖 sglang / 网络 / 真实 tokenizer；不碰
现有 `test_trajectory_manager.py`。

## §1 基础设施

### 语义化 token 词表

token ID 用带语义段的小整数，看 assertion 一眼知道分叉在哪。每个 message
渲染成 `[START, ...body, END]`：

```
system   : START=1000, END=1009, body 1001..1008
user     : START=2000, END=2009, body 2001..2008
assistant: START=9000, END=9009, body 9001..9008
tool     : START=3000, END=3009, body 3001..3008
gen-prompt 起手符 (add_generation_prompt): 9000
漂移哨兵段: 7000..7099（不属于任何 role，dump 里一眼认出是人为漂移）
```

反查表 `TOKEN_NAMES: dict[int, str]` 把每个 ID 翻译成可读名（如
`1000→"<sys>"`、`2001→"u:compute"`、`7001→"<DRIFT>"`）。dump 打印时把 ID
序列翻译成可读字符串。

### 构造助手（薄封装，不引入 DSL）

- `MsgTok`：给一个 message 分配固定的渲染 token 段，保证同一 message 在不同
  turn 渲染出相同 token（模拟干净 tokenizer）；漂移由测试显式注入。
- `render_prompt(messages) -> list[int]`：拼成 token 序列（含 add_generation_prompt）。
- `render_response(text) -> list[int]`：assistant 输出 token。
- `turn(prompt_ids, response_ids, finish_reason, logprobs=None)`：构造 `TurnRecord`。
- `drift(ids, at, sentinel=7001)`：在指定下标注入/替换哨兵 token，制造
  token-ID 漂移，返回新序列——让漂移点在测试代码里显式可见。

### 双态运行

- 每个 case 函数做严格 assertion。
- `main()` 顺序跑所有 case；每个 case 跑完用 `_dump_helpers.dump_tree_txt`
  打印 tree，再打印线性化出的每个 Sample（token 翻译成可读名 + loss_mask
  对齐展示）。
- 沿用现有文件的 `test_*` + `main()` 风格，可被 pytest 收集，也可直接
  `python -m` 跑供人眼审查。

## §2 Case 矩阵（按「层 × 分叉位置」组织）

### 组 1 — 路由树层（断言 tree 形状）

| # | Case | 分叉位置 | 预期树形 |
|---|------|---------|---------|
| 1.1 | 单 turn | 无 | system→user→assistant 一条链 |
| 1.2 | 干净多 turn（含 tool） | 无 | 一条链，每 message 一节点 |
| 1.3 | system 分叉 | system 不同 | root 下 2 子树 |
| 1.4 | user 分叉（共享 system） | user 不同 | system 共享，user 层 2 leaf |
| 1.5 | assistant message 分叉 | assistant 身份不同 | 共享 user，assistant 层 2 leaf |
| 1.6 | tool 分叉（同 assistant 不同 tool 结果） | tool 不同 | 共享 assistant，tool 层分叉 |
| 1.7 | token-only 漂移不分叉 | message 相同、prompt_ids 不同 | 树不分叉（DFS 忽略 token） |
| 1.8 | 多 tool message 逐节点挂载 | 一个 turn 多 tool | 每 tool 独立节点 |
| 1.9 | 跨 sid 隔离 | 不同 sid | 两棵独立树 |
| 1.10 | 空 response | 无 | assistant leaf messages=[]，turn_response_ids=[] |

### 组 2 — 线性化层（断言 tokens/loss_mask/logprobs/reward）

| # | Case | 触发 | 预期 |
|---|------|------|------|
| 2.1 | 单 turn 线性化 | — | tokens=p+r，loss 只覆盖 r |
| 2.2 | 干净多 turn 线性化 | LCP=cumulative | 1 Sample，prompt 尾 loss=0、resp loss=1 |
| 2.3 | drift case A（prompt 区漂移） | L 落在 prompt 区 | fork 成 2 Sample，不丢 token |
| 2.4 | drift case B1 短→replace | L 落在最近 resp 区、d<threshold | 1 Sample，丢漂移尾、realign |
| 2.5 | drift case B1 长→fork | d≥threshold | 2 Sample |
| 2.6 | drift case B1 threshold=0→fork | 强制 | 2 Sample |
| 2.7 | drift case B2（更早 turn resp 区漂移） | L 落在更早 resp span | 总是 fork |
| 2.8 | fork reward 均分 | N 段 | 每段 reward=R/N |
| 2.9 | 多 leaf reward 均分 | 树分叉 2 leaf | 各 R/2 |
| 2.10 | cross-leaf dedup | 共享 assistant 被 2 leaf 用 | 首 leaf 训练、次 leaf loss=0 |
| 2.11 | routing-only assistant 被过滤 | cc 重放未记录的 assistant | 不 raise，turn 全在一条链 |
| 2.12 | drop 后 sid 清空 | 二次调用 | 返回 [] |

### 组 3 — 组合 / 压力（两层叠加）

| # | Case | 组合形态 |
|---|------|---------|
| 3.1 | rewrite-merge 吸收短 assistant | message 层重写 + 短 resp → demote |
| 3.2 | rewrite-merge 长 assistant→fork | message 重写但 resp 长 |
| 3.3 | rewrite-merge threshold=0→fork | 关闭 |
| 3.4 | rewrite-merge 歧义候选→fork | 2 个短 leaf 候选 |
| 3.5 | rewrite-merge 后 match_key 更新 | 3 turn 跨 merge 节点正确下降 |
| 3.6 | 树分叉 + 其中一支 token 漂移 fork | 2 leaf，一支内部又 drift fork → 3 Sample |
| 3.7 | 深层嵌套多 leaf（≥3 leaf 共享多级前缀） | dedup 跨多 leaf 仅训练一次 |
| 3.8 | 长会话压力（~6–8 turn + 中途 B1 replace + 末尾 A fork） | 端到端多机制串联 |

总计约 30 个 case。组 1 验证「message 身份决定树」，组 2 验证「token ID 决定
线性化」，组 3 验证「两层叠加」。

## §3 断言与打印细节

### 断言强度（统一不变量）

- 树形 case：断言 `root.children` 结构、各层节点数、leaf 数、节点
  `role`/`messages`/`turn_index`/`turn_prompt_ids` 是否为 None。
- 线性化 case：断言 `len(samples)`、每个 `s.tokens`（完整序列）、`s.loss_mask`、
  `s.rollout_log_probs`、`s.reward`，以及全局不变量
  `len(loss_mask)==len(rollout_log_probs)==response_length` 且
  `sum(loss_mask)>0`（无全 mask 样本）。
- token 期望值用构造助手拼出来，不手敲魔数。

### 打印格式（`main()` 时，每个 case 之后）

```
=== CASE 1.4 user 分叉（共享 system）===
[tree]
  <dump_tree_txt 输出，token ID 经反查表翻译成可读名>
[samples] 2 个
  Sample#0 reward=1.0 resp_len=4
    tok : <sys> u:A </user> <gen> 9001 </asst>
    loss:  0    0    0       0     1    1
  Sample#1 ...
PASS 1.4
```

token 与 loss_mask 上下对齐，漂移 token 显示成 `<DRIFT>`，一眼看出分叉点
和训练区。

### 结尾

`main()` 顺序跑全部 case，全 PASS 后打印 `ALL E2E CASES PASSED (N cases)`。

## 非目标（YAGNI）

- 不引入 DSL / fluent builder。
- 不改动 `trajectory_manager.py`。
- 不依赖 sglang / 网络 / 真实 tokenizer。
- 不修改现有 `test_trajectory_manager.py`。
