# 用 FlashREINFORCE 在 MoE 模型上训练多轮 agent

这个示例用 [FlashREINFORCE](../flash_reinforce/README.md) 在 MoE 模型上训练一个会调用工具的 agent：
Moonlight-16B-A3B-Instruct 做 GSM8K，每轮之间可以调用计算器。生成函数 `calculator_agent.py` 是按模板写的，
你用 `--custom-generate-function-path` 把自己的多轮 agent 接入 slime 时可以照着改。本文也说明这类 agent
训练时要看哪些指标、它们正常时应该怎样变化。

整条流程在一台 8 卡 A100-80GB 上验证过。Moonlight 各跑了 40 次更新，用来对照的 1.5B dense 模型跑到了
1,024 次更新，详见[验证记录](#验证记录)。

## 目录

* [快速开始](#快速开始)
* [工作原理](#工作原理)
* [自己写多轮生成函数](#自己写多轮生成函数)
* [训练时要看的指标](#训练时要看的指标)
* [常见问题排查](#常见问题排查)
* [验证记录](#验证记录)
* [局限](#局限)

## 快速开始

```bash
hf download moonshotai/Moonlight-16B-A3B-Instruct --local-dir /root/models/Moonlight-16B-A3B-Instruct
hf download zhuzilin/gsm8k --repo-type dataset --local-dir /root/datasets/gsm8k
cd examples/flash_reinforce_agent
python prepare_data.py /root/datasets/gsm8k /root/datasets/gsm8k-calculator
cd ../..
bash examples/flash_reinforce_agent/run-moonlight-16B-A3B.sh
```

`prepare_data.py` 会把 GSM8K 原有的系统消息换成 agent 的系统提示，告诉模型怎么调用计算器。脚本在一台 8 卡
机器上运行：

* 4 张训练卡：专家并行 4，完整重计算，优化器状态放在 CPU 上（`--optimizer-cpu-offload`，16B 模型约需
  200GB 内存）。
* 4 个 SGLang 推理引擎，每个一张卡。

| 变量 | 默认值 | 含义 |
|---|---|---|
| `MODEL_DIR` | `/root/models/Moonlight-16B-A3B-Instruct` | HF 权重，直接加载，不需要转换 |
| `DATA_DIR` | `/root/datasets/gsm8k-calculator` | `prepare_data.py` 的输出 |
| `NUM_ROLLOUT` | 100 | 更新次数 |
| `POOL_SIZE` | 64 | 正在生成或等待组 batch 的 rollout 数 |
| `QUEUED_BATCHES` | 2 | 提前组好、等待训练的 batch 数 |
| `TRUST_REGION_DELTA` | 5e-3 | binary-KL 信赖域阈值；设为 `inf` 即关闭（普通的重要性加权 PG） |
| `CALCULATOR_LATENCY` | 0 | 每次调用计算器等待的秒数，用来模拟慢工具 |
| `DEBUG_ROLLOUT_DIR` | 不设 | 把每一步的 rollout（含路由）保存下来，便于检查 |
| `NUM_GPUS`、`ACTOR_GPUS`、`ROLLOUT_GPUS` | 8、4、4 | GPU 分配 |
| `WANDB_PROJECT`、`WANDB_MODE`、`WANDB_DIR` | 不设 | wandb 记录 |

## 工作原理

### 算法

FlashREINFORCE 不需要 critic，每个 prompt 只采一条 rollout：

* advantage 是奖励减去整个 rollout batch 的平均奖励。
* 每个 rollout batch 只做一次优化，策略梯度就是普通的 REINFORCE（`train/ppo_kl` 恒为 0）。
* 每个参与训练的 token 带一个不截断的重要性权重 `pi_train / pi_rollout`。
* 如果一条序列上采样 token 的推理端概率与训练端概率之间的平均 binary KL 超过 δ，这条序列就不产生梯度。

各部分的详细说明见 [`examples/flash_reinforce`](../flash_reinforce/README.md)，那里也复现了 NVIDIA molt 的配方。

### 异步 rollout

rollout 采用 [fully-async](../fully_async/README.md)，并按 molt 异步训练器的方式组 batch
（`--fully-async-pool-size`、`--fully-async-max-queued-batches`）：

* 池子里始终保持 `POOL_SIZE` 条 rollout，有的在生成，有的已经完成、在等待组 batch。
* 训练端每次取走下一个组好的 batch；组 batch 时每取走一条 rollout，就往池子里补一个新 prompt。
* 权重更新时，还有轨迹在生成中。所以每个 token 都是由比当前训练策略旧几次更新的策略生成的（落后的
  更新次数就是它的 staleness），一条轨迹里的不同轮也可能来自不同版本的策略。重要性权重和信赖域负责
  校正这些偏差。

### 一条轨迹的样子

模型以 `<calc>EXPR</calc>` 结束一轮（这是一个停止串）。环境计算表达式，把结果作为下一条用户消息返回，
格式按模型自己的 chat template 渲染。接着开始新的 assistant 轮，如此往复，直到模型用 `\boxed{}`
给出答案：

```
tokens    [prompt][turn 1 ... <calc>2+3</calc>][<|im_end|> user: <result>5</result> <|im_end|> assistant:][turn 2 ... \boxed{5} <|im_end|>]
loss mask          1 1 1 1 1 1 1 1 1 1 1 1 1 1  0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

整条轨迹是一个训练样本。结果消息用 `trainable=False` 追加，loss mask 为 0：它们只作为上下文，不进
loss，也不参与重要性权重和信赖域的计算。

### 一轮被权重更新打断

权重更新时 SGLang 引擎会暂停，所有在途请求都会被中止。`slime.rollout.sglang_rollout.generate_turn`
随即重发这一轮：带上 prompt、之前各轮和这一轮已生成的部分，token 预算减去已生成的数量。SGLang 会把重发的
请求挂起，等引擎恢复后在新权重下接着生成：

* 中止前生成的 token 保留原来的推理端 logprob。
* `Sample.weight_versions` 记下每个生成过 token 的策略版本。
* 之前各轮和工具状态都不会重做。

### 路由重放（R3）

开了 `--use-rollout-routing-replay` 后，SGLang 会返回每个位置被路由到的专家，训练时按这些路由重放，
保证训练端的前向用的专家和推理时一样。`generate_turn` 每次请求只要还没有路由的那些位置
（`routed_experts_start_len`）：

* 续写的那一轮保留之前 token 生成时的路由。
* 工具 token 的路由来自下一轮模型生成时的 prefill。

训练要求除最后一个位置外，每个位置都有路由。

## 自己写多轮生成函数

以 `calculator_agent.generate` 为模板：

```python
async def generate(args, sample, sampling_params):
    state = GenerateState(args)
    if not sample.tokens:
        sample.tokens = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    for turn in range(MAX_TURNS):
        output = await generate_turn(args, sample, {**sampling_params, "max_new_tokens": ..., "stop": [...]})
        if output["meta_info"]["finish_reason"]["type"] != "stop" or not is_tool_call(output["text"]):
            return sample  # the final answer, a truncated turn or a cancelled rollout
        message = render_tool_result(run_tool(output["text"]))  # chat-template text around the result
        if turn == MAX_TURNS - 1 or no_budget_left(message):
            sample.status = Sample.Status.TRUNCATED
            return sample  # end on the model's turn (see rule 3)
        sample.append_response_tokens(args, tokens=state.tokenizer.encode(message, add_special_tokens=False),
                                      trainable=False, text=message)
    return sample
```

规则：

1. **每一轮都用 `generate_turn` 生成。** 不要自己处理被中止的请求，也不要把整条轨迹重来。轨迹一旦比两次
   权重更新的间隔长，每次中止都重来的轨迹就永远跑不完，工具也会被重复执行。
2. **工具和环境的输出用 `trainable=False` 追加，** 按模型实际看到的样子分词。工具结果要按模型的
   chat template 作为消息返回。之前把结果直接接在模型的 `</calc>` 后面，Moonlight 就不给答案直接结束
   了，RL 很快学会了不用工具。
3. **每条轨迹都以模型的一轮结束。** 工具 token 的路由来自下一轮的 prefill。如果后面不会再有模型的一轮
   （轮数或预算用完），就不要追加工具输出，否则训练时 R3 的检查会报错（见[常见问题排查](#常见问题排查)）。
4. **一条轨迹只对应一个训练样本。** FlashREINFORCE 的基线是按样本求平均的，拆成多个样本的轨迹会被
   重复计入。
5. **只有真正失败时才返回 `Sample.Status.ABORTED`**（比如沙箱挂了）。调度器会重发被中止的 rollout，但
   如果连续 3 次中止都没有产生新 token，就丢弃它，计入 `rollout/fully_async/failed_rollouts`。
6. **请求并发要够池子用：** `--sglang-server-concurrency` 乘以引擎数要不小于 `POOL_SIZE`，否则一部分
   轮次会在 slime 里排队，而不是在生成。
7. **MoE 模型要开 `--use-rollout-routing-replay`。** 不开的话，SGLang 和 Megatron 的路由差异可能让
   大部分序列落在信赖域之外。

奖励函数（`--rm-type` 或 `--custom-rm-path`）看到的是 `sample.response`，包含整条轨迹和其中的工具消息。

## 训练时要看的指标

下面的参考值来自[验证记录](#验证记录)里的几次运行：

* **1.5B dense：** DeepSeek-R1-Distill-Qwen-1.5B，molt 的配方，1,024 次更新。
* **MoE：** Moonlight-16B-A3B 跑本示例，40 次更新。表中是中位数，需要时括号里给出范围。

这些指标会记到 wandb，也会打印在任务日志的 `step N`、`rollout N`、`perf N` 行里。

### 1. 学习进展

| 指标 | 含义 | 正常表现 | 1.5B dense | MoE agent |
|---|---|---|---|---|
| `eval/<数据集>` | 评测准确率（本示例每题采 1 次） | 上升 | AIME 平均 0.220 → 0.267 | GSM8K 0.61 → 0.89 |
| `rollout/raw_reward` | batch 的平均奖励 | 上升，逐步看会有波动 | 0.53 → 0.75 | 0.72 → 0.91 |
| `rollout/response_len/mean` | 回复 token 数，含工具消息 | 缓慢变化 | 6.5k → 4.6k（截断变少） | 220 → 275 |
| `rollout/truncated_ratio` | 被 token 预算截断的回复比例 | 保持低位或下降 | 41% → 10% | 0 |
| `train/entropy_loss` | 参与训练的 token 上的策略熵 | 缓慢下降 | 0.79 → 0.63 | 0.16 → 0.12 |

奖励下降的同时 `rollout/truncated_ratio` 上升，说明回复越来越长、撞到了预算上限。熵突然掉向 0 说明策略
变得确定，通常紧接着就会出现退化、重复的输出，`rollout/repetition_frac` 也能看出来。

### 2. 更新是否符合 FlashREINFORCE 的假设

| 指标 | 含义 | 正常表现 | 1.5B dense | MoE agent |
|---|---|---|---|---|
| `train/ppo_kl` | 训练前向与复用它作为旧 logprob 之间的 KL | 恒为 0 | 0 | 0 |
| `train/pg_clipfrac` | PPO 截断比例 | 恒为 0 | 0 | 0 |
| `train/tis` | 平均重要性权重 `pi_train / pi_rollout` | ≈ 1 | 1.000 | 1.000（0.989–1.002） |
| `train/train_rollout_logprob_abs_diff` | 每个 token 训练端与推理端 logprob 之差的绝对值均值 | 小且稳定 | 0.011 | 0.006（0.004–0.033） |
| `train/tis_binary_kl` | 采样 token 的平均 binary KL，即信赖域的判据 | 远低于 δ = 5e-3 | 1.6e-4，基本不变 | 2–3e-4（有一次运行前几步为 1.6e-3），尖峰到 6e-3 |
| `train/tis_seq_reject_frac` | 落在信赖域外的序列比例 | 小，且没有上升趋势 | 0 | 0（早期有一步到 0.19） |
| `train/grad_norm` | 裁剪（阈值 1.0）前的梯度范数 | 稳定，没有越来越大的尖峰 | 0.05 | 后期 0.32（早期最高 3） |

`train/ppo_kl` 或 `train/pg_clipfrac` 不为 0，说明旧 logprob 不再来自训练前向。设了 `--num-steps-per-rollout`
大于 1、`--use-rollout-logprobs`、`--get-mismatch-metrics` 或 `--kl-coef` 不为 0 都会这样，这时的更新就
不再是 FlashREINFORCE 了。

binary KL 和拒绝比例衡量推理引擎的概率离训练端有多远：

* **1.5B dense** 在 1,024 次更新里从没触到信赖域，binary KL 一直比 δ 低约 30 倍。
* **MoE** 会出现尖峰：即使开了 R3，个别步里也有百分之几的序列落在信赖域外，最多的一步是 19%。
* 要看**趋势**，不要只看单步。拒绝比例持续上升，或长期高于约 0.3，说明 batch 里真正在训练的部分已经
  很少了，见[常见问题排查](#常见问题排查)。

### 3. 异步与 staleness

| 指标 | 含义 | 正常表现 | 1.5B dense | MoE agent |
|---|---|---|---|---|
| `rollout/staleness/mean`、`/max` | 样本最早的 token 比训练步落后几次权重更新（由当前权重生成则为 0） | 大约是 池子 / batch + 排队 batch 数，且稳定 | 10（最大 12–19） | 3.2（最大可到 32） |
| `rollout/staleness/multi_version_frac` | 由多个权重版本生成的样本比例 | 有轮次跨越更新时大于 0 | 0.29 → 0.17 | 0（每轮很短）；加 10 秒工具延迟后 0–0.09 |
| `rollout/fully_async/queued_batches` | 训练端取 batch 时已提前组好的 batch 数 | 训练是瓶颈时会顶到上限 | 8 个里有 7 个 | 2 个里有 1 个 |
| `rollout/fully_async/pool_finished` | 池子里已完成、在等待的 rollout 数 | 训练是瓶颈时会很大 | 270–340 | 31 |
| `rollout/fully_async/failed_rollouts` | 被丢弃的 rollout 数（出错、反复中止） | 0 | 0 | 0 |
| `perf/wait_time_ratio` | 每步里等数据的时间占比 | 训练是瓶颈时很低 | 0.03–0.06 | 0.22 |
| `perf/step_time` | 每次更新的秒数 | 稳定 | 50–70 | 20–23 |
| `perf/update_weights_time` | 把权重推到推理引擎的秒数 | 小 | 0.2 | 1.6 |

这几个指标要一起看：

* **谁是瓶颈。** `queued_batches` 一直顶在上限、`pool_finished` 一直很高，说明训练是瓶颈：推理引擎在
  batch 之间空闲，staleness 也最大。`perf/wait_time_ratio` 很高、`queued_batches` 一直是 0，说明生成
  是瓶颈。
* **staleness。** 平均值大约是 池子 / batch 再加上排队的 batch 数。平均值一直在涨，或者 `max` 连续很多步
  远高于平均值，说明有 rollout 卡住了。
* **跨更新的轮次。** 每轮很长的 agent 里，`multi_version_frac` 应该大于 0。只有每轮都能在两次更新之间
  生成完时（比如这里的 GSM8K），它才会是 0。staleness 按样本里最早的 token 计算。

## 常见问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `train/tis_seq_reject_frac` 持续上升或高于约 0.3 | 推理端和训练端的前向不一致 | MoE 先确认开了 `--use-rollout-routing-replay`；看 `train/train_rollout_logprob_abs_diff`；不要设置 `SGLANG_RETURN_ORIGINAL_LOGPROB`；如果差异是模型本身固有的，调大 `TRUST_REGION_DELTA` |
| `ValueError: R3 sample N routed-experts rows=A, expected=B from len(tokens)-1` | 轨迹以追加的工具 token 结束，这些位置没有后续轮次来 prefill | 以模型的一轮结束（规则 3） |
| `ValueError: No dot product attention backend is available` | Moonlight 的 MLA 注意力（qk head 维度 192、v 128）在 A100 上没有 FlashAttention 2 或 cuDNN 内核 | 用 `--attention-backend auto`（脚本默认），A100 上会退回 unfused 实现；H100 上会用 FlashAttention 3 |
| 模型不再调用工具 | 工具结果破坏了对话格式，或工具确实没用 | 按规则 2 以对话消息返回结果；用 `DEBUG_ROLLOUT_DIR` 保存轨迹逐条查看 |
| 轨迹一直跑不完，`pool_finished` 一直是 0 | 生成函数在中止后把整条轨迹重来 | 用 `generate_turn`（规则 1） |
| `rollout/fully_async/failed_rollouts` 大于 0 | 生成函数抛了异常，或连续 3 次返回 `ABORTED` 且没有进展 | 在任务日志里找 `dropping a failed rollout` 和对应的报错栈 |
| `train/ppo_kl` 不为 0 | 旧 logprob 不是训练前向 | 用 `--num-steps-per-rollout 1`、`--kl-coef 0`，不要加 `--use-rollout-logprobs` 或 `--get-mismatch-metrics` |
| 训练卡显存不足 | 轨迹太长或 micro-batch 太大 | 调小 `--max-tokens-per-gpu`；保持完整重计算；用更多卡做专家并行 |

查看轨迹：设置 `DEBUG_ROLLOUT_DIR`，然后加载保存的文件：

```python
import torch
from slime.utils.types import Sample

samples = [Sample.from_dict(item) for item in torch.load("rollout_12.pt", weights_only=False)["samples"]]
sample = samples[0]
print(sample.response)                     # the whole trajectory, tool messages included
print(sample.loss_mask, sample.weight_versions)
assert sample.get_rollout_routed_experts_length() == len(sample.tokens) - 1  # with R3
```

## 验证记录

所有运行都在一台 8 卡 A100-80GB 上完成。

| 运行 | 配置 | 结果 |
|---|---|---|
| 1.5B dense，molt 配方 | DeepSeek-R1-Distill-Qwen-1.5B，[flash_reinforce](../flash_reinforce/README.md)，1,024 次更新 | AIME 2024/2025 平均 0.220 → 0.267；没有序列被信赖域拒绝；梯度范数约 0.05 |
| MoE agent | 本示例，40 次更新 | GSM8K 0.61 → 0.89。保存下来的每个样本都通过了检查：loss mask 恰好只在工具 token 上为 0；每个回复 token 对应一个 logprob；除最后一个位置外每个位置都有路由；权重版本单调不减 |
| MoE agent，慢工具 | `CALCULATOR_LATENCY=10`，40 次更新 | GSM8K 0.76 → 0.88；最多 9% 的样本由两个权重版本生成；同样的检查全部通过 |
| 一轮被权重更新打断 | 单个 SGLang 引擎上的 Moonlight：用 `generate_turn` 生成一轮最多 1,500 token 的回复，4 秒后暂停，以版本 2 重新加载权重，再恢复 | 生成到 598 token 时被中止；带着已生成部分重发，剩余预算 902；在版本 2 下正常生成完（`weight_versions == ["1", "2"]`）；1,505 个 token 有 1,504 行路由；断点前后文本连贯 |

## 局限

* MoE 只跑了 40 次更新，还没有测过它的长期稳定性。1.5B dense 跑到了 molt 12,000 次更新里的 1,024 次。
* 只测了单机（专家并行 4，优化器放 CPU），多机以及张量并行、流水线并行都没测过。
* A100 上 MLA 注意力走 unfused 实现：更慢，显存随上下文长度平方增长。agent 上下文很长时，建议用 H100
  或选 GQA 架构的 MoE。
* 停止串如果被续写从中间切开就匹配不到。`</calc>` 在 Moonlight 里是 3 个 token，可能被切开；停止
  token id 不会。
* 分布式 fully-async rollout（`--rollout-data-transport straw`）不支持这里的组 batch 方式。

## 文件

* `calculator_agent.py`：自定义生成函数（`calculator_agent.generate`）和计算器。
* `prepare_data.py`：生成带 agent 系统提示的 GSM8K。
* `run-moonlight-16B-A3B.sh`：训练脚本。
* 仓库根目录下的 `tests/test_flash_reinforce_agent.py`：CPU 测试，覆盖计算器、工具消息的 loss mask、
  续写的轮次，以及以模型的一轮结束。
