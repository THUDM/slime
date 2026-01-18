# 使用文档

## slime 参数简介

在使用 slime 时，传参主要是为了如下几件事：

1. 把集群中一部分 GPU 分配做训练，一部分分配做推理；
2. 训练的部分加载 megatron；
3. 推理部分加载 sglang；
4. 配置 RL 训练需要的超参。

按照这个顺序，我们需要配置这些参数：

### 集群资源分配

集群资源分配主要有这样的 4 个参数：

- `--actor-num-nodes`：RL 的 actor 训练需要多少节点；

- `--actor-num-gpus-per-node`：RL 的 actor 训练的每个节点有卡；

- `--rollout-num-gpus`：rollout （inference）一共需要多少卡；

- `--rollout-num-gpus-per-engine`：每个 inference engine 有多少卡，这个参数会比较像 sglang 的 `tp_size`，也就是在进行多机 serving 的时候，这个数值应该是总卡数，例如 2 机 16 卡 serving 一个模型，这里的值应该是 16。

  这里不像其他的 sglang 参数那样引入 `--sglang-tp-size` 是因为未来也许会考虑支持 sglang 的 dp_size 参数，也就是一个 engine 里面其实是有多个 sglang server 的（目前只支持 `--sglang-enable-dp-attention` 情况下的 `--sglang-dp-size`）。

在默认的配置下，我们会根据这些参数，通过 ray 给训练部分分配 `actor_num_nodes * actor_num_gpus_per_node` 张 GPU，给推理分配 `rollout_num_gpus` 张 GPU，也就是实现了训推分离。

当需要训推一体的时候，还需要配置上：

- `--colocate`：开启训推一体。开启后会忽略 `--rollout-num-gpus` 让训练和推理的卡数相等。

此外，slime 支持 Prefill 和 Decode 的分离部署 (PD Disaggregation)，可以通过设置 `--prefill-num-servers` 参数来指定用于 Prefill 的服务器数量。

### 选择训练后端

slime 支持多种训练后端，可以通过 `--train-backend` 参数进行选择：

- `megatron`（默认）：使用 Megatron-LM 作为训练后端，支持大规模模型的高效训练；
- `fsdp`（实验性）：使用 PyTorch FSDP 作为训练后端，可以直接加载 HuggingFace 格式权重，无需转换。

### 加载 megatron

megatron 与 sglang, vllm 或者 huggingface trainer 之类的工具不同，他不能直接读取 huggingface ckpt，而是需要用户配置好要训练的模型的参数，并且加载 megatron 自己的 ckpt。

一般来说，我们需要做 3 点准备：

- 配置模型参数
- 配置并行以及一些优化
- 配置需要加载的 ckpt

对于一些 megatron 的自定义以及 slime 引入 megatron 的原理，请见 megatron 使用方法一节。

#### 配置模型参数

这里以 qwen3 4B 为例，我们需要这些参数：

```bash
MODEL_ARGS=(
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --swiglu
   --vocab-size 151936
   --disable-bias-linear
   # attn head
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --kv-channels 128
   --qk-layernorm
   # norm
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   # rope
   --use-rotary-position-embeddings
   --rotary-base 1000000
)
```

我们在 [scripts/models](../../../scripts/models) 提供了常用模型的配置，可以直接复用。如果你也在使用 megatron 进行 pretrain/sft 的话，可以直接复用 pretrain/sft 中的模型配置。

注意：

- slime 会加载 `PYTHONPATH` 中的 megatron 的所有参数，所以可以在环境中的 megatron 里找参数以及参数的说明；
- slime 会使用 data packing (或称 varlen 或 thd) 进行训练，无需配置 `--seq-length` 或 `--max-positional-embedding`，这两个参数不会影响训练模型的最大 context length。

#### 设置各种并行与重计算

megatron 是目前优化最为齐全的训练框架，大家使用 megatron 的一个主要目的就是追求其卓越的性能，这里简单介绍一些 megatron 的并行和重计算的配置方法。

- 这里我们简单陈列 megatron 的并行策略，关于这些并行策略之间的 trade-off 请参考更专业的一些讨论：
  - `--tensor-model-parallel-size`：tp
  - `--sequence-parallel`：megatron 的 sp 是 tp 的一种优化，推荐在使用 tp 的时候一直开启 sp。
  - `--pipeline-model-parallel-size`: pp
  - `--context-parallel-size`：megatron 的 cp，也就是序列并行，一般对应 ring attention；
  - `--expert-model-parallel-size`：moe 的 ep，每张卡上有 `num_experts / ep_size` 个 expert；
  - `--expert-tensor-parallel-size`：megatron 支持 moe 的 expert 与其他部分采用不同的 tp_size，我们一般称为 etp。
- 对于重计算，megatron 中一般是配置如下的几个 flag：
  - `--recompute-granularity` 这个值可以选 full 或者 selective，full 就是完全重计算，selective 会少重计算一些，不配置就是不重算；
  - `--recompute-method`：一般用 uniform 就行；
  - `--recompute-num-layers`：多少层分一组来做重算，一般 1 就行。
  

#### 加载 megatron ckpt

megatron 支持多种其自定义的 ckpt 格式，这里介绍 2 种比较主流的格式，

- 曾经比较主流的 torch 格式（对应 `--ckpt-format torch`）；
- 现在推荐使用的 torch_dist 格式（对应  `--ckpt-format torch_dist`）

torch 格式是 megatron 的老存储格式，里面的结构大约是一些 `mp_rank_xxx` 的文件夹，每个文件夹对应了在对应的并行划分下，每个 rank 存储的 ckpt。也是因为如此，在加载 torch 格式的 ckpt 的时候，需要保证 ckpt 的并行策略和训练任务的并行策略是相同的。

我们推荐使用 torch_dist 格式 ckpt，因为 torch_dist 格式可以支持自动并行切分，也就是不同并行的训练任务都可以共用同一个 ckpt，会方便很多。torch_dist 这也是开源 megatron 目前的默认格式。torch_dist 格式的 ckpt 中一般是一堆 `.distcp` 文件。在使用 torch_dist 时，可以使用 [README](../../../README_zh.md) 中介绍的 ckpt 转化方法从 huggingface 转化为 torch_dist，反之亦然。

在存储结构上，megatron 的 ckpt 一般是这样的结构，这里假设存储的路径为 `/ckpt/`：

```bash
--/ckpt/
    |-- latest_checkpointed_iteration.txt
    |-- iter_0000100/
         |-- _0_0.distcp
         |-- _0_1.distcp
         |-- ...
    |-- iter_0000200/
    |-- iter_0000300/
    |-- ...
```

其中 `latest_checkpointed_iteration.txt` 中记录了训练最新的训练步。在加载模型时，不能直接传入 `/ckpt/iter_xxxxxxx`，而是要传入 `/ckpt/`，并用 `--ckpt-step` 来选取对应的训练步（如果不使用 `--ckpt-step`，则会通过 `latest_checkpointed_iteration.txt` 读取对应的训练步。）

在使用 slime 的时候，有 3 个参数用来加载和保存 ckpt：

- `--ref-load`：reference model 用的 megatron ckpt；
- `--load`：actor 用的 megatron ckpt，如果没有设置 `--load`，或者设置的目录不存在，目录中没有 `latest_checkpointed_iteration.txt`，都会直接从 `--ref-load` 的 ckpt 进行初始化；
- `--save`：actor 保存的路径。

注意：

- 不管进行何种方式存储 ckpt，即无论如何设置 `--ckpt-format`，megatron 都可以加载 torch 或 torch_dist 格式

### 加载 sglang

sglang 的加载非常简单，只需要：

- `--hf-checkpoint`：初始化 sglang 用的 huggingface ckpt；

注意：

- 在第一个训练步之前，slime 会把 megatron 里的参数同步给 sglang，所以 `--hf-checkpoint` 中不需要有最新的训练参数，在续训得时候也不需要更换 hf ckpt；
- sglang 默认会从 huggingface ckpt 中 `config.json` 读取模型的最大 context length，可以使用 `--sglang-context-length` 参数来对这个值进行覆盖，从而支持进行更长的推理；
- 在训推一体的训练过程中，虽然 megatron 和 sglang 会先后 offload，但是还是需要为对方留有一些空间，需要通过减小 `--sglang-mem-fraction-static` 来调整 sglang 的显存占用总量。
- slime 支持透传 sgl-router 的参数，方式是在原参数名前加上 `router` 前缀。例如，sgl-router 的 `--balance-abs-threshold` 参数需要设置为 `--router-balance-abs-threshold`。由于 sgl-router 默认使用 cache-aware routing，可能会导致请求分配不均衡的问题。可以通过设置 `--router-balance-abs-threshold 0` 来强制均衡分配，但这可能会影响多轮对话场景下 prefix cache 的命中率。

对于一些 sglang 的自定义以及 slime 引入 sglang 的原理，请见 sglang 使用方法一节。

### 数据格式

目前 slime 只支持加载 `.jsonl` 格式文件，即文件的每一行都是一个 json，一行数据的样例（展开后）为：

```json
{
  "prompt": [
    {
      "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
      "role": "user",
      "step_loss_mask": 1,
    }
  ],
  "label": "34"
}
```

对应的配置为：

```bash
  --input-key prompt
  --label-key label
  --apply-chat-template
```

请注意，这里的 `step_loss_mask`（默认值为 1）字段为 SFT 阶段提供，若设置为 0，则会将该轮 `loss_mask` 设置为 0；若设置为 1，则使用正常 `loss_mask`。
另外我们还提供了一个 metadata_key，默认为 `"metadata"`，读取后我们会把数据中的 metadata 加载进 slime，可能会对自定义数据生成或者自定义 reward model 有帮助。

### RL 训练需要的超参

- `--advantage-estimator`: 当前训练需要的 RL 算法，目前支持：
  - `grpo`（https://arxiv.org/abs/2402.03300）；
  - `gspo`（https://arxiv.org/abs/2507.18071）；
  - `reinforce_plus_plus` 与 `reinforce_plus_plus_baseline`（https://arxiv.org/abs/2501.03262）；
  - `ppo`（https://arxiv.org/abs/1707.06347）；
  - `on_policy_distillation`。
- `--calculate-per-token-loss`：slime 中默认的方案是 per sample loss，即 `mean(sum(sample_i) / len(sample_i))`，如果需要计算 per token loss，即 `sum(sum(sample_i)) / sum(len(sample_i))`，可以开启 `--calculate-per-token-loss`；
- `--use-tis`：如果需要开启 tis（https://fengyao.notion.site/off-policy-rl），可以开启这一设置；
- `--true-on-policy-mode`：开启 True On-Policy 模式，即在训练过程中严格保证数据是当前策略生成的。

#### GRPO 算法

GRPO（Group Relative Policy Optimization）是 DeepSeek-Math 中提出的一种 RL 算法，其核心思想是通过组内相对比较来计算 advantage，而不需要额外的 critic 模型。

使用 GRPO 时，需要设置：

```bash
--advantage-estimator grpo
```

GRPO 的主要特点：

- **无需 Critic 模型**：GRPO 通过对同一 prompt 采样多个 response，然后在组内计算相对 reward 来估计 advantage，避免了训练和维护 critic 模型的开销；
- **资源高效**：由于不需要 critic 模型，GPU 资源可以完全用于 actor 训练和推理；
- **简单易用**：配置简单，只需要设置 `--advantage-estimator grpo` 即可。

相关参数：

- `--n-samples-per-prompt`：每个 prompt 采样的 response 数量，用于组内比较；
- `--normalize-advantages`：是否对 advantage 进行归一化；
- `--eps-clip`：PPO 风格的 clip 范围。

#### PPO 算法

PPO（Proximal Policy Optimization）是经典的 RL 算法，使用 critic 模型来估计 value function，从而计算 advantage。

使用 PPO 时，需要设置：

```bash
--advantage-estimator ppo
```

**注意：PPO 的 Critic 和 Actor 是并列申请 GPU 的**，在资源分配时需要考虑这一点。具体来说：

- Critic 模型会独立占用一组 GPU，与 Actor 的 GPU 资源分开；
- 可以通过 `--critic-num-nodes` 和 `--critic-num-gpus-per-node` 来配置 critic 使用的资源；
- 如果不配置 critic 的资源参数，默认会使用与 actor 相同的资源配置。

集群资源分配示例：

```bash
# Actor 使用 1 个节点，4 张 GPU
--actor-num-nodes 1
--actor-num-gpus-per-node 4

# Critic 使用 1 个节点，4 张 GPU（与 Actor 并列）
--critic-num-nodes 1
--critic-num-gpus-per-node 4

# Rollout 使用 8 张 GPU
--rollout-num-gpus 8
```

在上述配置下，总共需要 `4 (actor) + 4 (critic) + 8 (rollout) = 16` 张 GPU。

PPO 相关参数：

- `--critic-load`：critic 模型的 checkpoint 路径；
- `--critic-save`：critic 模型的保存路径；
- `--critic-lr`：critic 模型的学习率；
- `--critic-lr-warmup-iters`：critic 模型的 warmup 步数；
- `--num-critic-only-steps`：训练开始时只训练 critic 的步数；
- `--eps-clip`：PPO clip 范围；
- `--value-clip`：value loss 的 clip 范围；
- `--kl-coef`：KL penalty 系数，用于 reward shaping。

## 自定义 rollout 函数

slime 支持不同程度的自定义数据生成（rollout）。

- 默认会使用 [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py) 中的 `generate_rollout` 函数进行数据生成。这个文件中实现了基于 sglang 的异步（asyncio）数据生成流程，并支持了例如 dynamic sampling，partial rollout 等功能；

- 可以通过 `--rollout-function-path` 参数，完全替换 sglang_rollout.py 中的 `generate_rollout`，只需要保证 `--rollout-function-path` 传入的函数签名满足：

  ```python
  def generate_rollout(args, rollout_id, data_buffer, evaluation=False) -> list[list[Sample]]:
      """
      Args:
          args: the whole args
          rollout_id: int, the id of the rollout, used for deterministic data generation
          data_buffer: the data buffer to store the generated samples
          evaluation: bool, whether the rollout is for evaluation or not
      
      Returns:
          list[list[Sample]]: a list of list of samples generated by the rollout
      """
          ...
          return samples
  ```

  其中：

  -  `args` 为整个 slime 运行使用的 args；
  - `rollout_id` 对应的是当前是第几次数据生成，用作保证续训时的数据顺序；
  - `data_buffer` 是 slime 中全局唯一的数据 buffer，可以用来获取初始 prompt，数据 id，将生成至一半的 sample 存储下来下次留作下次使用等；
  - `evaluation` 是否是当做 evaluation 使用。可以通过 `--eval-function-path` 单独配置 eval 的函数；
  -  返回的 `Sample` 类型见 [slime/utils/types.py](https://github.com/THUDM/slime/blob/main/slime/utils/types.py)，在实现时，需要保证
     -   `tokens`：prompt + response 的 token；
     -  `response_length`：response 的总长。对于多轮任务，则是除去第一轮 prompt，剩余的 token 长度；
     -  `reward`：这条数据的 reward；
     -  `truncated`：这条数据是否被截断了，类似于 sglang 中的 `finish_reason == length`。
     
     这几个参数被正确配置了。以及如果有工具调用或者多轮使用等场景，确保 `loss_mask` 是正确的：
     
     - `loss_mask` 应该和 `response_length` 一样长，其中需要算 loss 的 token 为 1，mask 掉的为 0
  
- 在一些情况下，可能只需要替换数据生成的逻辑，那么使用 `--custom-generate-function-path` 进行替换即可，这个函数一个简化版实现如下：

  ```python
  async def generate(args, sample: Sample, sampling_params) -> Sample:
      global TOKENIZER
      if TOKENIZER is None:
          TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
  
      # send request to router
      output = await post(
          f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
          {
              "text": sample.prompt,
              "sampling_params": sampling_params,
          }
      )
  
      prompt_tokens_ids = TOKENIZER(sample.prompt, add_special_tokens=False)["input_ids"]
      response_token_ids = TOKENIZER(output["text"], add_special_tokens=False)["input_ids"]
  
      # set sample
      sample.tokens = prompt_tokens_ids + response_token_ids
      sample.response_length = len(response_token_ids)
      sample.truncated = output["meta_info"]["finish_reason"]["type"] == "length"
      sample.response = output["text"]
      sample.aborted = output["meta_info"]["finish_reason"]["type"] == "abort"
  
      return sample
  ```

   更完备的版本请查看 [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py)。

- 有的时候，我们还需要支持自定义的 reward model，可以通过配置 `--custom-rm-path` 来进行配置。

## sglang 使用方法

slime 通过 `HttpServerEngineAdapter` 作为中介，实现了基于 sglang 的 server based engine。

### 参数配置

slime 通过引入 sglang 的 `ServerArgs.add_cli_args`，从而引入了几乎所有的 sglang 参数，在设置一个 sglang 参数的时候，需要在参数前加上 `--sglang` 的前缀，例如：

- 在训推一体的训练时，往往需要限制 `--mem-fraction-static`，这个参数需要转变为 `--sglang-mem-fraction-static`；
- 在训练中，希望 sglang 能推理超过 huggingface checkpoint 的 `config.json` 中标识的最长 context length，需要使用 `--context-length`，那么在 slime 中需要使用 `--sglang-context-length`；
- 在进行多机大 ep 推理的时候，需要 `--ep-size`、`--enable-dp-attention`、`--dp-size`、`--moe-a2a-backend deepep` 等，则可以对应地传入 `--sglang-ep-size`、`--sglang-enable-dp-attention`、`--sglang-dp-size`、`--sglang-moe-a2a-backend deepep` 。

有部分参与和 slime 的资源调度相关，会由 slime 自行配置，例如：

- `--tp-size` 在 slime 中会使用 `--rollout-num-gpus-per-engine`
- `--model-path` 在 slime 中会使用 `--hf-checkpoint`

sglang 参数引入 slime 的方式可以参考 [slime/backends/sglang_utils/arguments.py](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/arguments.py)。

### router 使用方法

slime 会用 [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) 来管理训练过程中的 sglang server。可以通过 `--sglang-router-ip` 与 `--sglang-router-port` 来配置 [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) 的地址。如果不进行配置，则会在集群中默认启动一个 router。

所有的 sglang server 在启动后，会通过 `/add_worker` 申请加入 router。在实际进行数据生成的时候，只需要向 router 发送 http 请求，router 会进行 load balancing 操作，将请求转发给 server 们。

当通过 `--sglang-router-ip` 与 `--sglang-router-port` 来配置传入一个外部的 router，此时 slime 不再会在内部启动一个 router，而是会把所有的 server 都注册在这个外部 router 上。这时可以利用这个外部的 router 地址来实现更复杂的数据生成流程。注意 router 是支持 openai compatible api 的。

## megatron 使用方法

slime 通过复用 `megatron.training` 目录下的常规函数，如 `parse_args`， `save_checkpoint`，`load_checkpoint`，从而实现对不同版本以及轻度魔改的 megatron 的支持。所以在使用时，需要保证 `PYTHONPATH` 中能访问到 megatron，例如在运行时加入 `export PYTHONPATH=/root/Megatron-LM`。

### 参数配置

slime 通过直接引入 `from megatron.training.arguments import parse_args` 引入了当前环境中 megatron 的所有参数。如果当前使用的 megatron 有在 `parse_args` 之外的参数，可以通过像 [train.py](https://github.com/THUDM/slime/blob/main/train.py) 中传入参数来进行配置，例如：

```python
if __name__ == "__main__":
    try:
        from pretrain_gpt import extra_args_provider
    except:
        extra_args_provider = None
    args = parse_args(extra_args_provider)
    train(args)
```

### 自定义参数

在一些定制版 megatron 的实现中，需要在初始化，或者训练步的前后进行特殊的操作。目前我们加入如下的插件：

- `--custom-megatron-init-path`：会增加一些 init 的调用；
- `--custom-megatron-before-log-prob-hook-path`：会在计算 log prob 之前调用；
- `--custom-megatron-before-train-step-hook-path`：会在每个训练步之前调用。可以考虑用这种方式混入特殊的训练 loss 之类的。
