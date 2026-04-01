# 常见 Q&A

1. **训练过程中为什么会出现乱码？**

   一般来说这种情况是 megatron 没有被正确加载。请检查 `--load` 或 `--ref-load` 是否有对应的 ckpt。注意 megatron 只能加载其中有 `latest_checkpointed_iteration.txt` 的目录。

   如果需要指定某个特定的 iter，可以查看当前 megatron 的使用方法，一般是可以通过 `--ckpt-step` 来指定步数。

1. **为什么我的任务一直卡在 ray 提交的页面上？**

   请先检查你需要跑的任务是训推一体的，还是训推分离的。

   如果是训推一体，即训练和推理共用 GPU，请检查

   - 是否设置了 `--colocate` 参数开启训推一体；
   - 当前任务的总卡数是否大于等于 `actor_num_nodes * actor_num_gpus_per_node`

   如果是训推分离，请检查：

   - 当前任务的总卡数是否大于等于 `actor_num_nodes * actor_num_gpus_per_node + rollout_num_gpus`

1. **为什么训着训着 OOM 了？`max_tokens_per_gpu` 是干什么用的？**

   OOM 往往是因为 `max_tokens_per_gpu` 设置过高了。 `max_tokens_per_gpu` 是指在训练过程中，每张 GPU 上最多可以放多少 token。如果担心 OOM 的话，可以先把这个值设成 `rollout_max_response_len / cp_size`，之后再为了提升训练效率来增大这个值。`--max-tokens-per-gpu` 只有在开启 `--use-dynamic-batch-size` 的情况下才会启用。

   如果 `max_tokens_per_gpu` 很小，还会 oom，可以检查一下是否单次生成的数据太长了，需要开启 cp（`--context-parallel-size`）。如果进行了自定义的数据生成，可以看一下是否在多轮生成的情况下，生成的总长度比预期的长很多。

1. **多机训练的时候，遇到了 transformers 库找不到某个模型的错误该怎么办？**

   这种情况一般是因为多个进程都在通过类似于 `AutoConfig.from_pretrained` 或者 `AutoModelForCausalLM.from_pretrained` 的方式读取本地文件，出现了文件系统的写冲突。可以通过设置 `--model-name` 缓解这一问题。

1. **如何续训？**

   直接将 `--load` 设置为 `--save` 的目录即可。

1. **batch size 是如何计算的？**

   一个 rollout 会用 `rollout_batch_size` 条 prompt，每一条会采 `n_samples_per_prompt` 条，所以一个 rollout 共 `rollout_batch_size * n_samples_per_prompt` 条数据。

   可以用 `--num-steps-per-rollout` 来决定每一个 rollout 跑多少步。这相当于是把 `global_batch_size` 设置成 `rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`。

1. **slime 是否进行了 data packing / varlen 处理？**

   data packing 是指在训练过程中，将长短不一的 sample 拼接到一起，从而提升训练的利用率。slime 默认会进行这样的操作。

1. **sglang 部分出现 `Max retries exceeded with url: /get_model_info (Caused by NewConnectionError` 的问题怎么办？**

   这个问题主要来源于单机内多个 sglang server 导致的端口冲突，目前我们仍在和 sglang 团队一起解决这个问题。一个临时的缓解方案是尽可能减少单机内的 sglang server 数量，例如设置 tp=8。

1. **grad norm 好高，训练训崩了怎么办？**

   首先请确保数据和模型是匹配的，例如说，如果数据是实现已经做好 chat template 的了，这个 chat template 是否和原模型一致。如果数据正确的话，可以参考 [debug 指南](../developer_guide/debug.md) 进行更深入的分析。

1. **我的 sglang 生成时间特别特别久，gpu 功率都打满了，跑了好久好没有输出是为什么？**

   请确认一下 `--hf-checkpoint` 对应的模型是否正确设置了 stop token，如果没有，可以通过 `--rollout-stop` 或者 `--rollout-stop-token-ids` 来进行设置。

1. **sglang 出现 an illegal memory access was encountered**

   根据 sglang 的文档（https://docs.sglang.ai/references/troubleshooting.html），有可能是 OOM 了，可以考虑缩小 `--sglang-mem-fraction-static`。

1. **出现 torch compile/inducer 的 `JSONDecodeError`**

   一般是 torch compile 读写 cache 出现的问题。可以考虑在 ray 的 env_var 里加上 `"TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"`。

1. **训练出现 grad NaN 或者 Inf 的情况**

   可以通过设置 `--no-check-for-nan-in-loss-and-grad` 来尝试跳过对应的训练步。
