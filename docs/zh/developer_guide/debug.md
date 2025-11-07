# Debug 指南

## 对齐精度

在开发 slime 的过程中，经常会需要检查模型的精度是否正确，可以通过以下方式检查：

1. 训练第一步
   1. rollout 的生成是否是人话，如果不是，有以下 2 种可能：
      - 参数没有正常加载。需要查看是否有 megatron 成功加载 ckpt 的日志；
      - 更新参数有误。可以查看是不是所有的参数都做了转换和参数对应，或者参数名是不是根据并行做了转换（例如 pp_size > 1 时，第二个 stage 提供的参数的 layer id 是不是正确的）。一个比较彻底的方法是在对应模型的 sglang 实现的 `load_weights` 中保存所有的参数，查看和加载的 ckpt 中是否一致；
      - 如果所有参数更新都正确，还出现问题，有可能是 sglang 里有一些特殊的 buffer 在 release 的时候被释放了；
      - 如果是用 pretrain 模型进行的测试，可以换成同结构模型的 instruct 版本，查看这种乱码是不是 pretrain 模型特有的。
   2. 查看打印的 rollout stats 的 `log_probs` 和 `ref_log_probs` 是否完全相等（即第一步 kl=0），且值较小
      - 如果不是完全相等的，一般是 transformer engine 中的某些 non-deterministic kernel 导致的，例如：
        - 在某些版本的 te 里，megatron 需要 `--attention-backend flash`，来强制使用 flash attention，从而避免 CP 下 fused attention 的数值不稳定；
      - 如果数值较大（例如 >1），一般有 2 种可能：
        - 如果值非常大，应该是训练配置有问题；
        - 如果值只是比 sft loss 的状态略大，例如 instruct 模型的 logprob 到了 0.8，有可能是数据不符合训练的 chat template，或者不符合冷启动的分布。
   3. 查看在推一训一（`num_steps_per_rollout == 1`），kl 是否为 0，grad_norm 是否较小
      - 基本上就是一些 megatron / te 相关的 bug，例如：
        - moe 需要开启 `--moe-permute-fusion`。

2. 训练第二步
   1. 对于训推一体，查看是否能正确加载第二步，是否会 OOM；

## 训练推理单独 debug

slime 支持将训练部分和推理部分分开进行调试，从而实现：

- 在调优/debug 推理部分时，只用少量卡就可以启动任务；
- 在调优/debug 训练部分时，可以保证模型输入固定，去除 rollout 的随机性。

具体来说，目前 slime 提供了如下的参数来进行分离调试：

1. `--debug-rollout-only`

   开启后，slime 将不会加载 megatron，只初始化 sglang ，可以用这个方法来进行推理部分的调试。

1. `--debug-train-only`

   开启后，slime 将不会加载 sglang，只初始化 megatron ，可以用这个方法来进行训练部分的调试。

2. `--save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

   开启后，会保存每次 rollout 的结果，可以和 `--debug-rollout-only` 配合使用。注意保存的方式为 `args.save_debug_rollout_data.format(rollout_id=rollout_id)`。

3. `--load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

   开启后，会从 `args.load_debug_rollout_data.format(rollout_id=rollout_id)` 来加载数据，并且不会初始化 sglang（自动设置 `debug_train_only=True`）。可以以这种方式来固定训练部分的输入，对训练部分进行调优，例如切换各种并行。
