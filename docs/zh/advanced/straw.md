# 使用 straw 持久化 rollout

[straw](https://github.com/zhuzilin/straw) 为 slime 提供基于文件系统的队列和打包张量存储，持久化 prompt 任务、partial rollout、已完成 sample 和训练 batch。多机上的生成与训练进程可以并行读写载荷，Ray 负责传递控制消息和引用。

当前支持的共享文件系统是 JuiceFS，所有节点必须以相同绝对路径挂载存储池。包括 NFS 在内的其他网络文件系统，需要单独验证锁、可见性和持久性。

## 启用方式

标准 slime 安装和 Docker 构建包含 `straw-queue`。已有环境可以在每个 rollout 和训练节点执行以下命令，同一任务使用相同版本：

```bash
pip install 'straw-queue>=0.1.2'
```

在训练命令中添加：

```bash
--rollout-data-transport straw \
--rollout-data-dir /shared/juicefs/jobs/my-run/rollout_data \
--rollout-queue-run-id my-run \
--rollout-storage-profile juicefs \
--rollout-storage-declaration /shared/juicefs/deployment.json
```

部署声明描述已经验证的挂载设置，不会修改挂载配置。其中包含 `direct_mount: true`、`writeback: false`、`open_cache: 0`、`readdir_cache: false`、`client_version` 和 `durability_description`。详见 straw 的[文件系统约定](https://github.com/zhuzilin/straw/blob/main/docs/FILESYSTEM.md)。本地 POSIX 开发和测试使用默认的 `local` profile。

新任务省略 `--rollout-data-dir` 时使用 `<save>/rollout_data`；未设置 `--save` 时必须显式指定共享目录。从 checkpoint 恢复时，可以从 checkpoint 中读取存储池路径。

存储方式和执行方式独立选择。默认使用 Ray `object-store` 传输和同步 rollout。分布式 fully async 还需要添加：

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

Fully async 在每个有 CPU 资源的 Ray 节点启动一个生成进程，分配总并发。快速 worker 可以提供训练 batch，无需等待所有 worker。权重同步期间暂停接收新任务，持久化在途结果后再恢复生成。

未安装 straw 时，默认 object-store 传输可以运行；显式选择 straw 则在启动时报错并给出安装命令。

## 数据流与调度

1. Worker 请求 prompt group，数据源从 dataset 读取数据，将 group 和更新后的游标一起保存到队列。
2. Worker 生成和打分。归还的 partial group 持久化后等待续跑；已接收的完整 group 可供训练使用。
3. Batch builder 执行 reward/conversion hook，将选中的 sample 分配给各训练 rank。训练结束后，slime 确认数据已消费；没有其他引用需要它们时即可回收存储。

Rollout 和训练数据共用存储池，R3、SC 等大张量可以跨阶段复用。队列通过可续租的 lease 将任务分配给 worker；worker 失联后，未完成任务在 lease 到期后可重新领取。

调度优先级为 **已完成 group → partial group → 新 prompt**。同类中，生成 token 的权重版本越旧越先取用；版本相同时按 FIFO 顺序，没有数值型版本的排在同类末尾。Staleness 为当前 serving 版本减去最旧生成 token 版本，因此新鲜 sample 的 staleness 为 0。调度排序不会自动丢弃过旧 sample。

straw 不支持 `--buffer-filter-path`。`--buffer-sort-by-staleness` 适用于内存数据源，straw 使用上述排序规则。Reward 和 sample-selection hook 可用；分布式 fully async 不支持 `--rollout-all-samples-process-path`。自定义 rollout 函数和 queue reader 的用法见[自定义功能](../get_started/customization.md)。

## 数据打包与支持类型

每个 writer 向 pack 文件追加多条 sample 和 tensor，不会为每个 sample 或 tensor 单独创建文件，从而减少共享存储上的小文件元数据开销。不可变张量引用允许 rollout、训练 batch 和 checkpoint 共享数据；修改时写入新记录，保留的引用始终指向原有内容。

Sample codec 支持嵌套 list、tuple、dict、标量、字节数据、NumPy 数组、PyTorch 张量、PIL 图像和 Sample 字段，包括受支持的自定义字段。R3 routes、SC top-k 和 ragged top-p 数据以 typed tensor 保存。不支持的 Python 对象和循环引用会在发布时报错。记录和张量读取都有校验和检查。

R3 使用 `--use-rollout-routing-replay`，SC 使用 `--use-score-centering`。选择 straw 传输后，这些张量随 sample group 一起持久化。R3 训练只读取当前 CP/TP rank 分配到的行。大 replay 张量在 batch 转换时保持 lazy，但选中的 Sample 元数据和普通字段仍会占用 manager 内存。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--rollout-queue-segment-mib` | `256` | Pack 轮转的目标大小，单位 MiB |
| `--rollout-io-concurrency` | `4` | 限制并发序列化与文件系统 I/O 提交 |
| `--rollout-queue-lease-seconds` | `300` | Worker 租期，活跃 reader 会续租 |

## 在线 GC

在线 GC 默认关闭，通过 `--rollout-queue-online-gc` 开启。slime 在任务和训练 batch 使用完成或被丢弃后通知 straw。只有所有任务、reader、checkpoint 和归档都释放引用后，straw 才会回收封存的 pack。一条存活记录就会保留整个 pack，活跃 writer 和日志历史也会占用空间。

保留的 checkpoint 和归档在 GC 开启时也能保护其载荷，因此回退不需要关闭 GC。数据不再需要时还应释放对应的存储保留关系；只删除索引文件不会释放所有权。不要手动删除活跃存储池中的 pack 文件。离线删除前，先停止所有 coordinator、writer 和 reader。

GC 失败会通过后续队列操作和关闭流程报告。容量耗尽时需要释放不再使用的保留数据或增加容量，不会自动 reset 队列。目前没有活跃 pack 和日志压缩功能。

## 恢复与 checkpoint

正常训练 checkpoint 同时保存模型和 rollout 状态，包括 dataset 游标、sample/group 编号、pending/partial 输入、ready group 和训练进度。已经保存在 straw 中的载荷通过引用保存，不会在每次 checkpoint 时重新复制。保留 checkpoint 会保留它依赖的数据。

重启前需要停止整个原任务，包括远端 worker。Save 目录锁会拒绝并发 coordinator，但不会终止遗留 reader。目前没有 coordinator 自动故障转移。

### 续跑或选择 step

使用相同的逻辑目录恢复最近一次完整 checkpoint：

```bash
--rollout-data-transport straw \
--load /shared/checkpoints/run \
--save /shared/checkpoints/run \
--save-interval 1
```

设置 `--save` 时，Megatron 要求同时指定正数的 `--save-interval`。示例每轮 rollout 保存一次，可按实际需求调整间隔。

如需恢复 rollout 7 结束后的状态，添加 `--ckpt-step 7`，下一轮从 rollout 8 开始。Dataset、模型/tokenizer 配置、straw run ID、storage profile 和 fully async worker 拓扑应与 checkpoint 一致。训练恢复需要保留 optimizer 和训练 RNG 状态。

恢复时创建独立队列，共享 checkpoint 中的不可变载荷。Pending/partial 输入和 ready group 顺序来自该 checkpoint，dataset 游标恢复到保存的位置；不会混入原任务在此之后产生的 sample。后续写入不会修改原 checkpoint。

当 `--save` 已包含任务时，输出写入独立的 `branches/<id>` 目录，由 `rollout/current.json` 记录当前分支。以后重启可以继续使用相同的逻辑 `--load` 和 `--save`。省略 `--load` 且 `--save` 存在当前分支时，也会自动续跑。启动日志会显示实际路径。不可变的 `.straw.json` debug 归档需要为每次运行指定新的输出路径。

| 选择方式 | 参数 |
|---|---|
| 当前分支历史中的某一步 | `--load /shared/checkpoints/run --ckpt-step 7` |
| 指定分支 | `--load /shared/checkpoints/run/branches/<id> --ckpt-step 7` |
| 指定完整 checkpoint | `--load /shared/checkpoints/run/rollout/committed_7.json` |
| 单独的输出目录 | `--save /shared/checkpoints/another-run` |

也可以修改逻辑 save 目录或当前分支中的 `latest_checkpointed_iteration.txt`，选择更早的 step。显式 `--ckpt-step` 或具体提交文件优先。自动选择只使用完整的模型与队列 checkpoint，不会跟随更晚但未完成的模型保存。沿父分支查找时只追溯到各分支起点；如需加载其他分支已放弃的后续历史，请指定具体提交文件。

### 缺失状态与恢复限制

如果模型存在但没有保存队列快照，则从空队列开始。存在 `rollout/global_dataset_state_dict_<step>.pt` 时恢复其中的 dataset 游标，否则从 offset 0 开始并打印警告。不会从其他 step 借用 pending、partial 或 ready 数据。模型缺失、快照不完整或损坏、载荷缺失、快照版本不受支持等情况会报错，不会静默退化成空队列。

尚无模型 checkpoint 时，使用相同 `--save` 和模型/输入配置重启原任务，可以恢复已持久化的 rollout 工作，但仅限于第一个训练 batch 尚未开始规划的阶段。之后恢复需要匹配的模型/optimizer 和 rollout checkpoint。

恢复依赖原 straw 存储池，单独复制 checkpoint 目录不会复制载荷。恢复边界是完成的 rollout 训练 batch，不是 optimizer microstep。GPU KV cache 和生成 RNG 状态不会恢复，新生成的 token 和随机 hook 结果可能不同。

增大 `--num-rollout` 时，如需沿用保存的 optimizer schedule，可使用 Megatron 的 `--use-checkpoint-opt-param-scheduler`。队列恢复不会覆盖 optimizer 设置。

## Debug 归档与按 key 查询

Debug 保存和加载参数支持 `.pt` 与 `.straw.json` 两种格式：

```bash
--save-debug-rollout-data '/shared/debug/rollout_{rollout_id}.straw.json'
# 在独立的只训练任务中加载，不启动 SGLang：
--load-debug-rollout-data '/shared/debug/rollout_{rollout_id}.straw.json'
```

| 格式 | 内容 | 存储要求 |
|---|---|---|
| `.pt` | Sample 数据和实际张量内容 | 独立文件 |
| `.straw.json` | 带 sample/task key 的不可变索引 | 需要保留所引用的 straw 存储池 |

索引归档独立保留数据，不受队列消费和 GC 影响。使用 straw 传输保存时复用已有张量，否则在索引旁创建 `straw-data` 存储池。每个 rollout 只有一个索引，sample 打包存储。Evaluation 使用 `eval_<id>` 替代 rollout ID。单独复制 JSON 索引不会复制载荷。使用 straw 做只训练回放时，自动使用归档的存储池和 run，覆盖 `--rollout-data-dir` 和 `--rollout-queue-run-id`，并在该可写存储池中新建独立队列，不改变原队列。完整回放（不抽样）直接复用已有 Sample 和张量记录，不重新写入。一次回放的所有归档必须属于同一存储池和 run。`--load-debug-rollout-data-subsample` 也适用于归档。

```python
from slime.data.archive import RolloutArchive
from slime.observability.rollout_data_utils import load_debug_rollout_data

with RolloutArchive('/shared/debug/rollout_7.straw.json') as archive:
    print(archive.keys())  # 按归档顺序返回 (sample key, 可选 task key)
    samples = archive.load_samples(sample_key='sample:42')
    group = archive.load_samples(task_key='prompt:21')
    archive.export_pt('/shared/debug/rollout_7.pt')

# 将 .pt 转为索引归档，在索引旁创建 straw-data 存储池。
samples = load_debug_rollout_data('/shared/debug/rollout_7.pt', rollout_id=7)
RolloutArchive.save('/shared/debug/imported_7.straw.json', samples, rollout_id=7)
```

Key 选择的是该归档保存的数据版本，找不到时抛出 `KeyError`。Compact trajectory 可能共享 sample index，因此查询返回列表。没有 index 的 sample 使用 `position:<序号>`；没有队列来源信息的 sample 没有 task key。

关闭归档只会关闭 reader，数据继续保留。所有 reader 使用完后，可以打开归档并调用 `archive.release()` 释放保留关系。其他 checkpoint、队列和归档的引用独立保留。导出的 `.pt` 在 straw 回收载荷后仍可读取。
