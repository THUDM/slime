# 使用 straw 持久化 rollout

[straw](https://github.com/zhuzilin/straw) 为 slime 提供基于文件系统的队列和打包张量存储，持久化 prompt 任务、partial rollout、已完成 sample 和训练 batch。多机上的生成与训练进程可以并行读写载荷，Ray 负责传递控制消息和引用。

当前支持的共享文件系统是 JuiceFS，所有节点必须以相同绝对路径挂载存储池。包括 NFS 在内的其他网络文件系统，需要单独验证锁、可见性和持久性。

## 启用方式

标准 slime 安装和 Docker 构建包含 `straw-queue`。已有环境可以在每个 rollout 和训练节点执行以下命令，同一任务使用相同版本：

```bash
pip install 'straw-queue>=0.1.3'
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
3. Batch builder 执行全局 reward/conversion hook，并将结果作为队列控制任务接收。转换后的 batch 只写一次，各训练 rank 按索引直接读取自己的 sample。训练结束后推进消费游标，移除未完成批次引用；恢复所需数据的保留时间由模型 checkpoint 边界单独决定。

Rollout 和训练数据共用存储池，R3、SC 等大张量可以跨阶段复用。队列通过可续租的 lease 将任务分配给 worker；worker 失联后，未完成任务在 lease 到期后可重新领取。

调度优先级为 **已完成 group → partial group → 新 prompt**。同类中，生成 token 的权重版本越旧越先取用；版本相同时按 FIFO 顺序，没有数值型版本的排在同类末尾。Staleness 为当前 serving 版本减去最旧生成 token 版本，因此新鲜 sample 的 staleness 为 0。调度排序不会自动丢弃过旧 sample。

straw 不支持 `--buffer-filter-path`。`--buffer-sort-by-staleness` 适用于内存数据源，straw 使用上述排序规则。Reward 和 sample-selection hook 可用；分布式 fully async 不支持 `--rollout-all-samples-process-path`。自定义 rollout 函数和 queue reader 的用法见[自定义功能](../get_started/customization.md)。

## 数据打包与支持类型

选择 straw 传输后，数据默认打包写入：每个写入进程将多个样本和张量追加到同一个文件（pack）中，减少共享存储上大量小文件的管理开销。slime 默认沿用 straw 的包大小，当前为 **1 GiB**；可以通过 `--rollout-queue-segment-mib` 指定其他大小，单位为 MiB。

包大小控制何时切换到新文件，不影响数据何时可读：每次发布的数据都可以立即读取，无需等待包写满。单次发布超过目标大小时会完整保留，不会为了满足包大小而拆开；显式封存也可能产生小于目标大小的包。

已写入的张量通过不可变引用在 rollout、训练批次和 checkpoint 之间共享。修改数据时会写入新记录，已有引用仍指向原来的内容。

Sample codec 支持嵌套 list、tuple、dict、标量、字节数据、NumPy 数组、PyTorch 张量、PIL 图像和 Sample 字段，包括受支持的自定义字段。R3 routes、SC top-k 和 ragged top-p 数据以 typed tensor 保存。不支持的 Python 对象和循环引用会在发布时报错。记录和张量读取都有校验和检查。

R3 使用 `--use-rollout-routing-replay`，SC 使用 `--use-score-centering`。选择 straw 传输后，这些张量随 sample group 一起持久化。R3 训练只读取当前 CP/TP rank 分配到的行。大 replay 张量在 batch 转换时保持 lazy，但选中的 Sample 元数据和普通字段仍会占用 manager 内存。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--rollout-queue-segment-mib` | 不设置，沿用 straw 默认值（当前为 `1024`） | 切换到新包的目标大小，单位 MiB；显式设置时必须为正数。 |
| `--rollout-io-concurrency` | `4` | 限制并发序列化与文件系统 I/O 提交 |
| `--rollout-queue-lease-seconds` | `300` | Worker 租期，活跃 reader 会续租 |

## 在线 GC

在线 GC 默认关闭，通过 `--rollout-queue-online-gc` 开启。slime 在任务和训练 batch 使用完成或被丢弃后通知 straw。只有所有任务、reader、checkpoint 和归档都释放引用后，straw 才会回收封存的 pack。一条存活记录就会保留整个 pack，活跃 writer 和日志历史也会占用空间。

保留的 checkpoint 和归档在 GC 开启时也能保护其载荷，因此回退不需要关闭 GC。数据不再需要时还应释放对应的存储保留关系；只删除索引文件不会释放所有权。不要手动删除活跃存储池中的 pack 文件。离线删除前，先停止所有 coordinator、writer 和 reader。

GC 失败会通过后续队列操作和关闭流程报告。容量耗尽时需要释放不再使用的保留数据或增加容量，不会自动 reset 队列。目前没有活跃 pack 和日志压缩功能。

## 恢复与 checkpoint

正常训练 checkpoint 同时保存模型和 rollout 状态，包括 dataset 游标、sample/group 编号、pending/partial 输入、ready group 和训练进度。已经保存在 straw 中的载荷通过引用保存，不会在每次 checkpoint 时重新复制。保留 checkpoint 会保留它依赖的数据。

本节介绍推理集群也需要重新启动时的恢复方式。启动前需要停止整个原任务，包括远端生成进程；保存目录的锁会拒绝同时运行的队列控制器，但不会替你停止遗留的读取进程。目前不支持队列控制器自动故障转移。

如果只是 Megatron 训练失败，Ray 和推理集群仍在运行，应按[容灾文档](fault-tolerance.md)重新提交训练任务，保留原来的推理服务和队列控制器。

### 续跑或选择 step

使用相同目录恢复最近一次完整 checkpoint：

```bash
--rollout-data-transport straw \
--load /shared/checkpoints/run \
--save /shared/checkpoints/run \
--save-interval 1
```

设置 `--save` 时，Megatron 要求同时指定正数的 `--save-interval`。示例每轮 rollout 保存一次，可按实际需求调整间隔。

例如加载 step 16 后继续训练 step 17：

```bash
--load /shared/checkpoints/run \
--save /shared/checkpoints/run \
--ckpt-step 16
```

恢复直接使用原保存目录和原队列。Step 17 及之后的 Straw 队列记录、source/builder checkpoint 和当前配置路径下的 debug 归档会清理，下一轮重新 rollout。Checkpoint 中预取但尚未训练的 ready group 和 partial response 也会丢弃生成结果，保留 prompt、sample ID 和 dataset 游标后重新生成。旧租约失效，旧结果不会进入新一轮训练。

Step 16 及之前的 checkpoint 保留，后续 step 可以在原路径重新保存。开启 GC 时释放被清理记录的存储所有权；pack 只有在不再被任何保留的 checkpoint 或归档引用时才会删除。模型权重文件不会由 Straw 清理。

也可以修改保存目录中的 `latest_checkpointed_iteration.txt` 选择更早的 step，或用 `--load /shared/checkpoints/run/rollout/committed_16.json` 指定完整 checkpoint。显式 `--ckpt-step` 或提交文件优先。自动续跑选择最近的完整模型与队列 checkpoint，不会采用更晚但未完成的模型保存。

Dataset、模型/tokenizer 配置、Straw run ID 和 storage profile 应与 checkpoint 一致；训练恢复需要保留 optimizer 和训练 RNG 状态。Fully async worker 会重新启动，不沿用旧 ready 队列。

### 缺失状态与恢复限制

队列快照缺失、损坏、版本不受支持或引用的数据丢失时，会打印 warning 并从空队列重新 rollout。优先使用快照中仍可读取的 dataset 游标，其次读取单独保存的 `rollout/global_dataset_state_dict_<step>.pt`；游标也不可用时，从数据集开头读。模型仍从指定 step 续训，不复用其他 step 的生成结果。每次保存队列 checkpoint 时都会单独保存 dataset 游标。

模型 checkpoint 损坏、缺少必要的 optimizer/RNG 状态或配置不匹配时仍会报错。

尚无模型 checkpoint 时，使用相同 `--save` 和模型/输入配置重启原任务，可以恢复已持久化的 rollout 工作，但仅限于第一个训练 batch 尚未开始规划的阶段。之后恢复需要匹配的模型/optimizer 和 rollout checkpoint。

完整恢复队列需要原 straw 存储池；单独复制 checkpoint 目录不会复制载荷，载荷不可用时按上述规则回退到 dataset 游标。恢复边界是完成的 rollout 训练 batch，不是 optimizer microstep。GPU KV cache 和生成 RNG 状态不会恢复，新生成的 token 和随机 hook 结果可能不同。

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
