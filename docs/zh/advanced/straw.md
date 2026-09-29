# 使用 straw 持久化 rollout

[straw](https://github.com/zhuzilin/straw) 为 slime 提供基于文件系统的持久化队列和打包张量存储。启用 `--rollout-data-transport straw` 后，prompt 任务、部分 rollout、已接收结果和训练 batch 都拥有持久化身份。多台机器上的生成与训练进程通过共享文件系统直接读写载荷；Ray 调度 actor，传递控制消息和小型引用。

当前共享文件系统目标为 JuiceFS。各节点需要在同一个绝对路径挂载同一存储池。包括 NFS 在内的其他网络文件系统，需要单独验证跨客户端锁、数据可见性和持久化语义。

## 架构发生了什么变化

| 职责 | Ray object-store 传输（默认） | straw 传输 |
|---|---|---|
| 数据源 | `RolloutDataSourceWithBuffer` | 基于持久化 prompt 任务的 `QueueDataSource` |
| 载荷交换 | Ray object reference | 指向共享 pack 的 `RecordSetRef` / 张量引用 |
| Worker 分配 | 内存中的 producer 状态 | Lease、attempt、持久化 continuation 和结果 receipt |
| Rollout 转训练数据 | `BatchBuilder` 执行 hook 和 DP 分片 | 同一转换流程，额外持久化选择计划与 ready batch |
| 数据生命周期 | 进程/object-store 所有权 | 显式 task、consumer、reader 和 checkpoint 所有权 |
| 清理 | Object-store 生命周期 | 可选在线 GC，回收没有所有者的封存 pack |

逻辑上分成 rollout 工作与结果、ready training batch 两个阶段。它们共享同一个 straw 存储池；只要任意阶段或 checkpoint 仍引用某个张量，该张量就保持存活，不需要为两个阶段各复制一份。slime controller 协调两个阶段，straw 用 Rust 实现存储、日志、任务接收和所有权协议。

```{mermaid}
flowchart LR
    D[数据集 producer] -->|cursor + 任务提交| Q[持久化 prompt 任务]
    Q -->|lease + 输入引用| W[分布式生成 worker]
    W -->|持久化部分 continuation| Q
    W -->|已接收 group 引用| R[Rollout 结果]
    R --> B[BatchBuilder: 选择计划与 DP 分片]
    B -->|ready batch 引用| T[训练 rank]
    T -->|训练完成| C[Queue controller: 进度与所有权]
    C --> G[straw 在线 GC]
    P[(JuiceFS 共享 pack 文件)] -. 载荷读写 .-> W
    P -. 共享张量引用 .-> B
    P -. 当前 rank 所需数据 .-> T
```

一个不自动重启的 Ray actor 持有本次运行的 coordinator，串行协调队列状态转换和数据集分配；载荷由各 producer 进程写入。因此，多机 I/O 可以并行，每个队列日志仍只有一个写入所有者。当前没有 coordinator 自动选主或故障接管。

## 端到端数据流

1. **提交 prompt。** 数据集 cursor 与 prompt 任务一起提交。Worker 领取 lease，从共享存储读取 prompt group；丢失 worker 不会永久丢弃已经预分配的数据集索引区间。
2. **生成与续跑。** 分布式 fully async 在每个符合资源条件的 Ray 节点上启动生成 worker。Worker 生成、评分并过滤完整 group，将部分 group 保存为持久化 continuation。R3 routes 和 SC rows 必须覆盖已保存的 token 前缀；服务端 capture 不完整时，从上一份一致的输入重试。
3. **接收结果。** 先持久化载荷，再接收引用。Receipt 标识已接收 group，因此即使 RPC 回复丢失也能恢复结果。任务重新分配会产生新 attempt，旧 attempt 无法接收替代结果。
4. **构造训练 batch。** `BatchBuilder` 在调用 reward/conversion hook 前持久化输入选择和转换配置，发布所有 DP shard 后才将整个 batch 标记为 ready。训练 reader 检查 batch、plan 和 rank 身份。Manager 仍会将选中的 sample 物化用于转换，其主机内存仍需纳入容量规划。
5. **确认完成。** 所有 rank 的训练调用返回后，Manager 报告 `training_completed`，推进运行时消费进度并释放 ready batch 容量。这与模型/优化器 checkpoint 持久化是两件事。

Ray 继续负责调度、RPC 和执行进程故障；slime 负责 Sample schema、reward、filter、batch 和训练；straw 负责字节、引用、持久化和存储生命周期。自定义 rollout 仍可返回 Sample 列表，由 Manager 发布为兼容 collection。接口见[自定义指南](../get_started/customization.md)。

## 队列调度

`QueueReader` 是 worker 的队列客户端，负责领取、归还 group 和为正在执行的 lease 续租。`QueueDataSource` 增加整个作业的 reader 创建、consumer 生命周期和 source checkpoint 管理；它继承同一套取样方法，没有另实现一层 buffer。`--rollout-data-transport straw` 会自动选择此数据源；显式指定 `--data-source-path` 时使用 `slime.rollout.queue_data_source.QueueDataSource`。

`QueueReader` 不再有本地 continuation buffer。`add_samples()` 先发布可用的续跑前缀，再通过 `yield_tasks()` 在每个有界批次的一次 WAL 事务中替换输入并归还任务。旧 lease 随之结束，任意 worker 都能用新 lease 领取这些输入。reader 本地只保留正在执行的样本和 lease，关闭时无需重写已归还的 group。

持久化顺序为 **已完成待交付 → partial → 新 prompt**。同类任务按 group 中最旧的数值型 `weight_versions` 排序：版本越旧，staleness 越高，越先取用；相同版本按提交/归还的 FIFO 顺序，没有有效版本的排在同类末尾。比较已有版本即可，无需每次更新模型权重都改写队列。staleness 定义为“当前 serving 权重版本 − 最旧生成权重版本”，新鲜样本为 0。排序本身不会自动丢弃过旧样本。这是在一个队列内建立索引，不是维护三个独立队列。

straw 明确拒绝 `--buffer-filter-path`；`--buffer-sort-by-staleness` 仍用于内存数据源，straw 始终采用上述顺序。reward 和样本筛选 hook 继续可用。已经接收的 group 如果显式放回，会创建独立的 delivery task，保留原 accepted 历史；消费或丢弃 delivery 时也会确认被其替代的已接收版本。fully async scheduler 仍保留有界的 ready result 窗口，其 checkpoint 状态和已接收结果引用单独持久化。

source checkpoint 统一保存共享 pending 任务、准确的不可变输入引用、排序 metadata 和已提交的 dataset producer 游标；worker checkpoint 保存 reader metadata。reader 快照格式为 v3，旧 v1/v2 本地 buffer 快照需要迁移，加载时明确报错。此功能需要带 `yield_tasks()` 和持久化排序字段的新版 `straw-queue` 构建，原 PyPI 0.1.0 wheel 尚无这些接口。

## 共享张量、R3 与 SC

每个 writer 进程复用 pack writer。slime adapter 默认按 **256 MiB** 目标大小轮换，由 `--rollout-queue-segment-mib` 配置。记录和 manifest 都嵌入 pack，**不会为每个 sample 或张量单独创建文件**，避免大量小文件带来的共享文件系统元数据开销。队列与所有权元数据使用追加日志。

已完成的 R3 routes 和 SC 张量，在 custom sample hook 之后随其 sample group 一起发布。后续 rollout、continuation 和训练数据发布复用不可变张量依赖；修改时写入新记录，旧引用保持有效。这是张量级共享与写时复制，不是 GPU 共享内存或内存页级写时复制。R3 训练只读取分配给当前 CP/TP rank 的行。大 bundle 会按 native 写入预算拆分；SC 也支持不启用 R3 的场景。

需要持久化 R3/SC 张量时，使用 `--rollout-data-transport straw`；object-store 传输将它们保存在内存中。不再提供独立的 spill hook、每个 sample 的张量文件或 spill 清理步骤。导入另一个 straw 存储池的引用时，先将数据复制到目标池，再接收依赖这些数据的结果。旧 `.pt` debug dump 直接保存张量内容，队列 GC 后仍可读取；新的 `.straw.json` 索引归档保留共享张量引用。

## Debug 归档与按 key 查询

现有 debug 参数兼容两种格式：

```bash
--save-debug-rollout-data '/shared/debug/rollout_{rollout_id}.straw.json'
# 在独立的只训练任务中加载，不启动 SGLang：
--load-debug-rollout-data '/shared/debug/rollout_{rollout_id}.straw.json'
```

`.pt` 保持原来的独立文件格式。`.straw.json` 是指向 straw pack 的不可变索引，在 straw 传输下共享已有 R3/SC 张量；object-store 传输下会在索引旁创建 `straw-data` 存储池。每份归档独立持有 GC 引用，需要保留相应存储池；单独复制 JSON 不会复制数据。每个 rollout 一个索引文件，样本按 chunk 打包，不会每个 sample 一个文件。评估仍使用 `eval_<id>` 文件名约定。

```python
from slime.utils.rollout_archive import RolloutArchive
from slime.observability.rollout_data_utils import load_debug_rollout_data

with RolloutArchive('/shared/debug/rollout_7.straw.json') as archive:
    print(archive.keys())  # 按归档顺序返回 sample key、可选的原 task key
    samples = archive.load_samples(sample_key='sample:42')
    group = archive.load_samples(task_key='prompt:21')
    archive.export_pt('/shared/debug/rollout_7.pt')

# 将旧格式导入新格式；不传 args 时，在索引旁创建 straw-data 存储池。
samples = load_debug_rollout_data('/shared/debug/rollout_7.pt', rollout_id=7)
RolloutArchive.save('/shared/debug/imported_7.straw.json', samples, rollout_id=7)
```

key 查找的是当前归档中的版本，不是实时队列中最新的任务。找不到 key 会抛出 `KeyError`。compact rollout 可能让多个样本共享同一个 index，所以接口始终返回列表；没有 index 的样本使用 `position:<序号>`。旧文件没有 task 来源信息时无法提供 task key。读取后恢复 Sample，回放张量保持 lazy，旧队列授权信息会被去掉，再进入原来的 BatchBuilder 转换流程。`--load-debug-rollout-data-subsample` 仍然有效。使用 straw 做只训练任务时，应另外指定可写的训练队列。

关闭 archive 只关闭 reader，保留归档数据。确认所有 reader 都结束后，可以显式调用 `archive.release()` 释放其 GC 所有权；其他 queue、archive、checkpoint 的持有关系不受影响。导出的 `.pt` 在原 straw 数据回收后仍可独立读取。

## Sample 编码与完整性校验

Adapter 将 sample 保存为 `slime.v1` JSON record：`{version: 1, tree: ...}`，通过显式标签编码容器、Sample 字段、NumPy 数组、图像和 rollout 引用。嵌套 group、支持的动态字段、多模态输入、R3、SC top-k 和 ragged top-p 都会保留。未知 Python 类型及循环引用明确报错；载荷协议不使用 pickle，也不会根据存储数据中的名称导入类。

Tensor 节点包含 shape、dtype、semantic kind、lazy/validated 标志，以及指向外层 manifest 依赖的 `{dependency: index, ordinal: index}`。物理位置保存在这些依赖 manifest 中，因此后续 sample 或训练 batch 发布可以共享相同的不可变张量记录。所有张量依赖必须先发布，才能接收所属 sample。

straw 的 `tensor.v1` 保存连续、行优先、小端的 typed bytes。Lazy reader 校验元数据及所读行覆盖的每个 chunk 的 checksum，不会校验未读取的 chunk；完整检查则校验整个 segment。张量描述符批量恢复，以复用共享 pack 索引。

## 启用方式

`requirements.txt` 已包含 `straw-queue`，标准 Docker 构建和正常安装 slime 时会自动安装。已有环境可以在每个生成和训练节点执行以下命令，同一任务的所有节点使用相同版本：

```bash
pip install straw-queue
```

未安装 straw 时，默认 Ray `object-store` 传输仍可运行，启动日志会提示安装命令。如果显式选择 `--rollout-data-transport straw`，则在启动检查阶段报错并给出同样的安装命令。

Python 导入名为 `straw`。在原有训练命令中追加以下参数，使用新的共享目录：

```bash
--rollout-data-transport straw \
--rollout-data-dir /shared/juicefs/jobs/my-run/rollout_data \
--rollout-queue-run-id my-run \
--rollout-storage-profile juicefs \
--rollout-storage-declaration /shared/juicefs/deployment.json
```

声明文件包含 `direct_mount: true`、`writeback: false`、`open_cache: 0`、`readdir_cache: false`、已部署的 `client_version` 和 `durability_description`。它记录经过验证的部署设置，不会自动探测或配置挂载。详见 straw 的[文件系统要求](https://github.com/zhuzilin/straw/blob/main/docs/FILESYSTEM_zh.md)。

默认 `local` 存储 profile 对应 POSIX 开发契约；在共享挂载上选择它，并不意味着已验证该存储服务的持久性。

存储方式和执行模式分别选择。默认传输仍为 `object-store`，默认 rollout 入口仍为同步模式。使用 straw 的分布式 fully async 时，额外添加：

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

未指定 `--rollout-data-dir` 时，straw 模式使用 `<save>/rollout_data`；没有 `--save` 则必须显式指定共享目录。`--rollout-io-concurrency` 限制事件循环之外的序列化和 I/O 并发，默认 4。Adapter 使用 worker 进程和有界线程，straw 本身不创建 I/O 进程池。`--rollout-queue-max-pending` 与 `--rollout-queue-max-inflight` 限制任务准入，默认均为 65,536；它们不是保留字节数或文件数配额。

权重同步期间，分布式 producer 暂停提交新任务，让在途请求完成并持久化；还有下一次训练 rollout 时恢复。权重同步本身仍是独立子系统。

关闭时，所有生成 worker 共用五分钟期限完成持久化写入。超时会报告错误，不会被视为成功关闭。

## 在线 GC 与所有权

在线 GC 默认关闭，添加 `--rollout-queue-online-gc` 后，straw 可在任务运行期间回收不再使用的封存 pack。slime 通过 straw API 提供使用完成和丢弃信号。只有 task、publication、queue、reader、checkpoint 等所有者全部释放后，pack 才能删除。仅 lease 超时不能证明 reader 已停止。一条存活记录就会保留整个 pack；活跃 writer、保留 checkpoint 和 WAL 历史仍会占用空间。

GC 失败会停止后台循环，后续 coordinator 操作传播原始原因，关闭时也会报告错误；它不会停止已在远端 worker 中运行的工作。数据保留供排查。容量用尽时应使用背压或由操作者明确决定处理方式，不能 reset 活跃队列或直接删除 pack 腾空间。离线删除目录要求所有参与者停止；仍需配置存储配额与应用保留策略。

已处理位置和过滤决策被显式记录，避免接收顺序中的空洞使已消费数据持续占用容量。独立且有界的 control task 配额允许 Manager 在生产背压时仍能提交 collection。

## 恢复与 checkpoint

| 故障或重启场景 | 支持行为与边界 |
|---|---|
| Worker 在结果接收前丢失 | 恢复最近一次持久化 continuation，以新 attempt 重新分配 |
| 已提交结果但回复丢失 | 恢复已有 receipt，不重复进行逻辑接收 |
| 第一个 batch 计划生成前整个任务停止 | 用相同逻辑 `--save`、初始模型和 rollout 配置重启，自动恢复已接收 group 与持久化前缀 |
| 已生成 batch 计划后重启 | 需要匹配的模型/优化器及 rollout checkpoint，单独恢复队列会明确报错 |

重启前必须停止整个旧任务，包括 coordinator 和 reader。队列 controller 对逻辑 save 目录加锁，直到关闭或退出时释放，拒绝并发任务；这不能证明遗留的远端 reader 已停止。尚未发布的推理可能重新执行；SGLang GPU KV cache 不会恢复。

Checkpoint 的 `<save>/rollout/queue_state_<rollout_id>.json` 保存 source 和已注册 scheduler consumer；`builder_state_<rollout_id>.json` 保存训练 consumer 视图。它们的引用保留相应存储依赖图。Continuation 增量持久化，checkpoint 引用已有数据，不会再次重写全部归还样本的 token。恢复旧 checkpoint 不会抹去 WAL 中后来已接收的历史。

部分 R3/SC 张量在 continuation 快照中保持 lazy 引用，生成追加新行时才加载。恢复 source 和 builder 状态会保留已保存的 warm groups 和行为策略版本，并记录哪些后续输出不属于恢复分支。恢复后的 pending task 独立于过滤决策保留存储引用。包含尚未重启 consumer 的完整 source 快照，必须先保留并持久记录，才能释放旧引用；保存失败保留旧引用，后台 GC 在恢复后的 consumer 状态提交后才启动。

### 从旧训练 checkpoint 创建分支

新的 source v2 快照保存 dataset offset、epoch、sample/group 编号、metadata，以及数据集内容与配置身份。正常 straw 训练先等待 actor/critic 同步保存，再在同一次 admission 暂停期间保存 source 和 builder。保存异常直接终止任务；所有保存调用返回后，由 driver 发布 `rollout/committed_<rollout_id>.json`。该标记列举模型文件，并绑定同一训练边界的 source/builder 索引。省略 optimizer 或训练 RNG 的 checkpoint 不能用于训练分支恢复。

straw 不保存或恢复 data source、生成 worker 和 batch 转换的 RNG 状态。数据集顺序通过保存的游标和 shuffle seed/epoch 恢复。尚未完成的转换会使用当前 RNG 状态重新执行 hook，因此随机 hook 的结果和重新生成的 token 可能在重启后不同。旧快照里的 RNG 字段会被忽略。训练 RNG 仍由训练后端的模型 checkpoint 管理。

checkpoint 按 `rollout_id + 1` 推导 serving 权重版本：第 0 轮 rollout 前会先同步一次权重，每轮 checkpoint 都在下一次同步前保存。恢复时用这个版本初始化 updater，再由首次同步发布恢复后的模型，供下一轮 rollout 使用。`save_model()` 和 `update_weights()` 都不再为 checkpoint 返回版本号。

例如已经训练到 rollout 10，希望从 rollout 7 保存后的状态继续：

```bash
--rollout-data-transport straw \
--load /shared/checkpoints/run \
--ckpt-step 7 \
--save /shared/checkpoints/run
```

正常使用只需要模型的 `--load`、`--save` 和可选的 `--ckpt-step`。保持原来的数据集、模型/tokenizer 配置、straw run ID、存储 profile 和 fully async worker 拓扑。省略 `--rollout-data-dir` 时从提交标记读取原共享池。step 对应保存的 rollout ID，以上例子从 rollout 8 开始。

`--save` 是逻辑目录。首次运行直接写入它；复用已有目录时，自动创建唯一的 `branches/<id>` 输出目录。逻辑目录的 `rollout/current.json` 指向当前物理目录，各分支的 `rollout/branch.json` 记录父 checkpoint 和队列 namespace。已有模型 checkpoint 和队列数据不被覆盖。启动日志会打印实际路径。显式 debug 输出模板保持原样；使用不可变 `.straw.json` 归档时，每次运行应指定新的 debug 路径。

之后用 `--load /shared/checkpoints/run --save /shared/checkpoints/run` 重启，会选当前分支最新的**联合提交 checkpoint**，即使模型 tracker 指向更晚但未完成的保存。若 `--save` 已有 current 指针，省略 `--load` 也会自动续跑。新分支尚未存出第一个 checkpoint 就停止，则沿记录恢复父 checkpoint。尚无任何模型 checkpoint 的首次运行，在 batch 计划生成前可以直接恢复原队列 WAL。

仍然支持手动指定：

| 选择方式 | 参数 |
|---|---|
| 当前分支历史中的某一步 | `--load /shared/checkpoints/run --ckpt-step 7` |
| 某个已保存分支 | `--load /shared/checkpoints/run/branches/<id> --ckpt-step 7` |
| 精确指定不可变提交，忽略 current 指针 | `--load /shared/checkpoints/run/rollout/committed_7.json` |
| 另外选择输出目录 | `--save /shared/checkpoints/another-run` |

也支持直接把 `latest_checkpointed_iteration.txt` 改成更早的 step。显式 `--ckpt-step` 或具体提交文件优先；可以修改逻辑 save 目录或当前物理分支中的 tracker。每次联合提交后，逻辑 tracker 都更新为当前分支保存的 step；恢复 step K 时，在开始生成前先将其设置为 K。current 指针同时记录上次观察到的值，以区分用户修改和启动时继承的旧值。物理 tracker 指向更大 step 但没有联合提交时，按未完成保存处理。手动选中旧模型却没有队列快照时，按下述空队列规则恢复；模型缺失或快照损坏仍然报错。

沿父分支查找时，最多回到每个分叉点，不会自动选中父分支被放弃的未来。需要恢复那段未来时，直接指定它的提交文件。

每次 checkpoint 恢复都在同一存储池内创建独立命名空间的队列。pending/partial 共享快照中的准确输入版本，ready group 保持顺序但重新签发 receipt。producer 恢复到保存的游标，旧 receipt position 和已完成 batch 不导入。修改时写入新记录，旧 checkpoint 仍可读取。这是应用层写时复制；模型权重在新训练保存时才产生新副本。父队列不会 reset 或截断。

如果指定的**模型存在，但确实没有保存队列快照**，则启动新空队列。有旧格式 `rollout/global_dataset_state_dict_<step>.pt` 时，恢复 offset、epoch、sample/group 编号和 metadata；没有时从 offset 0 开始并明确警告，不保证数据精确重放。不会借用其他 step 的 pending、partial 或 ready 数据。模型缺失、已有队列快照不完整或损坏、被引用 payload 缺失、快照版本不受支持等情况会报错，不能静默退化成空队列。

queue fork/resume 命令行参数已删除，统一使用模型的 load/save 参数选择恢复方式。模型加载后核对实际 iteration。随 Docker 提供的 Megatron patch 已处理 step 0，并阻止较新的非持久 checkpoint 覆盖显式选择的 step。

增加 `--num-rollout` 还会改变 Megatron 默认优化器调度器的总步数。若延长训练但保留已保存的调度器配置，使用已有的 `--use-checkpoint-opt-param-scheduler` 参数。队列恢复不会自动覆盖优化器设置。

恢复边界是完整 rollout 训练 batch，不是 batch 内任意 optimizer microstep。推理 KV cache 和服务端采样状态没有快照，因此不承诺继续生成的结果逐位一致。此命令暂不实现复制到不同 straw 存储池的独立物理副本。

## 验证与当前限制

CPU 测试覆盖 codec、R3/SC 传输、CP/TP 按行读取、worker 故障、continuation、checkpoint 视图和 GC 所有权。straw 独立验证 Rust/Python 核心、有界所有权模型和 native 崩溃边界。物理多客户端文件系统检查及真实 GPU 训练补充这些测试。

`tests/test_straw_fully_async_recovery.py` 启动两个本地 Ray 节点，以 SIGKILL 中断整个 driver、coordinator 和 generation 进程树，再启动新任务读取同一个 straw 存储池，不经过正常 checkpoint。测试覆盖已接收结果重放、token/logprob/R3/SC 前缀保持不变、新 lease 和 reward 只计算一次，并分别检查在线 GC 关闭与开启的情况。推理和 reward 使用 CPU fixture，不代表 optimizer 恢复验证。此测试在 `cpu-unittest` 中自动运行，安装方式见 [CI 配置](../developer_guide/ci.md)。

`tests/test_straw_checkpoint_fork.py` 是注册在 Megatron e2e matrix 中的四 GPU Qwen2.5-0.5B 回归测试。先进行三步 fully async 训练，保存 checkpoint 0/1/2，再从 checkpoint 1 向同一逻辑 save 目录恢复两次，验证直接修改模型 tracker 回退，并不指定 step 自动续跑当前分支；另验证缺少队列快照的旧模型从空队列和旧 dataset 游标恢复（编号从 0 开始）。在恢复后的生成启动前，精确比较完整 producer 游标、pending 输入引用和 ready group 顺序；同时检查训练 sample 一致、新旧 task key 隔离、父分支索引未修改、梯度有限且非零，以及 `.straw.json` 和导出的 `.pt` 两种格式的 GPU 只训练重放。测试不要求 GPU 梯度或新生成 token 逐位一致。

```bash
python tests/test_straw_checkpoint_fork.py
```

多机运行时，模型、prompt 数据和 `--work-dir` 必须共享；测试支持指定这些路径和 `--num-gpus-per-node`。与 checkpoint fork 功能本身一样，需要安装 native 层支持 `pending_tasks` 和 `yield_tasks` 的 straw 构建。

本地运行分布式 rollout 与中断恢复测试：

```bash
PYTHONPATH=. python tests/test_distributed_rollout.py
PYTHONPATH=. python tests/test_straw_fully_async_recovery.py
```

主要部署限制包括共享存储带宽、持久化延迟、coordinator 吞吐和 Manager 转换内存。目前尚无存活 pack 整理、日志压缩或 coordinator 自动接管。Rollout 参数见[使用指南](../get_started/usage.md)，存储验证及其覆盖边界见 straw 的[验证文档](https://github.com/zhuzilin/straw/blob/main/docs/VERIFICATION_zh.md)。
