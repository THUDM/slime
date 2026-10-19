# P2P 参数同步

通过 `--update-weight-transport p2p`，可以在独立的 Megatron 训练 GPU 和 SGLang 推理 GPU 之间同步完整参数。训练 rank 直接向推理 worker 提供转换后的 HF 张量，无需跨训推建立 NCCL 通信组，也不需要共享 checkpoint 目录。

当前数据传输使用 Mooncake Transfer Engine。对外后端、控制协议和 SGLang 加载器统一命名为 `p2p`，以后替换传输实现时可以保留训练配置。它与共卡 CUDA IPC 更新器内部的 expert P2P 重分配是两个入口。

## 环境准备

两个环境都安装同一份 slime，并分别安装匹配自身 CUDA 的 Mooncake wheel，建议固定相同版本。例如，两个 CUDA 12 环境可以使用：

```bash
conda run -n train python -m pip install mooncake-transfer-engine==0.3.11.post1
conda run -n serve python -m pip install mooncake-transfer-engine==0.3.11.post1
```

CUDA 13 环境改用 `mooncake-transfer-engine-cuda13`，每个环境只安装一种 Mooncake wheel。CUDA 兼容性、RDMA provider 等系统依赖见 [Mooncake 官方安装文档](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html)。当前实现需要 CUDA GPU、GPUDirect RDMA 支持和可达的 RDMA 网络；两端还需允许动态分配的 TCP 控制端口和 Mooncake RPC 端口互通。

两端可以各自保留 PyTorch、CUDA、NCCL 库，训练侧 TP collectives 仍使用训练环境。环境解耦并不免除两侧各自的驱动和原生库要求。其他参数更新后端不要求安装 Mooncake。

需要限制自动网卡选择时，在各节点设置 `SLIME_MOONCAKE_IB_DEVICES`，值为允许使用的网卡名称列表，以逗号分隔。传输实现从中选择靠近当前 CUDA 设备的网卡。按实际网络配置设置 `MC_GID_INDEX`。

## 使用

在训推分离的 Megatron 作业中添加：

```bash
--update-weight-mode full \
--update-weight-transport p2p \
--update-weight-buffer-size 536870912 \
--update-weight-p2p-timeout 120
```

slime 管理的 engine 会自动注册加载器。[外部 rollout engine](external-rollout-engines.md) 需要在推理环境安装 slime，并在启动 SGLang 时添加：

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --custom-weight-loader slime.backends.sglang_utils.p2p.load_weights \
  ...
```

超时参数限制传输、加载以及各 engine 控制阶段的等待时间，大模型或慢网络应适当调大。当前适配器要求推理 PP=1；推理 DP 必须为 1，或开启 DP attention。训练 PP/VPP、dense TP、expert TP 和 EP 复用现有权重命名与转换逻辑。训推 TP/EP 布局可以不同。模型和量化支持取决于已有的 Megatron → HF 转换与 SGLang 原生加载器，并未逐一验证所有组合。该加载器不更新 draft model 权重。

当前 `p2p` 适用于 Megatron 全量更新和独立的训推 GPU。Delta 更新继续使用 disk transport，共卡作业继续使用 CUDA IPC 更新器。

## SGLang Prefill/Decode 分离

PD 的 KV 传输和 `p2p` 参数更新可以同时选择 Mooncake。二者使用独立的 Transfer Engine、RPC session 和注册缓冲；参数更新不会借用或重置 SGLang 共享的 KV engine。不过它们仍然占用同一网卡的资源，进程级 `MC_*` 配置也会影响两者，独立 engine 不代表带宽或配置隔离。RPC 端口范围需要容纳两套 engine，并避开 HTTP/bootstrap 端口。

PD 下要求 `--flush-cache-interval 1`。先暂停全部 prefill/decode engine 并中止请求，确认缓存清空成功后才开始加载；全部 engine 加载成功、提交版本后才恢复生成。暂不支持 PD 通过 `in_place` 暂停跨更新保留旧 KV。使用外部 PD 服务时，更新模型的所有 prefill/decode engine 都必须接入 slime 的 rollout service。

开启 `--offload-rollout` 时，SGLang 必须在释放 KV 底层内存前注销注册，并在恢复后重新注册；相同虚拟地址不代表重映射后原有 RDMA 注册仍有效。应使用仓库配套的 SGLang 生命周期补丁，不要通过参数加载器重置共享 KV engine 来处理。参数传输缓冲会在更新之间独立释放。中止和排空请求的正确性仍依赖 SGLang 的 PD 实现，包括底层 KV 传输是否已完成。

## 数据流与显存

Dense 层分配给训练 DP/CP 副本，expert 保留在其 EP owner 上。训练侧 TP 汇聚后，每个张量只做一次 HF 转换；推理 rank 主动读取，再由 SGLang 原生加载器按自身 TP/EP 布局切分。**每个推理 rank 都接收完整 HF 张量流，包括自己不持有的 expert。** 这一取舍让传输层无需掌握模型切分规则，但会产生冗余网络传输。按目标 rank 分片发送可以作为后续优化。

TCP 传输 JSON 描述符、序号和确认消息，张量通过注册的 GPU 缓冲传输。发送端每个来源使用双缓冲，接收端每个来源使用一个缓冲，独立排空各来源，避免阻塞训练侧 collectives。原生模型加载器拿到拥有独立存储的副本，因此可以跨批次保留配对投影。

普通批次的每来源容量为 `max(1 MiB, update_weight_buffer_size / training_world_size)`。单个张量不会拆分，因此该参数**不是显存硬上限**。超大张量、转换临时空间、接收缓冲和加载器副本都会增加峰值。成功更新后会释放闲置的注册缓冲并重建传输连接，更新之间不保留 GPU 传输缓冲。

## 完成与失败

同步期间暂停生成，缓存处理遵循 `--flush-cache-interval`。传输检查版本、消费者身份、序号和张量边界。所有消费者确认读取后，发送缓冲才能覆盖；加载器完成 CUDA 读取后，接收缓冲才能复用。

全部训练和推理 worker 完成加载后，才进行权重后处理、提交版本并恢复生成。部分更新无法回滚。加载或提交版本失败时保持生成暂停，更新器拒绝重试；需要重启相关训练和推理进程，再执行完整同步。如果对端失败会阻塞训练 TP collective，监控线程会中止相关训练通信组。仍可能被 DMA 访问的缓冲保留到进程退出。

实现位于 `slime/utils/p2p/`：`stream.py` 管理协议，`buffers.py` 管理注册存储，`mooncake.py` 实现 `transport.py` 定义的数据传输接口。Megatron 权重归属、转换和 SGLang 原生加载适配分别保留在各自后端。

## 验证

CPU 测试覆盖流顺序、多来源并发、缓冲复用、读取失败、权重归属和版本提交顺序：

```bash
python tests/test_p2p_stream.py
python tests/test_p2p_mooncake_transport.py
python tests/test_p2p_weight_loader.py
python tests/test_p2p_weight_update.py
```

在具备四张合适 GPU、已安装 Mooncake 的机器上运行：

```bash
python tests/test_p2p_qwen3_moe.py
```

此 e2e 测试在本地生成小型 Qwen3 MoE checkpoint，用 TP=2/EP=2 的 Megatron 和 TP=2/EP=2 的 SGLang，经过真实后端连续同步参数。测试将推理 log probability 与独立保存的 HF checkpoint 比对，并确认训练实际修改了 expert 参数；无需下载模型或数据集。跨节点 RDMA 和不同运行环境的组合仍需在目标集群另行验证。

PD 变体需要六张 GPU：训练、prefill、decode 各两张。两组推理 engine 同时使用 Mooncake KV 传输和 P2P 参数更新：

```bash
python tests/test_p2p_qwen3_moe.py --pd
python tests/test_p2p_qwen3_moe.py --pd --offload-rollout
```

两种配置分别检查 KV 常驻和 offload/onload 时，重置参数传输 engine 后是否仍能连续生成。测试使用同步 rollout，不覆盖 PD 请求仍在传输时更新参数。CPU 测试通过不代表 RDMA 或 PD 兼容性已验证；应在目标软件版本和网络上运行这些 GPU 用例。
