# Qwen3.5 routed-expert INT4-QAT 适配记录

## 目标与边界

本适配只保证 Qwen3.5 MoE 的 routed experts 进入 INT4 fake-QAT 和 SGLang INT4 rollout：

```text
INT4：routed experts
BF16：attention、linear_attn、vision、shared expert、router、MTP、
      embedding、norm、lm_head
```

训练 actor 保持 BF16 Parameter，通过 Megatron grouped-expert 的 fake INT4 路径训练；SGLang 使用 compressed-tensors INT4 权重做 rollout。当前只支持 `expert-tensor-parallel-size=1`，不恢复已经删除的 NVIDIA Megatron-Bridge。

## 最终数据路径

```text
初始化
BF16 HF checkpoint ──native HF→Megatron──> Megatron actor
        │
        └──convert_hf_to_int4_direct.py
             ├── config 精确匹配 Qwen3.5 MoE
             ├── fused expert 3D→逐 expert 2D
             └── symmetric group-wise INT4 pack
                         │
                         └──> SGLang rollout checkpoint

每轮更新
Megatron local 3D expert Parameter
        │  保留 Parameter 的 TP 属性
        │  名字携带当前 EP rank 的 global expert offset
        ▼
TP/EP collective
        ▼
qwen3_5.py：恢复 offset，拆成逐 expert gate/up/down 2D
        ▼
compressed-tensors online pack
        ▼
SGLang：restore 原始布局 → load → repack kernel 布局
```

## 必要机制

### 严格模型 identity

只有下面两个字段同时匹配才进入 Qwen3.5 专用转换：

```text
model_type   = qwen3_5_moe
architecture = Qwen3_5MoeForConditionalGeneration
```

参数名字只识别 tensor layout，不再承担模型识别职责。

### 共享 expert layout

`slime/backends/qwen3_5_expert_layout.py` 是离线 converter 和在线 exporter 的共同实现，负责：

- 识别 HF `gate_up_proj` / `down_proj` fused key；
- 识别 Megatron `linear_fc1` / `linear_fc2` grouped-expert key；
- 将 `[E, 2F, H]` 拆成逐 expert gate/up `[F, H]`；
- 将 `[E, H, F]` 拆成逐 expert down `[H, F]`；
- 编解码 EP rank 的第一个 global expert id。

这层不能删除：若分别内联到离线和在线路径，两套命名及切分逻辑会重新出现并可能漂移。

### EP global expert id

Qwen3.5 grouped expert 的参数名没有本地 expert 编号。EP rank `r` 对应的起始编号为：

```text
first_expert_id = r * num_experts / ep_size
```

切分不能发生在 collective 之前，否则 tensor slice 会丢失 Megatron Parameter 上的 TP metadata。当前 `ParamInfo` interface 又没有 expert offset 字段，因此 offset 临时编码进内部名字，在 exporter 完成切分后移除；该后缀不会发给 SGLang。

### 离线与在线一致性

两条路径最终必须产生相同的名字与二维 shape：

```text
model.language_model.layers.L.mlp.experts.E.gate_proj.weight
model.language_model.layers.L.mlp.experts.E.up_proj.weight
model.language_model.layers.L.mlp.experts.E.down_proj.weight
```

量化后变为 `weight_packed`、`weight_scale` 和 `weight_shape`。初始 checkpoint 写入的 ignore rules 也用于每轮在线更新，从而保证只有 routed experts 被量化。

## 主线迁移

迁移没有直接恢复旧 mbridge 代码，而是落到主线已有 interface：

- 原生 HF→Megatron loader 初始化 BF16 actor；
- `named_params_and_buffers()` 负责 PP/EP 全局命名；
- `qwen3_5.py` 负责 Megatron→HF；
- `quantizer_compressed_tensors.py` 负责在线 INT4 pack；
- slime 的 SGLang patch 提供在线权重 restore/repack。

启动脚本中 `--hf-checkpoint` 指向 INT4 rollout checkpoint；`--ref-load` 指向原始 BF16 HF checkpoint，仅作为首次 actor 初始化回退；`--load` / `--save` 指向 Megatron 训练 checkpoint。

## 2026-09-04 消融结果

以“Qwen3.5-35B-A3B、EP=4、ETP=1 可以完成离线转换和在线同步”为不变量，删除以下设计：

| 删除项 | 原因 | 结果 |
|---|---|---|
| 转换前扫描所有 safetensors key/shape | config 已经精确限定模型，转换时仍会检查 fused name 和 3D shape | 少一次完整 checkpoint 遍历 |
| `_is_qwen35_fused_expert()` 等一次性包装函数 | 没有形成独立 interface，只转发一次调用 | 逻辑直接留在 converter |
| 合并通用和 Qwen3.5 ignore rules | Qwen3.5 精度范围固定，不需要可变组合 | 输出 config 使用固定 Qwen3.5 rules |
| 重复的 `conv1d`、`shared_experts` rule | 已分别被 `linear_attn`、`shared_expert` 覆盖 | 行为不变 |
| EP offset 的负数保护 | offset 只由非负 EP rank 公式产生 | 删除不可达分支 |
| 独立 style/test-fix 历史 | 属于提交组织噪音 | 重整时并入对应功能提交 |

以下设计经删除测试后保留：

- 共享 layout 模块：有离线、在线两个真实消费者；
- `model_type + architectures` 精确 gate：避免其他同名 MoE tensor 进入专用路径；
- EP offset：否则不同 EP rank 都从 expert 0 开始并互相覆盖；
- ETP=1 限制：当前没有验证 fused expert 的 ETP 分片轴；
- SGLang `post_process_weights`：compressed-tensors 在线更新必须 restore/load/repack。

## 验证边界

本地可完成 Python compile、shell syntax、patch syntax、静态调用链和 pre-commit。完整验收仍需在 H20 环境完成：

1. 转换后的 checkpoint 只有 routed experts 含 INT4 packed tensors；
2. 256 个 global expert id 完整且无 EP 覆盖；
3. 第一次在线权重更新通过 `post_process_weights`；
4. 至少完成两个 rollout→train→update iteration；
5. reward、loss、gradient 无 NaN/Inf；
6. checkpoint 可以恢复训练。
