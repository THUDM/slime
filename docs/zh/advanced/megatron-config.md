# Megatron Config：按角色覆盖训练参数

`--megatron-config-path` 是一个基于 YAML 的配置系统，用于在公共 Megatron CLI 参数之上，为不同训练角色追加覆盖。目前它主要用于 PPO 场景中的 actor / critic 配置。

与 `--sglang-config` 不同，`--megatron-config-path` 不负责部署、路由或 GPU 资源编排；它只负责决定“这个角色最终使用哪些训练参数”。

---

## 设计概览

默认情况下（不使用 `--megatron-config-path`），actor 和 critic 都直接继承命令行中的 Megatron / slime 参数。

使用 `--megatron-config-path` 后，配置会分成两层：

- **公共 CLI 参数**：定义共享的 Megatron 拓扑、资源申请和默认训练参数。
- **角色 YAML 覆盖**：只覆盖 actor / critic 之间真正不同的配置项。

**核心设计原则：**

- **CLI 是公共基线。** slime 会先解析普通命令行参数，再应用 YAML 中的角色覆盖。
- **缺失角色自动继承。** 如果某个 role 没有出现在 YAML 中，该角色会直接继承 CLI 参数。
- **资源申请仍由 CLI 控制。** YAML 中的 `num_nodes` 和 `num_gpus_per_node` 会被忽略；资源分配仍由 `--actor-num-*` / `--critic-num-*` 决定。

---

## 配置格式

配置文件是一个 YAML 文档，顶层 `megatron` 键包含一个角色定义列表：

```yaml
megatron:
  - name: default
    role: actor
    overrides:
      lr: 1e-6
      save: /path/to/actor_ckpt
  - name: default
    role: critic
    overrides:
      lr: 1e-5
      save: /path/to/critic_ckpt
```

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 可选 | 配置项标签。当前运行时不会依赖它，建议保留为 `default`，方便未来扩展。 |
| `role` | `str` | **必填** | 角色名。目前支持 `actor` 和 `critic`。 |
| `overrides` | `dict` | `{}` | 该角色的参数覆盖。会在公共 CLI 参数之上应用。 |
| `args` | `dict` | `{}` | `overrides` 的向后兼容别名。新配置建议统一使用 `overrides`。 |

> **注意：** `overrides` 中的 key 使用 argparse 属性名，而不是命令行 flag 名。例如写 `tensor_model_parallel_size`，而不是 `tensor-model-parallel-size`。

---

## 使用方式

一个典型的 PPO 用法如下：

```yaml
# megatron_ppo.yaml
megatron:
  - name: default
    role: actor
    overrides:
      lr: 1e-6
  - name: default
    role: critic
    overrides:
      lr: 1e-5
```

```bash
python train.py \
  --advantage-estimator ppo \
  --use-critic \
  --megatron-config-path megatron_ppo.yaml \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 8 \
  ...
```

在这个模式下：

- CLI 负责共享的并行策略和资源配置；
- YAML 负责 actor / critic 的差异项，比如 `lr`、`load`、`save`、optimizer 或 scheduler 相关参数。

### 只覆盖一个角色

你也可以只为一个角色写覆盖，另一个角色自动继承公共 CLI 参数。例如只调整 critic 学习率：

```yaml
megatron:
  - name: default
    role: critic
    overrides:
      lr: 1e-5
```

这时 actor 会继续直接使用命令行中的公共参数。

---

## 当前限制

- **目前只支持 PPO。** `--megatron-config-path` 当前主要用于 PPO 工作流中的 actor / critic 角色配置。对于 GRPO、REINFORCE++ 等不依赖 critic 的流程，目前不建议使用这套角色配置。
- **当前 PPO 下，actor 和 critic 的 Megatron 并行配置必须一致。** 特别是 `tensor_model_parallel_size`、`pipeline_model_parallel_size`、`context_parallel_size`、`expert_model_parallel_size`、`expert_tensor_parallel_size`、`sequence_parallel` 等拓扑相关参数，不应在 actor 和 critic 之间配置成不同的值。
- **推荐把并行相关参数继续放在 CLI 中。** 当前最稳妥的用法是：并行与资源参数写在公共 CLI 中，只在 YAML 中覆盖角色差异项，例如 `lr`、`load`、`save`、warmup、optimizer / scheduler 参数等。

如果你在 actor 和 critic 之间写入不同的并行拓扑，当前行为不受支持，可能导致初始化或训练过程出错。

---

## FAQ

### Q: 可以只写 actor 或只写 critic 吗？

可以。缺失角色会自动继承公共 CLI 参数，不需要把所有参数都重复写一遍。

### Q: 可以把 `--actor-num-nodes` 或 `--critic-num-gpus-per-node` 写进 YAML 吗？

不可以。当前资源分配和 placement group 仍由 CLI 参数控制，YAML 中对应字段会被忽略。