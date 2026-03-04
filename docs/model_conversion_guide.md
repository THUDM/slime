# InternVL Model Format Conversion for SLIME + SGLang

## 问题

SLIME 训练使用 HuggingFace 格式，SGLang 推理需要特定格式。两者权重命名不同，特别是：
1. Vision tower 命名：`vision_tower` vs `vision_model`
2. Projector 命名：`multi_modal_projector` vs `mlp1`
3. QKV 权重：HF 分离 (q_proj, k_proj, v_proj)，SGLang 合并 (qkv)

## 解决方案

### 方案 1: 训练后转换（推荐）

在每次保存 checkpoint 后，自动转换为 SGLang 格式。

#### 步骤：

1. **添加转换脚本到 SLIME**
   - 已创建：`slime/utils/model_converter.py`
   - 包含 `convert_hf_checkpoint_to_sglang()` 函数

2. **修改训练流程**

在 `train.py` 的 `save()` 函数中添加转换：

```python
def save(rollout_id):
    # 原有的保存逻辑
    if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
        actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)

    # 新增：转换为 SGLang 格式
    if args.convert_to_sglang and dist.get_rank() == 0:
        from slime.utils.model_converter import convert_hf_checkpoint_to_sglang

        hf_ckpt_dir = f"{args.save}/iter_{rollout_id+1:07d}/model"
        sglang_ckpt_dir = f"{args.save}/iter_{rollout_id+1:07d}/model_sglang"

        convert_hf_checkpoint_to_sglang(
            hf_checkpoint_dir=hf_ckpt_dir,
            sglang_save_dir=sglang_ckpt_dir,
            original_sglang_model_dir=args.sglang_model_path
        )
```

3. **添加命令行参数**

在 `slime/utils/arguments.py` 中添加：

```python
parser.add_argument(
    "--convert-to-sglang",
    action="store_true",
    help="Convert checkpoints to SGLang format after saving"
)
parser.add_argument(
    "--sglang-model-path",
    type=str,
    default=None,
    help="Path to original SGLang model (for config/tokenizer)"
)
```

4. **更新 SGLang 权重**

修改 `update_weights()` 使用转换后的 SGLang 格式权重。

### 方案 2: 修改 SGLang 支持 HF 格式（已部分完成）

我们已经修改了 `sglang/python/sglang/srt/models/internvl.py` 的 `load_weights()` 方法，添加了名称映射。

但 QKV 权重合并比较复杂，需要缓存 q/k/v 然后 concat。

## 使用你的转换脚本

你已经有完整的转换脚本，可以直接使用：

```bash
# 训练后手动转换
python your_convert_script.py \
    --custom_path /path/to/original/sglang/model \
    --hf_path /path/to/trained/checkpoint \
    --save_path /path/to/sglang/checkpoint

# 然后重启 SGLang 服务加载新权重
bash scripts/start_sglang_internvl.sh --model /path/to/sglang/checkpoint
```

## 推荐流程

1. **初始化**：使用原始 SGLang 格式模型启动服务
2. **训练**：SLIME 训练保存 HF 格式 checkpoint
3. **转换**：每次 checkpoint 后自动转换为 SGLang 格式
4. **更新**：通知 SGLang 重新加载转换后的权重

## 待办事项

- [ ] 将你的转换脚本集成到 `slime/utils/model_converter.py`
- [ ] 修改 `train.py` 添加自动转换逻辑
- [ ] 添加命令行参数 `--convert-to-sglang`
- [ ] 实现 SGLang 动态重新加载权重的机制
- [ ] 测试完整的训练-转换-推理流程
