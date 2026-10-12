# Prism 分卡 GRPO 与多 domain MOPD

这套示例使用原 `train.py`、Ray placement、rollout 分组、Megatron optimizer 和 checkpoint。学生生成仍走 Omni Python router；Megatron 在每批更新前重放轨迹并固定旧学生 logprob。Omni 返回的逐动作分数用于对齐检查，默认不作为 PPO 分母。

首版固定 BF16、TP=PP=CP=1、关闭 sequence parallel，支持 learner DP 和多个完整 rollout 副本。Prism 的全部 RVQ 与 sampled stop 参与训练。模型输入按真实时间行 packing，策略目标按真实动作计算；自然结束的 terminal stop 与零帧 stop 不会丢失。

第一版保留原 Megatron 计算核，旧学生分数以 Megatron 重算为默认。训练持续报告逐动作平均绝对误差，以及 `mean_sample_train_rollout_logprob_max_abs_diff`：每条轨迹先取真实动作的最大误差，再由原 microbatch/DP 报表平均，因而该值不是全 DP 的 global max。两个来源的 RVQ/stop 分数均保留在训练 payload 和可选 debug dump 中。严格的跨引擎精度对齐作为独立工作处理；GPU 工程闭环通过不代表满足任意更紧的 logprob 容差。

实际原 `train.py` 的 GPU 训练、保存、恢复与多副本验收范围及报告位置见 [VALIDATION.md](VALIDATION.md)。其中工程奖励与真实质量奖励的验收范围分别写明。

| 训练方式 | 原生训练路径 | 额外输入 |
| --- | --- | --- |
| GRPO | prompt 组奖励优势 → native PPO clipping | WER、SIM、全局指令奖励服务 |
| MOPD | native OPD advantage → native PPO clipping | 按 domain 路由的冻结 Prism 教师 |

MOPD 的 RVQ 和 stop 都使用学生实际选择动作的 logprob，不使用完整 stop KL。每条语音的目标归约为 `mean(RVQ losses) + STOP_LOSS_WEIGHT × mean(stop losses)`。这采用 slime 的 OPD/PPO 目标，不要求重现旧 VoxRL 的不同蒸馏 surrogate。

## 解释器与资产

需要两个可独立选择的 Python 环境：

- `TRAIN_PYTHON`：当前 slime、Ray、PyTorch、Megatron Core/training、Transformer Engine、FlashAttention 和对应训练依赖。
- `OMNI_PYTHON`：当前 SGLang-Omni、Prism pipeline 和 Omni Python router。由 Ray actor 用这个解释器启动完整 pipeline。

训练与 Omni 的 NCCL、CUDA 和 PyTorch 组合需要能够互相传输 GPU tensor。所有 Ray 节点均须可读取模型、pipeline YAML、数据和音频引用。若 Megatron 未安装到训练环境，可用 `MEGATRON_ROOT` 指定其源码目录；脚本将其加入训练 workers 的 `PYTHONPATH`。

新运行需要：

```bash
export TRAIN_PYTHON=/path/to/training-env/bin/python
export OMNI_PYTHON=/path/to/omni-env/bin/python
export HF_CHECKPOINT=/assets/prism-student-hf
export PRISM_PRETRAINED_CHECKPOINT=/assets/prism-native-dcp
export CODEC_MODEL=/assets/moss-audio-tokenizer-v2
export PROMPT_DATA=/data/prism-train.jsonl
export OUTPUT_DIR=/runs/prism-grpo
# 可选：固定 native iteration；未指定时使用 importer 的 checkpoint 选择规则。
export PRISM_PRETRAINED_STEP=100000
```

HF 制品与 native DCP 必须对应同一个学生结构、执行计划、tokenizer/codec 和参数精度。HF 制品提供 Omni 权重、processor、prompt protocol 和模型配置；native DCP initializer 只加载预训练模型，RL optimizer 从新状态开始。恢复 slime RL checkpoint 时使用 `RL_LOAD`，保留原 optimizer、scheduler 和训练进度。

## JSONL 输入

脚本使用 `--input-key script --metadata-key metadata`，不应用文本 chat template。实际 prompt 由 Omni processor 渲染。参见 [prompts.example.jsonl](prompts.example.jsonl)：

```json
{
  "script": "你好，欢迎来到今天的节目。",
  "metadata": {
    "domain": "chinese",
    "omni_prompt": {
      "task_type": "Full VoiceClone",
      "show_language": false,
      "language": "zh",
      "transcript_target": "你好，欢迎来到今天的节目。",
      "global_instruction": {"Gender": "female"},
      "references": [{"id": "audio1", "uri": "/assets/reference-zh.wav"}],
      "reference_targets": {"timbre": "audio1"}
    }
  }
}
```

每行必须显式提供 `task_type` 和布尔 `show_language`。`references` 的 `id` 对应实际参考 URI；使用 SIM 时，`reference_targets.timbre` 必须明确指向音色参考。`transcript_target` 与 `language` 供 WER/CER 客户端使用，不传给 ASR 解码器。

启用 JudgeCLAP 时，`global_instruction` 是实际交给 processor 的全局属性 object 或其 JSON 字符串。属性必须同时属于实际 Prism renderer 与 JudgeCLAP 支持的英文规范类别。第一版只评分全局属性，不把全局分数用于局部片段指令。空属性或普通自由文本不能作为这条 JudgeCLAP 路径的评分目标。

MOPD 使用 `metadata.domain` 选择教师。domain 的采样比例由数据记录比例及数据源决定，不由教师响应速度决定。显式 domain 没有教师映射时会报错。

## Ray 资源

以下单机例子保留 GPU 0 给 learner、GPU 1 给一个完整学生 pipeline。教师和奖励服务使用其他 GPU 或其他机器，并在 Ray 管理的学生资源之外运行。

```bash
CUDA_VISIBLE_DEVICES=0,1 "$TRAIN_PYTHON" -m ray.scripts.scripts start \
  --head --num-gpus=2 --num-cpus=16 \
  --dashboard-port=8265 --disable-usage-stats
export RAY_ADDRESS=auto
export ACTOR_GPUS_PER_NODE=1
export ROLLOUT_GPUS=1
export GPUS_PER_NODE=2
```

Ray 默认 worker 的 Python 由启动 head/node 的解释器决定，因此使用上面的 `TRAIN_PYTHON` 模块入口启动；`PATH` 中的裸 `ray` 命令可能属于另一个环境。多节点时，各节点的 Ray 进程也须从兼容的训练环境启动。已有集群的 workers 必须具备同一训练环境所需的 Python 依赖；如果不匹配，应另建专用训练 cluster，脚本不会自动停止或替换现有集群。`OMNI_PYTHON` 仅用于独立启动 Omni pipeline 与 router，不能修复 learner worker 的环境。

脚本连接已有 Ray cluster，不停止现有进程。一个 rollout GPU 对应一个完整 Prism pipeline；该 pipeline 内 preprocessing、AR 和 vocoder 共用这一张 rollout GPU。训练 GPU 与 rollout GPU 分离。

增加学生副本时设置 `ROLLOUT_GPUS=2`，并让 Ray 提供 learner 加上两个 rollout GPU。生成仍调用一个 router，权重通过 NCCL 安装到两个 AR 接收者，全部安装成功后才开放下一轮生成。`ACTOR_NODES × ACTOR_GPUS_PER_NODE` 是 learner DP 大小；`GLOBAL_BATCH_SIZE` 必须能被它整除。跨节点时显式填写每节点的 `GPUS_PER_NODE`。

## GRPO

先启动符合 `voxrl.scoring.v1` 的冻结服务。这里沿用现有评分协议：Qwen3-ASR-1.7B 提供转写，ReDimNet2-B6 提供 speaker embedding，JudgeCLAP adapter 与其 Qwen2.5-Omni 基座提供全局属性评分。服务部署和模型环境独立于 learner；启用的服务需要 `/info`，以及对应的 `/score` 或 `/embed`。

`configure.py` 从已运行服务的 `/info` 读取并固定 fingerprint，生成当前 run 的明确配置：

```bash
export RUN_CONFIG=/runs/prism-grpo-config
"$TRAIN_PYTHON" -m examples.prism.configure \
  --student "$HF_CHECKPOINT" --codec "$CODEC_MODEL" --output "$RUN_CONFIG" \
  --reward wer http://asr-host:8101 0.6 \
  --reward sim http://sim-host:8102 0.2 \
  --reward judge_clap http://judge-host:8103 0.2
export OMNI_CONFIG="$RUN_CONFIG/student.yaml"
export REWARD_CONFIG="$RUN_CONFIG/rewards.yaml"
export AUDIO_DIR=/shared/prism-grpo-audio
NUM_ROLLOUT=1 bash examples/prism/run-grpo.sh
```

上述权重只是示例，应按实验显式选择。只用 WER 与 SIM 时，省去 `--reward judge_clap ...`，无需提供可评分指令。配置 schema 参见 [rewards.example.yaml](rewards.example.yaml)。

`audio_transport: path` 要求评分服务能读取生成音频和 SIM 参考音频的相同绝对路径。服务不共享文件系统时，可在生成配置时加 `--audio-transport upload`；rollout/评分客户端仍需能够读取本地参考文件。评分服务故障会抛出错误，不会静默变成 reward 0。零帧 sampled stop 是合法轨迹，音频奖励为明确的无音频结果，stop 仍参与组优势训练。

GRPO 默认每个 prompt 生成四条、每批两个 prompt。可设置 `PROMPTS_PER_BATCH`、`SAMPLES_PER_PROMPT` 和 `GLOBAL_BATCH_SIZE`。组优势在完整 prompt group 上计算后再进入 DP 分片。

## 多 domain MOPD

本示例把教师作为外部、冻结的 score-only Omni 服务管理。分别启动两个教师，例如在两个独立终端：

```bash
export TEACHER_ZH=/assets/prism-teacher-zh
export TEACHER_EN=/assets/prism-teacher-en

CUDA_VISIBLE_DEVICES=2 "$OMNI_PYTHON" -m sglang_omni.cli serve \
  --config examples/prism/teacher-score.yaml --model-path "$TEACHER_ZH" \
  --host 0.0.0.0 --port 8201

# 另一个终端
CUDA_VISIBLE_DEVICES=3 "$OMNI_PYTHON" -m sglang_omni.cli serve \
  --config examples/prism/teacher-score.yaml --model-path "$TEACHER_EN" \
  --host 0.0.0.0 --port 8202
```

待教师健康后，在训练环境运行：

```bash
export RUN_CONFIG=/runs/prism-mopd-config
export OUTPUT_DIR=/runs/prism-mopd
"$TRAIN_PYTHON" -m examples.prism.configure \
  --student "$HF_CHECKPOINT" --codec "$CODEC_MODEL" --output "$RUN_CONFIG" \
  --teacher zh_teacher http://teacher-host:8201 "$TEACHER_ZH" \
  --teacher en_teacher http://teacher-host:8202 "$TEACHER_EN" \
  --domain chinese zh_teacher --domain english en_teacher
export OMNI_CONFIG="$RUN_CONFIG/student.yaml"
export TEACHER_CONFIG="$RUN_CONFIG/teachers.yaml"
NUM_ROLLOUT=1 bash examples/prism/run-mopd.sh
```

教师配置 schema 参见 [teachers.example.yaml](teachers.example.yaml)。`configure.py` 从本地教师 artifact 构造 `model_identity`，核验服务返回的身份，并固定 `/model_info` 中的 `teacher_weight_sha256`。该值是教师启动时对已加载 tensor 计算的 checksum，不是某个 safetensors 文件的 hash；训练期间不逐步重新 hash 全模型。存在认证时，可用 `--teacher-auth-token-env ENV_NAME` 把环境变量名称写入配置；运行 rollout hook 的进程需要有该变量。

教师可以比学生大，但必须有兼容的 Prism codec、动作空间、特殊 token、tokenizer 和 replay 语义。教师重放的是学生已经生成的相同 typed prompt 与动作前缀。MOPD 默认每个 prompt 一条轨迹、关闭波形输出，使用示例 `zero_reward` hook 提供零 task reward；原生 OPD 将教师相对旧学生的 selected-action 分数加入 advantage。`OPD_KL_COEF` 控制该项权重。

固定 `endpoint` 指向实际 score worker，直接调用 `/score_actions`。学生 router 的 `/generate` 负载均衡继续沿用。

也可以交给 slime 管理教师的 Ray 资源与生命周期。复制并填写 [managed-deployment.example.yaml](managed-deployment.example.yaml)，使用一张学生 rollout GPU、两张教师 GPU；两个教师必须设置 `update_weights: false` 并使用 score-only pipeline。教师绑定中用 `deployment: teacher_zh` 替代 `endpoint`，两者必须恰选一个。hook 从该 named router 的 `/workers` 找到可用冻结副本，再直连实际 worker 的 `/score_actions`，不会向 router 发送评分请求。

需要事先固定教师身份与已加载 tensor checksum。可对已启动、使用相同制品和 score 配置的教师进行一次显式检查，并让配置工具写入 managed 字段：

```bash
"$TRAIN_PYTHON" -m examples.prism.configure \
  --student "$HF_CHECKPOINT" --codec "$CODEC_MODEL" --output /runs/prism-managed-config \
  --teacher zh_teacher http://teacher-host:8201 "$TEACHER_ZH" \
  --teacher en_teacher http://teacher-host:8202 "$TEACHER_EN" \
  --domain chinese zh_teacher --domain english en_teacher \
  --deployment zh_teacher teacher_zh --deployment en_teacher teacher_en
```

把 managed deployment 模板中的 `omni_config` 和模型路径填为实际资产，确保 Ray 独占其管理的 GPU，再启动：

```bash
export OMNI_CONFIG=/runs/prism-managed-config/student.yaml
export TEACHER_CONFIG=/runs/prism-managed-config/teachers.yaml
export ROLLOUT_GPUS=3  # 包含学生与两个冻结教师
export GPUS_PER_NODE=4 # 再加一张 learner GPU
bash examples/prism/run-mopd.sh --sglang-config /runs/prism-managed-deployment.yaml
```

多个教师副本沿用相同绑定；hook 按 sample/group 编号确定性选择健康可用 worker。训练过程中不会自动信任新的教师 checksum，返回身份和固定绑定不一致时仍会报错。

## 恢复、配置检查与首轮验收

恢复时保持同一算法、采样和服务配置，使用 slime 的完整 RL checkpoint：

```bash
export RL_LOAD="$OUTPUT_DIR"
bash examples/prism/run-grpo.sh
# 或相应的 run-mopd.sh
```

`NUM_ROLLOUT` 沿用原框架的总 rollout 上限，续训时需要大于 checkpoint 中已经完成的 rollout。恢复 `iter_0000000` 时，下一次训练为 rollout 1，恢复后的首次完整学生发布为版本 2；下一次更新后发布版本 3。已有更高 serving/recovery 版本会保留。新模型初始化仍从 rollout 0 与首次发布版本 1 开始。

`PRISM_DRY_RUN=1 bash examples/prism/run-grpo.sh` 会检查必需输入并打印实际命令，不创建 GPU 作业。可在脚本末尾附加原 slime 参数，例如 `--save-debug-train-data '/runs/debug/train-{rollout_id}.pt'`。示例使用同步 `train.py` 和 `flush_cache_interval=1`；未启用 colocate、rollout offload、TIS、reference KL、GSPO 或完整 stop KL。

首个实际 GPU 验收应固定少量 prompt、短帧预算和 `NUM_ROLLOUT=1`，检查 native checkpoint 加载、所有 RVQ/stop 重放分数、梯度及完整权重发布，再增加 rollout 副本进行同样验证。CPU 接缝测试支持绑定 HF 配置而不加载权重：

```bash
SLIME_TEST_PRISM_HF_CHECKPOINT="$HF_CHECKPOINT" "$TRAIN_PYTHON" -m pytest \
  tests/test_prism_rollout_integration.py tests/test_policy_actions.py \
  tests/test_prism_model.py -q
```

仓库提供独立 GPU 工程验收入口。它先加载 native 学生并发布版本 1，再通过 Omni router 生成真实轨迹，核对更新前 logprob，执行一次 native policy objective/Megatron Adam 更新，发布版本 2，并核对更新后生成动作的 logprob。下面使用 learner GPU 0 和 Omni GPU 1；把 `--gpus` 改为 `1,2` 可在三张以上 GPU 上验证两个完整副本：

```bash
cat > /runs/prism-smoke-options.json <<'JSON'
{"alignment_atol": 0.05, "alignment_rtol": 0, "learner_max_frames": 8, "learning_rate": 0.001}
JSON

"$TRAIN_PYTHON" -m scripts.test_prism_omni \
  --hf-checkpoint "$HF_CHECKPOINT" --config "$OMNI_CONFIG" \
  --python "$OMNI_PYTHON" --gpus 1 \
  --artifact-dir /runs/prism-publication-smoke \
  --native-checkpoint "$PRISM_PRETRAINED_CHECKPOINT" \
  --mode publication \
  --publication-hook scripts.prism_publication_smoke:publication \
  --post-generation-hook scripts.prism_publication_smoke:verify_generated \
  --publication-args-json /runs/prism-smoke-options.json \
  --expected-version 2 --max-frames 8
```

`0.05` 是这里显式选定的 BF16 数值容差示例；结果会分别报告 RVQ 与 stop 的最大和平均误差。该工程检查使用一个明确的非零 advantage 来验证反向和权重更新，不调用实际 reward model，也不替代完整 prompt group 的 GRPO 或 domain 教师训练验收。独立脚本直接使用 engine 控制方法，没有执行 Ray placement；实际训练的资源生命周期仍由 `train.py` 管理。

`--mode static` 可单独检查 HF artifact、pipeline、router 和真实 rollout 协议，省去两个 hook 及 native checkpoint。这个模式不发布 learner 权重，结果会明确记录这一点。原始 `/generate` JSON、typed replay、更新前后分数和清理状态均写入结果目录。
