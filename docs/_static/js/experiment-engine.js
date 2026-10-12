/* Configuration and shell generation. No DOM, network, or runtime dependencies. */
(function(root) {
  'use strict';
  const models = {
    qwen38: {
      name: 'Qwen3.8-27B',
      mtp: true,
      size: '27B · Hybrid Dense',
      repo: 'Qwen/Qwen3.8-27B',
      model: 'qwen3.5-27B',
      recipe: 'qwen3.5-27B',
      nodes: 4,
      engine: 2,
      tp: 4,
      pp: 2,
      cp: 4,
      ep: 1,
      dp: 1,
      moe: false,
      hybrid: {
        linear: 'Gated DeltaNet',
        linearLayers: 48,
        attentionLayers: 16,
        ssmDtype: 'float32'
      },
      maxTokens: 8192,
      extra: ['--decoder-last-pipeline-num-layers 30']
    },
    deepseek: {
      name: 'DeepSeek-R1',
      mtp: true,
      size: '671B · A37B · MoE',
      repo: 'deepseek-ai/DeepSeek-R1',
      model: 'deepseek-v3',
      recipe: 'deepseek-r1',
      nodes: 16,
      engine: 64,
      tp: 8,
      pp: 4,
      cp: 4,
      ep: 32,
      dp: 8,
      moe: true,
      maxTokens: 8192,
      extra: ['--decoder-last-pipeline-num-layers 13']
    },
    flash: {
      name: 'GLM-4.7-Flash',
      mtp: true,
      size: '30B · A3B · MoE',
      repo: 'zai-org/GLM-4.7-Flash',
      model: 'glm4.7-30B-A3B',
      recipe: 'glm4.7-30B-A3B',
      nodes: 1,
      engine: 8,
      tp: 2,
      pp: 2,
      cp: 2,
      ep: 4,
      dp: 8,
      moe: true,
      maxTokens: 8192,
      extra: ['--decoder-last-pipeline-num-layers 23']
    },
    dense: {
      name: 'GLM-4-9B',
      size: '9B · Dense',
      repo: 'THUDM/GLM-Z1-9B-0414',
      model: 'glm4-9B',
      recipe: 'glm4-9B',
      nodes: 1,
      engine: 2,
      tp: 2,
      pp: 1,
      cp: 2,
      ep: 1,
      dp: 1,
      moe: false,
      maxTokens: 4608,
      extra: []
    },
    glm47: {
      name: 'GLM-4.7',
      mtp: true,
      size: '355B · A32B · MoE',
      repo: 'zai-org/GLM-4.7',
      model: 'glm4.5-355B-A32B',
      recipe: 'glm4.7-355B-A32B',
      nodes: 8,
      engine: 32,
      tp: 8,
      pp: 4,
      cp: 2,
      ep: 16,
      dp: 4,
      moe: true,
      maxTokens: 8192,
      extra: []
    },
    glm5: {
      name: 'GLM-5.3',
      mtp: true,
      size: '744B · A40B · MoE',
      repo: 'zai-org/GLM-5.3',
      model: 'glm5.2-744B-A40B',
      recipe: 'glm5.2-744B-A40B',
      nodes: 32,
      engine: 64,
      tp: 4,
      pp: 8,
      cp: 8,
      ep: 32,
      dp: 64,
      moe: true,
      maxTokens: 8192,
      extra: ['--decoder-first-pipeline-num-layers 14', '--decoder-last-pipeline-num-layers 16',
        '--data-pad-size-multiplier 1024', '--log-probs-chunk-size 16384'
      ]
    },
  };
  const defaults = {
    straw: false,
    partial: false,
    maskPartial: false,
    queueDir: '',
    queueRun: '',
    declaration: '',
    cpuAdam: true,
    recompute: true,
    model: 'flash',
    hardware: 'unspecified',
    task: 'math',
    layout: 'colocate',
    schedule: 'sync',
    precision: 'bf16',
    kv: 'auto',
    mambaDtype: 'auto',
    mambaRatio: '',
    correction: 'tis',
    r3: false,
    sc: false,
    deterministic: false,
    pd: false,
    hicache: false,
    speculative: false,
    draftSource: 'builtin',
    draft: '',
    specSteps: 3,
    nodes: 1,
    rollout: 8,
    prefill: 8,
    concurrency: 128,
    response: 4096,
    rounds: 3,
    transport: 'nccl',
    hf: '/data/GLM-4.7-Flash',
    train: '/data/GLM-4.7-Flash_torch_dist',
    data: '/data/dapo-math-17k/dapo-math-17k.jsonl',
    save: '/data/slime-experiment',
    megatron: '/root/Megatron-LM',
    endpoints: '',
    shared: '/shared/slime-weights',
    local: '/local/slime-checkpoint',
    ib: '',
    generate: '',
    reward: '',
    step: 0
  };
  const enums = {
    hardware: ['unspecified', 'hopper', 'blackwell'],
    model: Object.keys(models),
    task: ['math', 'custom'],
    layout: ['colocate', 'separate', 'external'],
    schedule: ['sync', 'async'],
    precision: ['bf16', 'fp8', 'fp8train', 'int4'],
    kv: ['auto', 'fp8'],
    mambaDtype: ['auto', 'float32', 'bfloat16'],
    draftSource: ['builtin', 'external'],
    correction: ['none', 'tis', 'icepop'],
    transport: ['nccl', 'disk', 'delta']
  };
  const numberLimits = {
    nodes: [1, 512],
    rollout: [1, 8192],
    prefill: [1, 8192],
    concurrency: [1, 8192],
    response: [128, 131072],
    rounds: [1, 100000],
    specSteps: [1, 8],
    step: [0, 5]
  };
  const textKeys = ['hf', 'train', 'data', 'save', 'megatron', 'endpoints', 'shared', 'local', 'ib',
    'generate',
    'reward', 'queueDir', 'queueRun', 'declaration', 'mambaRatio', 'draft'
  ];

  function sanitize(input) {
    if (input?.schema === 1 && input.choices) input = input.choices;
    const s = {
      ...defaults
    };
    if (input && Object.hasOwn(models, input.model)) {
      const m = models[input.model];
      Object.assign(s, {
        nodes: m.nodes,
        rollout: m.nodes * 8,
        prefill: m.engine,
        draftSource: m.mtp ? 'builtin' : 'external',
        specSteps: input.model === 'glm5' ? 4 : 3,
        hf: '/data/' + m.name,
        train: '/data/' + m.name + '_torch_dist'
      });
    }
    if (!input || typeof input !== 'object') return s;
    for (const [key, values] of Object.entries(enums))
      if (values.includes(input[key])) s[key] = input[key];
    for (const key of Object.keys(defaults).filter(k => typeof defaults[k] === 'boolean'))
      if (typeof input[key] === 'boolean') s[key] = input[key];
    for (const [key, [lo, hi]] of Object.entries(numberLimits))
      if (Number.isInteger(Number(input[key])) && Number(input[key]) >= lo && Number(input[key]) <= hi) s[key] =
        Number(input[key]);
    for (const key of textKeys)
      if (typeof input[key] === 'string' && input[key].length <= 2048) s[key] = input[key].replace(/[\x00-\x1f\x7f]/g,
        '').trim();
    if (!models[s.model].hybrid) {
      s.mambaDtype = 'auto';
      s.mambaRatio = '';
    }
    return s;
  }

  function change(state, key, value) {
    const s = {
      ...state,
      [key]: value
    };
    if (key === 'model') {
      const m = models[value];
      s.nodes = m.nodes;
      s.rollout = m.nodes * 8;
      s.prefill = m.engine;
      s.cpuAdam = m.moe || value === 'qwen38';
      s.hf = '/data/' + m.name;
      s.train = s.hf + '_torch_dist';
      s.kv = 'auto';
      s.mambaDtype = 'auto';
      s.mambaRatio = '';
      s.draftSource = m.mtp ? 'builtin' : 'external';
      s.draft = '';
      s.specSteps = value === 'glm5' ? 4 : 3;
      if (!m.moe) s.r3 = false;
      if (value === 'glm5' || value === 'deepseek') {
        s.precision = 'fp8';
        s.kv = 'fp8';
      }
    }
    if (key === 'partial' && !value) s.maskPartial = false;
    // Dependency changes remain one undoable action, announced by the UI.
    if (key === 'schedule' && value === 'async' && s.layout === 'colocate') s.layout = 'separate';
    if (key === 'layout' && value === 'colocate') {
      s.schedule = 'sync';
      s.transport = 'nccl';
    }
    if (key === 'layout' && value === 'external') s.transport = 'disk';
    return sanitize(s);
  }

  function preset(name) {
    if (name === 'scale') return {
      ...change(defaults, 'model', 'glm5'),
      pd: true,
      precision: 'fp8',
      kv: 'fp8',
      correction: 'icepop'
    };
    if (name === 'agent') return {
      ...defaults,
      task: 'custom',
      layout: 'separate',
      schedule: 'async',
      straw: true,
      partial: true,
      precision: 'fp8',
      pd: true,
      hicache: true,
      rollout: 16
    };
    return {
      ...defaults
    };
  }
  // Always quote user supplied strings as shell literals, never executable substitutions.
  const quote = value => "'" + String(value).replace(/'/g, "'\"'\"'") + "'";

  function validate(s, lang = 'en') {
    const m = models[s.model],
      errors = [],
      notes = [],
      issues = [];
    const t = (en, zh) => lang === 'zh' ? zh : en;
    const reject = (code, message, fields, actions = []) => {
      errors.push(message);
      issues.push({code, message, fields, actions});
    };
    const total = s.nodes * 8,
      rollout = s.layout === 'colocate' ? total : s.rollout;
    if (total % (m.tp * m.pp * m.cp) !== 0 || total % (m.ep * m.pp) !== 0) errors.push(t(
      'Training GPUs must be divisible by TP × PP × CP and EP × PP for this recipe.',
      '此模型的训练 GPU 数必须同时整除 TP × PP × CP 和 EP × PP。'));
    if (s.layout !== 'external' && rollout % m.engine !== 0) errors.push(t(
      `Rollout GPUs must be a multiple of ${m.engine} (GPUs per engine).`,
      `Rollout GPU 数必须是 ${m.engine}（每个引擎的 GPU 数）的倍数。`));
    if (s.layout !== 'external' && s.pd && (s.prefill % m.engine !== 0 || s.prefill >= rollout)) {
      const prefill = Math.ceil(s.prefill / m.engine) * m.engine;
      const needed = Math.max(rollout, prefill + m.engine);
      const changes = {prefill};
      if (s.layout === 'colocate') {
        // Find an integer node count satisfying this model's training AND serving groups.
        let nodes = Math.max(s.nodes, Math.ceil(needed / 8));
        while (nodes <= numberLimits.nodes[1] &&
          (nodes * 8 % (m.tp * m.pp * m.cp) || nodes * 8 % (m.ep * m.pp) || nodes * 8 % m.engine)) nodes++;
        changes.nodes = nodes;
      } else changes.rollout = Math.ceil(needed / m.engine) * m.engine;
      const actions = [{id: 'pd-off', label: t('Keep resources; turn PD off', '保留资源分配，关闭 PD'), changes: {pd: false}}];
      if (Object.entries(changes).every(([key, value]) => value <= numberLimits[key][1])) actions.push({
        id: 'pd-resources', label: t('Allocate complete prefill / decode engines', '分配完整的 prefill / decode 引擎'), changes
      });
      reject('pd-capacity', t(
        `PD needs complete prefill and decode engines, each using ${m.engine} GPUs in this recipe.`,
        `此配方的 PD 需要完整的 prefill 和 decode 引擎，每个引擎使用 ${m.engine} 张 GPU。`),
        ['prefill', s.layout === 'colocate' ? 'nodes' : 'rollout'], actions);
    }
    if (s.layout === 'colocate' && s.schedule === 'async') errors.push(t(
      'Fully async requires separate training and rollout GPUs.', 'Fully async 需要独立的训练与 rollout GPU。'));
    if (s.layout === 'colocate' && s.transport !== 'nccl') errors.push(t(
      'This colocated recipe uses full weight updates through CUDA IPC; choose the default transport.',
      '此同卡配方使用 CUDA IPC 全量更新，请选择默认传输。'));
    if (s.r3 && !m.moe) errors.push(t('R3 requires a MoE model.', 'R3 仅适用于 MoE 模型。'));
    if (s.deterministic && s.schedule === 'async') errors.push(t(
      'A fully async queue does not reproduce the same sample ordering. Use synchronous execution for this reproducibility recipe.',
      'Fully async 队列不能复现相同的样本顺序；此可复现配方请选择同步训练。'));
    for (const [key, en, zh] of [
        ['hf', 'HF checkpoint', 'HF checkpoint'],
        ['train', 'training checkpoint', '训练 checkpoint'],
        ['data', 'prompt data', 'prompt 数据'],
        ['save', 'output directory', '输出目录'],
        ['megatron', 'Megatron directory', 'Megatron 目录']
      ])
      if (!s[key]) reject('path-' + key, t(`Enter the ${en} path.`, `请填写${zh}路径。`), [key]);
    if (s.task === 'custom') {
      if (!/^\w+(?:\.\w+)+$/.test(s.generate)) errors.push(t(
        'Enter an importable generation hook, e.g. my_agent.generate.', '请填写可导入的生成 hook，例如 my_agent.generate。'));
      if (!/^\w+(?:\.\w+)+$/.test(s.reward)) errors.push(t('Enter an importable reward hook, e.g. my_reward.reward.',
        '请填写可导入的奖励 hook，例如 my_reward.reward。'));
    }
    if (s.layout === 'external') {
      if (!s.endpoints || !s.endpoints.split(/\s+/).every(v => /^(?:https?:\/\/)?[a-zA-Z0-9._-]+:\d{1,5}$/.test(v) &&
          Number(v.split(':').at(-1)) > 0 && Number(v.split(':').at(-1)) < 65536)) reject('endpoints', t(
        'Enter engine addresses as host:port separated by spaces (not a router URL).',
        '填写以空格分隔的引擎 host:port（不是 router 地址）。'), ['endpoints']);
      notes.push(t(
        'External SGLang is configured by its owner. Use the generated serving reference on every engine; slime discovers existing regular/prefill/decode roles and cannot switch them from the training CLI.',
        '外部 SGLang 由部署方配置。请将生成的 serving 参考应用到各引擎；slime 发现已有 regular/prefill/decode 角色，无法通过训练 CLI 改变它们。'));
    }
    if (s.transport !== 'nccl' && !s.shared) errors.push(t(
      'Disk transport needs a shared path visible to both clusters.', '磁盘传输需要两个集群都可见的共享路径。'));
    if (s.transport === 'delta' && !s.local) errors.push(t(
      'Delta transport needs a local checkpoint directory on every serving host.',
      'Delta 传输需要每台推理主机上的本地 checkpoint 目录。'));
    if (s.pd && !s.ib) reject('rdma', t(
      'Enter the RDMA device(s) for this Mooncake PD recipe, e.g. mlx5_0. Check these names on every host.',
      '此 Mooncake PD 配方需要 RDMA 网卡名，例如 mlx5_0；请核对每台主机。'), ['ib']);
    if (s.speculative) {
      if (s.draftSource === 'builtin' && !m.mtp) errors.push(t(
        'This model has no built-in MTP recipe here. Select a separate EAGLE head and provide its checkpoint.',
        '此模型没有内置 MTP 配方，请选择独立 EAGLE 头并填写 checkpoint。'));
      if (s.draftSource === 'external' && !s.draft) errors.push(t(
        'Enter the separate EAGLE head checkpoint path or Hugging Face model ID.',
        '请填写独立 EAGLE 投机头的 checkpoint 路径或 Hugging Face 模型 ID。'));
      notes.push(t(
        'EAGLE proposes tokens, then the target verifies them. Use a compatible draft head, with its weights available on every serving host. The builder configures inference only; separate draft-head training is not enabled. Track spec_accept_rate, spec_accept_length and rollout time as RL changes the target.',
        'EAGLE 先提出候选 token，再由目标模型验证。投机头须与目标模型匹配，且权重需供所有推理主机访问。向导只配置推理，不开启独立投机头训练。随 RL 更新，观察 spec_accept_rate、spec_accept_length 和 rollout 耗时。'));
    }
    if (s.straw) {
      if (!s.queueDir || !s.queueDir.startsWith('/')) errors.push(t(
        'straw needs an absolute shared JuiceFS data directory.', 'straw 需要绝对路径的共享 JuiceFS 数据目录。'));
      if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(s.queueRun)) errors.push(t(
        'Enter a straw run ID using letters, digits, dots, underscores or hyphens.',
        '填写 straw run ID，使用字母、数字、点、下划线或连字符。'));
      if (!s.declaration || !s.declaration.startsWith('/')) errors.push(t(
        'Enter the absolute path to your verified JuiceFS deployment declaration.', '填写已验证的 JuiceFS 部署声明文件绝对路径。'));
      notes.push(t(
        'straw persists payloads and queue state on shared JuiceFS. Every node needs the same mount and straw-queue package. With fully async it activates one generation process per eligible Ray node; reserve CPU resources and keep topology stable for recovery. The short recipe saves every round.',
        'straw 在共享 JuiceFS 保存数据和队列。各节点需相同挂载与 straw-queue 依赖；与 fully async 组合后，每个符合条件的 Ray 节点启动生成进程，需预留 CPU 并保持恢复时拓扑一致。短实验每轮保存。'
        ));
    }
    if (s.partial) notes.push(t(
      'Partial rollout preserves unfinished prefixes and resumes them later; the synchronous recipe oversamples 16 groups for a target of 8. A response can contain multiple weight versions. Masking old prefix tokens does not make the remaining prefixes on-policy.',
      'Partial rollout 保留未完成前缀，之后续跑；同步配方超采样 16 组、目标 8 组。一个回复可能混合权重版本；mask 旧 token 也不会让后续 prefix 自动变成 on-policy。'));
    if (s.maskPartial && !s.partial) errors.push(t('Prefix masking requires partial rollout.',
      '旧前缀 masking 需要启用 partial rollout。'));
    if (s.schedule === 'async') notes.push(t(
      'Rollouts continue across weight updates. Monitor staleness/mean and staleness/max; run evaluation separately from this continuous queue.',
      'Rollout 跨权重更新持续生成。监控 staleness/mean、staleness/max；在此连续队列之外单独运行评估。'));
    if (s.correction === 'none' && (s.schedule === 'async' || s.precision !== 'bf16')) notes.push(t(
      'You chose stale or quantized rollouts without importance correction. Track train_rollout_logprob_abs_diff and reward before scaling.',
      '当前使用过期或量化 rollout，但未启用重要性修正。扩容前检查 train_rollout_logprob_abs_diff 和 reward。'));
    if (s.model === 'qwen38') notes.push(t(
      'Qwen3.8-27B uses the qwen3_5 architecture and matches the existing 27B model dimensions. This text-RL recipe reuses that backend; the checkpoint still needs GPU validation.',
      'Qwen3.8-27B 使用 qwen3_5 架构，尺寸匹配现有 27B 配置。此文本 RL 配方复用该后端，checkpoint 仍需 GPU 验证。'));
    if (m.hybrid) {
      notes.push(t(
        'Hybrid cache has two pools: attention KV and recurrent state. KV dtype applies only to attention layers. Mamba SSM dtype and its memory ratio are independent; convolution state is separate again. Check backend support and log-prob accuracy for each change.',
        'Hybrid cache 分为 attention KV 和递归状态两个池。KV dtype 只作用于 attention 层；Mamba SSM 精度与内存比例独立，卷积状态又是另一部分。每项调整都需核对后端支持与 log-prob 精度。'
        ));
      if (s.mambaRatio && (!/^(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s.mambaRatio) ||
          !Number.isFinite(Number(s.mambaRatio)) || Number(s.mambaRatio) <= 0)) errors.push(t(
        'Mamba / attention KV memory ratio must be a positive number, or left blank for the serving default.',
        'Mamba / attention KV 内存比例须为正数，或留空沿用推理栈默认值。'));
    }
    if (s.sc) notes.push(t(
      'SC selects REINFORCE with GRPO advantages and no standard-deviation normalization. Sampling is fixed to temperature=1, top_p=1, top_k=-1, without penalties, constraints, or streaming.',
      'SC 使用 REINFORCE、GRPO advantage，关闭标准差归一化。采样固定为 temperature=1、top_p=1、top_k=-1，不使用惩罚、约束解码或 streaming。'));
    if (s.hicache) notes.push(t(
      'HiCache consumes host RAM as well as GPU cache. The recipe uses a host/device cache ratio of 2; PD enables it on prefill only. Cache reuse is limited by weight-update invalidation.',
      'HiCache 还会占用主机内存；配方的 host/device cache 比例为 2，PD 时只在 prefill 启用。权重更新后的缓存失效会限制复用收益。'));
    if (s.deterministic) notes.push(t(
      'Deterministic execution is not cross-engine equality. Install the deterministic SGLang/Megatron stack and remove FlashAttention 3 as documented. Exact GLM-5 alignment has a separate six-layer EP8 regression gate.',
      '确定性执行不等于训推数值相等。按文档安装 deterministic SGLang/Megatron 依赖并移除 FlashAttention 3。GLM-5 精确对齐另有六层 EP8 回归验证。'));
    if (s.cpuAdam) notes.push(t(
      'CPU Adam offloads optimizer work and state to host memory. Budget host RAM and CPU↔GPU bandwidth; it does not offload activations or KV cache.',
      'CPU Adam 将优化器计算与状态移到主机，需预算 CPU 内存和 CPU↔GPU 带宽；它不会卸载激活或 KV cache。'));
    if (!s.recompute) notes.push(t(
      'Activation recomputation is disabled. Retaining intermediate activations increases GPU memory use, especially for long sequences.',
      '已关闭激活重计算。保留中间激活会增加显存，长序列尤其明显。'));
    if (s.precision === 'fp8train') notes.push(t(
      'FP8 training is experimental. This recipe omits fp8-param-gather to keep the CPU Adam offload path compatible.',
      'FP8 训练为实验性功能；此配方不启用 fp8-param-gather，以兼容 CPU Adam offload。'));
    if (s.precision === 'int4') notes.push(t(
      'INT4 rollout is beta and has no maintained GLM end-to-end recipe here. Validate your model and installed kernels with a short run before scaling; training remains BF16.',
      'INT4 rollout 为 beta，仓库尚无维护中的 GLM 端到端配方。扩容前先用短实验验证模型和当前 kernel；训练仍为 BF16。'));
    if (s.hardware === 'blackwell') notes.push(t(
      'B200/B300 uses the CUDA 13 image, with different kernel builds. GPU CI primarily covers H100/H200; this hardware choice does not certify the selected model or memory fit.',
      'B200/B300 使用 CUDA 13 镜像，kernel 构建不同。GPU CI 主要覆盖 H100/H200；选择硬件不会认证此模型配方或显存容量。'));
    return {
      errors,
      issues,
      notes,
      total,
      rollout,
      localGpus: s.layout === 'colocate' ? total : s.layout === 'external' ? total : total + rollout
    };
  }

  function servingArgs(s) {
    const m = models[s.model],
      args = [`--rollout-num-gpus-per-engine ${m.engine}`, '--sglang-mem-fraction-static 0.7'];
    if (m.dp > 1) args.push('--sglang-enable-dp-attention', `--sglang-dp-size ${m.dp}`, '--sglang-enable-dp-lm-head',
      '--sglang-moe-dense-tp-size 1');
    if (['glm47', 'glm5', 'deepseek'].includes(s.model)) args.push(`--sglang-ep-size ${m.engine}`);
    if (m.hybrid) {
      args.push('--sglang-mamba-scheduler-strategy extra_buffer');
      if (s.mambaDtype !== 'auto') args.push(`--sglang-mamba-ssm-dtype ${s.mambaDtype}`);
      if (s.mambaRatio) args.push(`--sglang-mamba-full-memory-ratio ${quote(s.mambaRatio)}`);
    }
    if (s.model === 'deepseek') args.push('--sglang-moe-a2a-backend deepep', '--sglang-deepep-mode auto');
    if (s.model === 'glm5') args.push('--sglang-moe-a2a-backend deepep', '--sglang-deepep-mode auto',
      '--sglang-page-size 64', '--sglang-nsa-decode-backend flashmla_kv',
      '--sglang-nsa-prefill-backend flashmla_sparse', '--sglang-attention-backend nsa',
      '--sglang-disable-overlap-schedule');
    if (s.kv === 'fp8') args.push('--sglang-kv-cache-dtype fp8_e4m3');
    if (s.speculative) {
      args.push('--sglang-speculative-algorithm EAGLE', `--sglang-speculative-num-steps ${s.specSteps}`,
        '--sglang-speculative-eagle-topk 1', `--sglang-speculative-num-draft-tokens ${s.specSteps + 1}`);
      if (s.draftSource === 'external') args.push(`--sglang-speculative-draft-model-path ${quote(s.draft)}`);
      else if (s.model === 'glm5') args.push('--sglang-speculative-draft-attention-backend nsa');
    }
    if (s.deterministic) {
      args.push('--sglang-enable-deterministic-inference');
      if (s.model === 'dense') args.push('--sglang-attention-backend flashinfer');
    }
    if (s.hicache) args.push('--sglang-enable-hierarchical-cache', '--sglang-hicache-ratio 2');
    if (s.pd) args.push('--sglang-disaggregation-transfer-backend mooncake',
      `--sglang-disaggregation-ib-device ${quote(s.ib)}`);
    return args;
  }

  function yaml(s) {
    if (!s.pd || s.layout === 'external') return '';
    const v = validate(s),
      m = models[s.model];
    let out = `sglang:\n  - name: default\n    update_weights: true\n    server_groups:\n`;
    for (const [role, count] of [
        ['prefill', s.prefill],
        ['decode', v.rollout - s.prefill]
      ]) {
      out += `      - worker_type: ${role}\n        num_gpus: ${count}\n        num_gpus_per_engine: ${m.engine}\n`;
      const overrides = [];
      if (s.hicache) overrides.push(`enable_hierarchical_cache: ${role==='prefill'}`);
      if (s.model === 'glm5') overrides.push(`deepep_mode: ${role==='decode'?'low_latency':'auto'}`,
        `load_balance_method: ${role==='decode'?'round_robin':'follow_bootstrap_room'}`);
      if (s.model === 'glm5' && role === 'decode') overrides.push('moe_runner_backend: deep_gemm',
        'disable_overlap_schedule: true');
      if (overrides.length) out += '        overrides:\n' + overrides.map(a => '          ' + a).join('\n') + '\n';
    }
    return out;
  }

  function fieldLabel(key, lang = 'en') {
    const labels = {
      model: ['Model', '模型'], hardware: ['Hardware context', '硬件条件'], task: ['Learning signal', '训练信号'],
      layout: ['GPU placement', 'GPU 布局'], schedule: ['Schedule', '调度'], nodes: ['Training nodes × 8 GPU', '训练节点 × 8 GPU'],
      rollout: ['Rollout GPUs', 'Rollout GPU'], prefill: ['Prefill GPUs', 'Prefill GPU'],
      precision: ['Weight precision', '权重精度'], kv: ['Attention KV dtype', 'Attention KV 精度'],
      mambaDtype: ['SSM dtype', 'SSM 精度'], mambaRatio: ['State / KV memory ratio', '状态 / KV 内存比例'],
      correction: ['Behavior correction', 'Behavior 修正'], r3: ['R3', 'R3'], sc: ['SC', 'SC'],
      deterministic: ['Deterministic execution', '确定性执行'], cpuAdam: ['CPU Adam', 'CPU Adam'],
      recompute: ['Activation recomputation', '激活重计算'], pd: ['PD', 'PD'], hicache: ['HiCache', 'HiCache'],
      speculative: ['EAGLE', 'EAGLE'], draftSource: ['Draft head source', '投机头来源'], draft: ['Draft head path', '投机头路径'],
      specSteps: ['Draft steps', '草拟步数'], partial: ['Partial rollout', 'Partial rollout'],
      maskPartial: ['Mask old prefix', 'Mask 旧前缀'], straw: ['straw', 'straw'],
      transport: ['Weight transport', '权重传输'], concurrency: ['Concurrency', '并发数'], response: ['Response tokens', '回复 token 数'],
      rounds: ['Training rounds', '训练轮数'], hf: ['HF checkpoint', 'HF checkpoint'], train: ['Training checkpoint', '训练 checkpoint'],
      data: ['Prompt data', 'Prompt 数据'], save: ['Output path', '输出路径'], megatron: ['Megatron path', 'Megatron 路径'],
      endpoints: ['Engine addresses', '引擎地址'], shared: ['Shared weight path', '共享权重路径'], local: ['Serving checkpoint path', '推理 checkpoint 路径'],
      ib: ['RDMA devices', 'RDMA 网卡'], generate: ['Generation hook', '生成 hook'], reward: ['Reward hook', '奖励 hook'],
      queueDir: ['straw data path', 'straw 数据路径'], queueRun: ['straw run ID', 'straw run ID'], declaration: ['JuiceFS declaration', 'JuiceFS 声明']
    };
    return labels[key]?.[lang === 'zh' ? 1 : 0] || key;
  }

  function compare(a, b, lang = 'en') {
    a = sanitize(a);
    b = sanitize(b);
    const t = (en, zh) => lang === 'zh' ? zh : en;
    const differences = Object.keys(defaults).filter(key => key !== 'step' && a[key] !== b[key])
      .map(key => ({key, label: fieldLabel(key, lang), before: a[key], after: b[key]}));
    const changed = key => differences.some(item => item.key === key);
    const insights = [];
    if (changed('layout') || changed('transport')) insights.push(t(
      'Colocation reuses one GPU pool in alternating phases. Separate pools reserve training and rollout GPUs independently; external engine GPUs are outside the Ray allocation. Separate pools may still run synchronously. Compare idle time, weight-transfer time and total reserved GPUs.',
      '同卡布局在两个阶段复用 GPU；分离布局分别占用训练与 rollout GPU，外部引擎 GPU 不计入本地 Ray 分配。分离布局仍可同步执行。比较空闲时间、权重传输耗时和总占用 GPU。'));
    if (changed('schedule') || changed('partial') || changed('straw')) insights.push(t(
      'Fully async overlaps generation and training on separate GPUs, but mixes weight versions. Partial rollout saves unfinished prefixes; straw persists queue payloads and enables distributed producers with fully async. Measure staleness/mean, staleness/max, queue waits, rollout time and held-out reward.',
      'Fully async 在独立 GPU 上重叠生成与训练，但会混合权重版本。Partial rollout 保留未完成前缀；straw 持久化队列数据，搭配 fully async 启用分布式 producer。观察 staleness/mean、staleness/max、队列等待、rollout 耗时和独立评估 reward。'));
    if (['precision', 'kv', 'mambaDtype', 'mambaRatio'].some(changed)) insights.push(t(
      'BF16 → FP8 changes rollout weight storage and arithmetic; FP8 training also changes the trainer. Attention KV and recurrent state have independent budgets. Measure train_rollout_logprob_abs_diff, reward, rollout time and peak memory; bit width alone does not predict total memory or speed.',
      'BF16 → FP8 改变 rollout 权重存储与运算，FP8 training 还改变训练侧。Attention KV 与递归状态各有独立预算。观察 train_rollout_logprob_abs_diff、reward、rollout 耗时及显存峰值；位宽不能直接换算总显存或速度。'));
    if (changed('correction')) insights.push(t(
      'For the same illustrative ratio p/q = 0.4/0.1 = 4, TIS with upper bound 2 uses weight 2; ICE-POP outside [0.5, 2] uses 0. This is behavior correction, separate from PPO update clipping. Compare clipping/masking fractions, mismatch and held-out reward.',
      '同一示例 p/q = 0.4/0.1 = 4：TIS 上界 2 时权重为 2；ICE-POP 在 [0.5, 2] 外权重为 0。这是 behavior 修正，与 PPO 更新裁剪独立。比较 clipping/masking 比例、训推差异与独立评估 reward。'));
    if (['pd', 'prefill', 'hicache', 'speculative', 'draftSource', 'draft', 'specSteps'].some(changed)) insights.push(t(
      'PD separates prefill and decode with cache transfer; HiCache trades host memory and I/O for prefix reuse; EAGLE adds draft work and batch verification. Measure prefill/decode time, cache hit rate, host RAM, spec_accept_rate and spec_accept_length for your workload.',
      'PD 拆分 prefill/decode 并传输缓存；HiCache 用主机内存与 I/O 换取前缀复用；EAGLE 增加草拟与批量验证。按工作负载测量 prefill/decode 耗时、缓存命中率、主机内存、spec_accept_rate 和 spec_accept_length。'));
    if (['cpuAdam', 'recompute', 'nodes', 'rollout', 'response', 'concurrency'].some(changed)) insights.push(t(
      'More GPUs and lower concurrency can relieve different constraints. CPU Adam uses host RAM and transfers; recomputation saves activations by repeating work. Measure peak GPU/host memory and train/rollout time; allocation is not a memory-fit guarantee.',
      '增加 GPU 和降低并发缓解不同约束。CPU Adam 需要主机内存与传输，重计算通过重复计算节省激活。测量 GPU/主机内存峰值及训推耗时；分配数量不保证显存足够。'));
    if (differences.length && !insights.length) insights.push(t(
      'Check the changed fields and each configuration’s assumptions. Run a short controlled experiment and compare held-out reward and stage timings.',
      '核对变化字段及两套配置的条件，用短对照实验比较独立评估 reward 和各阶段耗时。'));
    return {differences, insights, a: validate(a, lang), b: validate(b, lang)};
  }

  function shareChoices(state) {
    const shared = {...sanitize(state)};
    for (const key of ['hf', 'train', 'data', 'save', 'megatron', 'endpoints', 'shared', 'local', 'ib',
      'generate', 'reward', 'queueDir', 'queueRun', 'declaration', 'draft']) delete shared[key];
    return shared;
  }

  function sourceRecord(s, build = {}) {
    const commit = /^[a-f0-9]{40}$/.test(build.commit || '') ? build.commit : null;
    const recipe = 'scripts/run-' + models[s.model].recipe + '.sh';
    const model = 'scripts/models/' + models[s.model].model + '.sh';
    const source = path => ({path, sha256: build.sourceHashes?.[path] || null,
      url: commit ? `https://github.com/THUDM/slime/blob/${commit}/${path}` : null});
    return {
      schema: 1,
      page: {commit, dirty: typeof build.dirty === 'boolean' ? build.dirty : null,
        generator: source('docs/_static/js/experiment-engine.js')},
      recipe: source(recipe), modelConfig: source(model),
      environment: {recipeVersions: build.environment || null,
        suggestedImageTag: s.hardware === 'unspecified' ? null : 'slimerl/slime:latest-' + (s.hardware === 'blackwell' ? 'cu130' : 'cu129'),
        imageDigest: null, installedCommit: null},
      checkpoint: {id: models[s.model].repo, revision: null},
      staticChecksPassed: validate(s).errors.length === 0,
      runtimeEvidence: []
    };
  }

  function generate(s, lang = 'en', build = {}) {
    s = sanitize(s);
    const m = models[s.model],
      v = validate(s, lang),
      t = (en, zh) => lang === 'zh' ? zh : en;
    const provenance = sourceRecord(s, build);
    const arg = [`--actor-num-nodes ${s.nodes}`, '--actor-num-gpus-per-node 8'];
    if (s.layout === 'colocate') arg.push('--colocate');
    if (s.layout === 'external') arg.push('--rollout-external-engine-addrs ' + s.endpoints.split(/\s+/).filter(
      Boolean).map(quote).join(' '));
    else arg.push(`--rollout-num-gpus ${v.rollout}`);
    arg.push('--hf-checkpoint "$ROLLOUT_MODEL"', '--ref-load "$TRAIN_CHECKPOINT"', '--save "$SAVE_DIR"',
      `--save-interval ${s.straw?1:20}`, '--prompt-data "$PROMPT_DATA"', '--input-key prompt', '--label-key label',
      '--apply-chat-template', '--rollout-shuffle');
    if (s.task === 'math') arg.push('--rm-type deepscaler');
    else arg.push('--custom-generate-function-path ' + quote(s.generate), '--custom-rm-path ' + quote(s.reward));
    arg.push(`--num-rollout ${s.rounds}`, '--rollout-batch-size 8', '--n-samples-per-prompt 8',
      '--global-batch-size 64', `--rollout-max-response-len ${s.response}`, '--rollout-temperature 1',
      '--rollout-top-p 1', '--rollout-top-k -1');
    if (s.schedule === 'async') arg.push(
      '--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async',
      '--skip-eval-before-train');
    if (s.straw) arg.push('--rollout-data-transport straw', `--rollout-data-dir ${quote(s.queueDir)}`,
      `--rollout-queue-run-id ${quote(s.queueRun)}`, '--rollout-storage-profile juicefs',
      `--rollout-storage-declaration ${quote(s.declaration)}`);
    if (s.partial) {
      arg.push('--partial-rollout');
      if (s.schedule === 'sync') arg.push('--over-sampling-batch-size 16');
    }
    if (s.maskPartial) arg.push('--mask-offpolicy-in-partial-rollout');
    arg.push(`--sglang-server-concurrency ${s.concurrency}`, '--advantage-estimator grpo', '--kl-coef 0',
      '--entropy-coef 0');
    if (s.sc) arg.push('--use-score-centering', '--score-centering-top-k 128', '--pg-loss-type reinforce',
      '--disable-grpo-std-normalization', '--calculate-per-token-loss');
    else arg.push('--eps-clip 0.2', '--eps-clip-high 0.28');
    if (s.correction !== 'none') arg.push('--use-tis', `--tis-clip-low ${s.correction==='icepop'?0.5:0}`,
      '--tis-clip 2');
    if (s.correction === 'icepop') arg.push(
      '--custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function');
    if (s.r3) arg.push('--use-rollout-routing-replay');
    if (s.deterministic) arg.push('--deterministic-mode');
    arg.push(`--tensor-model-parallel-size ${m.tp}`, '--sequence-parallel', `--pipeline-model-parallel-size ${m.pp}`,
      `--context-parallel-size ${m.cp}`, `--expert-model-parallel-size ${m.ep}`, '--expert-tensor-parallel-size 1',
      ...m.extra, ...(s.recompute ? ['--recompute-granularity full', '--recompute-method uniform',
        '--recompute-num-layers 1'
      ] : []), '--use-dynamic-batch-size', `--max-tokens-per-gpu ${m.maxTokens}`);
    arg.push('--optimizer adam', '--lr 1e-6', '--lr-decay-style constant', '--weight-decay 0.1', '--adam-beta1 0.9',
      '--adam-beta2 0.98');
    if (s.cpuAdam) arg.push('--optimizer-cpu-offload', '--overlap-cpu-optimizer-d2h-h2d',
      '--use-precision-aware-optimizer');
    if (s.precision === 'fp8train') arg.push('--fp8-format e4m3', '--fp8-recipe blockwise');
    arg.push('--attention-dropout 0', '--hidden-dropout 0', '--accumulate-allreduce-grads-in-fp32',
      '--attention-softmax-in-fp32', '--attention-backend flash');
    if (m.moe) arg.push(s.model === 'glm5' ? '--moe-token-dispatcher-type alltoall' :
      '--moe-token-dispatcher-type flex');
    if (m.moe && s.model !== 'glm5') arg.push('--moe-enable-deepep');
    arg.push(...servingArgs(s));
    if (s.pd && s.layout !== 'external') arg.push('--sglang-config "$LAB_CONFIG/sglang.yaml"');
    if (s.transport !== 'nccl') {
      arg.push(`--update-weight-mode ${s.transport==='delta'?'delta':'full'}`, '--update-weight-transport disk',
        `--update-weight-disk-dir ${quote(s.shared)}`);
      if (s.transport === 'delta') arg.push(`--update-weight-local-checkpoint-dir ${quote(s.local)}`);
    }
    const env = {
      PYTHONPATH: s.megatron,
      CUDA_DEVICE_MAX_CONNECTIONS: '1',
      NVSHMEM_DISABLE_NCCL: '1',
      NCCL_NVLS_ENABLE: '0',
      PYTHONUNBUFFERED: '1'
    };
    if (s.deterministic) Object.assign(env, {
      NCCL_ALGO: 'Ring',
      NVTE_ALLOW_NONDETERMINISTIC_ALGO: '0',
      CUBLAS_WORKSPACE_CONFIG: ':4096:8'
    });
    if (s.precision === 'fp8train') env.NVTE_FP8_BLOCK_SCALING_FP32_SCALES = '1';
    if (s.sc) env.SGLANG_RETURN_ORIGINAL_LOGPROB = '0';
    const config = yaml(s);
    const header =
      `#!/usr/bin/env bash\n# Generated by slime experiment lab. Run from the slime repository root.\n# Recipe basis: scripts/run-${m.recipe}.sh\n# ${m.name} | ${s.layout} | ${s.schedule} | ${s.precision} | ${s.correction}\n# Allocated Ray GPUs: ${v.localGpus}. Memory fit depends on your GPU and context.\n# Usage: bash experiment.sh prepare | convert | check | train\nset -euo pipefail\n\nHF_MODEL=${quote(s.hf)}\nTRAIN_CHECKPOINT=${quote(s.train)}\nPROMPT_DATA=${quote(s.data)}\nSAVE_DIR=${quote(s.save)}\nMEGATRON_DIR=${quote(s.megatron)}\nROLLOUT_MODEL="$HF_MODEL${s.precision==='bf16'?'':s.precision==='int4'?'-int4':'-fp8'}"\n# Same repository and data paths must be accessible on every participating node.\nexport PYTHONPATH="$PWD:$MEGATRON_DIR\${PYTHONPATH:+:$PYTHONPATH}"\nsource ${quote('scripts/models/'+m.model+'.sh')}\n\n`;
    const conversionNodes = ['glm5', 'glm47', 'deepseek'].includes(s.model) ? 4 : 1;
    const conversionArgs = s.model === 'glm5' ?
      '--tensor-model-parallel-size 8 --pipeline-model-parallel-size 2 --decoder-last-pipeline-num-layers 40 --expert-model-parallel-size 16 --expert-tensor-parallel-size 1' :
      s.model === 'deepseek' ?
      '--tensor-model-parallel-size 1 --pipeline-model-parallel-size 8 --decoder-first-pipeline-num-layers 7 --decoder-last-pipeline-num-layers 6 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1' :
      s.model === 'glm47' ?
      '--tensor-model-parallel-size 8 --pipeline-model-parallel-size 4 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1' :
      '';
    const prep =
      `if [[ "\${1:-check}" == prepare ]]; then\n  # Run once on the shared filesystem. Conversion is a separate distributed step.\n${['deepseek','glm5'].includes(s.model)?`  hf download ${quote(m.repo)} --local-dir "$HF_MODEL-source-fp8"\n  python tools/fp8_cast_bf16.py --input-fp8-hf-path "$HF_MODEL-source-fp8" --output-bf16-hf-path "$HF_MODEL"\n  python - "$HF_MODEL/config.json" <<'SLIME_BF16_CONFIG'\nimport json, sys\nfrom pathlib import Path\np = Path(sys.argv[1])\ncfg = json.loads(p.read_text())\ncfg.pop('quantization_config', None)\np.write_text(json.dumps(cfg, indent=2))\nSLIME_BF16_CONFIG\n`:`  hf download ${quote(m.repo)} --local-dir "$HF_MODEL"\n`}${s.task==='math'?'  hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "$(dirname -- "$PROMPT_DATA")"\n':'  # Supply your prompt JSONL and importable custom hooks before training.\n'}${s.precision==='bf16'?'':s.precision==='int4'?'  python tools/convert_hf_to_int4_direct.py --model-dir "$HF_MODEL" --save-dir "$ROLLOUT_MODEL"\n':'  python tools/convert_hf_to_fp8.py --model-dir "$HF_MODEL" --save-dir "$ROLLOUT_MODEL" \\\n    --strategy block --block-size 128 128 --max-workers 4\n'}  printf '%s\\n' 'Next: run bash experiment.sh convert on ${conversionNodes} node(s), 8 GPUs each.'\n  exit 0\nfi\n\nif [[ "\${1:-check}" == convert ]]; then\n${conversionNodes>1?'  : "\${CONVERT_MASTER_ADDR:?Set the conversion head IP on all 4 nodes}"\n  : "\${CONVERT_NODE_RANK:?Set this node rank: 0, 1, 2, or 3}"\n':''}  torchrun --nproc-per-node 8 ${conversionNodes===1?'--standalone':'--nnodes 4 --node-rank "$CONVERT_NODE_RANK" --master-addr "$CONVERT_MASTER_ADDR" --master-port 12345'} \\\n    tools/convert_hf_to_torch_dist.py "\${MODEL_ARGS[@]}" \\\n    ${conversionArgs?conversionArgs+' \\\n    ':''}--hf-checkpoint "$HF_MODEL" --save "$TRAIN_CHECKPOINT"\n  exit 0\nfi\n\nfor path in "$ROLLOUT_MODEL/config.json" "$TRAIN_CHECKPOINT/latest_checkpointed_iteration.txt" "$PROMPT_DATA"${s.straw?' '+quote(s.declaration):''}; do\n  if [[ ! -f "$path" ]]; then\n    printf 'Missing input: %s\\nRun prepare and convert, or edit the paths at the top of this script.\\n' "$path" >&2\n    exit 1\n  fi\ndone\n\nif [[ "\${1:-check}" == check ]]; then\n  printf '%s\\n' 'Input files exist. Verify Ray capacity, serving dependencies, and the selected model recipe before training.'\n  exit 0\nfi\nif [[ "\${1:-}" != train ]]; then\n  printf '%s\\n' 'Usage: bash experiment.sh prepare | convert | check | train' >&2\n  exit 1\nfi\n\n`;
    const runtime = JSON.stringify({
      env_vars: env
    }, null, 2);
    const launch =
      `# Ray must already be running. Join workers before submitting this job.\n# RAY_DASHBOARD can point at an existing head (default: localhost).\nLAB_CONFIG=$(mktemp -d "$PWD/.slime-lab.XXXXXX")\ntrap 'rm -rf -- "$LAB_CONFIG"' EXIT\ncat > "$LAB_CONFIG/runtime-env.json" <<'SLIME_RUNTIME'\n${runtime}\nSLIME_RUNTIME\npython - "$LAB_CONFIG/runtime-env.json" "$PWD" <<'SLIME_PATH'\nimport json, sys\nfrom pathlib import Path\np = Path(sys.argv[1])\ncfg = json.loads(p.read_text())\ncfg['env_vars']['PYTHONPATH'] = sys.argv[2] + ':' + cfg['env_vars']['PYTHONPATH']\np.write_text(json.dumps(cfg))\nSLIME_PATH\n${config?`cat > "$LAB_CONFIG/sglang.yaml" <<'SLIME_SGLANG'\n${config}SLIME_SGLANG\n`:''}\nray job submit --address="\${RAY_DASHBOARD:-http://127.0.0.1:8265}" \\\n  --runtime-env "$LAB_CONFIG/runtime-env.json" -- python train.py \\\n  "\${MODEL_ARGS[@]}" \\\n  ${arg.join(' \\\n  ')}\n`;
    const external = s.layout === 'external' ?
      `# External SGLang launch reference — apply before submitting experiment.sh.\n# One command per engine; multi-node engines also need --nnodes, --node-rank\n# and --dist-init-addr. Use each engine's real host, ports and visible devices.\n# Model, dtype, TP/EP/DP, cache and routed-expert reporting must match the trainer.\n# Pass engine host:port addresses, not a router URL, to slime.\n\npython -m sglang.launch_server \\\n  --model-path ${quote(s.hf+(s.precision==='bf16'?'':s.precision==='int4'?'-int4':'-fp8'))} \\\n  --tp-size ${m.engine} \\\n  ${servingArgs(s).filter(a=>!a.startsWith('--rollout-')).map(a=>a.replace('--sglang-','--')).concat(s.r3?['--enable-return-routed-experts']:[],s.pd?['--disaggregation-mode prefill']:[]).join(' \\\n  ')}\n\n${s.pd?'# Launch a matching decode group with --disaggregation-mode decode.\n# Remove --enable-hierarchical-cache and --hicache-ratio on decode.\n# Configure bootstrap ports, RDMA routing, and multi-node engine ranks.\n':''}# Disk sync: expose ${s.shared} at the SAME path on trainer and serving hosts.\n${s.transport==='delta'?'# Delta requires slime-patched SGLang /pull_weights and local checkpoint paths.\n':''}` :
      '';
    const plan = {
      schema: 1,
      choices: s,
      model: m.name,
      validation: v,
      provenance
    };
    return {
      validation: v,
      provenance,
      files: {
        'experiment.sh': header.replace('# Recipe basis:',
          `# Generator source commit: ${provenance.page.commit || 'not recorded'}; local changes: ${provenance.page.dirty ?? 'unknown'}\n# Source: ${provenance.recipe.url || 'commit not recorded'}\n# Runtime image digest and model revision: not recorded. Verify the installed checkout with git rev-parse HEAD.\n# Static checks do not establish GPU memory fit or a tested training run.\n# Recipe basis:`) + prep + launch,
        ...(config ? {
          'sglang.yaml': config
        } : {}),
        ...(external ? {
          'serving-reference.sh': external
        } : {}),
        'experiment.json': JSON.stringify(plan, null, 2) + '\n'
      }
    };
  }
  root.SlimeExperiment = {
    models,
    defaults,
    enums,
    numberLimits,
    sanitize,
    change,
    preset,
    validate,
    generate,
    fieldLabel,
    compare,
    shareChoices,
    sourceRecord,
    quote
  };
})(typeof window === 'undefined' ? globalThis : window);
