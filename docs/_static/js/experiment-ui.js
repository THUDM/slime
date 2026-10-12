(function() {
  'use strict';
  const E = window.SlimeExperiment,
    $ = id => document.getElementById(id);
  if (!E || !$('interactive-lab')) return;
  const lang = document.documentElement.lang.startsWith('zh') ? 'zh' : 'en',
    t = (en, zh) => lang === 'zh' ? zh : en;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  } [c]));
  const steps = [
    [t('Model', '模型'), t('Choose your starting character.', '选一个你的初始角色。'), t(
      'Start with a model and a task. Everything else follows.', '先选模型和任务，后面的配方会随之变化。')],
    [t('Placement', '训推布局'), t('Give your model a home.', '让训练与推理各就各位。'), t(
      'GPU ownership and scheduling are two separate choices.', 'GPU 如何分配，和任务如何调度，是两个独立选择。')],
    [t('Precision', '精度'), t('Find your precision sweet spot.', '找到精度与效率的平衡。'), t(
      'Choose precision for weights, attention KV, and hybrid recurrent state separately.',
      '分别选择权重、attention KV 与 hybrid 递归状态的精度。')],
    [t('Consistency', '训推一致'), t('Keep the two worlds in tune.', '让两个世界保持合拍。'), t(
      'Reduce numerical mismatch, correct its effect, or combine the two.', '可以减少数值差异，也可以修正它对训练的影响。')],
    [t('Serving', '推理引擎'), t('Build your rollout engine.', '搭好你的 rollout 引擎。'), t(
      'Tune for your workload. More switches do not always mean a faster run.', '根据负载调优。开关越多，并不总是越快。')],
    [t('Launch', '启动'), t('Connect the last few pieces.', '接好最后几块积木。'), t(
      'Use paths visible to your cluster. The shell includes preparation and a short three-round run by default.',
      '填写集群可见的路径。shell 默认包含准备步骤和三轮短实验。')],
  ];
  const theory = {
    grpo: {
      title: 'GRPO',
      formula: 'Aᵢ = (Rᵢ − mean(R)) / (std(R) + ε)',
      text: t(
        'Compare multiple answers to the same prompt instead of training a critic. The recipe uses 8 answers per prompt; Score Centering switches off standard-deviation normalization and uses REINFORCE.',
        '用同一 prompt 的多次回答估计相对优势，不额外训练 critic。配方每个 prompt 采样 8 次；启用 SC 时改用 REINFORCE 并关闭标准差归一化。'),
      paper: 'https://arxiv.org/abs/2402.03300',
      page: 'advanced/policy-mismatch.html#grpo'
    },
    tis: {
      title: 'TIS · Truncated Importance Sampling',
      formula: 'w = clip(p_old(a | h) / q(a | h), L, U)',
      text: t(
        'Rollout samples come from q, while PPO uses the old trainer p_old. Importance weights bridge that gap; clipping limits extreme weights but introduces bias. This recipe uses L=0, U=2. With SC the numerator is the current trainer and the weight is detached.',
        '样本来自 rollout 分布 q，PPO 的旧策略来自训练器 p_old。重要性权重连接两者；裁剪限制极端权重，同时引入偏差。配方使用 L=0、U=2。与 SC 组合时，分子改为当前训练策略，并对权重停止梯度。'
      ),
      paper: 'https://openreview.net/forum?id=8MHqvb4lK9',
      page: 'advanced/policy-mismatch.html#tis'
    },
    icepop: {
      title: 'ICE-POP',
      formula: 'w = (p / q) · 1{ L ≤ p / q ≤ U }',
      text: t(
        'slime’s built-in IcePop callback masks weights outside the interval instead of clipping them to its boundary. The recipe uses [0.5, 2]. Track the masked fraction: excessive masking can discard the learning signal.',
        'slime 内置 IcePop callback 将区间外的权重置零，而不是裁剪到边界。配方使用 [0.5, 2]。需要观察被 mask 的比例，避免丢掉太多学习信号。'),
      paper: 'https://arxiv.org/abs/2510.18855',
      page: 'advanced/policy-mismatch.html#icepop'
    },
    r3: {
      title: 'R3 · Rollout Routing Replay',
      formula: 'MoE(x) = Σᵢ∈TopK_rollout gᵢ(x) · Expertᵢ(x)',
      text: t(
        'Tiny router differences can select different experts. Replay the expert IDs recorded during rollout on the training side. This aligns the discrete route, while numerical differences inside experts can remain. Extra route tensors cost memory and transfer bandwidth.',
        '微小的 router 差异可能选中不同专家。R3 在训练时重放 rollout 记录的 expert IDs，对齐离散路由；专家内部仍可能有数值差异。保存路由张量会增加内存和传输开销。'),
      paper: 'https://arxiv.org/abs/2510.11370',
      page: 'advanced/policy-mismatch.html#r3'
    },
    sc: {
      title: 'SC · Score Centering',
      formula: 'g = A · [wₐ ∇ log pₐ − Σᵥ qᵥ wᵥ ∇ log pᵥ]',
      text: t(
        'Subtract the sampler’s expected (possibly importance-weighted) score at each prefix. The centered score has zero mean, removing a drift term. This is not an unbiased on-policy gradient. slime stores the sampler top-128 head and approximates the tail; it requires REINFORCE, not PPO clipping.',
        '逐个 prefix 减去采样器下（可加重要性权重的）score 期望，使中心化后的 score 均值为零，消除漂移项；这不等于无偏 on-policy 梯度。slime 保存 sampler top-128 head 并近似 tail，使用 REINFORCE，而非 PPO clipping。'
      ),
      paper: 'https://arxiv.org/abs/2609.20807',
      page: 'advanced/policy-mismatch.html#sc'
    },
    deterministic: {
      title: t('Deterministic execution', '确定性执行'),
      formula: 'repeatable(p) ∧ repeatable(q) ⇏ p = q',
      text: t(
        'Fixing each engine’s execution order improves reproducibility. Cross-engine log-prob equality also requires aligned kernels, precision, reductions, and routers. The maintained GLM-5 exact-alignment gate is six-layer EP8; do not extrapolate that guarantee to a full model.',
        '固定各引擎的执行顺序可以改善可复现性。要让训推 log-prob 相等，还需要对齐 kernel、精度、归约和路由。现有 GLM-5 精确对齐验证为六层 EP8，不能直接推广为全模型保证。'
        ),
      paper: 'https://lmsys.org/blog/2025-09-22-sglang-deterministic/',
      page: 'advanced/rl-systems.html#deterministic'
    },
    precision: {
      title: t('Precision & quantization', '精度与量化'),
      formula: 'q(a | h) = softmax(f_quantized(h))ₐ',
      text: t(
        'Lower precision changes the rollout distribution. Weight precision, attention KV dtype and hybrid recurrent-state dtype are independent choices. FP8 attention KV requires model/backend support; Mamba SSM state and convolution state have separate storage. Compare the two cache pools below.',
        '低精度会改变 rollout 分布。权重、attention KV 和 hybrid 递归状态分别选择精度。FP8 attention KV 需模型与后端支持；Mamba SSM 和卷积状态各有独立存储。下面分别说明两类缓存。'
      ),
      paper: 'https://arxiv.org/abs/2209.05433',
      page: 'advanced/rl-systems.html#precision'
    },
    async: {
      title: t('Synchronous vs fully async', '同步与 Fully async'),
      formula: 'T_sync ≈ T_rollout + T_train + T_sync_weights',
      text: t(
        'Synchronous execution makes experiment ordering easier to understand. Fully async keeps a warm queue of generations across steps, reducing waits for long tails but mixing weight versions. The animation is explanatory, not a performance prediction. Track queue age and measured throughput.',
        '同步执行更容易理解实验顺序。Fully async 跨 step 保持生成队列，减少长尾等待，但会混合不同权重版本。右侧动画用于解释流程，不预测性能；请测量队列年龄和实际吞吐。'),
      paper: 'https://arxiv.org/abs/2509.19128',
      page: 'advanced/rl-systems.html#async'
    },
    placement: {
      title: t('GPU placement & weight updates', 'GPU 布局与权重更新'),
      formula: 'separate GPUs = training pool + rollout pool',
      text: t(
        'Colocated engines take turns on one GPU pool. Disaggregated engines own separate pools; external engines are also managed outside slime. External weight updates can use NCCL or a shared filesystem; delta updates require disk transport and patched serving endpoints.',
        'Colocated 在同一组 GPU 上轮流工作；分离布局各占一组 GPU；external 还把引擎生命周期交给外部系统。外部权重更新可使用 NCCL 或共享文件系统，delta 需要磁盘传输和带补丁的推理接口。'
      ),
      paper: 'https://arxiv.org/abs/2409.19256',
      page: 'advanced/rl-systems.html#placement'
    },
    pd: {
      title: 'PD · Prefill / Decode',
      formula: 'throughput ≤ min(nₚ μₚ, n_d μ_d, B_network / KV_bytes)',
      text: t(
        'Separate prompt processing from token generation and size each pool independently. This can reduce interference on long prompts, but adds KV-transfer and deployment costs. It is a serving split, independent of the training/rollout GPU split.',
        '将 prompt 处理与逐 token 生成拆开，分别扩容。长 prompt 下可能减少干扰，但增加 KV 传输与部署成本。它是推理内部拆分，独立于训练 / rollout 的 GPU 布局。'
        ),
      paper: 'https://arxiv.org/abs/2401.09670',
      page: 'advanced/rl-systems.html#pd'
    },
    eagle: {
      title: 'EAGLE · ' + t('Speculative decoding', '投机采样'),
      text: t(
        'A draft head proposes candidates; the target model verifies them and commits an accepted prefix. A separate head must match the target model. Monitor acceptance and rollout time as RL updates the target.',
        '投机头提出候选，目标模型验证并提交接受的前缀。独立投机头必须匹配目标模型；随 RL 更新，观察接受率和 rollout 耗时。'),
      paper: 'https://arxiv.org/abs/2401.15077',
      page: 'advanced/speculative-decoding.html'
    },
    hicache: {
      title: 'HiCache',
      formula: 'saved time ≈ hit_rate × prefill_cost − cache_IO',
      text: t(
          'Reuse prefixes through a hierarchy of GPU and host memory, optionally storage. It helps repeated prefixes, not unrelated prompts. This recipe configures host memory only, and prefill only under PD; model-weight updates limit how long cache entries remain valid.',
          '在 GPU、主机内存乃至存储之间复用 prefix。它适合重复前缀，不一定帮助无关 prompt。此配方只配置主机缓存，PD 下仅在 prefill 启用；模型权重更新会限制缓存有效期。'
          ),
      paper: 'https://docs.sglang.io/docs/advanced_features/hicache_best_practices',
      page: 'advanced/rl-systems.html#hicache'
    },
  };
  theory.parallel = {
    title: t('Parallelism & memory', '并行与内存'),
    formula: 'N = TP × PP × CP × DP;  EP uses an expert process-group factorization',
    text: t(
      'TP splits a layer, PP splits layers, EP splits experts, CP splits tokens. DP replicates work across different samples; the distributed optimizer shards its state. CPU Adam moves optimizer work to host memory. These save different kinds of memory and introduce different communication.',
      'TP 拆层内张量，PP 拆模型层，EP 拆专家，CP 拆序列。DP 用副本处理不同样本，distributed optimizer 对状态分片。CPU Adam 将优化器工作移到主机。它们节省的内存和引入的通信各不相同。'
    ),
    paper: 'https://arxiv.org/abs/2104.04473',
    page: 'advanced/parallelism-memory.html'
  };
  theory.partial = {
    title: 'Partial rollout',
    formula: 'q(y | x) = ∏ₜ q_version(t)(yₜ | x, y₍<t₎)',
    text: t(
      'When a synchronous batch is ready, save unfinished responses and resume after the update. Reuse generated tokens instead of restarting. A response can span weight versions; old-token masking changes the loss, not the prefix distribution.',
      '同步 batch 凑齐后保存未完成回复，更新后继续生成，复用已经生成的 token。一个回复可能跨权重版本；mask 旧 token 改变 loss，不会改变 prefix 的采样分布。'),
    paper: 'https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py',
    page: 'advanced/rollout-scheduling.html#partial'
  };
  theory.straw = {
    title: 'straw · distributed fully async',
    formula: 'producers → partial / ready queue → batch → trainer',
    text: t(
      'straw stores queue state and packed tensors. Combining it with fully async selects distributed generation processes, one per eligible Ray node. Storage, continuation and scheduling are separate choices. Recovery uses model-and-queue checkpoints.',
      'straw 保存队列状态与打包张量；搭配 fully async 后启用分布式生成，每个符合条件的 Ray 节点一个进程。存储、续跑和调度是独立选择，恢复以模型与队列共同 checkpoint 为边界。'
      ),
    paper: 'https://github.com/zhuzilin/straw',
    page: 'advanced/rollout-scheduling.html#distributed'
  };
  const history = [],
    future = [];
  let state = {
      ...E.defaults
    },
    artifact = 'experiment.sh',
    generated, timer = null,
    phase = 0,
    toastTimer, parallelView = 'tp',
    cpView = 'zigzag';
  const hashParams = new URLSearchParams(location.hash.slice(1));
  const build = JSON.parse($('experiment-build').textContent);
  let baseline = null;
  try {
    const encoded = hashParams.get('experiment');
    if (encoded) state = E.sanitize(JSON.parse(encoded));
    else {
      const saved = localStorage.getItem('slime-experiment-v1');
      if (saved) state = E.sanitize(JSON.parse(saved));
    }
    const savedBaseline = encoded ? hashParams.get('baseline') : localStorage.getItem('slime-baseline-v1');
    if (savedBaseline) baseline = E.sanitize(JSON.parse(savedBaseline));
  } catch (_) {}

  function toast(message) {
    $('toast').textContent = message;
    $('toast').classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 4500);
  }

  function persist() {
    try {
      localStorage.setItem('slime-experiment-v1', JSON.stringify(state));
    } catch (_) {}
  }

  function commit(next, message) {
    if (JSON.stringify(next) === JSON.stringify(state)) return;
    history.push({
      ...state
    });
    if (history.length > 100) history.shift();
    future.length = 0;
    state = next;
    stop();
    render();
    persist();
    if (message) toast(message);
  }

  function options(key, title, items, single = false) {
    return `<fieldset class="field-group"><legend>${title}</legend><div class="option-grid ${single?'single':''}">${items.map(([value,label,description])=>`<label class="option-card"><input type="radio" name="${key}" value="${value}" ${state[key]===value?'checked':''}><span class="option-title">${label}</span><span class="option-description">${description}</span></label>`).join('')}</div></fieldset>`;
  }

  function toggle(key, label, desc) {
    return `<label class="toggle-row"><input type="checkbox" name="${key}" ${state[key]?'checked':''}><span><strong>${label}</strong><small>${desc}</small></span></label>`;
  }

  function input(key, label, type = 'text', help = '', wide = false) {
    return `<label class="input-label ${wide?'wide':''}">${label}<input name="${key}" type="${type}" value="${esc(state[key])}" ${type==='number'?`min="${E.numberLimits[key][0]}" max="${E.numberLimits[key][1]}" step="1"`:''} spellcheck="false" autocomplete="off">${help?`<small>${help}</small>`:''}</label>`;
  }

  function form() {
    const m = E.models[state.model];
    switch (state.step) {
      case 0:
        return options('model', t('Your model', '你的模型'), Object.entries(E.models).sort(([a], [b]) => Number(
            ['qwen38',
              'deepseek'
            ].includes(a)) - Number(['qwen38', 'deepseek'].includes(b))).map(([k, v]) => [k, v.name, v
            .size
          ])) +
          options('task', t('Your learning signal', '你的学习信号'), [
            ['math', t('Verifiable math', '可验证数学'), t('Prompt + label JSONL, built-in deepscaler reward.',
              'Prompt + label JSONL，内置 deepscaler 奖励。')],
            ['custom', t('My agent / reward', '我的 Agent / 奖励'), t(
              'Plug in your generation and reward functions.',
              '接入自己的生成和奖励函数。')]
          ]) +
          `<p class="option-description">${t('The compact starting recipe uses GRPO, 8 responses per prompt, and 3 rounds.','起步配方使用 GRPO，每个 prompt 8 次采样，先训练 3 轮。')}</p>`;
      case 1:
        return `<label class="input-label">${E.fieldLabel('hardware', lang)}<select name="hardware">${[['unspecified', t('Not specified', '未指定')], ['hopper', 'H100 / H200'], ['blackwell', 'B200 / B300']].map(([value, label]) => `<option value="${value}" ${state.hardware === value ? 'selected' : ''}>${label}</option>`).join('')}</select><small>${t('Shared with the installation image selector. Records conditions, not available GPU capacity or a validated model list.', '与安装区的镜像选择关联。只记录硬件条件，不代表可用卡数或经过验证的模型列表。')}</small></label>` + options('layout', t('Where do the engines live?', '引擎部署在哪里？'), [
            ['colocate', 'Colocated', t(
              'One pool. Rollout and training take turns; weights use CUDA IPC.',
              '同一组 GPU，训推轮流执行；CUDA IPC 同步权重。')],
            ['separate', 'Disaggregated', t('Separate GPU pools, both managed by slime.',
              '各自独立 GPU 池，由 slime 管理。')],
            ['external', 'External SGLang', t('Attach engines already running in another cluster.',
              '连接已部署的外部 SGLang 引擎。')]
          ], true) + options('schedule', t('How does work progress?', '任务如何推进？'), [
            ['sync', t('Synchronous', '同步训练'), t('Finish a batch, train, update weights.',
              '完成一批 rollout，再训练并更新权重。')],
            ['async', 'Fully async', t('A warm queue across steps. Add straw for distributed workers.',
              '跨 step 持续生成；搭配 straw 启用分布式 workers。')]
          ]) +
          `<div class="inputs-grid">${input('nodes',t('Training nodes × 8 GPUs','训练节点数 × 8 GPU'),'number')}${state.layout!=='colocate'?input('rollout',state.layout==='external'?t('External GPUs (planning only)','外部 GPU 数（规划用）'):t('Rollout GPUs','Rollout GPU 数'),'number'):''}</div>` +
          schedulingPanel() + `<details class="resource-options" data-detail="resources"><summary>${t('GPU recommendation, memory & parallelism', 'GPU 推荐、显存与并行')} · ${E.validate(state).localGpus} GPU</summary>` + resourcePanel() + toggle('cpuAdam', 'CPU Adam · ' + t('save GPU memory',
            '节省显存'), t(
            'Offload optimizer states and compute; uses host RAM and transfer bandwidth.',
            '优化器状态与计算 offload 到 CPU，需要主机内存与传输带宽。')) + toggle('recompute', t('Recompute activations',
            '激活重计算'), t(
            'Keep fewer intermediates; recompute them during backward.', '少存中间激活，在 backward 时重新计算。')) +
          parallelPanel() + '</details>';
      case 2:
        return options('precision', t('Training → rollout weights', '训练 → 推理权重'), [
            ['bf16', 'BF16 → BF16', t('An easy baseline for measuring mismatch.', '适合建立基线，先测量训推差异。')],
            ['fp8', 'BF16 → FP8', t('Maintained large-MoE path. Convert the HF checkpoint.',
              '大 MoE 维护路径，需转换 HF checkpoint。')],
            ['fp8train', 'FP8 → FP8', t('Experimental. Changes the training numerical path.',
              '实验性，改变训练侧数值路径。')],
            ['int4', 'BF16 → INT4', t('Beta. Validate GLM kernel support first.',
              'Beta，先验证 GLM kernel 支持。')]
          ]) + (m.hybrid ?
            `<p class="notice">${t(
          'This hybrid model has 16 attention layers and 48 Gated DeltaNet layers. Their caches use separate settings.',
          '这个 hybrid 模型有 16 层 attention 和 48 层 Gated DeltaNet，两类缓存分别配置。')}
          <a href="advanced/rl-systems.html#hybrid-cache">${t('How the two caches work →', '看懂两类缓存 →')}</a></p>` : '') +
          options('kv', t('Rollout attention KV cache', 'Rollout attention 层 KV cache'), [
            ['auto', t('Model default', '模型默认'), t('Use the serving stack’s default KV dtype.',
              '沿用推理栈默认 KV dtype。')],
            ['fp8', 'FP8 E4M3', t(
              'Attention KV only. Requires support from this model’s attention backend and GPU.',
              '只作用于 attention KV，需当前模型的 attention 后端与 GPU 支持。')]
          ]) + (m.hybrid ? options('mambaDtype', t('Mamba / linear-attention recurrent state',
            'Mamba / 线性注意力递归状态'), [
            ['auto', t('Model default · FP32', '模型默认 · FP32'), t(
              'Qwen3.8-27B config selects FP32 SSM state. Independent of attention KV.',
              'Qwen3.8-27B 配置采用 FP32 SSM 状态，与 attention KV 独立。')],
            ['float32', 'FP32', t(
              'Explicit 32-bit recurrent state; convolution state keeps its own dtype.',
              '显式使用 32 bit 递归状态；卷积状态保留自身的 dtype。')],
            ['bfloat16', 'BF16', t(
              'Smaller recurrent state. Validate long-sequence accuracy and kernel support.',
              '减少递归状态存储；验证长序列精度与 kernel 支持。')]
          ]) + input('mambaRatio', t('Mamba / attention KV memory ratio (optional)',
              'Mamba / attention KV 内存比例（选填）'),
            'text', t(
              'State-pool bytes ÷ attention-KV-pool bytes, e.g. 0.9. Leave blank for the serving default; this is not a percentage of all GPU memory.',
              '状态池字节数 ÷ attention KV 池字节数，例如 0.9。留空沿用推理栈默认值；这不是总显存百分比。')) : '');
      case 3:
        return options('correction', t('Importance correction', '重要性修正'), [
            ['none', t('Observe only', '暂不修正'), t('Baseline. Watch log-prob mismatch and reward.',
              '作为对照，观察 log-prob 差异与 reward。')],
            ['tis', 'TIS', t('Clamp weights to [0, 2].', '将权重裁剪到 [0, 2]。')],
            ['icepop', 'ICE-POP', t('Zero weights outside [0.5, 2].', '将 [0.5, 2] 之外的权重置零。')]
          ]) + toggle('r3', 'R3 · ' + t('Replay MoE routes', '重放 MoE 路由'), m.moe ? t(
            'Align expert selection; can combine with correction.', '对齐专家选择，可与重要性修正组合。') : t(
            'Requires MoE; this dense model has no expert routes.', '仅适用于 MoE；当前 dense 模型没有专家路由。')) +
          toggle('sc',
            'SC · Score Centering', t('Center weighted scores; selects REINFORCE and saves sampler top-k.',
              '中心化加权 score，改用 REINFORCE 并保存 sampler top-k。')) + toggle('deterministic', t(
            'Deterministic execution',
            '确定性执行'), t('Control execution order. Requires the deterministic dependency stack.',
            '控制执行顺序，需要 deterministic 依赖栈。'));
      case 4:
        return toggle('pd', t('Split prefill & decode', '分离 prefill 与 decode'), t(
            'Separate serving stages. Useful when prompt processing delays decoding.',
            '拆分推理阶段，适合 prefill 干扰 decode 的场景。')) + toggle('hicache', 'HiCache', t(
            'Reuse repeated prefixes in host RAM. Extra memory and I/O budget.',
            '在主机内存复用重复前缀，需要额外内存与 I/O 预算。')) +
          toggle('speculative', 'EAGLE · ' + t('Speculative decoding', '投机采样'), t(
            'Draft several candidates, then verify them together with the target model.',
            '先提出多个候选 token，再由目标模型一起验证。')) +
          (state.speculative ? options('draftSource', t('Draft head', '投机采样头'), [
            ...(m.mtp ? [['builtin', t('Checkpoint MTP head', 'Checkpoint 内置 MTP 头'), t(
              'Use the target checkpoint’s supported prediction layers.', '使用目标 checkpoint 自带的预测层。')]] : []),
            ['external', t('Separate EAGLE head', '独立 EAGLE 头'), t(
              'Load a compatible head from a local path or Hugging Face model ID.',
              '从本地路径或 Hugging Face 模型 ID 加载匹配的投机头。')]
          ]) + (state.draftSource === 'external' ? input('draft', t('EAGLE head checkpoint', 'EAGLE 投机头 checkpoint'),
            'text', t('Path or model ID, available on every serving host. The head is loaded for inference.',
              '路径或模型 ID，需供每台推理主机访问；投机头用于推理加载。'), true) : '') +
          input('specSteps', t('Draft steps', '投机步数'), 'number', t(
            'Top-k is fixed to 1; verification budget is steps + 1. Start small and measure acceptance.',
            'Top-k 固定为 1，验证 token 预算为步数 + 1；先从小值开始测量接受率。')) : '') +
          `<div class="inputs-grid">${state.pd&&state.layout!=='external'?input('prefill',t(`Prefill GPUs (engine = ${m.engine})`,`Prefill GPU（每引擎 ${m.engine}）`),'number'):''}${state.pd?input('ib',t('RDMA devices','RDMA 网卡'),'text','mlx5_0,mlx5_1'):''}${input('concurrency',t('Concurrency per engine (global budget basis)','每引擎并发（全局预算基数）'),'number')}${input('response',t('Max response tokens','最大回复 token 数'),'number')}</div>` +
          (state.layout !== 'colocate' ? options('transport', t('Weight updates', '权重更新'), [
            ['nccl', 'NCCL', t('Reachable, compatible GPU communication group.', '可建立兼容的 GPU 通信组。')],
            ['disk', t('Full · shared disk', '全量 · 共享盘'), t('A complete checkpoint every sync.',
              '每次同步传输完整 checkpoint。')],
            ['delta', t('Delta · shared disk', '增量 · 共享盘'), t(
              'Changed bytes + local checkpoint. Patched serving required.',
              '变化字节 + 本地 checkpoint，需要推理补丁。')]
          ]) : '');
      default:
        return `<div class="inputs-grid">${input('hf',t('Original BF16 HF checkpoint','原始 BF16 HF checkpoint'),'text',t('prepare downloads here; FP8 / INT4 gets its own sibling directory.','prepare 下载到此处；FP8 / INT4 使用旁边的独立目录。'),true)}${input('train',t('Megatron checkpoint','Megatron checkpoint'),'text','',true)}${input('data',t('Prompt JSONL','Prompt JSONL'),'text',t('Use prompt and label keys for math.','数学数据使用 prompt 和 label 字段。'),true)}${input('save',t('Output directory','输出目录'),'text')}${input('megatron',t('Megatron-LM repository','Megatron-LM 仓库'),'text')}${input('rounds',t('Training rounds','训练轮数'),'number')}${state.layout==='external'?input('endpoints',t('External engine addresses','外部引擎地址'),'text','host1:30000 host2:30000',true):''}${state.transport!=='nccl'?input('shared',t('Shared weight directory','共享权重目录'),'text','',true):''}${state.transport==='delta'?input('local',t('Serving host local checkpoint','推理主机本地 checkpoint'),'text','',true):''}${state.task==='custom'?input('generate',t('Generation hook','生成 hook'),'text','my_agent.generate')+input('reward',t('Reward hook','奖励 hook'),'text','my_reward.reward'):''}</div>`;
    }
  }
  const helpKeys = [
    ['grpo'],
    ['placement', 'async', 'partial', 'straw', 'parallel'],
    ['precision'],
    ['tis', 'icepop', 'r3', 'sc', 'deterministic'],
    ['pd', 'hicache', 'eagle'],
    ['placement']
  ];

  function schedulingPanel() {
    const async = state.schedule === 'async', distributed = async && state.straw;
    return toggle('partial', 'Partial rollout', t(
        'Keep unfinished prefixes and resume them after an interruption.',
        '保留未完成前缀，在中断后继续生成。')) + (state.partial ? toggle('maskPartial', t('Mask earlier prefix tokens',
        'Mask 已有前缀 token'), t('Exclude their loss; does not remove prefix-distribution mismatch.',
        '不计算旧 token 的 loss；不会消除 prefix 分布差异。')) : '') + toggle('straw', 'straw · ' + t(
        'persistent rollout storage',
        '持久 rollout 存储'), t(
        'Shared JuiceFS queue + tensor packs. With fully async: distributed generation workers.',
        '共享 JuiceFS 队列与张量 packs；搭配 fully async：分布式生成 workers。')) + (state.straw ?
        `<div class="inputs-grid">${input('queueDir',t('Shared JuiceFS rollout directory','共享 JuiceFS rollout 目录'),'text','/shared/juicefs/jobs/my-run/rollout_data',true)}${input('queueRun','Run ID','text','my-run')}${input('declaration',t('JuiceFS declaration JSON','JuiceFS 声明 JSON'),'text','/shared/juicefs/deployment.json')}</div>` :
        '') +
      `<details class="schedule-explorer" data-detail="schedule"><summary>${distributed?'Distributed fully async':async?'Fully async':state.partial?t('Synchronous + partial rollout','同步 + partial rollout'):t('Synchronous baseline','同步基线')} · ${t('see the timeline', '查看时间线')}</summary><div class="schedule-lanes"><div><small>${t('Generation','生成')}</small><span class="lane-gen">${distributed?'worker 0 / 1 / …': 'SGLang'}</span><span class="lane-gen ${async?'':'lane-muted'}">${async?t('continues during training','训练时持续生成'):t('pause for update','暂停并更新')}</span></div><div><small>${t('Training','训练')}</small><span class="lane-muted">${t('ready batch','就绪 batch')}</span><span class="lane-train">Megatron</span><span class="lane-update">weights ↺</span></div>${state.partial?`<div><small>${t('Long response','长回复')}</small><span class="lane-gen">prefix · v₀</span><span class="lane-muted">${t('save','保存')}</span><span class="lane-update">suffix · v₁</span></div>`:''}</div><p>${distributed?t('Autonomous producers feed one global queue; the trainer need not wait for every worker. Admission pauses and in-flight results are persisted around weight updates.','多个 producer 向全局队列供数，训练不必等待每个 worker。权重更新时暂停接收新任务，并保存进行中的结果。'):async?t('A manager-local background worker keeps the queue warm across rounds. Select straw to distribute the producer processes.','管理进程内的后台 worker 跨轮次维持队列；选择 straw 可将 producer 进程分布到多机。'):state.partial?t('Oversample 16 groups, train with 8 accepted complete groups, save unfinished groups for continuation. Partial responses are not directly used as completed training examples.','超采样 16 组，用 8 组已完成且被接受的样本训练；未完成组留待续跑，不直接当作完整训练样本。'):t('Complete the selected batch before the training phase. Long responses can hold the batch open.','先完成当前 batch，再进入训练阶段；长回复可能让整批等待。')}</p><small>${t('Conceptual timeline; widths are not measured durations.','概念时间线，宽度不代表实际耗时。')}</small><a href="advanced/rollout-scheduling.html">${t('Compare scheduling, continuation & recovery →','对比调度、续跑与恢复 →')}</a></details>`;
  }

  function resourcePanel() {
    const m = E.models[state.model],
      gpus = m.nodes * 8,
      total = E.validate(state).localGpus;
    return `<div class="resource-card"><div><small>${t('RECIPE STARTING POINT','配方起步参考')}</small><strong>${gpus} GPU <span>/ ${m.nodes} × 8</span></strong></div><p>${t(`Train TP${m.tp} · PP${m.pp} · EP${m.ep} · CP${m.cp}. Serving: ${m.engine} GPUs per engine.`,`训练 TP${m.tp} · PP${m.pp} · EP${m.ep} · CP${m.cp}。推理每引擎 ${m.engine} 张卡。`)}</p><p>${t('Based on repository H100/H200 recipes, with a short response budget; not a minimum or a capacity guarantee.','基于仓库 H100/H200 配方，以短回复起步；不是最小卡数或容量保证。')} <a href="advanced/parallelism-memory.html">${t('See assumptions →','看推荐依据 →')}</a></p><div class="resource-total">${t('Your local reservation','当前本地资源总量')} <b>${total} GPU</b></div></div>`;
  }

  function parallelPanel() {
    const m = E.models[state.model];
    return `<details class="parallel-explorer" data-detail="parallel"><summary>${t('What are TP / PP / EP / CP / ZeRO?','TP / PP / EP / CP / ZeRO 是什么？')}</summary><div class="parallel-tabs" aria-label="${t('Parallelism explanations','并行方式解释')}">${['tp','pp','ep','cp','zero','memory'].map(k=>`<button type="button" data-parallel="${k}" aria-pressed="${parallelView===k}">${k==='memory'?t('Memory','内存'):k==='zero'?'ZeRO':k.toUpperCase()}</button>`).join('')}</div><div id="parallel-diagram">${parallelDiagram()}</div><p class="parallel-current">${t('Current recipe','当前配方')}: TP${m.tp} · PP${m.pp} · EP${m.ep} · CP${m.cp} · ${state.model==='glm5'?'allgather CP':'zigzag CP'} · ${t('distributed optimizer enabled','distributed optimizer 已开启')}</p><a href="advanced/parallelism-memory.html" target="_blank" rel="noopener">${t('Diagrams, formulas & papers ↗','图示、公式与论文 ↗')}</a></details>`;
  }

  function parallelDiagram() {
    const block = (text, cls = '') => `<span class="parallel-block ${cls}">${text}</span>`;
    let caption = '',
      diagram = '';
    if (parallelView === 'tp') {
      caption = t(
        'Split one matrix across GPUs. Every GPU works on the same tokens, then communicates partial results. Useful when individual layers are large; fast interconnect matters.',
        '把同一个矩阵拆给多张 GPU。各卡处理同一批 token，再通信合并部分结果。适合单层很大，需要高速互联。');
      diagram =
        `<div class="parallel-flow">${block('X')} → <div class="matrix-shards">${block('W₀','blue')}${block('W₁','yellow')}${block('W₂','pink')}${block('W₃','green')}</div> → ${block('Y')}</div>`;
    }
    if (parallelView === 'pp') {
      caption = t(
        'Split model layers into stages. Microbatches move through stages; each stage holds fewer layers, but an empty stage is a pipeline bubble.',
        '按模型层分成多个 stage。Microbatch 依次流过，每卡保存更少的层；等待数据的空闲阶段就是 pipeline bubble。');
      diagram =
        `<div class="parallel-flow">${block('Layers 0–3','blue')} → ${block('4–7','yellow')} → ${block('8–11','pink')}</div><small>${t('Illustrative stages; actual layer counts follow the selected model.','示意分层；实际层数按当前模型配置。')}</small>`;
    }
    if (parallelView === 'ep') {
      caption = t(
        'Place different MoE experts on different GPUs. The router sends tokens to selected experts through all-to-all dispatch, then combines their outputs. EP is not another independent factor to multiply into TP × PP × CP × DP.',
        '将不同 MoE 专家放在不同卡上。Router 通过 all-to-all 将 token 发给选中专家，再合并输出。EP 不是在 TP × PP × CP × DP 之外再乘一次的独立维度。'
        );
      diagram =
        `<div class="parallel-flow">${block('router')} ⇄ <div class="matrix-shards">${block('Expert 0','blue')}${block('Expert 1','yellow')}${block('Expert 2','pink')}${block('Expert 3','green')}</div></div>`;
    }
    if (parallelView === 'cp') {
      caption = cpView === 'allgather' ? t(
        'slime allgather CP splits a packed token stream into contiguous rank chunks. The DSA path gathers K/V or index data as needed. With CP > 1 this mode is supported only by the guarded DSA architectures, not every model.',
        'slime 的 allgather CP 将打包 token 流连续切给各 rank；DSA 路径按需收集 K/V 或 index 数据。CP > 1 时仅支持代码允许的 DSA 架构，不能给所有模型随意打开。'
      ) : t(
        'The default zigzag layout gives each rank one early and one late chunk per sequence, balancing causal attention work. The attention backend exchanges K/V; zigzag describes the token layout, not a universal communication algorithm.',
        '默认 zigzag 布局把每条序列的一段前部、一段后部分给同一 rank，平衡 causal attention 计算量。Attention backend 交换 K/V；zigzag 描述 token 布局，不等于某一种固定通信算法。'
      );
      const ids = cpView === 'allgather' ? [
        [0, 1, 2, 3],
        [4, 5, 6, 7]
      ] : [
        [0, 1, 6, 7],
        [2, 3, 4, 5]
      ];
      diagram =
        `<div class="cp-switch"><button type="button" data-cp="zigzag" aria-pressed="${cpView==='zigzag'}">Zigzag</button><button type="button" data-cp="allgather" aria-pressed="${cpView==='allgather'}">Allgather</button></div><div class="token-sequence">${Array.from({length:8},(_,i)=>block(i,'token')).join('')}</div><div class="cp-ranks">${ids.map((v,i)=>`<div><small>Rank ${i}</small>${v.map(n=>block(n,i?'yellow':'blue')).join('')}</div>`).join('')}</div><small>${t('Illustration: one sequence, CP=2. These buttons explore the layout; the model recipe selects its supported mode.','示意：单序列 CP=2。按钮仅切换图示；实际配方采用模型支持的模式。')}</small>`;
    }
    if (parallelView === 'zero') {
      caption = t(
        'Data-parallel replicas process different samples. Shard optimizer state between them rather than storing every state on every GPU. slime enables Megatron’s distributed optimizer by default; this is not automatically ZeRO-3 parameter sharding.',
        '数据并行副本处理不同样本。优化器状态分片保存，避免每卡存一整份。slime 默认开启 Megatron distributed optimizer；这不自动等于 ZeRO-3 参数分片。');
      diagram =
        `<div class="cp-ranks">${[0,1].map(i=>`<div><small>DP ${i}</small>${block('model','blue')}${block('state '+i,'yellow')}</div>`).join('')}</div><small>reduce-scatter → optimizer → all-gather</small>`;
    }
    if (parallelView === 'memory') {
      const p = {
        flash: 30,
        dense: 9,
        glm47: 355,
        glm5: 744,
        deepseek: 671,
        qwen38: 27
      } [state.model];
      caption = t(
        'A whole-model tensor ledger before sharding: BF16 weights use about 2 bytes/parameter; two FP32 Adam moments use 8. CPU Adam moves optimizer work to the host. Gradients, master weights, activations, buffers and KV cache are additional. This is not a per-GPU memory estimate.',
        '分片前的全模型张量账本：BF16 权重约 2 字节/参数，两份 FP32 Adam moments 共 8 字节/参数。CPU Adam 将优化器工作移到主机。梯度、master weights、激活、buffers 和 KV cache 还需另算；这不是每卡显存估计。'
      );
      diagram =
        `<div class="memory-ledger"><div>${block('BF16 weights','blue')}<strong>≈ ${2*p} GB</strong><small>GPU · ${t('before TP / PP / EP','TP / PP / EP 分片前')}</small></div><div>${block('Adam m + v',state.cpuAdam?'green':'pink')}<strong>≈ ${8*p} GB</strong><small>${state.cpuAdam?'CPU':'GPU'} · ${t('before optimizer sharding','优化器分片前')}</small></div></div><small>${t('Approximate decimal GB; master-weight dtype and optimizer implementation can change storage.','十进制 GB 近似；master weight 精度和优化器实现会影响实际存储。')}</small>`;
    }
    return `${diagram}<p>${caption}</p>`;
  }

  function drawWorld() {
    const m = E.models[state.model],
      v = generated.validation;
    $('world').innerHTML = window.SlimeDiagrams.render(state, m, v, lang);
    $('world').classList.remove('enter');
    void $('world').offsetWidth;
    $('world').classList.add('enter');
    $('simulation-status').textContent = '';
    $('summary').innerHTML = [
      [t('Local Ray GPUs', '本地 Ray GPU'), v.localGpus],
      [t('Schedule', '调度'), state.schedule === 'async' ? (state.straw ? 'Distributed async' :
          'Fully async') : state
        .partial ? t('Sync + partial', '同步 + partial') : t('Synchronous', '同步')
      ],
      [t('Correction', '修正'), [state.correction === 'none' ? '—' : state.correction.toUpperCase(), state
        .r3 ? 'R3' :
        '', state.sc ? 'SC' : ''
      ].filter(Boolean).join(' + ')]
    ].map(([k, v]) => `<div><small>${k}</small><strong>${esc(v)}</strong></div>`).join('');
    const notes = [];
    if (state.layout === 'colocate') notes.push(t(
      'One GPU pool, two phases. Training and rollout release memory for each other.',
      '同一组 GPU，两段流程；训练和 rollout 交替释放显存。'));
    else if (state.schedule === 'async') notes.push(t(
      'Generation stays in flight while training progresses. Keep an eye on stale samples.',
      '训练推进时生成仍在继续，需要关注样本的过期程度。'));
    else notes.push(t(
      'Two GPU pools, synchronous steps. Switch to fully async when long-tail waits matter.',
      '两个 GPU 池，同步执行。长尾等待明显时，可尝试 fully async。'));
    if (generated.validation.errors.length) notes.push(
      `<span class="warning">${t(`${generated.validation.errors.length} setup item(s) to resolve before export.`, `导出前还有 ${generated.validation.errors.length} 项配置需要补齐。`)}</span>`
    );
    if (generated.validation.issues.length) notes.push(`<a href="#repair-options">${t('Review repair options and missing fields →', '查看修复方案与待填字段 →')}</a>`);

    $('advice').innerHTML = notes.map(n => `<p>${n}</p>`).join('');
  }

  function showArtifact() {
    if (!generated.files[artifact]) artifact = 'experiment.sh';
    $('artifact-tabs').innerHTML = Object.keys(generated.files).map(name =>
      `<button id="tab-${name.replace(/\W/g,'-')}" type="button" role="tab" data-artifact="${name}" aria-controls="artifact-panel" aria-selected="${name===artifact}" tabindex="${name===artifact?0:-1}">${name}</button>`
    ).join('');
    $('artifact-panel').setAttribute('aria-labelledby', 'tab-' + artifact.replace(/\W/g, '-'));
    const blocked = generated.validation.errors.length > 0 && artifact !== 'experiment.json';
    $('artifact-code').textContent = blocked ? t(
      '# Complete the setup items above to generate an executable artifact.\n# You can still download experiment.json to save your choices.',
      '# 补齐上方配置后生成可执行文件。\n# 仍可下载 experiment.json 保存当前选择。') : generated.files[artifact];
    $('download').disabled = blocked;
    $('copy').disabled = blocked;
  }

  const expandedDetails = new Set();
  function render() {
    document.querySelectorAll('[data-detail]').forEach(detail => {
      if (detail.open) expandedDetails.add(detail.dataset.detail);
      else expandedDetails.delete(detail.dataset.detail);
    });
    generated = E.generate(state, lang, build);
    $('steps').innerHTML = steps.map((step, i) =>
      `<button type="button" data-step="${i}" ${state.step===i?'aria-current="step"':''}><span>${String(i+1).padStart(2,'0')}</span>${step[0]}</button>`
    ).join('');
    $('step-number').textContent = String(state.step + 1).padStart(2, '0');
    $('step-title').textContent = steps[state.step][1];
    $('step-description').textContent = steps[state.step][2];
    $('choices').innerHTML = form();
    $('step-help').innerHTML = helpKeys[state.step].map(key =>
      `<button type="button" class="learn-button" data-learn="${key}">↗ ${key === 'straw' ? 'straw' : key.toUpperCase()} · ${t('why & math','原理与推导')}</button>`
    ).join('');
    $('previous').disabled = state.step === 0;
    $('next').textContent = state.step === 5 ? t('See my script ↓', '查看启动脚本 ↓') : t('Next →', '下一步 →');
    $('step-count').textContent = `${state.step+1} / 6`;
    $('undo').disabled = !history.length;
    $('redo').disabled = !future.length;
    drawWorld();
    document.querySelectorAll('[data-detail]').forEach(detail => {
      detail.open = expandedDetails.has(detail.dataset.detail);
    });
    const v = generated.validation;
    $('issues').innerHTML = (v.errors.length ?
      `<div class="notice error"><strong>${t('Finish these settings to export','补齐这些配置即可导出')}</strong><ul>${v.errors.map(e=>`<li>${esc(e)}</li>`).join('')}</ul></div>` :
      '') + (v.notes.length ?
      `<details class="notice"><summary>${t('Recipe notes','配方说明')} · ${v.notes.length}</summary><ul>${v.notes.map(n=>`<li>${esc(n)}</li>`).join('')}</ul></details>` :
      '');
    showArtifact();
    renderComparison();
    renderRepairs();
    renderSource();
    if (state.hardware !== 'unspecified') {
      $('install-image').value = state.hardware === 'blackwell' ? 'slimerl/slime:latest-cu130' : 'slimerl/slime:latest-cu129';
      $('install-image-value').textContent = $('install-image').value;
    }
    $('run-steps').innerHTML =
      `<li><strong>${t('Prepare weights & data.','准备模型与数据。')}</strong> ${t('Use the','先准备')} <a href="#install">${t('environment above','上方的运行环境')}</a>${t(', save experiment.sh in the slime checkout, then run ', '，将 experiment.sh 保存到 slime 仓库根目录，然后运行 ')}<code>bash experiment.sh prepare</code> → <code>bash experiment.sh convert</code>${t('. Existing BF16 / torch_dist paths can be used directly.', '。也可直接填写已有 BF16 / torch_dist 路径。')}</li><li><strong>${t('Connect the cluster.','连接集群。')}</strong> ${t(`Make ${v.localGpus} Ray GPUs available. Start the head with ray start --head; join workers with ray start --address=HEAD:6379. Use the same repo and paths on all nodes.`, `准备 ${v.localGpus} 张 Ray GPU。主节点执行 ray start --head；worker 使用 ray start --address=HEAD:6379 加入。各节点保持相同仓库和路径。`)} ${state.layout==='external'?t('Deploy engines using serving-reference.sh first.','先参照 serving-reference.sh 部署引擎。'):''}</li><li><strong>${t('Run & inspect.','启动并观察。')}</strong> <code>bash experiment.sh check</code> → <code>bash experiment.sh train</code>. ${t('Then check reward and train_rollout_logprob_abs_diff. Add a held-out evaluation before a long run.','观察 reward 与 train_rollout_logprob_abs_diff；长时间训练前添加独立验证集。')} <a href="${generated.provenance.recipe.url || 'https://github.com/THUDM/slime/blob/main/scripts/run-' + E.models[state.model].recipe + '.sh'}">${t('Recipe source ↗','配方源码 ↗')}</a></li>`;
  }

  const displayValue = value => esc(typeof value === 'boolean' ? value ? t('on', '开') : t('off', '关') : value === '' ? '—' : value);
  const resourceSummary = (s, v) => `${v.localGpus} ${t('local Ray GPUs', '本地 Ray GPU')} · ${v.total} train · ${v.rollout} rollout${s.layout === 'colocate' ? ' (' + t('shared pool', '共享池') + ')' : s.layout === 'external' ? ' (' + t('external pool, not counted locally', '外部池，不计入本地') + ')' : ''}`;
  const differencesHtml = diff => `<ul class="config-differences">${diff.map(d => `<li><strong>${esc(d.label)}</strong><span>${displayValue(d.before)} → ${displayValue(d.after)}</span></li>`).join('')}</ul>`;
  let comparisonSide = 'b';

  function renderComparison() {
    for (const id of ['restore-baseline', 'share-comparison', 'clear-baseline']) $(id).disabled = !baseline;
    $('comparison-count').textContent = '';
    if (!baseline) {
      $('comparison-content').innerHTML = `<p>${t('Save A, then change any choice in B to compare the tradeoffs.', '先保存 A，再修改 B 的任意选项，即可对照取舍。')}</p>`;
      return;
    }
    const comparison = E.compare(baseline, state, lang);
    $('comparison-count').textContent = `· ${comparison.differences.length} ${t('changes', '项变化')}`;
    $('comparison-content').innerHTML = `<div class="comparison-switch" aria-label="${t('View configuration', '查看配置')}">${['a', 'b'].map(side => `<button type="button" data-compare-side="${side}" aria-pressed="${comparisonSide === side}">${side.toUpperCase()} · ${side === 'a' ? t('Saved baseline', '已存基线') : t('Current / export', '当前 / 导出')}</button>`).join('')}</div>
      <div class="comparison-pair">${[['a', baseline], ['b', state]].map(([side, s]) => `<section class="comparison-config ${comparisonSide === side ? 'selected' : ''}" data-compare-config="${side}"><h3>${side.toUpperCase()} · ${esc(E.models[s.model].name)} · ${esc(s.schedule)}</h3><p>${resourceSummary(s, comparison[side])}</p><div class="world">${window.SlimeDiagrams.topology(s, E.models[s.model], comparison[side], lang)}</div><p>${comparison[side].errors.length ? `${comparison[side].errors.length} ${t('unresolved items', '项待解决')}` : t('Static checks passed', '静态检查通过')}</p>${comparison[side].errors.length ? `<ul>${comparison[side].errors.map(error => `<li>${esc(error)}</li>`).join('')}</ul>` : ''}</section>`).join('')}</div>
      ${comparison.differences.length ? differencesHtml(comparison.differences) : `<p>${t('A and B have the same configuration.', 'A 与 B 的配置相同。')}</p>`}
      <div class="comparison-insights">${comparison.insights.map(text => `<p>${esc(text)}</p>`).join('')}</div><p class="comparison-note">${t('Mechanisms and allocation counts, not measured speed or a memory-fit prediction. Current B remains the export target.', '展示机制与资源分配，不预测实测速度或显存容量。当前 B 始终是导出目标。')}</p>`;
  }

  function renderRepairs() {
    $('repair-options').innerHTML = generated.validation.issues.map(issue => `<section class="repair-issue"><h3>${esc(issue.message)}</h3><div class="repair-jumps">${issue.fields.map(field => `<button type="button" data-repair-target="${field}">${t('Edit', '填写 / 修改')} ${esc(E.fieldLabel(field, lang))} →</button>`).join('')}</div>${issue.actions.map(action => {
      const next = E.sanitize({...state, ...action.changes});
      const comparison = E.compare(state, next, lang);
      return `<details class="repair-preview"><summary>${t('Preview', '预览')} · ${esc(action.label)}</summary>${differencesHtml(comparison.differences)}<p>${resourceSummary(state, comparison.a)} → ${resourceSummary(next, comparison.b)}</p><p>${t('Allocation only; check available hardware and memory before applying.', '仅为资源分配；应用前核对可用硬件与显存条件。')}</p>${comparison.b.errors.length ? `<strong>${t('Still required after applying:', '应用后仍需解决：')}</strong><ul>${comparison.b.errors.map(error => `<li>${esc(error)}</li>`).join('')}</ul>` : `<p>${t('Static constraints would pass. GPU execution remains unverified.', '静态约束将通过；GPU 运行仍待验证。')}</p>`}<button type="button" class="button light" data-repair-issue="${issue.code}" data-repair-action="${action.id}">${t('Apply this repair', '应用此修复')}</button></details>`;
    }).join('')}</section>`).join('');
  }

  function renderSource() {
    const p = generated.provenance;
    const unknown = t('Not recorded', '未记录');
    const sourceLink = source => source.url ? `<a href="${esc(source.url)}">${esc(source.path)} ↗</a>` : esc(source.path);
    $('recipe-source-content').innerHTML = `<p><strong>${p.staticChecksPassed ? t('Static checks passed', '静态检查通过') : t('Static checks have unresolved items', '静态检查有待解决项')}</strong> · ${t('No measured run attached to this configuration.', '此配置未附实测记录。')}</p><dl>
      <dt>${t('Page / generator checkout', '页面 / 生成器 checkout')}</dt><dd>${esc(p.page.commit || unknown)} · ${p.page.dirty === null ? unknown : p.page.dirty ? t('local changes present; commit is the base revision', '含本地修改；commit 为基础版本') : t('clean checkout', '干净 checkout')}</dd>
      <dt>${t('Recipe basis', '配方依据')}</dt><dd>${sourceLink(p.recipe)}<br>SHA-256: ${esc(p.recipe.sha256 || unknown)}</dd>
      <dt>${t('Model dimensions', '模型尺寸配置')}</dt><dd>${sourceLink(p.modelConfig)}</dd>
      <dt>${t('Recorded Dockerfile defaults', '记录的 Dockerfile 默认版本')}</dt><dd>${Object.entries(p.environment.recipeVersions || {}).map(([k, v]) => `${esc(k)}: ${esc(v || unknown)}`).join('<br>') || unknown}</dd>
      <dt>${t('Suggested image / runtime digest', '建议镜像 / 实际 digest')}</dt><dd>${esc(p.environment.suggestedImageTag || unknown)} / ${unknown}</dd>
      <dt>${t('Checkpoint ID / revision', 'Checkpoint ID / revision')}</dt><dd>${esc(p.checkpoint.id)} / ${unknown}</dd>
      <dt>${t('Installed slime commit', '服务器实际 slime commit')}</dt><dd>${unknown} · <code>git rev-parse HEAD</code></dd></dl>
      <p>${t('These records describe the sources used by this page, not the software installed on your servers. Local source hashes are included in experiment.json; dirty builds can differ from the linked commit. A script basis does not certify your modified combination. Record image digest, checkpoint revision and short-run results before scaling.', '这些记录描述页面采用的源码，不代表服务器实际安装版本。experiment.json 包含本地源码哈希；含本地修改的构建可能与 commit 链接不同。配方来源不等于对修改后组合的认证；扩容前记录镜像 digest、checkpoint revision 和短实验结果。')}</p>`;
  }

  function saveBaseline() {
    try {
      if (baseline) localStorage.setItem('slime-baseline-v1', JSON.stringify(baseline));
      else localStorage.removeItem('slime-baseline-v1');
    } catch (_) {}
    renderComparison();
  }
  $('save-baseline').addEventListener('click', () => {
    baseline = E.sanitize(state);
    saveBaseline();
    toast(t('Baseline A saved. Further edits change B only.', '基线 A 已保存，后续编辑只改变 B。'));
  });
  $('clear-baseline').addEventListener('click', () => {baseline = null; saveBaseline();});
  $('restore-baseline').addEventListener('click', () => {
    if (baseline) commit(E.sanitize({...baseline, step: state.step}), t('A restored into B. Undo restores the previous B.', 'A 已恢复到 B；撤销可找回之前的 B。'));
  });
  $('comparison-content').addEventListener('click', event => {
    const button = event.target.closest('[data-compare-side]');
    if (!button) return;
    comparisonSide = button.dataset.compareSide;
    renderComparison();
    document.querySelector(`[data-compare-side="${comparisonSide}"]`).focus({preventScroll: true});
  });
  $('repair-options').addEventListener('click', event => {
    const field = event.target.closest('[data-repair-target]')?.dataset.repairTarget;
    if (field) {
      navigate(['nodes', 'rollout'].includes(field) ? 1 : ['prefill', 'ib'].includes(field) ? 4 : 5);
      document.querySelector(`[name="${field}"]`)?.focus();
      return;
    }
    const button = event.target.closest('[data-repair-action]');
    if (!button) return;
    const action = E.validate(state, lang).issues.find(issue => issue.code === button.dataset.repairIssue)?.actions.find(action => action.id === button.dataset.repairAction);
    if (action) {
      commit(E.sanitize({...state, ...action.changes}), t('Repair applied. Undo restores all changed fields.', '修复已应用，撤销可一起恢复所有修改字段。'));
      $('recipe-title').setAttribute('tabindex', '-1');
      $('recipe-title').focus({preventScroll: true});
    }
  });

  function navigate(step) {
    commit({
      ...state,
      step
    });
    $('steps').scrollIntoView({behavior: 'smooth', block: 'start'});
    $('step-title').focus({
      preventScroll: true
    });
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    $('world').classList.remove('running');
    $('play').setAttribute('aria-pressed', 'false');
    $('play').textContent = '▶ ' + t('Play a round', '演示一轮');
  }
  const walkthroughState = E.preset('first');
  let walkthroughStage = 0, walkthroughTimer = null;
  const walkthroughSteps = [
    [t('Prompt', '输入'), t('Start with one question.', '从一个问题开始。'), t('For this small example, ask “What is 6 × 7?” The starting model already knows how to generate text. RL uses feedback on its sampled answers to update its behavior.', '这个例子问「6 × 7 等于多少？」。起始模型已经能生成文本，RL 用采样回答得到的反馈来更新它的行为。')],
    [t('Generate', '生成'), t('SGLang samples eight answers.', 'SGLang 采样八个回答。'), t('Sample eight responses to the same prompt. Answers can repeat or be wrong. Save the generated tokens and rollout log-probabilities for training.', '对同一个 prompt 采样八次。回答可能重复，也可能出错；保存生成 token 及 rollout log-prob，供训练使用。')],
    [t('Reward', '奖励'), t('A verifier scores each answer.', '验证器给每个回答打分。'), t('In this illustrative binary reward, answer 42 earns 1 and the other answers earn 0. A coding task could instead run tests. The reward function determines which behavior receives a useful learning signal.', '这个二值奖励示例中，答案 42 得 1 分，其他答案得 0 分。代码任务可以运行测试打分。奖励函数决定哪些行为获得有效学习信号。')],
    [t('Train', '训练'), t('Relative scores become a learning signal.', '相对得分变成训练信号。'), t('Four correct and four incorrect answers give mean reward 0.5. With the repository’s sample-standard-deviation normalization, advantages are about +0.94 and −0.94. Megatron uses these signals with GRPO, clipping and the configured corrections to update model weights.', '四个正确、四个错误，平均 reward 为 0.5。按仓库采用的样本标准差归一化，advantage 约为 +0.94 和 −0.94。Megatron 结合 GRPO、裁剪和配置的修正，用这些信号更新模型权重。')],
    [t('Sync', '同步'), t('New weights return to SGLang.', '新权重回到 SGLang。'), t('After this synchronous training phase, send the updated weights to SGLang for the next batch. Reward improvement is measured over runs, not guaranteed for the next answer. Later you can explore separate GPU pools, partial rollout and fully async scheduling.', '这轮同步训练结束后，将新权重交给 SGLang，用于下一批生成。Reward 改善要通过实验评估，不保证下一次回答更好。接下来可以探索独立 GPU 池、partial rollout 和 fully async 调度。')]
  ];
  function pauseWalkthrough() {
    clearInterval(walkthroughTimer);
    walkthroughTimer = null;
    $('walkthrough-play').setAttribute('aria-pressed', 'false');
    $('walkthrough-play').textContent = '▶ ' + t('Continue', '继续');
  }
  function renderWalkthrough() {
    $('walkthrough-stages').innerHTML = walkthroughSteps.map(([label], stage) => `<button type="button" data-walkthrough-stage="${stage}" ${stage === walkthroughStage ? 'aria-current="step"' : ''}>${stage + 1} · ${label}</button>`).join('');
    $('walkthrough-stage-title').textContent = walkthroughSteps[walkthroughStage][1];
    $('walkthrough-explanation').textContent = walkthroughSteps[walkthroughStage][2];
    $('walkthrough-world').innerHTML = window.SlimeDiagrams.walkthrough(walkthroughState, walkthroughStage, lang);
    $('walkthrough-back').disabled = walkthroughStage === 0;
    $('walkthrough-next').disabled = walkthroughStage === 4;
  }
  function playWalkthrough() {
    if (walkthroughTimer) {pauseWalkthrough(); return;}
    if (walkthroughStage === 4) walkthroughStage = 0;
    stop();
    renderWalkthrough();
    $('walkthrough-play').setAttribute('aria-pressed', 'true');
    $('walkthrough-play').textContent = 'Ⅱ ' + t('Pause', '暂停');
    walkthroughTimer = setInterval(() => {
      if (walkthroughStage === 4) {pauseWalkthrough(); return;}
      walkthroughStage++;
      renderWalkthrough();
    }, 15000);
  }
  function openWalkthrough() {
    $('learn').hidden = false;
    renderWalkthrough();
    $('walkthrough-title').focus({preventScroll: true});
    $('learn').scrollIntoView({behavior: 'smooth'});
  }
  document.querySelectorAll('a[href="#learn"]').forEach(link => link.addEventListener('click', openWalkthrough));
  window.addEventListener('hashchange', () => {if (location.hash === '#learn') openWalkthrough(); else pauseWalkthrough();});
  document.addEventListener('visibilitychange', () => {if (document.hidden) pauseWalkthrough();});
  $('walkthrough-play').addEventListener('click', playWalkthrough);
  $('walkthrough-stages').addEventListener('click', event => {
    const button = event.target.closest('[data-walkthrough-stage]');
    if (!button) return;
    pauseWalkthrough();
    walkthroughStage = Number(button.dataset.walkthroughStage);
    renderWalkthrough();
    document.querySelector(`[data-walkthrough-stage="${walkthroughStage}"]`).focus({preventScroll: true});
  });
  for (const [id, delta] of [['walkthrough-back', -1], ['walkthrough-next', 1], ['walkthrough-restart', 0]]) {
    $(id).addEventListener('click', () => {
      pauseWalkthrough();
      walkthroughStage = delta ? Math.min(4, Math.max(0, walkthroughStage + delta)) : 0;
      renderWalkthrough();
    });
  }
  $('adopt-walkthrough').addEventListener('click', () => {
    pauseWalkthrough();
    commit(E.sanitize(walkthroughState), t('GRPO starting point adopted. Undo restores your experiment.', '已采用 GRPO 起点，撤销可恢复你的实验。'));
    location.hash = 'lab';
  });
  if (location.hash === '#learn') openWalkthrough();
  $('play').addEventListener('click', () => {
    pauseWalkthrough();
    if (timer) {
      stop();
      $('simulation-status').textContent = t('Paused. You can change any piece.', '已暂停，可以修改任何一块。');
      return;
    }
    phase = 0;
    $('world').classList.add('running');
    $('play').setAttribute('aria-pressed', 'true');
    $('play').textContent = 'Ⅱ ' + t('Pause', '暂停');
    const advance = () => {
      if (state.step === 4) {
        const stages = [
          ['prefill', state.hicache ? t('Look up prefixes in GPU / CPU cache, then process uncached tokens.',
            '在 GPU / CPU cache 查找前缀，再处理未命中的 token。') : t('Read the prompt using the engine’s GPU cache.',
            '读取 prompt，使用引擎内的 GPU cache。')],
          ...(state.speculative ? [
            ['draft', t('The EAGLE head proposes candidate tokens.', 'EAGLE 投机头提出候选 token。')],
            ['verify', t('The target model verifies the candidate block.', '目标模型批量验证候选 token。')],
            ['commit', t('Commit the accepted prefix and a corrected or bonus token, then continue.',
              '提交接受的前缀与修正或额外 token，然后继续生成。')]
          ] : [['decode', t('The target model decodes one token, then repeats.', '目标模型解码一个 token，再继续下一步。')]])
        ];
        if (state.pd) stages.splice(1, 0, ['transfer', t('Transfer the prefill cache to the decode engine.',
          '将 prefill 缓存传输给 decode 引擎。')]);
        const [stage, label] = stages[phase % stages.length];
        $('simulation-status').textContent = label;
        $('world').querySelectorAll('[data-serving-phase]').forEach(node =>
          node.classList.toggle('active', node.dataset.servingPhase === stage));
        phase++;
        return;
      }
      const async = state.schedule === 'async';
      const labels = async ? [t('Rollouts enter the warm queue; new generations keep running.',
        'Rollout 进入预热队列；新的生成持续运行。'), t(
        'Megatron trains a ready batch while SGLang keeps generating.',
        'Megatron 训练就绪 batch，SGLang 继续生成。'), t(
        'Updated weights reach SGLang; queued samples can be older.', '新权重到达 SGLang；队列里可能仍有旧样本。'
        )] : [t(
          'SGLang generates a batch; rewards are evaluated.', 'SGLang 生成一批样本，并计算奖励。'), state
        .partial ? t(
          'Save unfinished prefixes, then train on complete groups.', '保存未完成前缀，再用已完成组训练。') : t(
          'Megatron trains on the completed batch.', 'Megatron 使用完成的 batch 训练。'), state.partial ?
        t(
          'Sync weights; resume saved prefixes with the new version.', '同步权重，用新版本续跑已保存的前缀。') : t(
          'New weights are synchronized before the next batch.', '同步新权重，然后进入下一批。')
      ];
      $('simulation-status').textContent = labels[phase % 3];
      $('world').querySelector('.trainer').classList.toggle('active', async || phase % 3 === 1);
      $('world').querySelectorAll('.sampler').forEach(node => node.classList.toggle('active', async ||
        phase % 3 === 0));
      phase++;
    };
    advance();
    timer = setInterval(advance, 2000);
  });
  $('choices').addEventListener('submit', e => e.preventDefault());
  $('choices').addEventListener('click', e => {
    const b = e.target.closest('[data-parallel], [data-cp]');
    if (!b) return;
    if (b.dataset.parallel) parallelView = b.dataset.parallel;
    if (b.dataset.cp) cpView = b.dataset.cp;
    const host = document.querySelector('.parallel-explorer');
    host.outerHTML = parallelPanel();
    document.querySelector('.parallel-explorer').open = true;
    document.querySelector(b.dataset.parallel ? `[data-parallel="${parallelView}"]` :
        `[data-cp="${cpView}"]`)
      .focus({
        preventScroll: true
      });
  });
  $('choices').addEventListener('change', e => {
    const input = e.target;
    if (!input.name) return;
    const value = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input
        .value) :
      input.value;
    let message;
    if (input.type === 'number' && (!Number.isInteger(value) || value < E.numberLimits[input.name][0] ||
        value > E
        .numberLimits[input.name][1])) {
      toast(t(
        `Use a whole number from ${E.numberLimits[input.name][0]} to ${E.numberLimits[input.name][1]}.`,
        `请输入 ${E.numberLimits[input.name][0]} 到 ${E.numberLimits[input.name][1]} 的整数。`));
      render();
      return;
    }
    if (input.name === 'schedule' && value === 'async' && state.layout === 'colocate') message = t(
      'Moved to separate GPUs to keep rollout alive. Undo restores both choices.',
      '已切换到独立 GPU 以持续 rollout；撤销会同时恢复两个选择。');
    if (input.name === 'layout' && value === 'colocate' && state.schedule === 'async') message = t(
      'Colocation uses synchronous execution. Undo restores the async layout.',
      '同卡布局采用同步执行；撤销可恢复异步布局。');
    const focusName = input.name,
      focusValue = input.value;
    commit(E.change(state, input.name, value), message);
    const next = Array.from($('choices').elements).find(el => el.name === focusName && (el.type !==
      'radio' || el
      .value === focusValue));
    if (next) next.focus({
      preventScroll: true
    });
  });
  $('steps').addEventListener('click', e => {
    const b = e.target.closest('[data-step]');
    if (b) navigate(Number(b.dataset.step));
  });
  $('previous').addEventListener('click', () => navigate(Math.max(0, state.step - 1)));
  $('next').addEventListener('click', () => {
    if (state.step < 5) navigate(state.step + 1);
    else $('recipe').scrollIntoView({
      behavior: 'smooth'
    });
  });
  document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => commit(E.preset(
    b.dataset
    .preset), t('Starting point loaded. Your previous choices are one undo away.',
    '已载入起点；撤销即可恢复之前的选择。'))));
  $('undo').addEventListener('click', () => {
    if (!history.length) return;
    future.push({
      ...state
    });
    state = history.pop();
    stop();
    render();
    persist();
    toast(t('Choice undone.', '已撤销。'));
  });
  $('redo').addEventListener('click', () => {
    if (!future.length) return;
    history.push({
      ...state
    });
    state = future.pop();
    stop();
    render();
    persist();
    toast(t('Choice restored.', '已重做。'));
  });
  $('reset').addEventListener('click', () => commit({
    ...E.defaults
  }, t('Reset to the starting experiment. Undo is available.', '已重置为起步实验，可以撤销。')));
  document.addEventListener('keydown', e => {
    if (/input|textarea|select/i.test(e.target.tagName) || $('learn-dialog').open) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      (e.shiftKey ? $('redo') : $('undo')).click();
    }
  });
  $('artifact-tabs').addEventListener('click', e => {
    const b = e.target.closest('[data-artifact]');
    if (b) {
      artifact = b.dataset.artifact;
      showArtifact();
      document.querySelector('[role=tab][aria-selected=true]').focus();
    }
  });
  $('artifact-tabs').addEventListener('keydown', e => {
    const keys = Object.keys(generated.files);
    let i = keys.indexOf(artifact);
    if (e.key === 'ArrowRight') i = (i + 1) % keys.length;
    else if (e.key === 'ArrowLeft') i = (i - 1 + keys.length) % keys.length;
    else if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = keys.length - 1;
    else return;
    e.preventDefault();
    artifact = keys[i];
    showArtifact();
    document.querySelector('[role=tab][aria-selected=true]').focus();
  });
  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast(t('Copied.', '已复制。'));
    } catch (_) {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed';
      area.style.left = '-9999px';
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand('copy');
      area.remove();
      toast(ok ? t('Copied.', '已复制。') : t('Clipboard unavailable. Select and copy the text manually.',
        '剪贴板不可用，请选中文字手动复制。'));
    }
  }
  $('install-methods').addEventListener('change', e => {
    $('install-docker').hidden = e.target.value !== 'docker';
    $('install-conda').hidden = e.target.value !== 'conda';
  });
  $('install-image').addEventListener('change', e => {
    $('install-image-value').textContent = e.target.value;
    commit(E.change(state, 'hardware', e.target.value.endsWith('cu130') ? 'blackwell' : 'hopper'),
      t('Hardware context carried into the builder. Model and GPU counts are unchanged.', '硬件条件已带入配置器，模型和 GPU 数量保持不变。'));
  });
  document.querySelectorAll('[data-install-copy]').forEach(button => {
    button.hidden = false;
    button.addEventListener('click', () => copy($(button.dataset.installCopy).textContent));
  });
  $('install-docker').hidden = !document.querySelector('[name="install-method"][value="docker"]').checked;
  $('install-conda').hidden = !document.querySelector('[name="install-method"][value="conda"]').checked;
  $('install-image-value').textContent = $('install-image').value;
  $('install-methods').hidden = false;
  $('install-image-control').hidden = false;
  $('copy').addEventListener('click', () => copy(generated.files[artifact]));
  $('load-experiment').addEventListener('click', () => $('import-experiment').click());
  $('import-experiment').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw new Error('size');
      const data = JSON.parse(await file.text());
      if (!data || (data.schema !== undefined && data.schema !== 1) || !Object.hasOwn(E.models, (data.choices || data).model)) throw new Error('schema');
      commit(E.sanitize(data), t('Choices loaded into B. Source records are regenerated for this page; Undo is available.', '已将选择载入 B，来源记录按当前页面重新生成；可以撤销。'));
    } catch (_) {
      toast(t('Use a schema 1 experiment.json or a saved choices object (up to 1 MiB).', '请选择 schema 1 的 experiment.json 或已保存的 choices 对象，大小不超过 1 MiB。'));
    }
    event.target.value = '';
  });
  $('download').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([generated.files[artifact]], {
      type: artifact.endsWith('.json') ? 'application/json' : 'text/plain;charset=utf-8'
    }));
    const a = document.createElement('a');
    a.href = url;
    a.download = artifact;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(t('Downloaded.', '已下载。'));
  });
  $('share').addEventListener('click', () => {
    const url = new URL(location.href);
    url.hash = new URLSearchParams({
      experiment: JSON.stringify(E.shareChoices(state))
    }).toString();
    copy(url.href);
    toast(t('Share link copied. Machine paths and addresses are excluded.', '分享链接已复制，不包含机器路径与地址。'));
  });
  $('share-comparison').addEventListener('click', () => {
    if (!baseline) return;
    const url = new URL(location.href);
    url.hash = new URLSearchParams({experiment: JSON.stringify(E.shareChoices(state)), baseline: JSON.stringify(E.shareChoices(baseline))}).toString();
    copy(url.href);
    toast(t('A/B link copied; both configurations exclude private paths and addresses.', 'A/B 链接已复制，两套配置均不含私有路径与地址。'));
  });
  $('step-help').addEventListener('click', e => {
    const b = e.target.closest('[data-learn]');
    if (!b) return;
    const info = theory[b.dataset.learn];
    window.SlimeReader.open(info.page, b, info);
  });
  const preview = document.querySelector('.preview');
  const previewPosition = document.createComment('preview position');
  preview.before(previewPosition);
  const mobile = matchMedia('(max-width: 720px)');
  let labVisible = false;
  const updatePreviewButton = () => {
    $('mobile-preview').hidden = !mobile.matches || !labVisible;
  };
  new IntersectionObserver(entries => {
    labVisible = entries[0].isIntersecting;
    updatePreviewButton();
  }).observe($('lab'));
  mobile.addEventListener('change', () => {
    if (!mobile.matches && $('mobile-preview-dialog').open) $('mobile-preview-dialog').close();
    updatePreviewButton();
  });
  $('mobile-preview').addEventListener('click', () => {
    $('mobile-preview-host').append(preview);
    $('mobile-preview-dialog').showModal();
  });
  $('close-preview').addEventListener('click', () => $('mobile-preview-dialog').close());
  $('mobile-preview-dialog').addEventListener('close', () => {
    previewPosition.after(preview);
    stop();
    $('mobile-preview').focus({
      preventScroll: true
    });
  });
  $('interactive-lab').hidden = false;
  $('recipe').hidden = false;
  render();
})();
