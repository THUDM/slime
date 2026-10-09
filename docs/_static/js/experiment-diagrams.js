/* Explanatory diagrams: the values illustrate a mechanism, not measured performance. */
(function(root) {
  'use strict';
  const esc = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  } [c]));

  function render(s, m, v, lang) {
    const t = (en, zh) => lang === 'zh' ? zh : en;
    const chip = (text, color = '') => `<span class="diagram-chip ${color}">${esc(text)}</span>`;
    const arrow = '<b class="diagram-arrow" aria-hidden="true">→</b>';
    const row = (...items) => `<div class="diagram-row">${items.join(arrow)}</div>`;
    const gpu = count =>
      `<div class="chips" aria-hidden="true">${'<i class="gpu"></i>'.repeat(Math.min(8, count))}</div>`;
    const trainer =
      `<div class="world-node trainer"><strong>Megatron</strong><small>${s.precision === 'fp8train' ? 'FP8' : 'BF16'} · ${v.total} GPU</small>${gpu(v.total)}</div>`;
    const precision = s.precision === 'bf16' ? 'BF16' : s.precision === 'int4' ? 'INT4' : 'FP8';
    const serving = (label, count) =>
      `<div class="world-node sampler"><strong>${esc(label)}</strong><small>SGLang · ${precision}${count ? ' · '+count+' GPU' : ''}</small>${s.pd ? '<div class="pd-nodes"><span>Prefill</span><span>Decode</span></div>' : gpu(count || m.engine)}${s.hicache ? '<div class="cache-tier">↳ HiCache · CPU</div>' : ''}</div>`;
    let topology;
    if (s.layout === 'external') {
      topology =
        `<div class="external-map" data-topology="external"><svg viewBox="0 0 400 280" preserveAspectRatio="none" aria-hidden="true"><path d="M80 140 Q180 10 315 50 M80 140 Q220 100 325 180 M80 140 Q140 280 210 260"/></svg><div class="external-trainer">${trainer}<small>${t('slime training job', 'slime 训练任务')}</small></div><div class="external-cluster cluster-a">${serving(t('Cluster A', '集群 A'))}</div><div class="external-cluster cluster-b">${serving(t('Cluster B', '集群 B'))}</div><div class="external-cluster cluster-c">${serving(t('Cluster …', '集群 …'))}</div></div><p class="diagram-caption">${t('Independently managed fleets, connected by endpoint. Locations are illustrative; external engines may also be in one datacenter.', '通过 endpoint 连接独立管理的 GPU 集群。位置仅为示意；external 引擎也可以位于同一机房。')}</p>`;
    } else {
      topology =
        `<div data-topology="${s.layout}" class="${s.layout === 'colocate' ? 'world-colocate' : 'managed-pools'}"><div class="world-nodes">${trainer}<div class="flow-arrow">${s.schedule === 'async' ? '⇄' : '⇢'}</div>${serving('Rollout', v.rollout)}</div><span class="pool-label">${s.layout === 'colocate' ? t('one GPU pool · alternating phases', '同一 GPU 池 · 交替工作') : t('slime manages two dedicated GPU pools', 'slime 统一管理 · 两个独立 GPU 池')}</span></div>`;
    }
    let lesson;
    let title;
    switch (s.step) {
      case 0:
        title = t('Where does the learning signal come from?', '训练信号从哪里来？');
        lesson = s.task === 'math' ?
          `<div data-signal="math">${row(chip('Prompt', 'blue'), chip('8 × '+t('answer', '回答')), chip(t('Math verifier', '数学验证器'), 'yellow'))}<div class="reward-samples">${[1,0,1,1,0,0,1,0].map(n => chip(n, n ? 'green' : '')).join('')}</div>${row(chip('reward'), chip(s.sc ? 'REINFORCE + SC' : 'GRPO', 'blue'))}</div><p>${t('One prompt, eight answers. Compare verifier scores within the group to compute advantages.', '同一个 prompt 生成八个回答；比较组内验证器得分，计算相对优势。')}</p>` :
          `<div data-signal="custom">${row(chip('Agent', 'blue'), chip(t('Tool / environment', '工具 / 环境'), 'yellow'))}<div class="feedback-return">↶ ${t('observe → act → observe → …', '观察 → 行动 → 观察 → …')} ↶</div>${row(chip(t('Trajectory + loss mask', '轨迹 + loss mask')), chip(t('Your reward function', '自定义奖励'), 'green'))}</div><p>${t('Collect tool results, verifier feedback, or environment rewards. Train on model-generated tokens in the trajectory.', '收集工具结果、验证器反馈或环境奖励；对轨迹中模型生成的 token 进行训练。')}</p>`;
        lesson +=
          `<div class="architecture-strip" data-architecture="${m.moe ? 'moe' : 'dense'}">${chip(m.moe ? 'MoE · router' : 'Dense', 'blue')}${m.moe ? ' ↗ '+chip('E₁')+' '+chip('E₂')+' '+chip('…') : ' → '+chip(t('all layers', '完整网络'))}<small>${esc(m.size)}</small></div>`;
        break;
      case 1:
        title = t('Follow a sample across updates', '跟着样本走过权重更新');
        lesson =
          `<div class="schedule-lanes" data-schedule="${s.schedule}"><div><b>rollout</b>${chip('v₀', 'blue')}${chip(s.partial ? t('save prefix', '保存前缀') : t('ready', '就绪'), 'yellow')}${s.schedule === 'async' ? chip('v₀ / v₁', 'blue') : chip(t('wait', '等待'))}</div><div><b>train</b>${chip(s.schedule === 'async' ? t('warm queue', '预热队列') : t('wait', '等待'))}${chip('update → v₁', 'green')}${chip(s.partial ? t('resume', '续跑') : t('next batch', '下一批'), 'blue')}</div></div><p>${s.schedule === 'async' ? t('Rollout stays in flight during training. Samples in the queue may use older weights.', '训练时 rollout 持续进行，队列中的样本可能来自旧权重。') : t('Complete groups train together; synchronize weights before the next generation phase.', '完整样本组一起训练；同步权重后再进入下一段生成。')}</p>${s.partial ? `<div class="prefix-tokens">${chip('prefix · v₀', s.maskPartial ? 'masked' : 'yellow')}${chip('continuation · v₁', 'blue')}<small>${s.maskPartial ? t('Old tokens masked in loss; still used as context.', '旧 token 不计 loss，仍作为上下文。') : t('Keep old tokens and their rollout log-probs.', '保留旧 token 及其 rollout log-prob。')}</small></div>` : ''}${row(chip('Adam m + v'), chip(s.cpuAdam ? 'CPU RAM' : 'GPU HBM', s.cpuAdam ? 'green' : 'yellow'))}<p>${s.recompute ? t('Recompute activations during backward to save GPU memory.', '反向传播时重新计算激活，节省 GPU 显存。') : t('Keep activations for backward; this uses more GPU memory.', '保留激活供反向传播使用，需要更多 GPU 显存。')}</p>`;
        break;
      case 2: {
        title = t('Different tensors, different precision', '不同张量，各选精度');
        const bar = (label, bits, format, cache = '') =>
          `<div class="bit-row"${cache ? ` data-cache="${cache}"` : ''}><span>${label}</span><div class="bit-bar" style="--bits:${bits};--max-bits:${m.hybrid ? 32 : 16}">${'<i></i>'.repeat(bits)}</div><b>${format}</b></div>`;
        const ssm = s.mambaDtype === 'auto' ? m.hybrid?.ssmDtype : s.mambaDtype;
        lesson =
          `<div data-precision="${s.precision}">${bar('Train', s.precision === 'fp8train' ? 8 : 16, s.precision === 'fp8train' ? 'FP8' : 'BF16')}${bar('Rollout', precision === 'BF16' ? 16 : precision === 'FP8' ? 8 : 4, precision)}${bar('Attn KV', s.kv === 'fp8' ? 8 : 16, s.kv === 'fp8' ? 'FP8' : 'auto*', 'attention')}${m.hybrid ? bar('SSM', ssm === 'float32' ? 32 : 16, ssm === 'float32' ? 'FP32' : 'BF16', 'ssm') : ''}</div><p>${t('Each cell represents one bit per element, before scales and metadata. *Auto KV follows the serving stack; the 16-bit row is illustrative. These rows show dtype, not total pool memory.', '每格表示每元素一 bit，不含 scale 和元数据。*Auto KV 由推理栈决定，图中以 16 bit 为例。各行展示 dtype，不代表整个缓存池大小。')}</p>`;
        if (m.hybrid) lesson +=
          `<div class="hybrid-cache-map"><div><strong>${m.hybrid.attentionLayers} × Attention</strong><span>${t('More tokens → more KV entries', 'token 增多 → KV 条目增多')}</span></div><div><strong>${m.hybrid.linearLayers} × ${esc(m.hybrid.linear)}</strong><span>hₜ₋₁ → hₜ</span><small>${t('Recurrent state + convolution state', '递归状态 + 卷积状态')}</small></div></div><p>${t('Mamba cache keeps recurrent states and snapshots, with a fixed state shape per slot. Its SSM dtype does not change with FP8 attention KV; convolution state has its own dtype.', 'Mamba cache 保存递归状态与快照，每个 slot 的状态形状固定。SSM 精度不跟随 attention KV 的 FP8 开关改变；卷积状态也有自己的 dtype。')}</p><p>${t('State / attention KV memory ratio', '状态 / attention KV 内存比例')}: ${esc(s.mambaRatio || t('serving default', '推理栈默认'))}</p>`;
        break;
      }
      case 3: {
        title = t('Make a mismatch visible', '让训推差异看得见');
        const weight = s.correction === 'tis' ? 2 : s.correction === 'icepop' ? 0 : 1;
        lesson =
          `<div class="probability-demo" data-correction="${s.correction}"><div><span>q(a) = 0.1</span><i style="--prob:10%"></i></div><div><span>p(a) = 0.4</span><i style="--prob:40%"></i></div></div>${row(chip('p / q = 4'), chip(s.correction === 'none' ? t('no correction', '无修正') : s.correction.toUpperCase(), 'blue'), chip('w = '+weight, weight ? 'green' : 'masked'))}<p>${t('Illustrative probabilities: TIS clips to 2; ICE-POP drops ratios outside [0.5, 2]; no correction uses weight 1.', '示例概率：TIS 裁剪至 2；ICE-POP 丢弃 [0.5, 2] 外的比率；无修正使用权重 1。')}</p>`;
        if (m.moe) lesson +=
          `<div class="route-demo" data-r3="${s.r3}"><div>Rollout → ${chip('E₁', 'blue')} ${chip('E₃', 'yellow')}</div><div>Train → ${chip('E₁', 'blue')} ${chip(s.r3 ? 'E₃' : 'E₂', s.r3 ? 'yellow' : '')}</div><small>${s.r3 ? t('R3 replays the recorded expert IDs.', 'R3 重放记录的 expert IDs。') : t('Without replay, router rounding can choose different experts.', '未重放时，router 数值误差可能选到不同专家。')}</small></div>`;
        lesson +=
          `<div class="score-demo" data-sc="${s.sc}">${s.sc ? row(chip('score'), chip('− E_q[score]', 'yellow'), chip(t('zero mean', '均值为零'), 'green')) : row(chip('PPO ratio'), chip(t('clipped objective', '裁剪目标'), 'blue'))}</div><p>${s.sc ? t('SC centers the weighted score per prefix; it uses REINFORCE and a top-128 sampler approximation.', 'SC 在每个 prefix 中心化加权 score，使用 REINFORCE 与 sampler top-128 近似。') : t('PPO clipping controls policy changes; TIS / ICE-POP correct a different ratio, between trainer and sampler.', 'PPO clipping 控制策略更新幅度；TIS / ICE-POP 修正训练器与采样器之间的另一种比率。')}</p><div class="determinism-demo">${chip(s.deterministic ? 'run A = run B*' : 'run A ≈ run B', s.deterministic ? 'green' : '')}<small>${t('*Repeatability is scoped to supported kernels; it does not imply equality across engines.', '*可重复性取决于支持的 kernel，不代表跨引擎数值相等。')}</small></div>`;
        break;
      }
      case 4: {
        title = t('Inside the serving fleet', '走进推理集群内部');
        const cache = `<div class="serving-cache" data-hicache="${s.hicache}"><div data-cache-tier="gpu">${chip('GPU', 'blue')}<span>${t('Prefix cache', '前缀缓存')}</span></div>${s.hicache ? `<b aria-hidden="true">⇄</b><div data-cache-tier="cpu">${chip('CPU', 'yellow')}<span>HiCache</span></div>` : ''}</div><small>${s.hicache ? (s.pd ? t('Enabled on prefill workers only.', '仅在 prefill worker 启用。') : t('Reuse prefixes through GPU ↔ host RAM.', '通过 GPU ↔ 主机内存复用前缀。')) : t('GPU prefix cache only; no host cache tier.', '只有 GPU 前缀缓存，无主机缓存层。')}</small>`;
        const prefill = `<div class="serving-stage" data-serving-phase="prefill"><h5>Prefill</h5><p>${t('Read the prompt; reuse cached prefixes.', '读取 prompt，复用已缓存的前缀。')}</p>${cache}</div>`;
        const shownDrafts = Math.min(s.specSteps, 3);
        const proposed = Array.from({length: shownDrafts}, (_, i) => chip('d' + (i + 1), 'yellow')).join(' ');
        const accepted = Array.from({length: Math.max(0, shownDrafts - 1)}, (_, i) => chip('✓ d' + (i + 1), 'green')).join(' ');
        const decoding = s.speculative ?
          `<div class="draft-stage" data-serving-phase="draft"><strong>${s.draftSource === 'external' ? t('Separate EAGLE head', '独立 EAGLE 头') : t('Checkpoint MTP head', 'Checkpoint MTP 头')}</strong><small>${t('Propose', '提出候选')} · ${s.specSteps} ${t('steps', '步')} · top-k 1</small><div class="serving-tokens">${proposed}${s.specSteps > shownDrafts ? chip('…') : ''}</div></div><div class="serving-connector" aria-hidden="true">↓</div><div class="verify-stage" data-serving-phase="verify"><strong>${t('Target model · batch verification', '目标模型 · 批量验证')}</strong><div class="serving-tokens">${accepted} ${chip('× d' + shownDrafts, 'rejected')}</div></div><div class="serving-connector" aria-hidden="true">↓</div><div class="commit-stage" data-serving-phase="commit"><strong>${t('Commit accepted prefix + corrected token', '提交接受的前缀 + 修正 token')}</strong><div class="serving-tokens">${accepted} ${chip('c', 'blue')}</div><small>${t('Illustrative rejection round; acceptance varies by request. All-accepted rounds can emit a bonus token.', '图示为一次拒绝示例，接受长度随请求变化；全部接受时可多输出一个 token。')}</small></div>` :
          `<div class="verify-stage" data-serving-phase="decode"><strong>${t('Target model · one token at a time', '目标模型 · 逐 token 解码')}</strong>${row(chip('t₁', 'green'), chip('t₂', 'green'), chip('t₃', 'green'))}<small>${t('One target decoding step for each new token.', '每产生一个新 token，执行一次目标模型解码。')}</small></div>`;
        const decode = `<div class="serving-stage"><h5>Decode ${s.speculative ? '· EAGLE' : ''}</h5>${decoding}</div>`;
        const pools = s.pd ?
          `<div class="serving-pool prefill-pool"><div class="serving-pool-label">${t('Prefill engine pool', 'Prefill 引擎池')}</div>${prefill}</div><div class="serving-transfer" data-serving-phase="transfer"><span>↓</span><strong>${m.hybrid ? t('KV + recurrent state transfer', 'KV + 递归状态传输') : 'KV transfer'}</strong><small>Mooncake · RDMA</small></div><div class="serving-pool decode-pool"><div class="serving-pool-label">${t('Decode engine pool', 'Decode 引擎池')}</div>${decode}</div>` :
          `<div class="serving-pool combined-pool"><div class="serving-pool-label">${t('One engine · prefill + decode', '同一引擎 · prefill + decode')}</div>${prefill}<div class="serving-connector">↓ ${t('Cache stays in the engine', '缓存在引擎内复用')}</div>${decode}</div>`;
        lesson =
          `<div class="serving-scene" data-pd="${s.pd}" data-speculative="${s.speculative}"><div class="serving-prompt">${chip('Prompt', 'blue')} ↓</div>${pools}</div><p>${t('Concurrency', '并发上限')}: ${s.concurrency} · ${t('Response limit', '回复上限')}: ${s.response} tokens</p>`;
        break;
      }
      default:
        title = t('Your path to a running experiment', '让实验真正跑起来');
        lesson = row(chip('prepare', 'blue'), chip('convert', 'yellow'), chip('check'), chip('train',
            'green')) +
          `<p>${v.errors.length ? t('Complete the highlighted settings before exporting the shell.', '补齐标出的配置后即可导出 shell。') : t('Export experiment.sh, then run these stages in your prepared environment.', '导出 experiment.sh，在准备好的环境中依次运行这些步骤。')}</p><div class="diagram-files">${chip('experiment.sh')}${s.pd ? chip('sglang.yaml') : ''}${s.layout === 'external' ? chip('serving-reference.sh') : ''}${chip('experiment.json')}</div>`;
    }
    const explanation = `<section class="world-lesson" data-lesson="${s.step}"><h4>${title}</h4>${lesson}</section>`;
    const placement = `${topology}${s.straw ? `<div class="queue-pool"><strong>straw</strong> · ${s.schedule === 'async' ? 'worker 0 / 1 / … → ' : ''}partial → ready → batch <small>JuiceFS · ${t('persistent payloads + queues', '持久数据 + 队列')}</small></div>` : ''}`;
    return `<div class="world-top">${esc(m.name)} · ${esc(m.size)}</div>${s.step === 4 ? explanation + placement : placement + explanation}`;
  }
  const api = {
    render
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SlimeDiagrams = api;
})(typeof window !== 'undefined' ? window : globalThis);
