// Documentation generator tests; no GPU or training dependencies required.
const {
  test
} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {
  spawnSync
} = require('node:child_process');
require('../_static/js/experiment-engine.js');
const E = globalThis.SlimeExperiment;
const repo = path.resolve(__dirname, '../..');
const shell = state => E.generate(state).files['experiment.sh'];

test('memory controls alter optimizer and activation settings independently', () => {
  const cpu = shell({
    ...E.defaults,
    cpuAdam: true,
    recompute: false
  });
  assert.match(cpu, /--optimizer-cpu-offload/);
  assert.doesNotMatch(cpu, /--recompute-granularity/);
  const gpu = shell({
    ...E.defaults,
    cpuAdam: false,
    recompute: true
  });
  assert.doesNotMatch(gpu, /--optimizer-cpu-offload|--overlap-cpu-optimizer-d2h-h2d/);
  assert.match(gpu, /--recompute-granularity full/);
});

test('straw storage and partial continuation compose with either scheduling mode', () => {
  assert.equal(E.validate({
    ...E.defaults,
    straw: true
  }).errors.length, 3);
  for (const schedule of ['sync', 'async']) {
    const state = {
      ...E.defaults,
      schedule,
      layout: 'separate',
      straw: true,
      partial: true,
      maskPartial: true,
      queueDir: '/shared/juicefs/my run',
      queueRun: 'run-1',
      declaration: '/shared/juicefs/deployment.json'
    };
    assert.deepEqual(E.validate(state).errors, []);
    const script = shell(state);
    assert.match(script, /--rollout-data-transport straw/);
    assert.match(script, /--rollout-storage-profile juicefs/);
    assert.match(script, /--rollout-data-dir '\/shared\/juicefs\/my run'/);
    assert.match(script, /--save-interval 1(?:\s|\\)/);
    assert.match(script, /--partial-rollout/);
    assert.match(script, /--mask-offpolicy-in-partial-rollout/);
    assert.equal(script.includes('--over-sampling-batch-size 16'), schedule === 'sync');
    assert.equal(script.includes('generate_rollout_fully_async'), schedule === 'async');
    assert.equal(spawnSync('bash', ['-n'], {
      input: script
    }).status, 0);
    assert.equal(E.change(state, 'partial', false).maskPartial, false);
  }
  assert.doesNotMatch(shell(E.defaults), /--rollout-data-transport straw|--partial-rollout/);
});

test('the default and every model baseline have valid parallel layouts and existing architecture sources',
() => {
    assert.deepEqual(E.validate(E.defaults).errors, []);
    assert.deepEqual(Object.keys(E.models).filter(k => k.startsWith('qwen')), ['qwen38']);
    for (const model of Object.keys(E.models)) {
      const s = E.change(E.defaults, 'model', model),
        m = E.models[model];
      assert.deepEqual(E.validate(s).errors, [], model);
      assert.ok(fs.existsSync(path.join(repo, 'scripts/models', m.model + '.sh')));
      assert.ok(fs.existsSync(path.join(repo, 'scripts', 'run-' + m.recipe + '.sh')));
    }
  });

test('layout transitions enforce GPU residency in one reversible state change', () => {
  const old = {
    ...E.defaults
  };
  const async = E.change(old, 'schedule', 'async');
  assert.equal(async.layout, 'separate');
  assert.equal(old.layout, 'colocate');
  assert.equal(E.change(async, 'layout', 'colocate').schedule, 'sync');
  const external = E.change(old, 'layout', 'external');
  assert.equal(external.transport, 'disk');
  assert.ok(E.validate(external).errors.some(s => s.includes('addresses')));
});

test('PD partitions whole engines and external ownership excludes managed YAML', () => {
  const invalid = {
    ...E.defaults,
    pd: true,
    prefill: 8,
    ib: 'mlx5_0'
  };
  assert.ok(E.validate(invalid).errors.some(s => s.includes('PD needs')));
  const pd = {
    ...invalid,
    layout: 'separate',
    rollout: 16
  };
  const generated = E.generate(pd);
  assert.deepEqual(generated.validation.errors, []);
  assert.match(generated.files['sglang.yaml'], /worker_type: prefill\n\s+num_gpus: 8/);
  assert.match(generated.files['sglang.yaml'], /worker_type: decode\n\s+num_gpus: 8/);
  const external = E.generate({
    ...pd,
    layout: 'external',
    endpoints: 'worker-a:30000 worker-b:30001'
  });
  assert.ok(!external.files['sglang.yaml']);
  assert.doesNotMatch(external.files['experiment.sh'], /--sglang-config /);
  assert.match(external.files['serving-reference.sh'], /--disaggregation-mode prefill/);
});

test('SC uses current-policy REINFORCE with compatible sampling and composes with built-in corrections',
() => {
  for (const correction of ['none', 'tis', 'icepop']) {
    const generated = shell({
      ...E.defaults,
      sc: true,
      correction
    });
    assert.match(generated, /--pg-loss-type reinforce/);
    assert.match(generated, /--disable-grpo-std-normalization/);
    assert.match(generated, /--rollout-top-p 1/);
    assert.match(generated, /--rollout-top-k -1/);
    assert.doesNotMatch(generated, /--eps-clip /);
  }
  assert.doesNotMatch(shell({
    ...E.defaults,
    correction: 'none'
  }), /--use-tis/);
  assert.match(shell({
    ...E.defaults,
    correction: 'icepop'
  }), /--tis-clip-low 0.5/);
  assert.match(shell({
    ...E.defaults,
    correction: 'icepop'
  }), /loss.icepop_function/);
});

test('dense R3, async determinism, missing hooks and transport requirements are explained', () => {
  assert.ok(E.validate({
    ...E.change(E.defaults, 'model', 'dense'),
    r3: true
  }).errors.some(s => s.includes('MoE')));
  assert.ok(E.validate({
    ...E.defaults,
    layout: 'separate',
    schedule: 'async',
    deterministic: true
  }).errors.some(s => s.includes('ordering')));
  assert.equal(E.validate({
    ...E.defaults,
    task: 'custom'
  }).errors.length, 2);
  assert.ok(E.validate({
    ...E.defaults,
    layout: 'external',
    transport: 'delta',
    endpoints: 'node:30000',
    local: ''
  }).errors.some(s => s.includes('local checkpoint')));
  assert.ok(E.validate({
    ...E.defaults,
    pd: true,
    ib: ''
  }).errors.some(s => s.includes('RDMA')));
});

test('cache, routing replay, and precision settings survive export at the right ownership boundary', () => {
  const s = {
    ...E.defaults,
    layout: 'external',
    endpoints: 'node:30000',
    transport: 'disk',
    r3: true,
    hicache: true,
    kv: 'fp8',
    precision: 'fp8'
  };
  const files = E.generate(s).files;
  assert.match(files['serving-reference.sh'], /--enable-return-routed-experts/);
  assert.match(files['serving-reference.sh'], /--enable-hierarchical-cache/);
  assert.match(files['serving-reference.sh'], /--kv-cache-dtype fp8_e4m3/);
  assert.match(files['experiment.sh'], /--update-weight-transport disk/);
  assert.match(files['experiment.sh'], /ROLLOUT_MODEL="\$HF_MODEL-fp8"/);
  const pd = E.generate({
    ...E.defaults,
    layout: 'separate',
    rollout: 16,
    pd: true,
    ib: 'mlx5_0',
    hicache: true
  });
  assert.match(pd.files['sglang.yaml'], /enable_hierarchical_cache: false/);
});

test('all generated recipe variants are syntactically valid bash', () => {
  let count = 0;
  for (const model of Object.keys(E.models))
    for (const precision of E.enums.precision)
      for (const layout of E.enums.layout)
        for (const correction of E.enums.correction) {
          const s = {
            ...E.change(E.defaults, 'model', model),
            precision,
            layout,
            correction,
            endpoints: 'worker-a:30000',
            pd: layout === 'separate',
            rollout: E.models[model].engine * 2,
            ib: 'mlx5_0',
            hicache: true,
            sc: true
          };
          for (const [name, body] of Object.entries(E.generate(s).files))
            if (name.endsWith('.sh')) {
              const result = spawnSync('bash', ['-n'], {
                input: body,
                encoding: 'utf8'
              });
              assert.equal(result.status, 0, `${model} ${precision} ${layout}: ${result.stderr}`);
              count++;
            }
        }
  assert.equal(count, 288);
});

test('shared choices are schema-limited and restore paths for the selected model', () => {
  const s = E.sanitize({
    model: 'qwen38',
    evil: '<script>',
    step: 99,
    layout: 'unknown',
    hf: 'hello\nworld'
  });
  assert.equal(s.train, '/data/Qwen3.8-27B_torch_dist');
  assert.equal(s.hf, 'helloworld');
  assert.equal(s.step, 0);
  assert.equal(s.layout, 'colocate');
  assert.ok(!('evil' in s));
  assert.equal(E.sanitize({
    model: '__proto__'
  }).model, 'flash');
});

test('exported shell preserves literal user paths and submits the complete configuration', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'slime-lab-'));
  try {
    fs.mkdirSync(path.join(dir, 'scripts/models'), {
      recursive: true
    });
    fs.mkdirSync(path.join(dir, 'bin'));
    const weird = "model 'quoted' $(touch PWNED) `touch PWNED2`";
    const hf = path.join(dir, weird),
      train = path.join(dir, 'train ckpt'),
      data = path.join(dir, 'prompts.jsonl');
    fs.mkdirSync(hf);
    fs.writeFileSync(path.join(hf, 'config.json'), '{}');
    fs.mkdirSync(train);
    fs.writeFileSync(path.join(train, 'latest_checkpointed_iteration.txt'), '1');
    fs.writeFileSync(data, '{}');
    fs.writeFileSync(path.join(dir, 'scripts/models/glm4.7-30B-A3B.sh'),
    'MODEL_ARGS=(--num-layers 47)\n');
    fs.writeFileSync(path.join(dir, 'bin/ray'),
    '#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE"\n', {
      mode: 0o755
    });
    const s = {
      ...E.defaults,
      hf,
      train,
      data,
      megatron: path.join(dir, 'Megatron')
    };
    fs.writeFileSync(path.join(dir, 'experiment.sh'), shell(s));
    const result = spawnSync('bash', ['experiment.sh', 'train'], {
      cwd: dir,
      encoding: 'utf8',
      env: {
        ...process.env,
        PATH: path.join(dir, 'bin') + ':' + process.env.PATH,
        CAPTURE: path.join(dir, 'args')
      }
    });
    assert.equal(result.status, 0, result.stderr);
    const args = fs.readFileSync(path.join(dir, 'args'), 'utf8').split('\0');
    assert.equal(args[args.indexOf('--hf-checkpoint') + 1], hf);
    assert.equal(args[args.indexOf('--ref-load') + 1], train);
    assert.ok(args.includes('train.py'));
    assert.ok(args.includes('--num-layers'));
    assert.ok(!fs.existsSync(path.join(dir, 'PWNED')));
    assert.ok(!fs.existsSync(path.join(dir, 'PWNED2')));
    assert.ok(!fs.readdirSync(dir).some(n => n.startsWith('.slime-lab.')));
  } finally {
    fs.rmSync(dir, {
      recursive: true,
      force: true
    });
  }
});

test('the GLM-5.3 export prepares the current checkpoint as BF16 before conversion', () => {
  const script = shell(E.change(E.defaults, 'model', 'glm5'));
  assert.match(script, /hf download 'zai-org\/GLM-5\.3'/);
  assert.match(script, /tools\/fp8_cast_bf16.py/);
  assert.match(script, /cfg\.pop\('quantization_config', None\)/);
  assert.doesNotMatch(script, /hf download 'zai-org\/GLM-5\.2'/);
});

test('diagrams explain distinct signal paths, ownership, precision and corrections', () => {
  const D = require('../_static/js/experiment-diagrams.js');
  const draw = patch => {
    const state = {
      ...E.defaults,
      ...patch
    };
    return D.render(state, E.models[state.model], E.validate(state), 'en');
  };
  const math = draw({
      task: 'math'
    }),
    custom = draw({
      task: 'custom'
    });
  assert.match(math, /Math verifier/);
  assert.match(custom, /Tool \/ environment/);
  assert.doesNotMatch(custom, /Math verifier/);
  assert.match(draw({
    layout: 'external'
  }), /external-map/);
  assert.equal((draw({
    layout: 'external'
  }).match(/world-node sampler/g) || []).length, 3);
  assert.equal((draw({
    layout: 'separate'
  }).match(/world-node sampler/g) || []).length, 1);
  assert.doesNotMatch(draw({
    layout: 'separate'
  }), /external-map/);
  assert.match(draw({
    step: 2,
    precision: 'int4'
  }), /--bits:4/);
  assert.match(draw({
    step: 3,
    correction: 'tis'
  }), /w = 2/);
  assert.match(draw({
    step: 3,
    correction: 'icepop'
  }), /w = 0/);
  assert.match(draw({
    step: 3,
    correction: 'none'
  }), /w = 1/);
  assert.match(draw({
    step: 3,
    r3: true
  }), /R3 replays/);
  assert.match(draw({
    step: 3,
    sc: true
  }), /zero mean/);
  assert.match(draw({
    step: 1,
    partial: true,
    maskPartial: true
  }), /Old tokens masked/);
  assert.match(draw({
    step: 4,
    pd: true,
    hicache: true
  }), /KV transfer/);
  assert.match(draw({
    step: 4,
    pd: true,
    hicache: true
  }), /Enabled on prefill workers only/);
  assert.doesNotMatch(draw({
    step: 1,
    straw: true
  }), /STRAW|Straw/);
});
