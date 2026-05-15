# Megatron Config: Role-Based Training Overrides

`--megatron-config-path` is a YAML-based configuration system for applying role-specific overrides on top of the shared Megatron CLI arguments. Today it is mainly intended for PPO actor / critic configuration.

Unlike `--sglang-config`, `--megatron-config-path` does not manage deployment, routing, or GPU orchestration. Its only job is to decide which training arguments each role should finally use.

---

## Design Overview

By default, when `--megatron-config-path` is not used, both actor and critic inherit the Megatron / slime CLI arguments directly.

With `--megatron-config-path`, the configuration is split into two layers:

- **Shared CLI arguments** define the common Megatron topology, resource allocation, and default training parameters.
- **Role-level YAML overrides** only specify the fields that should differ between actor and critic.

**Key design principles:**

- **CLI remains the shared baseline.** slime first parses the normal CLI arguments, then applies the YAML role overrides.
- **Missing roles inherit automatically.** If a role is absent from the YAML file, it simply keeps the CLI arguments unchanged.
- **Resource allocation is still controlled by CLI.** `num_nodes` and `num_gpus_per_node` in YAML are ignored; placement is still controlled by `--actor-num-*` / `--critic-num-*`.

---

## Config Format

The config file is a YAML document whose top-level `megatron` key contains a list of role entries:

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

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Optional | Label for this entry. The runtime does not depend on it today, but keeping `default` is recommended for forward compatibility. |
| `role` | `str` | **Required** | Role name. Currently supported values are `actor` and `critic`. |
| `overrides` | `dict` | `{}` | Role-specific argument overrides applied on top of the shared CLI arguments. |
| `args` | `dict` | `{}` | Backward-compatible alias for `overrides`. New configs should prefer `overrides`. |

> **Note:** Keys inside `overrides` use argparse attribute names, not CLI flag names. For example, use `tensor_model_parallel_size` rather than `tensor-model-parallel-size`.

---

## Usage Pattern

A typical PPO setup looks like this:

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

In this setup:

- CLI defines the shared topology and resource layout.
- YAML defines the role-specific differences, such as `lr`, `load`, `save`, or optimizer / scheduler parameters.

### Overriding Only One Role

You can also override only one role and let the other inherit the shared CLI configuration. For example, changing only the critic learning rate:

```yaml
megatron:
  - name: default
    role: critic
    overrides:
      lr: 1e-5
```

In this case the actor keeps the shared CLI arguments unchanged.

---

## Current Limitations

- **PPO only for now.** `--megatron-config-path` is currently intended for PPO actor / critic role configuration. It is not the recommended interface for GRPO, REINFORCE++, and other critic-free workflows.
- **Actor and critic must use the same Megatron parallel topology in current PPO.** In particular, topology-related settings such as `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `expert_model_parallel_size`, `expert_tensor_parallel_size`, and `sequence_parallel` should not differ between actor and critic.
- **Keep topology-related settings on CLI.** The safest current pattern is to keep parallelism and resource arguments in the shared CLI configuration, and only put role-specific differences in YAML, such as `lr`, `load`, `save`, warmup, and optimizer / scheduler settings.

If you configure different parallel topologies for actor and critic, the behavior is currently unsupported and may fail during initialization or training.

---

## FAQ

### Q: Can I provide only an actor entry or only a critic entry?

Yes. Missing roles automatically inherit the shared CLI arguments, so you do not need to duplicate everything in YAML.

### Q: Can I move `--actor-num-nodes` or `--critic-num-gpus-per-node` into YAML?

No. Resource allocation and placement groups are still controlled by CLI arguments, and the corresponding YAML fields are ignored.