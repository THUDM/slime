# Slime jobs on Iris

Slime's Iris adapter submits one or more GPU tasks from the current checkout. Iris owns
workspace transfer, scheduling, retries, and logs; the selected image owns CUDA,
Ray, SGLang, Megatron, and Slime dependencies.

Design: https://echo.oa.dev/wiki/327

## Before launch

Use a clean, committed checkout so the submitted workspace has an attributable
revision. Install the `marin-iris` package from `marin-community/marin`, authenticate
to the target cluster, and choose an immutable task-image digest. For an isolated
launch environment, prefix the commands below with:

```bash
uv run --with 'marin-iris @ git+https://github.com/marin-community/marin.git#subdirectory=lib/iris'
```

Keep credentials in the local environment and forward only their names with
`--secret-env`.

Always inspect a dry run first:

```bash
python -m infra.iris.launcher \
  --cluster cw-rno2a \
  --task-image registry.example/slime@sha256:<digest> \
  --job-name slime-smoke \
  --gpus-per-node 8 \
  --secret-env WANDB_API_KEY \
  --dry-run \
  -- bash -lc 'nvidia-smi && python -c "import slime, ray, torch; print(torch.cuda.device_count())"'
```

The dry-run output lists environment variable names but never their values. Remove
`--dry-run` only after reviewing the image, resources, retry policy, and command.

## Operational boundary

For multi-node jobs, set `--nodes N` and provide a unique shared `--rendezvous-dir`
on S3 or a filesystem mounted by every task. The launcher gang-schedules the
GPU replicas and wraps the command with `infra.iris.ray_runtime`: rank zero starts
the Ray head and runs the command, while the remaining ranks join as Ray workers.
The command must attach to the existing cluster through `RAY_ADDRESS`; it must not
start another local Ray head. Use a distinct rendezvous directory for every job.

The launcher uses Iris's default container security profile. It does not request a
Docker socket or privileged container. Sandbox services used by coding-agent
rollouts must therefore be remote services supported by their own backend.

The default setup list is empty: Iris uses the image as-is. Use repeated
`--setup-command` only for small, reviewed setup steps. Prefer rebuilding the image
over installing large GPU packages at task start.

## Diagnose and stop

Use the Iris CLI to inspect the exact job ID printed at submission. A scheduled job
is not healthy until the task logs show visible GPUs, Ray initialization, rollout
progress, and advancing training steps. Check pending reasons before changing
resources; image-pull, quota, unschedulable, and application failures require
different responses.

`Ctrl-C` stops the local log stream; use Iris's explicit job command when an
authorized cancellation is required. Capture the job ID, terminal state, relevant
log excerpt, task image digest, and Slime revision before cleanup.
