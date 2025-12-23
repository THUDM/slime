# Terminal Bench Eval (Slime)

This folder wires Terminal Bench (TB) into Slime as an eval delegate. The TB
run happens on the host via the `tb` CLI, and Slime reads back `accuracy` and
`n_resolved`.

This guide is written for ML/algorithm folks who just want it to run.

## What runs where

- Slime runs your training/eval loop.
- Slime calls the TB delegate client.
- The TB delegate server (`tb_server.py`) runs `tb run ...` on the host.
- The server reads the latest TB JSON results and returns metrics to Slime.

## Prereqs

This setup assumes `slime/` and `terminal-bench/` are sibling directories under
`/mnt/data/xinyu/program/slime-tb`, and you only need one venv at
`/mnt/data/xinyu/program/slime-tb/.venv`.

1) A working OpenAI-compatible inference endpoint, e.g.:
   - `http://127.0.0.1:30001/v1`

2) Terminal Bench installed and its `tb` CLI available.
   - Use the same venv for both Slime and TerminalBench.

3) TerminalBench Eval Server dependencies (Slime side only).
   - Install with:
     ```bash
     uv pip install -r ../slime/examples/eval/terminal_bench/requirements.txt
     ```
   - Do not assume these dependencies exist in `terminal-bench/`.
   - Do not put this `requirements.txt` under `terminal-bench/`.

4) A Slime eval config file that includes `eval.datasets`.
   - Slime requires at least one dataset under `eval.datasets`.
   - You can reuse your existing eval config; just add the delegate section.

## Step 1: Start the inference server (sglang)

Example:

```bash
python3 -m sglang.launch_server \
  --model-path /data/models/OpenThinker-Agent-v1 \
  --served-model-name openai/qwen3-8b \
  --port 30001 \
  --host 0.0.0.0
```

Notes:
- The `served-model-name` should match what TB sends (`openai/<model>`).
- If your model name is different, update `model_name` in the delegate config.

## Step 2: Start the TB server

Run on the host (same machine where `tb` works):

```bash
cd /mnt/data/xinyu/program/slime-tb/terminal-bench

python slime/examples/eval/terminal_bench/tb_server.py \
  --host 0.0.0.0 --port 9050 \
  --output-root /tmp/tb-eval
```

What it does:
- Uses `OPENAI_API_KEY=EMPTY`
- Runs `tb run -a terminus-2 -m openai/<model> ... --n-concurrent 8`
- Waits for completion, then returns `accuracy` and `n_resolved`

## Step 3: Quick sanity check (curl, async)

Run a single task (e.g. `hello-world`). The server returns a `job_id`
immediately, then you poll the status endpoint.

```bash
# Submit job
curl -X POST http://localhost:9050/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"qwen3-8b","api_base":"http://127.0.0.1:30001/v1","dataset_path":"/mnt/data/xinyu/program/slime-tb/terminal-bench/tasks","task_id":"hello-world","n_concurrent":1}'
```

The response includes `job_id` and `status_url`, for example:

```json
{"job_id":"...","status":"queued","status_url":"/status/<job_id>", ...}
```

Poll status until `completed`:

```bash
curl http://localhost:9050/status/<job_id>
```

Where to check outputs:
- Logs: `/mnt/data/xinyu/program/slime-tb/tb_eval_logs/<run_id>.log`
- Results: `/tmp/tb-eval/<run_id>/results.json`

## Step 4: Configure Slime eval

You need an eval config. Example:

```yaml
eval:
  # Slime still needs normal eval datasets (can be any small one).
  datasets:
    - name: aime
      path: /root/datasets/aime-2024/aime-2024.jsonl
      rm_type: math

  # TB delegate config.
  delegate:
    - name: terminal_bench
      url: http://localhost:9050         # "/evaluate" auto-added if missing
      timeout_secs: 1200                 # 20 minutes
      model_name: qwen3-8b
      api_base: http://127.0.0.1:30001/v1
      dataset_path: /mnt/data/xinyu/program/slime-tb/terminal-bench/tasks
      n_tasks: 10
      n_concurrent: 1
      # Optional: run specific tasks instead of n_tasks
      # task_ids: ["hello-world"]
      # task_id: "hello-world"
```

Notes:
- `model_name` is auto-normalized to `openai/<model>` if you omit the prefix.
- The TB client auto-adds `/evaluate` if you give a bare host:port.
- `task_id` / `task_ids` overrides `n_tasks` when provided.
- `dataset_path` lets you run from any working directory.

## Step 5: Tell Slime to use the delegate rollout

Add this to your training/eval command:

```bash
--eval-config /path/to/your_eval_config.yaml \
--eval-function-path examples.eval.eval_delegate_rollout.generate_rollout
```

This makes Slime call the TB delegate during evaluation.

## Quick sanity check (eval-only)

If you just want to verify the TB integration, run a quick eval-only pass
(you still need your normal Slime args for model/data/etc.):

```bash
python slime/train.py \
  --num-rollout 0 \
  --eval-interval 1 \
  --eval-config /path/to/your_eval_config.yaml \
  --eval-function-path examples.eval.eval_delegate_rollout.generate_rollout \
  ...other required args...
```

## Common gotchas

- 404 from TB server: use `url: http://localhost:9050` or `.../evaluate`.
- Timeouts: keep `timeout_secs` large (TB tasks can compile code).
- No TB metrics: check `/tmp/tb-eval/<run_id>/results.json` and poll `/status/<job_id>`.
- No output in terminal: tail the log at `/mnt/data/xinyu/program/slime-tb/tb_eval_logs/<run_id>.log`.

## Reference: the CLI command it runs

The server is aligned with:

```bash
OPENAI_API_KEY=EMPTY tb run -a terminus-2 -m openai/qwen3-8b \
  --agent-kwarg api_base=http://127.0.0.1:30001/v1 \
  --n-concurrent 1
```
