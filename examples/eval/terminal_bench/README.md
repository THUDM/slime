# Terminal Bench Eval

This folder wires Terminal Bench (TB) into Slime as an eval delegate. The TB run happens on the host via the `tb` CLI, and Slime reads back aggregated metrics such as `accuracy`, `n_resolved`, `n_unresolved`, `pass_at_k/*`, and token stats like `total_input_tokens_mean/median` and `total_output_tokens_mean/median`.

## What runs where

- Slime runs your training/eval loop inside the Docker container.
- Slime calls the TB delegate client.
- The TB delegate server (`tb_server.py`) runs `tb run ...` on the host.
- The server reads the latest TB JSON results and returns metrics to Slime.

## 1) Get the code (host)

```bash
mkdir slime-tb
cd slime-tb
git clone https://github.com/THUDM/slime.git
git clone https://github.com/laude-institute/terminal-bench
```

## 2) Launch the Slime container

```bash
docker run \
  -itd \
  --gpus all \
  --shm-size 32g \
  --network host \
  --ipc=host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -v /mnt/data/.cache:/root/.cache \
  -v $(pwd):/shared/slime-tb \
  --name <slime container name> \
  slimerl/slime:latest \
  /bin/bash
```

## 3) Inside the Slime container

```bash
docker exec -it <slime container name> /bin/bash
```

## 4) Terminal Bench environment (host)

Run on the machine that will host `tb_server.py` (where you cloned both repos):

```bash
# Host machine terminal (outside Docker)
uv venv --python 3.13 .venv
source .venv/bin/activate

uv pip install terminal-bench/.
uv pip install -r slime/examples/eval/terminal_bench/requirements.txt
```

Notes:
- Use your local repo paths if they are not `./slime` and `./terminal-bench`.

## 5) Start the Terminal Bench server

Run on the host (same machine where `tb` works):

```bash
python slime/examples/eval/terminal_bench/tb_server.py \
  --host 0.0.0.0 --port 9051 \
  --output-root tb_eval_output
```

What it does:
- Uses `OPENAI_API_KEY=EMPTY`
- Runs `tb run -a terminus-2 -m openai/<model> ... --n-concurrent 8`
- Waits for completion, then returns `accuracy`, `n_resolved`,
  `n_unresolved`, `pass_at_k/*`, and token stats such as
  `total_input_tokens_mean/median` and `total_output_tokens_mean/median`

## 6) Run the eval script (example)

If you use the provided Qwen eval launcher (`run-eval-tb-qwen.sh`), follow the steps below to run Terminal-Bench evaluation.

First, update the `dataset_path` in `eval_tb_example.yaml` to the local path of `terminal-bench/tasks` on your host (not an internal Docker-only path). 

Then download the HuggingFace model checkpoint inside the Slime container:

```bash
huggingface-cli download open-thoughts/OpenThinker-Agent-v1 \
--local-dir /root/.cache/OpenThinker-Agent-v1
```

After downloading, convert the HuggingFace checkpoint to Slime's torch distributed format. From the Slime root directory, run:

```bash
cd /shared/slime-tb/slime
source scripts/models/qwen3-8B.sh

export PYTHONPATH=/root/Megatron-LM:/shared/slime-tb/slime

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/.cache/OpenThinker-Agent-v1 \
  --save /root/.cache/OpenThinker-Agent-v1_torch_dist
```

Finally, run the following command inside the Slime container:

```bash
bash slime/examples/eval/scripts/run-eval-tb-qwen.sh 2>&1 | tee run.log
```

For convenience, you can restrict the evaluation scope in `eval_tb_example.yaml`, either by specifying a single task or multiple tasks (`task_ids`), or by limiting the number of tasks via `n_tasks`.

## 7) Common Issues

When running Slime inside a Docker container with `--network host`, Ray may encounter port conflicts due to shared networking with the host.

In some cases, this manifests as Ray failing to start or reporting Redis- or session-related errors. This can usually be resolved by explicitly assigning unused ports when starting the Ray head node, for example by setting a non-default `--port` and `--dashboard-port`.

In more severe cases, Ray job submission may fail with errors indicating that no available agent can accept jobs. This typically happens when the dashboard agent or runtime environment agent ports are also in conflict. In such situations, explicitly specifying the agent-related ports (e.g. `--dashboard-agent-listen-port`, `--dashboard-agent-grpc-port`, and `--runtime-env-agent-port`) when starting Ray can resolve the issue.

If the TB server cannot connect to the Slime server through the sglang router (`InternalServerError`), check which address is actually listening on the router port (e.g. 30005 in this example) and update the `api_base` in `eval_tb_example.yaml` accordingly:

```bash
ss -lntp | grep 30005
```

You may see `Parser warnings`, `Context length exceeded`, `Command 1 should end with newline`, `Harness execution failed` in `tb_server.py` logs. They are warnings from Terminal Bench and can be ignored if runs proceed normally.