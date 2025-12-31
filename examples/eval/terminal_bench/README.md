# Terminal Bench Eval (Slime)

This folder wires Terminal Bench (TB) into Slime as an eval delegate. The TB run happens on the host via the `tb` CLI, and Slime reads back aggregated metrics such as `accuracy`, `n_resolved`, `n_unresolved`, `pass_at_k/*`, and token stats like `total_input_tokens_mean/median` and `total_output_tokens_mean/median`.

## What runs where

- Slime runs your training/eval loop inside the Docker container.
- Slime calls the TB delegate client.
- The TB delegate server (`tb_server.py`) runs `tb run ...` on the host.
- The server reads the latest TB JSON results and returns metrics to Slime.

## Prereqs

1) Docker with GPU access.
2) `uv` installed on the host.
3) Terminal Bench installed and its `tb` CLI available on the machine that runs
   `tb_server.py`.
4) The Slime repo available on the machine that runs `tb_server.py`.
5) A Slime eval config file that includes `eval.datasets`.
   - Slime requires at least one dataset under `eval.datasets`.
   - You can reuse your existing eval config; just add the delegate section.

## 1) Get the code (host)

```bash
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
  -v ~/.cache:/root/.cache \
  -v $(pwd)/slime:/opt/slime \
  -v $(pwd)/terminal-bench:/opt/terminal-bench \
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
cd /opt/slime
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
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

If the TB server cannot connect to the Slime server through the sglang router, check which address is actually listening on the router port (e.g. 30005 in this example) and update the `api_base` in `eval_tb_example.yaml` accordingly:

```bash
ss -lntp | grep 30005
```