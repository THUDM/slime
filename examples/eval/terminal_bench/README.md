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
git clone --branch xinyu/quick_start https://github.com/XinyuJiangCMU/slime.git
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

## 5) Start the TB server

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

If you use the provided Qwen eval launcher:

```bash
bash slime/examples/eval/scripts/run_eval_tb_qwen.sh 2>&1 | tee run.log
```
