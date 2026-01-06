# Docs

## Prerequisites
- A writable host directory for cached data (`/data/.cache`)
- Choose descriptive container names to replace the placeholders (`<slime container name>`, `<env container name>`).

## 1) Prepare host network
```bash
docker network create skills-net
```

## 2) Launch the slime container
```bash
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/.cache:/root/.cache \
  -v /dev/shm:/shm \
  --ipc=host \
  --privileged \
  --network skills-net \
  --name <slime container name> \
  slimerl/slime:latest \
  /bin/bash
```

## 3) Launch the Skills container
```bash
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/.cache:/root/.cache \
  -v /dev/shm:/shm \
  --ipc=host \
  --privileged \
  --network skills-net \
  --name <env container name> \
  --network-alias skills_server \
  guapisolo/nemoskills:0.7.1 \
  /bin/bash
```

## 4) Inside the Skills container
Clone repos and install the Skills package:
```bash
git clone -b slime_skills https://github.com/guapisolo/slime.git /opt/slime
git clone -b slime https://github.com/guapisolo/Skills.git /opt/Skills

cd /opt/Skills
pip install -e .
```

Download/prepare datasets:
```bash
cd /opt/Skills/nemo_skills/dataset
python3 aime25/prepare.py
python3 hle/prepare.py
python3 arena-hard/prepare.py
```

Start the skills server:
```bash
cd /opt/slime
python examples/eval/nemo_skills/skills_server.py \
  --host 0.0.0.0 \
  --port 9050 \
  --output-root /opt/skills-eval \
  --config-dir examples/eval/nemo_skills/config \
  --cluster local_cluster \
  --max-concurrent-requests 512 \
  --openai-model-name slime-openai-model
```

You can now connect to the server at `skills_server:9050` from within the `skills-net` Docker network. The server always proxies evaluation traffic to an OpenAI-compatible sglang router (Slime starts and manage the router), so adjust `--openai-model-name` and `--max-concurrent-requests` as needed for your deployment.
