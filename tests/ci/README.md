# Doc about CI

## Configure GitHub secrets

https://github.com/slimerl/slime/settings/secrets/actions

* `WANDB_API_KEY`: get from https://wandb.ai/authorize

## Setup new GitHub runners

### Step 1: Env

Write `.env` mimicking `.env.example`.
The token can be found at https://github.com/slimerl/slime/settings/actions/runners/new?arch=x64&os=linux.

WARN: The `GITHUB_RUNNER_TOKEN` changes after a while.

### Step 2: Prepare `/home/runner/externals`

```shell
docker run --rm -it --privileged --pid=host -v /:/host_root ubuntu /bin/bash -c 'rm -rf /host_root/home/runner/externals && mkdir -p /host_root/home/runner/externals && chmod -R 777 /host_root/home/runner/externals'
docker run -d --name temp-runner ghcr.io/actions/actions-runner:2.328.0 tail -f /dev/null
docker cp temp-runner:/home/runner/externals/. /home/runner/externals
docker rm -f temp-runner
ls -alh /home/runner/externals
```

### Step 3: Run

```shell
cd /data/tom/primary_synced/slime/tests/ci/github_runner
docker compose up -d
```

### Debugging

Logs

```shell
docker compose logs -f
```

Exec

```shell
docker exec -it github_runner-runner-1 /bin/bash
```
