# Fault Tolerance

To ensure long-term, stable RL training, slime enables a certain level of fault tolerance by default. This section introduces the design philosophy behind fault tolerance in slime.

To enable the fault tolerance function in slime, please set `--use-fault-tolerance`.

## Rollout Fault Tolerance

During the rollout process, slime periodically sends heartbeat requests (`/health_generate`) to all SGLang servers. If a heartbeat times out, that SGLang server will be stopped and **immediately restarted**.

### Instant Restart with Remote Weight Loading

When fault tolerance is enabled and a rollout engine fails, slime will:
1. **Immediately kill** the failed engine
2. **Immediately query** the router to find active workers
3. **Immediately restart** a new engine, loading weights directly from an active engine using SGLang's transfer engine

This instant fault tolerance mechanism ensures training continuity - failed engines are replaced immediately without waiting for the current rollout round to complete.

The new engine is started with remote weight loading parameters:
- `--load-format remote_instance`
- `--remote-instance-weight-loader-backend transfer_engine`
- `--remote-instance-weight-loader-seed-instance-ip <active_worker_ip>`
- `--remote-instance-weight-loader-seed-instance-service-port <active_worker_port>`

This leverages SGLang's built-in capability to bootstrap a new engine from an existing one, avoiding the overhead of loading model weights from storage and enabling fast recovery.

### Configuration Parameters

- `--rollout-health-check-first-wait`: Since some large MoE models require compilation on their first run, slime will wait for `rollout_health_check_first_wait` seconds before the first rollout to start sending heartbeats. Defaults to 300s.
- `--rollout-health-check-interval`: The interval between heartbeat checks. Defaults to 10s.
- `--rollout-health-check-timeout`: The timeout limit for a heartbeat request. Defaults to 5s.
