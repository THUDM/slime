import sys
from types import SimpleNamespace

import slime.utils.external_utils.command_utils as U


def test_get_default_swanlab_args(monkeypatch, tmp_path):
    monkeypatch.setenv("SWANLAB_API_KEY", "test-key")

    args = U.get_default_swanlab_args(str(tmp_path / "test_short.py"), run_name_prefix="prefix", run_id="run123")

    assert "--use-swanlab" in args
    assert "--swanlab-project slime-test_short" in args
    assert "--swanlab-group prefix_run123" in args
    assert "--swanlab-key 'test-key'" in args
    assert "--disable-swanlab-random-suffix" in args


def test_get_default_tracking_args_can_target_wandb(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")

    args = U.get_default_tracking_args(str(tmp_path / "test_short.py"), run_name_prefix="prefix", run_id="run123", backend="wandb")

    assert "--use-wandb" in args
    assert "--wandb-project slime-test_short" in args
    assert "--wandb-group prefix_run123" in args
    assert "--wandb-key 'wandb-key'" in args
    assert "--disable-wandb-random-suffix" in args


def test_logging_utils_distributes_to_wandb_and_swanlab(monkeypatch):
    import slime.utils.logging_utils as logging_utils

    wandb_calls = {"log": [], "finish": 0}
    swanlab_calls = {"log": [], "finish": 0, "init": []}

    wandb_stub = SimpleNamespace(
        run=SimpleNamespace(id="wandb-run"),
        log=lambda metrics: wandb_calls["log"].append(metrics),
        finish=lambda: wandb_calls.__setitem__("finish", wandb_calls["finish"] + 1),
    )
    swanlab_stub = SimpleNamespace(
        log=lambda metrics, step=None: swanlab_calls["log"].append((metrics, step)),
        finish=lambda: swanlab_calls.__setitem__("finish", swanlab_calls["finish"] + 1),
        init=lambda **kwargs: swanlab_calls["init"].append(kwargs),
        get_run=lambda: SimpleNamespace(id="swanlab-run"),
        util=SimpleNamespace(generate_id=lambda: "abc123"),
    )

    monkeypatch.setitem(sys.modules, "swanlab", swanlab_stub)
    monkeypatch.setattr(logging_utils, "wandb", wandb_stub)
    monkeypatch.setattr(logging_utils, "swanlab_utils", SimpleNamespace(
        init_swanlab_primary=lambda args: swanlab_calls["init"].append({"primary": True}),
        init_swanlab_secondary=lambda args: swanlab_calls["init"].append({"secondary": True}),
        reinit_swanlab_primary_with_open_metrics=lambda args, router_addr: swanlab_calls["init"].append(
            {"router_addr": router_addr}
        ),
        finish_swanlab=lambda args: swanlab_calls.__setitem__("finish", swanlab_calls["finish"] + 1),
    ))
    monkeypatch.setitem(logging_utils.__dict__, "_TensorboardAdapter", lambda args: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setitem(logging_utils.__dict__, "wandb_utils", SimpleNamespace(
        init_wandb_primary=lambda args, **kwargs: None,
        init_wandb_secondary=lambda args, **kwargs: None,
        reinit_wandb_primary_with_open_metrics=lambda args, router_addr: None,
    ))

    args = SimpleNamespace(use_wandb=True, use_swanlab=True, use_tensorboard=False)
    logging_utils.log(args, {"rollout/step": 3, "metric": 1.5}, step_key="rollout/step")

    assert wandb_calls["log"] == [{"rollout/step": 3, "metric": 1.5}]
    assert swanlab_calls["log"] == [({"metric": 1.5}, 3)]


def test_swanlab_open_metrics_monitor_collects_and_logs(monkeypatch):
    import slime.utils.swanlab_utils as swanlab_utils

    logged = []

    monkeypatch.setattr(swanlab_utils, "swanlab", SimpleNamespace(log=lambda metrics, step=None: logged.append((metrics, step))))
    monkeypatch.setattr(
        swanlab_utils,
        "httpx",
        SimpleNamespace(get=lambda url, timeout=10.0: SimpleNamespace(
            raise_for_status=lambda: None,
            text="""# HELP sglang_requests_total requests
sglang_requests_total{engine=rollout-0} 12
sglang_latency_seconds 0.5
""",
        )),
    )

    monitor = swanlab_utils._SwanlabOpenMetricsMonitor(args=SimpleNamespace(), router_addr="http://127.0.0.1:8000", interval_s=1)
    monitor.poll_once()

    assert logged == [({"sglang_engine/sglang_requests_total_engine_rollout_0": 12.0, "sglang_engine/sglang_latency_seconds": 0.5}, 0)]
