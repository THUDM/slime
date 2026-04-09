from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def load_rollout_manager_module():
    log_calls: list[tuple[object, dict, str]] = []

    class DummyPlacementGroupSchedulingStrategy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyLock:
        @classmethod
        def options(cls, **kwargs):
            return cls()

        def remote(self):
            return object()

    class DummySample:
        class Status:
            TRUNCATED = object()

    def fake_log(args, metrics, step_key):
        log_calls.append((args, dict(metrics), step_key))

    def fake_compute_rollout_step(args, rollout_id):
        return rollout_id * 10 + 1

    def fake_group_by(*args, **kwargs):
        return {}

    def fake_load_function(*args, **kwargs):
        return None

    modules = {
        "ray": types.ModuleType("ray"),
        "ray.util": _package("ray.util"),
        "ray.util.scheduling_strategies": types.ModuleType("ray.util.scheduling_strategies"),
        "torch": types.ModuleType("torch"),
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.constants": types.ModuleType("sglang.srt.constants"),
        "slime": _package("slime"),
        "slime.backends": _package("slime.backends"),
        "slime.backends.sglang_utils": _package("slime.backends.sglang_utils"),
        "slime.backends.sglang_utils.sglang_config": types.ModuleType("slime.backends.sglang_utils.sglang_config"),
        "slime.backends.sglang_utils.sglang_engine": types.ModuleType("slime.backends.sglang_utils.sglang_engine"),
        "slime.rollout": _package("slime.rollout"),
        "slime.rollout.base_types": types.ModuleType("slime.rollout.base_types"),
        "slime.utils": _package("slime.utils"),
        "slime.utils.health_monitor": types.ModuleType("slime.utils.health_monitor"),
        "slime.utils.http_utils": types.ModuleType("slime.utils.http_utils"),
        "slime.utils.logging_utils": types.ModuleType("slime.utils.logging_utils"),
        "slime.utils.metric_utils": types.ModuleType("slime.utils.metric_utils"),
        "slime.utils.misc": types.ModuleType("slime.utils.misc"),
        "slime.utils.seqlen_balancing": types.ModuleType("slime.utils.seqlen_balancing"),
        "slime.utils.types": types.ModuleType("slime.utils.types"),
        "slime.ray": _package("slime.ray"),
        "slime.ray.utils": types.ModuleType("slime.ray.utils"),
    }

    modules["ray"].remote = lambda obj=None, **kwargs: obj
    modules["ray"].get = lambda value: value
    modules["ray"].util = modules["ray.util"]
    modules["ray.util"].scheduling_strategies = modules["ray.util.scheduling_strategies"]
    modules["ray.util.scheduling_strategies"].PlacementGroupSchedulingStrategy = DummyPlacementGroupSchedulingStrategy

    modules["torch"].load = lambda *args, **kwargs: {"samples": []}

    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].constants = modules["sglang.srt.constants"]
    modules["sglang.srt.constants"].GPU_MEMORY_TYPE_CUDA_GRAPH = 0
    modules["sglang.srt.constants"].GPU_MEMORY_TYPE_KV_CACHE = 1
    modules["sglang.srt.constants"].GPU_MEMORY_TYPE_WEIGHTS = 2

    dummy_cls = type("Dummy", (), {})
    modules["slime.backends"].sglang_utils = modules["slime.backends.sglang_utils"]
    modules["slime.backends.sglang_utils"].sglang_config = modules["slime.backends.sglang_utils.sglang_config"]
    modules["slime.backends.sglang_utils"].sglang_engine = modules["slime.backends.sglang_utils.sglang_engine"]
    modules["slime.backends.sglang_utils.sglang_config"].ModelConfig = dummy_cls
    modules["slime.backends.sglang_utils.sglang_config"].ServerGroupConfig = dummy_cls
    modules["slime.backends.sglang_utils.sglang_config"].SglangConfig = dummy_cls
    modules["slime.backends.sglang_utils.sglang_engine"].SGLangEngine = dummy_cls

    modules["slime.rollout"].base_types = modules["slime.rollout.base_types"]
    modules["slime.rollout.base_types"].call_rollout_fn = lambda *args, **kwargs: None

    modules["slime"].utils = modules["slime.utils"]
    modules["slime.utils"].health_monitor = modules["slime.utils.health_monitor"]
    modules["slime.utils"].http_utils = modules["slime.utils.http_utils"]
    modules["slime.utils"].logging_utils = modules["slime.utils.logging_utils"]
    modules["slime.utils"].metric_utils = modules["slime.utils.metric_utils"]
    modules["slime.utils"].misc = modules["slime.utils.misc"]
    modules["slime.utils"].seqlen_balancing = modules["slime.utils.seqlen_balancing"]
    modules["slime.utils"].types = modules["slime.utils.types"]

    modules["slime.utils.health_monitor"].RolloutHealthMonitor = dummy_cls
    modules["slime.utils.http_utils"]._wrap_ipv6 = lambda value: value
    modules["slime.utils.http_utils"].find_available_port = lambda *args, **kwargs: 0
    modules["slime.utils.http_utils"].get_host_info = lambda *args, **kwargs: ("127.0.0.1", "localhost")
    modules["slime.utils.http_utils"].init_http_client = lambda *args, **kwargs: None
    modules["slime.utils.logging_utils"].configure_logger = lambda *args, **kwargs: None
    modules["slime.utils.logging_utils"].finish_tracking = lambda *args, **kwargs: None
    modules["slime.utils.logging_utils"].init_tracking = lambda *args, **kwargs: None
    modules["slime.utils.logging_utils"].log = fake_log
    modules["slime.utils.metric_utils"].compute_pass_rate = lambda *args, **kwargs: {}
    modules["slime.utils.metric_utils"].compute_rollout_step = fake_compute_rollout_step
    modules["slime.utils.metric_utils"].compute_statistics = lambda values: {}
    modules["slime.utils.metric_utils"].dict_add_prefix = lambda d, prefix: {f"{prefix}{k}": v for k, v in d.items()}
    modules["slime.utils.metric_utils"].has_repetition = lambda text: False
    modules["slime.utils.misc"].Box = dict
    modules["slime.utils.misc"].group_by = fake_group_by
    modules["slime.utils.misc"].load_function = fake_load_function
    modules["slime.utils.seqlen_balancing"].get_seqlen_balanced_partitions = lambda *args, **kwargs: []
    modules["slime.utils.types"].Sample = DummySample

    modules["slime"].ray = modules["slime.ray"]
    modules["slime.ray"].utils = modules["slime.ray.utils"]
    modules["slime.ray.utils"].NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    modules["slime.ray.utils"].Lock = DummyLock

    saved_modules = {name: sys.modules.get(name) for name in modules}
    saved_test_module = sys.modules.get("slime.ray.rollout_under_test")

    try:
        sys.modules.update(modules)
        spec = importlib.util.spec_from_file_location(
            "slime.ray.rollout_under_test",
            Path(__file__).resolve().parents[1] / "slime" / "ray" / "rollout.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, log_calls
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        if saved_test_module is None:
            sys.modules.pop("slime.ray.rollout_under_test", None)
        else:
            sys.modules["slime.ray.rollout_under_test"] = saved_test_module


def make_args(**overrides):
    defaults = dict(
        use_wandb=False,
        use_tensorboard=False,
        wandb_always_use_train_step=False,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        global_batch_size=8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_log_fully_async_metrics_uses_dedicated_step_axis():
    module, log_calls = load_rollout_manager_module()
    args = make_args()
    fake_manager = SimpleNamespace(args=args, rollout_id=7, _fully_async_log_step=0)

    module.RolloutManager._log_fully_async_metrics(
        fake_manager,
        {
            "fully_async/count/stale_samples_processed": 3,
            "fully_async/partial/total_partial_num": 2,
            "stale_samples": 99,
        },
    )

    module.RolloutManager._log_fully_async_metrics(
        fake_manager,
        {
            "fully_async/count/stale_samples_processed": 5,
        },
    )

    assert len(log_calls) == 2
    _, metrics, step_key = log_calls[0]
    assert step_key == "fully_async/step"
    assert metrics["fully_async/step"] == 0
    assert metrics["fully_async/count/stale_samples_processed"] == 3
    assert metrics["fully_async/partial/total_partial_num"] == 2
    assert "rollout/step" not in metrics
    assert "stale_samples" not in metrics
    _, second_metrics, second_step_key = log_calls[1]
    assert second_step_key == "fully_async/step"
    assert second_metrics["fully_async/step"] == 1
    assert second_metrics["fully_async/count/stale_samples_processed"] == 5
    assert fake_manager._fully_async_log_step == 2


def test_log_fully_async_metrics_skips_initial_sync_and_empty_payloads():
    module, log_calls = load_rollout_manager_module()
    args = make_args()

    module.RolloutManager._log_fully_async_metrics(
        SimpleNamespace(args=args, rollout_id=-1),
        {"fully_async/count/stale_samples_processed": 1},
    )
    module.RolloutManager._log_fully_async_metrics(SimpleNamespace(args=args, rollout_id=3), None)
    module.RolloutManager._log_fully_async_metrics(SimpleNamespace(args=args, rollout_id=3), {"stale_samples": 1})

    assert log_calls == []


def test_after_weight_update_logs_hook_metrics():
    module, log_calls = load_rollout_manager_module()
    args = make_args()
    runtime_updates = []
    hook_calls = []
    hook_result = {
        "fully_async/count/stale_samples_processed": 4,
        "fully_async/partial/total_partial_num": 1,
    }
    fake_manager = SimpleNamespace(
        args=args,
        rollout_id=5,
        _fully_async_log_step=0,
        update_runtime_state=lambda **metadata: runtime_updates.append(metadata),
        _call_generate_rollout_hook=lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or hook_result,
    )
    fake_manager._log_fully_async_metrics = lambda result: module.RolloutManager._log_fully_async_metrics(
        fake_manager, result
    )

    result = module.RolloutManager.after_weight_update(fake_manager, policy_version=3)

    assert result is hook_result
    assert runtime_updates == [{"current_policy_version": 3}]
    assert hook_calls == [("after_weight_update", {"policy_version": 3})]
    assert len(log_calls) == 1
    _, metrics, step_key = log_calls[0]
    assert step_key == "fully_async/step"
    assert metrics["fully_async/step"] == 0
    assert fake_manager._fully_async_log_step == 1


def test_dispose_flushes_tail_window_metrics():
    module, log_calls = load_rollout_manager_module()
    args = make_args()
    hook_calls = []
    stop_calls = []
    hook_results = {
        "flush_metrics": {"fully_async/partial/total_partial_num": 3},
        "shutdown_worker": None,
    }
    fake_manager = SimpleNamespace(
        args=args,
        rollout_id=2,
        _fully_async_log_step=0,
        _health_monitors=[
            SimpleNamespace(stop=lambda: stop_calls.append("first")),
            SimpleNamespace(stop=lambda: stop_calls.append("second")),
        ],
        _call_generate_rollout_hook=lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs))
        or hook_results[hook_name],
    )
    fake_manager._log_fully_async_metrics = lambda result: module.RolloutManager._log_fully_async_metrics(
        fake_manager, result
    )

    module.RolloutManager.dispose(fake_manager)

    assert hook_calls == [("flush_metrics", {}), ("shutdown_worker", {})]
    assert len(log_calls) == 1
    _, metrics, step_key = log_calls[0]
    assert step_key == "fully_async/step"
    assert metrics["fully_async/step"] == 0
    assert metrics["fully_async/partial/total_partial_num"] == 3
    assert stop_calls == ["first", "second"]
    assert fake_manager._fully_async_log_step == 1
