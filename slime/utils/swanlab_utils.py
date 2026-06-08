import logging
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass

import httpx

try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None

logger = logging.getLogger(__name__)

_OPEN_METRICS_MONITOR = None


def _require_swanlab():
    if swanlab is None:
        raise ImportError("swanlab is not installed. Please install it with: pip install swanlab")


def _is_offline_mode(args) -> bool:
    if args.swanlab_mode:
        return args.swanlab_mode in {"offline", "local", "disabled"}
    return os.environ.get("SWANLAB_MODE") in {"offline", "local", "disabled"}


def _maybe_login(args):
    if _is_offline_mode(args):
        return
    if args.swanlab_key is not None or args.swanlab_host is not None or args.swanlab_web_host is not None:
        swanlab.login(api_key=args.swanlab_key, host=args.swanlab_host, web_host=args.swanlab_web_host)


def _sanitize_metric_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name).strip("_")


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        metric_expr = parts[0]
        value_text = parts[1]

        try:
            value = float(value_text)
        except ValueError:
            continue

        if "{" in metric_expr:
            name, labels_text = metric_expr.split("{", 1)
            labels_text = labels_text.rstrip("}")
            label_parts = []
            for item in labels_text.split(","):
                if not item or "=" not in item:
                    continue
                key, raw_value = item.split("=", 1)
                label_parts.append(f"{_sanitize_metric_name(key)}_{_sanitize_metric_name(raw_value.strip('\\"'))}")
            metric_name = "_".join([name, *label_parts]) if label_parts else name
        else:
            metric_name = metric_expr

        metrics[_sanitize_metric_name(metric_name)] = value

    return metrics


@dataclass
class _SwanlabOpenMetricsMonitor:
    args: object
    router_addr: str
    interval_s: int

    def __post_init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._poll_step = 0

    def start(self):
        if self._thread is not None:
            return self

        self._thread = threading.Thread(target=self._run, name="swanlab-open-metrics", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.interval_s)

    def poll_once(self):
        try:
            response = httpx.get(f"{self.router_addr}/engine_metrics", timeout=10.0)
            response.raise_for_status()
            metrics = _parse_prometheus_metrics(response.text)
            if not metrics:
                return

            payload = {f"sglang_engine/{key}": value for key, value in metrics.items()}
            swanlab.log(payload, step=self._poll_step)
            self._poll_step += 1
        except Exception:
            logger.exception("Failed to collect SwanLab open metrics from %s", self.router_addr)


def _build_init_kwargs(args, primary: bool):
    group = args.swanlab_group
    experiment_name = args.swanlab_experiment_name or group

    if args.swanlab_random_suffix and group:
        group = f"{group}_{swanlab.util.generate_id()}"
        if experiment_name is None:
            experiment_name = group

    init_kwargs = {
        "project": args.swanlab_project,
        "workspace": args.swanlab_workspace,
        "group": group,
        "experiment_name": experiment_name,
        "config": _compute_config_for_logging(args),
        "mode": args.swanlab_mode,
        "id": getattr(args, "swanlab_run_id", None),
        "resume": "allow" if getattr(args, "swanlab_run_id", None) is not None else None,
        "reinit": True,
    }

    if args.swanlab_mode in (None, "cloud"):
        init_kwargs["parallel"] = "shared"

    if args.swanlab_dir:
        os.makedirs(args.swanlab_dir, exist_ok=True)
        init_kwargs["logdir"] = args.swanlab_dir
        logger.info(f"SwanLab logs will be stored in: {args.swanlab_dir}")

    if primary and getattr(args, "rank", 0) != 0:
        init_kwargs["reinit"] = True

    return init_kwargs


def init_swanlab_primary(args):
    if not args.use_swanlab:
        args.swanlab_run_id = None
        return

    _require_swanlab()
    _maybe_login(args)

    init_kwargs = _build_init_kwargs(args, primary=True)
    swanlab.init(**init_kwargs)
    args.swanlab_run_id = swanlab.get_run().id


def reinit_swanlab_primary_with_open_metrics(args, router_addr):
    global _OPEN_METRICS_MONITOR

    if not args.use_swanlab:
        return
    if router_addr is None:
        return

    _require_swanlab()
    if _is_offline_mode(args):
        logger.info("SwanLab open metrics disabled in offline/local/disabled mode.")
        return

    if _OPEN_METRICS_MONITOR is not None:
        _OPEN_METRICS_MONITOR.stop()

    logger.info(f"Starting SwanLab open metrics monitor at {router_addr}.")
    _OPEN_METRICS_MONITOR = _SwanlabOpenMetricsMonitor(
        args=args,
        router_addr=router_addr,
        interval_s=max(int(getattr(args, "swanlab_open_metrics_interval", 10)), 1),
    ).start()


def init_swanlab_secondary(args):
    if not args.use_swanlab:
        return

    wandb_run_id = getattr(args, "swanlab_run_id", None)
    if wandb_run_id is None:
        return

    _require_swanlab()
    _maybe_login(args)

    init_kwargs = _build_init_kwargs(args, primary=False)
    init_kwargs["id"] = wandb_run_id
    init_kwargs["resume"] = "allow"
    swanlab.init(**init_kwargs)


def finish_swanlab(args):
    global _OPEN_METRICS_MONITOR

    if not args.use_swanlab:
        return
    if swanlab is None:
        return
    try:
        if _OPEN_METRICS_MONITOR is not None:
            _OPEN_METRICS_MONITOR.stop()
            _OPEN_METRICS_MONITOR = None
        run = swanlab.get_run()
        if run is not None:
            swanlab.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish SwanLab run")


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    whitelist_env_vars = ["SLURM_JOB_ID"]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    return output
