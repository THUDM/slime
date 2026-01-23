import os
import socket
from copy import deepcopy
import wandb

def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode."""
    if hasattr(args, "wandb_mode") and args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"

def _get_unique_dir(base_dir, rank, process_name):
    """
    为每个进程生成唯一的日志目录，防止 Offline 模式下的文件锁冲突。
    结构: base_dir/offline_logs/rank_{rank}_{process_name}_{pid}
    """
    if base_dir is None:
        base_dir = "./wandb"
    
    # 获取当前进程PID，确保唯一性
    pid = os.getpid()
    unique_sub = f"rank_{rank}_{process_name}_{pid}"
    return os.path.join(base_dir, "offline_logs", unique_sub)

def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            print("W&B offline mode enabled. Data will be saved locally.")

    offline = _is_offline_mode(args)

    # Login if online
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Group and Name setup
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Config setup
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "job_type": "primary",  # 明确标记为主进程
        "config": _compute_config_for_logging(args),
    }

    # Settings setup
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
        # 【关键修复】Offline模式下，主进程也使用独立目录，避免与同节点的其他进程冲突
        if args.wandb_dir:
            unique_dir = _get_unique_dir(args.wandb_dir, args.rank, "primary")
            os.makedirs(unique_dir, exist_ok=True)
            init_kwargs["dir"] = unique_dir
            print(f"[WandB Primary] Logging to unique offline dir: {unique_dir}")
        elif args.wandb_dir: # Fallback if not offline but dir specified
             os.makedirs(args.wandb_dir, exist_ok=True)
             init_kwargs["dir"] = args.wandb_dir
    else:
        # Online mode: use shared settings
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)
        if args.wandb_dir:
            os.makedirs(args.wandb_dir, exist_ok=True)
            init_kwargs["dir"] = args.wandb_dir

    run = wandb.init(**init_kwargs)
    _init_wandb_common()

    # Save Run ID
    args.wandb_run_id = run.id
    print(f"[WandB] Primary Run ID: {args.wandb_run_id}")


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    whitelist_env_vars = ["SLURM_JOB_ID"]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    return output


def init_wandb_secondary(args, wandb_run_id, router_addr=None):
    """
    Initialize W&B for secondary processes (RolloutManager, Trainers, etc.).
    """
    if wandb_run_id is None:
        return

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # === Settings Configuration ===
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    # SGLang metrics forwarding
    if args.sglang_enable_metrics and router_addr is not None:
        print(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={"sgl_engine": f"{router_addr}/engine_metrics"},
            x_stats_open_metrics_filters={"sgl_engine.*": {}},
        )

    # === 【关键修复】构建完整的 init 参数 ===
    # 1. 获取 Group (与 Primary 保持一致)
    # 注意：args.wandb_group 可能在 Primary 中被修改过（加了随机后缀），
    # 但由于 args 在 Ray 中传递通常是拷贝，这里最好重新构建或确保 args.wandb_group 是最终值。
    # 假设 args.wandb_group 是基础组名，我们尽量保持一致。
    # 如果 Primary 修改了 args 对象并传递过来最好，否则这里使用基础 group 也没问题，
    # 只要 ID 相同，W&B 会自动归类。
    group_name = args.wandb_group 
    
    # 2. 构建 Name (增加 Rank 标识，方便在 UI 中区分来源)
    # 尝试获取 rank，如果没有则使用 PID
    rank = getattr(args, "rank", "unknown")
    run_name = f"{group_name}-worker-{rank}-{os.getpid()}"

    init_kwargs = {
        "id": wandb_run_id,          # 强制使用相同 ID
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group_name,         # 【新增】确保分组正确
        "name": run_name,            # 【新增】确保名字唯一
        "job_type": "secondary",     # 【新增】标记为从属进程
        "config": args.__dict__,
        "resume": "allow",           # 允许恢复/合并
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # === 【关键修复】Offline 模式下的目录隔离 ===
    if args.wandb_dir:
        if offline:
            # 在 Offline 模式下，强制为每个进程使用独立的子目录。
            # 这样每个进程都会生成自己的 wandb-events.jsonl，互不干扰。
            # wandb sync 时会自动合并具有相同 ID 的运行记录。
            unique_dir = _get_unique_dir(args.wandb_dir, rank, "secondary")
            os.makedirs(unique_dir, exist_ok=True)
            init_kwargs["dir"] = unique_dir
            print(f"[WandB Secondary] Logging to unique offline dir: {unique_dir}")
        else:
            # Online Shared 模式下，通常需要指向同一个目录以便 Service 发现，
            # 或者由 Service 处理。
            os.makedirs(args.wandb_dir, exist_ok=True)
            init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)
    _init_wandb_common()


def _init_wandb_common():
    """Define metrics for custom plotting."""
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")

    # Train batch metrics (samples used for training, may be from buffer)
    # Use train/step for consistency with train/ metrics
    wandb.define_metric("train_batch/*", step_metric="train/step")

    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")

    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    wandb.define_metric("perf/*", step_metric="rollout/step")

    # Buffer metrics (use train/step for consistency)
    wandb.define_metric("buffer/*", step_metric="train/step") 

def get_wandb_offline_dir(args):
    """Get the directory where offline W&B data is stored."""
    if _is_offline_mode(args):
        if args and hasattr(args, "wandb_dir") and args.wandb_dir:
            return args.wandb_dir
        else:
            return os.path.expanduser("~/wandb")
    return None