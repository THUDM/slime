# Add these arguments to slime/utils/arguments.py
# Insert after line 852 (after --buffer-max-size)

# === Buffer Sampling Strategy Configuration ===
parser.add_argument(
    "--buffer-sampling-strategy",
    type=str,
    choices=["fifo_staleness", "priority", "random", "reservoir", "custom"],
    default="fifo_staleness",
    help=(
        "Buffer sampling strategy: "
        "'fifo_staleness': FIFO with staleness filtering (DEFAULT, recommended for most cases); "
        "'priority': Priority-based sampling using reward/advantage metrics; "
        "'random': Random sampling with staleness filtering; "
        "'reservoir': Reservoir sampling for uniform distribution; "
        "'custom': Custom strategy (requires --buffer-sampling-custom-path)"
    ),
)

parser.add_argument(
    "--buffer-sampling-custom-path",
    type=str,
    default=None,
    help=(
        "Path to custom sampling strategy class (only used when buffer_sampling_strategy=custom). "
        "Should be a subclass of BaseSamplingStrategy. "
        "Example: 'my_module.MyCustomStrategy'"
    ),
)

parser.add_argument(
    "--buffer-remove-on-sample",
    type=lambda x: x.lower() in ["true", "1", "yes"],
    default=True,
    help=(
        "Whether to remove samples from buffer after sampling. "
        "If False, samples can be reused (controlled by --buffer-reuse-samples). "
        "Default: True (each sample used once)."
    ),
)

parser.add_argument(
    "--buffer-reuse-samples",
    type=int,
    default=1,
    help=(
        "Maximum times a sample can be reused (only effective when --buffer-remove-on-sample=False). "
        "0 means unlimited reuse. "
        "Default: 1 (no reuse, equivalent to remove-on-sample=True)"
    ),
)

# === Priority Sampling Configuration ===
parser.add_argument(
    "--buffer-priority-metric",
    type=str,
    choices=["reward", "advantage", "custom"],
    default="reward",
    help=(
        "Metric for priority sampling (only used when buffer_sampling_strategy=priority): "
        "'reward': Sample based on reward values; "
        "'advantage': Sample based on advantage values (requires pre-computation); "
        "'custom': Custom metric (requires --buffer-priority-custom-path)"
    ),
)

parser.add_argument(
    "--buffer-priority-custom-path",
    type=str,
    default=None,
    help=(
        "Path to custom priority metric function (only used when buffer_priority_metric=custom). "
        "Function signature: def metric(group: List[Sample]) -> float"
    ),
)

parser.add_argument(
    "--buffer-priority-weight",
    type=float,
    default=1.0,
    help="Weight for priority metric in scoring (only for priority sampling). Default: 1.0",
)

parser.add_argument(
    "--buffer-staleness-penalty",
    type=float,
    default=0.1,
    help=(
        "Staleness penalty coefficient for priority sampling. "
        "Score = reward × priority_weight - staleness × staleness_penalty. "
        "Default: 0.1"
    ),
)

# === Random Sampling Configuration ===
parser.add_argument(
    "--buffer-random-seed",
    type=int,
    default=None,
    help="Random seed for random/reservoir sampling strategies. Default: None (non-deterministic)",
)
