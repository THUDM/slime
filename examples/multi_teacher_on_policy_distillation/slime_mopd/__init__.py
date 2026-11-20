"""MOPD: teacher-routed multi-teacher on-policy distillation hooks for slime.

Data model: one prompt file whose rows carry a ``domain`` label (``math`` /
``code`` / ``ifeval`` in the example) and one teacher per label. The data source
stamps the label on every sample and routing maps it onto the teacher registry,
so there is no per-file dataset binding table.

slime's core code stays unchanged: the package couples to it only through
existing hook arguments, so it runs on any backend slime supports.

Hooks (sglang teacher backend, one teacher server per ``rm_url``)::

    --custom-rm-path slime_mopd.rm.mopd_reward_func
    --custom-reward-post-process-path slime_mopd.post_process.mopd_post_process
    --data-source-path slime_mopd.data_source.MopdRolloutDataSource
    --custom-config-path /path/to/mopd.yaml     # provides the top-level `mopd` block
    --use-opd --opd-type sglang --opd-kl-coef 1.0

Megatron teacher backend (teachers loaded from checkpoints, no teacher servers)
additionally uses the custom convert hook, the zero reward, and the injected
actor::

    --custom-convert-samples-to-train-data-path \
        slime_mopd.train_data.mopd_convert_samples_to_train_data
    --custom-rm-path slime_mopd.rm.mopd_zero_reward
    --use-opd --opd-type megatron
    python -m slime_mopd.train_mopd ...

See ``README.md`` in this directory and ``run-qwen3.5-9B-mopd.sh``.
"""

from slime_mopd.config import MopdConfig, MopdConfigError, TeacherConfig, load_mopd_config
from slime_mopd.routing import RoutingError, resolve_teacher, resolve_teacher_id

__all__ = [
    "MopdConfig",
    "MopdConfigError",
    "RoutingError",
    "TeacherConfig",
    "load_mopd_config",
    "resolve_teacher",
    "resolve_teacher_id",
]

__version__ = "0.1.0"
