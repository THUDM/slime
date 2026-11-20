"""Custom sample->train_data conversion for MOPD teacher routing.

slime's built-in conversion (``slime.data.batch_builder.BatchBuilder.convert``)
does not carry a per-record MOPD teacher annotation,
and the per-rank split only forwards a fixed key whitelist, so a brand-new dict
key would be dropped before it reaches the actor. This hook runs at rollout time
-- where the full sample (prompt and metadata) still exists -- resolves each
sample's teacher from the MOPD config, and writes the resolved ids into the
whitelisted ``source_names`` field. The actor's routing then reads that id
directly (``config.has_teacher(source)``) instead of re-deriving it from the
reduced train-side fields.

Only ``source_names`` is (over)written: it is in slime's rollout-log ignore set,
whereas a non-numeric field such as ``prompt`` would break
``train_metric_utils.log_rollout_data`` (it does ``sum(value)`` on non-tensor
per-sample lists). Prompt-based routing still works because resolution happens
here, while the real sample prompt is still available.

Use with::

    --custom-convert-samples-to-train-data-path \
        slime_mopd.train_data.mopd_convert_samples_to_train_data
"""

from __future__ import annotations

import types
from typing import Any

from slime_mopd.config import load_mopd_config
from slime_mopd.routing import resolve_teacher_id, set_sample_teacher

from slime.data.batch_builder import BatchBuilder
from slime.utils.misc import load_function


def _default_convert(args: Any, samples: list[Any]) -> dict[str, Any]:
    """Run slime's built-in conversion via a minimal shim ``self``.

    ``BatchBuilder.convert`` only touches ``self.args``, the two custom hooks,
    and ``self._post_process_rewards``; the shim supplies exactly those, so the
    conversion logic stays slime's and remains in sync with upstream.
    """

    shim = types.SimpleNamespace()
    shim.args = args
    shim.custom_convert_samples_to_train_data_func = None
    reward_post_process_path = getattr(args, "custom_reward_post_process_path", None)
    shim.custom_reward_post_process_func = (
        load_function(reward_post_process_path) if reward_post_process_path is not None else None
    )
    shim._post_process_rewards = types.MethodType(BatchBuilder._post_process_rewards, shim)
    return BatchBuilder.convert(shim, samples)


def resolve_train_teacher_ids(config: Any, args: Any, samples: list[Any]) -> list[str]:
    """Resolve one teacher id per sample, caching it on the sample."""

    teacher_ids: list[str] = []
    for sample in samples:
        teacher_id = getattr(sample, "mopd_teacher_id", None)
        if not teacher_id:
            teacher_id = resolve_teacher_id(config, args, sample)
            set_sample_teacher(sample, teacher_id)
        teacher_ids.append(str(teacher_id))
    return teacher_ids


def mopd_convert_samples_to_train_data(args: Any, samples: list[Any]) -> dict[str, Any]:
    """slime custom-convert hook: per-record teacher ids into train_data."""

    train_data = _default_convert(args, samples)
    config = load_mopd_config(args)
    train_data["source_names"] = resolve_train_teacher_ids(config, args, samples)
    return train_data
