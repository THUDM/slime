# `examples/common`

Shared code for training examples lives here only when it is part of the
training or data-preparation pipeline and is reused by multiple examples.

Current layout:

- `dataset_registry.py`, `dataset_selection.py`
  Shared dataset registry and resolution helpers.
- `prepare_pool_data.py`, `prepare_runtime_dataset.py`
  Shared data-prep entrypoints.
- `pool_data_utils.py`, `pool_runtime_semantics.py`, `eval_prep_utils.py`
  Shared runtime materialization and eval-prep helpers.
- `multidomain_shared.py`, `log_rollout.py`
  Shared Python entry modules used directly by training jobs.
- `dataset_queries.py`, `system_queries.py`
  Small Python CLIs used by shell scripts. Dataset/data-prep queries and
  system/Ray helpers are kept separate on purpose.
- `ray_bootstrap_utils.sh`
  Ray address and hostname/bootstrap helpers.
- `training_runner_utils.sh`
  Shared Ray cluster orchestration for training runners.
- `training_prep_utils.sh`
  Shared training-preparation helpers used by multiple runners.

If a file here stops serving multiple examples or stops belonging to the
training/data pipeline, move or delete it.
