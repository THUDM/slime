import ast
import dataclasses
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]


def _training_args(**overrides):
    values = dict(
        lr=2e-6,
        kl_coef=0.1,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        critic_num_nodes=1,
        critic_num_gpus_per_node=1,
        use_critic=False,
        megatron_config_path=None,
        start_rollout_id=None,
        rollout_global_dataset=False,
        num_rollout=1,
    )
    values.update(overrides)
    return Namespace(**values)


class _DummyTrainGroup:
    def __init__(self, args, role="actor"):
        self.args = args
        self.role = role
        self.create_calls = 0

    def create(self, rollout_manager=None):
        self.create_calls += 1
        return [3]


def _install_allocate_train_group(monkeypatch, allocated):
    from slime.ray import placement_group as placement_group_module

    def fake_allocate_train_group(
        args,
        num_nodes,
        num_gpus_per_node,
        pg,
        role="actor",
        with_ref=False,
        with_opd_teacher=False,
        actor_cls=None,
    ):
        allocated.append(role)
        return _DummyTrainGroup(args, role=role)

    monkeypatch.setattr(placement_group_module, "allocate_train_group", fake_allocate_train_group)
    monkeypatch.setattr(placement_group_module.ray, "get", lambda value: value)
    return placement_group_module


def test_create_training_models_skips_critic_when_num_rollout_is_zero(monkeypatch):
    allocated = []
    placement_group_module = _install_allocate_train_group(monkeypatch, allocated)
    args = _training_args(use_critic=True, num_rollout=0)

    actor_model, critic_model = placement_group_module.create_training_models(
        args,
        {"actor": None, "critic": None},
        object(),
    )

    assert allocated == ["actor"]
    assert critic_model is None
    assert actor_model.create_calls == 1
    assert args.start_rollout_id == 3


def test_create_training_models_still_builds_critic_when_training(monkeypatch):
    allocated = []
    placement_group_module = _install_allocate_train_group(monkeypatch, allocated)
    args = _training_args(use_critic=True, num_rollout=4)

    actor_model, critic_model = placement_group_module.create_training_models(
        args,
        {"actor": None, "critic": None},
        object(),
    )

    assert allocated == ["actor", "critic"]
    assert critic_model is not None
    assert critic_model.create_calls == 1
    assert actor_model.create_calls == 1
    assert args.start_rollout_id == 3


def test_train_eval_only_pushes_weights_then_evals_without_training():
    calls = []

    class Actor:
        def update_weights(self):
            calls.append("update_weights")

        def async_train(self, *args, **kwargs):
            raise AssertionError("eval-only must not train")

        def save_model(self, *args, **kwargs):
            raise AssertionError("eval-only must not save")

        def clear_memory(self):
            raise AssertionError("eval-only must not enter the train loop")

        def create(self):
            raise AssertionError("eval-only must not recreate the actor")

    class Remote:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            calls.append((self.name, args, kwargs))
            return (self.name, args, kwargs)

    class RolloutManager:
        def __init__(self):
            self.eval = Remote("eval")
            self.dispose = Remote("dispose")
            self.generate = Remote("generate")
            self.onload_weights = Remote("onload_weights")
            self.onload_kv = Remote("onload_kv")
            self.offload = Remote("offload")
            self.save = Remote("save")
            self.check_weights = Remote("check_weights")

    actor = Actor()
    rollout_manager = RolloutManager()

    def fake_create_training_models(args, pgs, rollout_manager):
        args.start_rollout_id = 9
        calls.append("create_training_models")
        return actor, None

    tree = ast.parse((REPO_ROOT / "train.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    ns = {
        "ray": types.SimpleNamespace(get=lambda value: value),
        "configure_logger": lambda: None,
        "init_tracking": lambda args: None,
        "finish_tracking": lambda args: None,
        "create_placement_groups": lambda args: {"rollout": object()},
        "create_rollout_manager": lambda args, pg: (rollout_manager, None),
        "create_training_models": fake_create_training_models,
        "should_run_periodic_action": lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("eval-only must not enter the train loop")
        ),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(REPO_ROOT / "train.py"), "exec"), ns)

    args = Namespace(
        release_train=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=0,
        eval_interval=1,
        start_rollout_id=None,
        use_critic=True,
        offload_train=True,
        save_interval=1,
    )
    ns["train"](args)

    assert calls == [
        "create_training_models",
        "update_weights",
        ("eval", (), {"rollout_id": 0}),
        ("dispose", (), {}),
    ]


def _load_model_module(monkeypatch):
    megatron_modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.distributed": types.ModuleType("megatron.core.distributed"),
        "megatron.core.enums": types.ModuleType("megatron.core.enums"),
        "megatron.core.models": types.ModuleType("megatron.core.models"),
        "megatron.core.models.gpt": types.ModuleType("megatron.core.models.gpt"),
        "megatron.core.optimizer": types.ModuleType("megatron.core.optimizer"),
        "megatron.core.optimizer.optimizer": types.ModuleType("megatron.core.optimizer.optimizer"),
        "megatron.core.optimizer_param_scheduler": types.ModuleType("megatron.core.optimizer_param_scheduler"),
        "megatron.core.pipeline_parallel": types.ModuleType("megatron.core.pipeline_parallel"),
        "megatron.core.pipeline_parallel.utils": types.ModuleType("megatron.core.pipeline_parallel.utils"),
        "megatron.core.utils": types.ModuleType("megatron.core.utils"),
        "megatron.training": types.ModuleType("megatron.training"),
        "megatron.training.global_vars": types.ModuleType("megatron.training.global_vars"),
        "megatron.training.training": types.ModuleType("megatron.training.training"),
    }
    megatron_modules["megatron.core"].mpu = types.SimpleNamespace()
    megatron_modules["megatron.core.distributed"].DistributedDataParallel = type("DDP", (), {})
    megatron_modules["megatron.core.distributed"].finalize_model_grads = lambda *args, **kwargs: None
    megatron_modules["megatron.core.enums"].ModelType = types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")
    megatron_modules["megatron.core.models.gpt"].GPTModel = type("GPTModel", (), {})

    @dataclasses.dataclass
    class OptimizerConfig:
        pass

    megatron_modules["megatron.core.optimizer"].OptimizerConfig = OptimizerConfig
    megatron_modules["megatron.core.optimizer"].get_megatron_optimizer = lambda **kwargs: object()
    megatron_modules["megatron.core.optimizer.optimizer"].MegatronOptimizer = type("MegatronOptimizer", (), {})
    megatron_modules["megatron.core.optimizer_param_scheduler"].OptimizerParamScheduler = type(
        "OptimizerParamScheduler", (), {}
    )
    megatron_modules["megatron.core.pipeline_parallel"].get_forward_backward_func = lambda: None
    megatron_modules["megatron.core.pipeline_parallel.utils"].unwrap_model = lambda model: model
    megatron_modules["megatron.core.utils"].get_model_config = lambda model: None
    megatron_modules["megatron.core.utils"].unwrap_model = lambda model: model
    megatron_modules["megatron.training.global_vars"].get_args = lambda: None
    megatron_modules["megatron.training.training"].get_model = lambda *args, **kwargs: ["model"]
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)
    logging_utils = types.ModuleType("slime.utils.logging_utils")
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils)
    wandb_mod = types.ModuleType("wandb")
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
    for name, module in megatron_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    pkg_name = "isolated_megatron_utils"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(REPO_ROOT / "slime" / "backends" / "megatron_utils")]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)

    checkpoint = types.ModuleType(f"{pkg_name}.checkpoint")
    checkpoint.load_checkpoint = lambda *args, **kwargs: (0, None)
    checkpoint.save_checkpoint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, checkpoint.__name__, checkpoint)

    cp_utils = types.ModuleType(f"{pkg_name}.cp_utils")
    cp_utils.reduce_train_step_metrics = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, cp_utils.__name__, cp_utils)

    data = types.ModuleType(f"{pkg_name}.data")
    data.DataIterator = object
    data.get_batch = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, data.__name__, data)

    loss = types.ModuleType(f"{pkg_name}.loss")
    loss.ROLLOUT_TOP_P_TOKEN_KEYS = ()
    loss.get_rollout_top_p_logprob_kwargs = lambda *args, **kwargs: None
    loss.loss_function = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, loss.__name__, loss)

    model_provider = types.ModuleType(f"{pkg_name}.model_provider")
    model_provider.get_model_provider_func = lambda args, role: object()
    monkeypatch.setitem(sys.modules, model_provider.__name__, model_provider)

    module_name = f"{pkg_name}.model"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "model.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_setup_model_and_optimizer_skips_train_stack_when_num_rollout_is_zero(monkeypatch):
    module = _load_model_module(monkeypatch)
    calls = []

    def fake_get_model(*args, **kwargs):
        calls.append("get_model")
        return ["model"]

    def fake_get_megatron_optimizer(**kwargs):
        calls.append("optimizer")
        return object()

    def fake_get_optimizer_param_scheduler(args, optimizer):
        calls.append("scheduler")
        return object()

    monkeypatch.setattr(module, "get_model", fake_get_model)
    monkeypatch.setattr(module, "get_megatron_optimizer", fake_get_megatron_optimizer)
    monkeypatch.setattr(module, "get_optimizer_param_scheduler", fake_get_optimizer_param_scheduler)

    args = Namespace(
        num_rollout=0,
        moe_use_upcycling=False,
        load="ckpt",
        pretrained_checkpoint=None,
        no_load_optim=False,
    )
    model, optimizer, scheduler = module.setup_model_and_optimizer(args, "actor")

    assert calls == ["get_model"]
    assert model == ["model"]
    assert optimizer is None
    assert scheduler is None
    assert args.no_load_optim is True


def test_setup_model_and_optimizer_builds_train_stack_when_training(monkeypatch):
    module = _load_model_module(monkeypatch)
    calls = []

    def fake_get_model(*args, **kwargs):
        calls.append("get_model")
        return ["model"]

    def fake_get_megatron_optimizer(**kwargs):
        calls.append("optimizer")
        return object()

    def fake_get_optimizer_param_scheduler(args, optimizer):
        calls.append("scheduler")
        return object()

    monkeypatch.setattr(module, "get_model", fake_get_model)
    monkeypatch.setattr(module, "get_megatron_optimizer", fake_get_megatron_optimizer)
    monkeypatch.setattr(module, "get_optimizer_param_scheduler", fake_get_optimizer_param_scheduler)

    args = Namespace(
        num_rollout=4,
        moe_use_upcycling=False,
        load="ckpt",
        pretrained_checkpoint=None,
        use_stateless_adam=False,
        enable_gloo_process_groups=False,
    )
    model, optimizer, scheduler = module.setup_model_and_optimizer(args, "actor")

    assert calls == ["get_model", "optimizer", "scheduler"]
    assert model == ["model"]
    assert optimizer is not None
    assert scheduler is not None


def test_initialize_model_and_optimizer_loads_checkpoint_without_optimizer(monkeypatch):
    module = _load_model_module(monkeypatch)
    loads = []

    class Chunk:
        role = None

    monkeypatch.setattr(module, "setup_model_and_optimizer", lambda args, role: ([Chunk()], None, None))
    monkeypatch.setattr(module, "clear_memory", lambda: None)
    monkeypatch.setattr(module, "_critic_output_layer_needs_reinit", lambda args, model, role: False)

    def fake_load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        loads.append((optimizer, opt_param_scheduler, kwargs.get("skip_load_to_model_and_opt")))
        return 0, None

    monkeypatch.setattr(module, "load_checkpoint", fake_load_checkpoint)

    args = Namespace(num_rollout=0, fp16=False, bf16=False)
    model, optimizer, scheduler, iteration = module.initialize_model_and_optimizer(args, "actor")

    assert optimizer is None
    assert scheduler is None
    assert iteration == 0
    assert model[0].role == "actor"
    assert loads == [(None, None, False)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
