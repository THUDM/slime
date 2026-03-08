"""Configuration dataclasses for SGLang engine deployment."""

import dataclasses
import logging

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerGroupConfig:
    """Configuration for a single server group.

    Attributes:
        worker_type: One of "regular", "prefill", "decode", or "placeholder".
                     "placeholder" reserves GPU slots without creating engines.
        num_gpus: Total number of GPUs for this group.
        num_gpus_per_engine: GPUs per engine for this group.  Overrides the
                             model-level or global ``--rollout-num-gpus-per-engine``.
        overrides: Optional dict of SGLang ``ServerArgs`` field overrides.
                   These are applied on top of the base CLI ``--sglang-*``
                   arguments in ``_compute_server_args``.
    """

    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        valid_types = {"regular", "prefill", "decode", "placeholder"}
        assert (
            self.worker_type in valid_types
        ), f"Invalid worker_type '{self.worker_type}', must be one of {valid_types}"
        assert self.num_gpus > 0, f"num_gpus must be > 0, got {self.num_gpus}"


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single model deployment.

    Attributes:
        name: Unique name for this model (e.g. "actor", "reward").
        model_path: HF checkpoint path.  Falls back to ``args.hf_checkpoint``.
        num_gpus_per_engine: Default GPUs per engine for all groups in this
                             model.  Individual groups can override.
        server_groups: Server group configurations for this model.
        update_weights: Whether this model receives weight updates from
                        training.  Set to ``False`` for frozen models
                        (reference, reward, etc.).  When ``None`` (default),
                        automatically inferred in ``resolve()``: ``True`` if
                        model_path matches ``args.hf_checkpoint``, ``False``
                        otherwise.
    """

    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    server_groups: list[ServerGroupConfig] = dataclasses.field(default_factory=list)
    update_weights: bool | None = None

    def resolve(self, args) -> None:
        """Resolve per-group defaults from model-level then args-level values."""
        default_gpus_per_engine = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
        default_model_path = self.model_path or args.hf_checkpoint
        for g in self.server_groups:
            if g.num_gpus_per_engine is None:
                g.num_gpus_per_engine = default_gpus_per_engine
            # Inject model_path into overrides so _compute_server_args picks it up.
            if "model_path" not in g.overrides:
                g.overrides["model_path"] = default_model_path

        # Validate: all server groups within a model must share the same model_path.
        if self.server_groups:
            model_paths = {g.overrides["model_path"] for g in self.server_groups}
            assert len(model_paths) == 1, (
                f"Model '{self.name}' has server groups with different model_path values: "
                f"{model_paths}. All server groups within a model must use the same model_path."
            )
            effective_model_path = model_paths.pop()
        else:
            effective_model_path = default_model_path

        # Auto-infer update_weights when not explicitly set.
        if self.update_weights is None:
            if effective_model_path != args.hf_checkpoint:
                logger.warning(
                    f"Model '{self.name}' uses model_path='{effective_model_path}' which differs "
                    f"from hf_checkpoint='{args.hf_checkpoint}'. Defaulting update_weights to False. "
                    f"Set update_weights explicitly in the config to suppress this warning."
                )
                self.update_weights = False
            else:
                self.update_weights = True

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(g.worker_type in ("prefill", "decode") for g in self.server_groups)

    @property
    def total_num_gpus(self) -> int:
        return sum(g.num_gpus for g in self.server_groups)


@dataclasses.dataclass
class SglangConfig:
    """Configuration for SGLang engine deployment.

    Loaded from ``--sglang-config`` YAML file.

    **Config format**::

        sglang:
          - name: actor
            model_path: /path/to/actor
            update_weights: true          # receives training weight updates (default)
            num_gpus_per_engine: 2
            server_groups:
              - worker_type: prefill
                num_gpus: 4
                num_gpus_per_engine: 2
              - worker_type: decode
                num_gpus: 8
                num_gpus_per_engine: 4
          - name: ref
            model_path: /path/to/ref
            update_weights: false          # frozen, no weight updates
            server_groups:
              - worker_type: regular
                num_gpus: 4

    Each model gets its own router.  ``placeholder`` groups reserve GPU
    slots without creating engines.  ``overrides`` are ``ServerArgs``
    field names applied on top of the base ``--sglang-*`` CLI args.

    Set ``update_weights: false`` for frozen models (reference, reward,
    etc.) that should not receive weight updates from training.

    .. note::

       ``engine_groups`` is accepted as a backward-compatible alias for
       ``server_groups`` in the YAML config.
    """

    models: list[ModelConfig]

    @staticmethod
    def from_yaml(path: str) -> "SglangConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        assert "sglang" in data, (
            f"sglang config must have a 'sglang' key, got {list(data.keys())}. "
            f"Wrap your server_groups inside a model entry under 'sglang'."
        )
        models = []
        for m in data["sglang"]:
            # Accept both "server_groups" and legacy "engine_groups".
            raw_groups = m.get("server_groups") or m.get("engine_groups") or []
            groups = [ServerGroupConfig(**g) for g in raw_groups]
            models.append(
                ModelConfig(
                    name=m["name"],
                    model_path=m.get("model_path"),
                    num_gpus_per_engine=m.get("num_gpus_per_engine"),
                    server_groups=groups,
                    update_weights=m.get("update_weights"),
                )
            )
        return SglangConfig(models=models)

    @staticmethod
    def from_prefill_num_servers(args) -> "SglangConfig":
        """Build a config equivalent to the legacy --prefill-num-servers flag."""
        total_gpus = args.rollout_num_gpus
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = total_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {total_gpus}, prefill {prefill_gpus}"
        return SglangConfig(
            models=[
                ModelConfig(
                    name="default",
                    server_groups=[
                        ServerGroupConfig(worker_type="prefill", num_gpus=prefill_gpus),
                        ServerGroupConfig(worker_type="decode", num_gpus=decode_gpus),
                    ],
                )
            ]
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(m.has_pd_disaggregation for m in self.models)

    @property
    def total_num_gpus(self) -> int:
        return sum(m.total_num_gpus for m in self.models)
