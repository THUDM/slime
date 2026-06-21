from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from slime.utils.misc import decode_int32_meta_array

_TOP_P_TOKEN_ID_META_KEYS = ("top_p_token_ids", "top_p_kept_token_ids")
_TOP_P_TOKEN_OFFSET_META_KEYS = ("top_p_token_offsets", "top_p_kept_token_offsets")


def _extract_rollout_top_p_token_data(
    meta_info: dict[str, Any],
    *,
    expected_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    token_ids = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
    offsets = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
    if token_ids is None and offsets is None:
        return None
    if token_ids is None or offsets is None:
        raise ValueError("SGLang top-p token replay must include both token ids and offsets.")
    if offsets.numel() == 0 or int(offsets[0]) != 0:
        raise ValueError(f"SGLang top-p token offsets must start with 0, got {offsets[:1].tolist()}.")
    if int(offsets[-1]) != token_ids.numel():
        raise ValueError(
            "SGLang top-p token ids/offsets mismatch: "
            f"offsets[-1]={int(offsets[-1])}, len(token_ids)={token_ids.numel()}."
        )
    if expected_num_tokens is not None and offsets.numel() != expected_num_tokens + 1:
        raise ValueError(
            "SGLang top-p token offsets length must equal generated token count + 1: "
            f"len(offsets)={offsets.numel()}, generated={expected_num_tokens}."
        )
    return token_ids, offsets


def _merge_rollout_top_p_token_data(
    base_token_ids: list[int] | torch.Tensor | None,
    base_offsets: list[int] | torch.Tensor | None,
    token_ids: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_token_ids = torch.as_tensor([] if base_token_ids is None else base_token_ids, dtype=torch.int32).reshape(-1)
    base_offsets = torch.as_tensor([0] if base_offsets is None else base_offsets, dtype=torch.int32).reshape(-1)
    base_offset = int(base_offsets[-1])
    return (
        torch.cat([base_token_ids, token_ids]),
        torch.cat([base_offsets, offsets[1:] + base_offset]),
    )


def _pad_rollout_top_p_offsets(
    token_ids: list[int] | torch.Tensor | None,
    offsets: list[int] | torch.Tensor | None,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if offsets is None or token_ids is None:
        raise ValueError("Cannot append empty top-p spans without existing token ids and offsets.")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    token_ids = torch.as_tensor(token_ids, dtype=torch.int32).reshape(-1)
    offsets = torch.as_tensor(offsets, dtype=torch.int32).reshape(-1)
    if offsets.numel() == 0:
        raise ValueError("Cannot append empty top-p spans to empty offsets.")
    if num_tokens == 0:
        return token_ids, offsets
    empty_offsets = offsets.new_full((num_tokens,), int(offsets[-1]))
    return token_ids, torch.cat([offsets, empty_offsets])


def _to_int_list(tokens) -> list[int]:
    if tokens is None:
        return []
    if torch.is_tensor(tokens):
        return [int(token) for token in tokens.detach().cpu().reshape(-1).tolist()]
    return [int(token) for token in tokens]


def _to_float_list(values) -> list[float] | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]
    return [float(value) for value in values]


def _numel(value) -> int:
    return int(torch.as_tensor(value).reshape(-1).numel())


@dataclass
class Sample:
    """The sample generated"""

    group_index: int | None = None
    index: int | None = None
    # Id of the rollout this sample came from. Defaults to ``None`` and the
    # downstream pipeline falls back to ``index`` (so the default rollout
    # path, where one execution = one training sample, sees rollout_id ==
    # index). Compact / subagent paths that split one rollout execution into
    # multiple training samples should set the same ``rollout_id`` on every
    # sibling, so loss aggregation averages within the rollout instead of
    # over-counting it.
    rollout_id: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] | None = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] | None = None  # processed multimodal data, e.g. pixel_values, etc.
    multimodal_train_input_id: str | None = None
    apply_chat_template_kwargs: dict = field(default_factory=dict)
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    # Ragged top-p nucleus token ids replayed from rollout sampling. For response
    # token i, kept ids are rollout_top_p_token_ids[offsets[i]:offsets[i + 1]].
    rollout_top_p_token_ids: list[int] | torch.Tensor | None = None
    rollout_top_p_token_offsets: list[int] | torch.Tensor | None = None
    rollout_routed_experts: list[list[int]] | torch.Tensor | None = None  # Routed experts from rollout engine
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    custom_rm_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    # Session ID for consistent hashing routing (used when router policy is consistent_hashing)
    session_id: str | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        spec_accept_token_num: int = 0
        spec_draft_token_num: int = 0
        spec_verify_ct: int = 0
        completion_token_num: int = 0

        @property
        def spec_accept_rate(self) -> float:
            return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

        @property
        def spec_accept_length(self) -> float:
            return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            self.spec_accept_token_num += meta_info.get("spec_accept_token_num", 0)
            self.spec_draft_token_num += meta_info.get("spec_draft_token_num", 0)
            self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
            self.completion_token_num += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            return {
                "spec_accept_token_num": self.spec_accept_token_num,
                "spec_draft_token_num": self.spec_draft_token_num,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_token_num": self.completion_token_num,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.SpecInfo()
            info.spec_accept_token_num = data.get("spec_accept_token_num", 0)
            info.spec_draft_token_num = data.get("spec_draft_token_num", 0)
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_token_num = data.get("completion_token_num", 0)
            return info

    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        cached_tokens: int = 0
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def append_response_tokens(
        self,
        args=None,
        *,
        tokens=None,
        log_probs=None,
        trainable: bool = True,
        meta_info: dict | None = None,
        text: str | None = None,
        update_terminal_info: bool = True,
    ):
        """
        Append response-side tokens and keep training metadata aligned.

        Model-generated tokens should pass ``trainable=True`` plus SGLang
        ``meta_info`` and log probabilities. Tool/environment tokens should pass
        ``trainable=False``; they receive loss-mask zeros and empty top-p spans
        when top-p replay is active.
        """
        tokens = _to_int_list(tokens)
        log_probs = _to_float_list(log_probs)
        if log_probs is not None and len(log_probs) != len(tokens):
            raise ValueError(f"log_probs length {len(log_probs)} != tokens length {len(tokens)}")
        if tokens and trainable and log_probs is None:
            raise ValueError("trainable response tokens require rollout log probabilities.")
        if tokens and not trainable:
            if log_probs is not None:
                raise ValueError("non-trainable response tokens should not pass rollout log probabilities.")
            log_probs = [0.0] * len(tokens)

        if text is not None:
            self.response += text

        previous_response_length = self.response_length
        if tokens:
            self.tokens += tokens
            self.response_length += len(tokens)
            if self.loss_mask is None:
                self.loss_mask = [1] * previous_response_length
            self.loss_mask += [1 if trainable else 0] * len(tokens)

        if log_probs is not None:
            if self.rollout_log_probs is None:
                if trainable and previous_response_length:
                    raise ValueError(
                        "Cannot append trainable rollout log probabilities to a sample with existing response "
                        "tokens but no existing rollout_log_probs."
                    )
                self.rollout_log_probs = [0.0] * previous_response_length
            self.rollout_log_probs += log_probs

        should_pad_top_p = bool(tokens and not trainable)
        if meta_info is not None or should_pad_top_p:
            self._apply_meta_info(
                args,
                meta_info or {},
                new_token_count=len(tokens),
                pad_missing_top_p=should_pad_top_p,
                update_terminal_info=update_terminal_info,
            )

        self._validate_response_metadata_lengths()

    def _apply_meta_info(
        self,
        args,
        meta_info: dict,
        *,
        new_token_count: int = 0,
        pad_missing_top_p: bool = False,
        update_terminal_info: bool = True,
    ) -> None:
        applied_top_p_data = False
        if new_token_count:
            top_p_data = _extract_rollout_top_p_token_data(meta_info, expected_num_tokens=new_token_count)
            if top_p_data is not None:
                applied_top_p_data = True
                base_token_ids, base_offsets = self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets
                if base_token_ids is None and base_offsets is None:
                    self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets = top_p_data
                else:
                    self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets = _merge_rollout_top_p_token_data(
                        base_token_ids,
                        base_offsets,
                        *top_p_data,
                    )

        if (
            pad_missing_top_p
            and new_token_count
            and self.rollout_top_p_token_offsets is not None
            and not applied_top_p_data
        ):
            self.rollout_top_p_token_ids, self.rollout_top_p_token_offsets = _pad_rollout_top_p_offsets(
                self.rollout_top_p_token_ids,
                self.rollout_top_p_token_offsets,
                new_token_count,
            )

        routed_experts = decode_int32_meta_array(meta_info, "routed_experts")
        if routed_experts is not None:
            if args is None:
                raise ValueError("args is required to decode routed experts metadata.")
            self.rollout_routed_experts = routed_experts.reshape(
                len(self.tokens) - 1,
                args.num_layers,
                args.moe_router_topk,
            )

        if not update_terminal_info or "finish_reason" not in meta_info:
            return

        if getattr(args, "sglang_speculative_algorithm", False):
            # cannot directly use spec info from sglang because of partial rollout.
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        self.prefix_cache_info.add(meta_info=meta_info)

        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED

    def _validate_response_metadata_lengths(self):
        if self.loss_mask is not None and len(self.loss_mask) != self.response_length:
            raise ValueError(f"loss_mask length {len(self.loss_mask)} != response_length {self.response_length}")

        if self.rollout_log_probs is not None and len(self.rollout_log_probs) != self.response_length:
            raise ValueError(
                f"rollout_log_probs length {len(self.rollout_log_probs)} != response_length {self.response_length}"
            )

        if self.rollout_top_p_token_ids is None and self.rollout_top_p_token_offsets is None:
            return
        if self.rollout_top_p_token_ids is None or self.rollout_top_p_token_offsets is None:
            raise ValueError("rollout top-p replay must include both token ids and offsets.")

        offsets = torch.as_tensor(self.rollout_top_p_token_offsets, dtype=torch.int32).reshape(-1)
        if offsets.numel() != self.response_length + 1:
            raise ValueError(
                "rollout_top_p_token_offsets length must equal response_length + 1: "
                f"len(offsets)={offsets.numel()}, response_length={self.response_length}."
            )
        token_id_count = _numel(self.rollout_top_p_token_ids)
        if int(offsets[-1]) != token_id_count:
            raise ValueError(
                "rollout top-p token ids/offsets mismatch: "
                f"offsets[-1]={int(offsets[-1])}, len(token_ids)={token_id_count}."
            )


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]


@dataclass
class MultimodalType:
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        return next((m for m in cls.all() if m.name == name), None)
