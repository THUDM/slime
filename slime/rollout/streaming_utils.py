"""Helpers for consuming SGLang streaming responses."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from slime.utils.misc import decode_int32_meta_array

_TOP_P_TOKEN_ID_META_KEYS = ("top_p_token_ids", "top_p_kept_token_ids")
_TOP_P_TOKEN_OFFSET_META_KEYS = ("top_p_token_offsets", "top_p_kept_token_offsets")


def _has_full_response_top_p_metadata(
    meta_info: dict[str, Any],
    *,
    new_token_count: int,
    reported_length: int,
) -> bool:
    token_ids = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
    offsets = decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
    if token_ids is None or offsets is None or offsets.numel() == new_token_count + 1:
        return False
    if offsets.numel() != reported_length + 1:
        raise ValueError(
            "Incremental SGLang top-p metadata must describe either the current chunk or the full response: "
            f"offsets={offsets.numel()}, chunk_tokens={new_token_count}, reported_tokens={reported_length}."
        )

    return True


@dataclass(frozen=True)
class SGLangStreamUpdate:
    """A normalized update ready to apply to the current HTTP call state."""

    replace_call_state: bool
    tokens: list[int]
    log_probs: list[float]
    meta_info: dict[str, Any]


@dataclass
class SGLangStreamAccumulator:
    """Validate and accumulate configured SGLang stream chunks.

    SGLang's ``output_token_logprobs_length`` is cumulative in both modes.
    In cumulative mode each chunk must contain that many logprob pairs; in
    incremental mode the current and prior chunk lengths must sum to it.
    """

    output_mode: Literal["incremental", "cumulative"]
    output_length: int = 0
    tokens: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    _text_chunks: list[str] = field(default_factory=list)
    _cumulative_text: str | None = None
    _decode_text: bool = False

    def add(self, chunk: dict[str, Any]) -> SGLangStreamUpdate:
        meta_info = chunk.get("meta_info") or {}
        if "output_token_logprobs_length" not in meta_info:
            raise ValueError("SGLang streaming responses must include output_token_logprobs_length.")

        pairs = meta_info.get("output_token_logprobs") or []
        chunk_tokens = [item[1] for item in pairs]
        chunk_log_probs = [item[0] for item in pairs]
        reported_length = int(meta_info["output_token_logprobs_length"])

        previous_length = self.output_length
        cumulative = self.output_mode == "cumulative"
        expected_length = len(chunk_tokens) if cumulative else previous_length + len(chunk_tokens)
        if expected_length != reported_length:
            raise ValueError(
                f"SGLang {self.output_mode} streaming output has inconsistent output_token_logprobs_length: "
                f"received={len(chunk_tokens)}, previous={previous_length}, "
                f"expected={expected_length}, reported={reported_length}."
            )
        if reported_length < previous_length:
            raise ValueError(
                "SGLang cumulative streaming output length decreased: "
                f"previous={previous_length}, reported={reported_length}."
            )

        if cumulative:
            new_tokens = chunk_tokens[previous_length:]
            new_log_probs = chunk_log_probs[previous_length:]
            self.tokens.extend(new_tokens)
            self.log_probs.extend(new_log_probs)
        else:
            new_tokens = chunk_tokens
            new_log_probs = chunk_log_probs
            self.tokens.extend(new_tokens)
            self.log_probs.extend(new_log_probs)

        full_top_p_metadata = _has_full_response_top_p_metadata(
            meta_info,
            new_token_count=len(new_tokens),
            reported_length=reported_length,
        )

        chunk_text = chunk.get("text")
        if cumulative:
            if chunk_text is not None:
                self._cumulative_text = chunk_text
            elif new_tokens:
                self._decode_text = True
        else:
            if chunk_text is None:
                self._decode_text = True
            elif chunk_text:
                self._text_chunks.append(chunk_text)

        self.output_length = reported_length
        return SGLangStreamUpdate(
            replace_call_state=full_top_p_metadata,
            tokens=list(self.tokens) if full_top_p_metadata else new_tokens,
            log_probs=list(self.log_probs) if full_top_p_metadata else new_log_probs,
            meta_info=meta_info,
        )

    def response_text(self, decode: Callable[[list[int]], str]) -> str:
        """Materialize response text once after completion or cancellation."""
        if self.output_mode == "cumulative":
            if self._cumulative_text is not None and not self._decode_text:
                return self._cumulative_text
            return decode(self.tokens)
        if self._decode_text:
            return decode(self.tokens)
        return "".join(self._text_chunks)
