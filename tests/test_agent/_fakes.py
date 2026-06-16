"""Shared CPU-only fakes for the agent test suite.

These stand in for the four real external boundaries of an agent rollout so the
whole pipeline (generate -> sandbox -> harness -> adapter HTTP -> sglang ->
trajectory) can run deterministically on CPU, no GPU / E2B / sglang /
checkpoint required. The code under test stays real; only these edges are faked:

  * :class:`FakeTokenizer`     -- word-level chat-template render + decode that
                                  round-trips (``decode(encode(t)) == t``), so a
                                  scripted model reply survives the
                                  encode->generate->decode->parse round trip.
  * :class:`ScriptedTokenizer` -- pre-baked prompt-id queue + id->text decode,
                                  for adapter unit tests that assert exact ids.
  * :class:`FakeSGLangServer`  -- a real aiohttp ``/generate`` upstream returning
                                  scripted ``output_token_logprobs`` per turn
                                  (exercises the real HTTP path in
                                  ``common.call_sglang_generate``).
  * :func:`fake_call_sglang_generate` -- a drop-in for
                                  ``common.call_sglang_generate`` (monkeypatch)
                                  that skips HTTP and yields scripted TurnRecords.
  * :class:`FakeSandbox`       -- an in-memory :class:`slime.agent.sandbox.Sandbox`
                                  that records every ``exec`` and drives the
                                  detached-launch/poll handshake via ``on_launch``.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from aiohttp import web

from slime.agent.adapters.common import TurnRecord

# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

# Fixed ids for the structural chat-template markers (kept out of the dynamic
# word band so decode can drop them as "special").
_ROLE_BEGIN = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
_ROLE_END = 5
_GEN = 3  # add_generation_prompt marker == assistant-begin (mirrors a real template)
_SPECIAL_IDS = set(_ROLE_BEGIN.values()) | {_ROLE_END, _GEN}
_WORD_BASE = 100  # dynamic word ids start here, never colliding with specials


class FakeTokenizer:
    """Deterministic word-level tokenizer with a round-tripping decode.

    Each distinct whitespace-delimited word gets a stable id (>= ``_WORD_BASE``)
    on first sight, so ``decode(encode(text)) == text`` for the content words.
    ``apply_chat_template`` frames each message as ``[ROLE_BEGIN, *words, END]``
    and appends the generation marker, the same shape a real template emits --
    enough for the manager to see clean prefix extensions when an assistant turn
    is replayed verbatim on the next request.
    """

    def __init__(self, outputs: dict[tuple[int, ...], str] | None = None) -> None:
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}
        # Explicit output-id -> text map for decode, decoupled from encode: lets a
        # test script the model server to return fixed ids and still control the
        # decoded reply text (mirrors the historic ToyTokenizer.outputs pattern).
        self._outputs = dict(outputs or {})
        self.rendered: list[tuple[list[dict], list[dict] | None]] = []

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            tid = _WORD_BASE + len(self._vocab)
            self._vocab[word] = tid
            self._inv[tid] = word
        return self._vocab[word]

    def encode(self, text: str) -> list[int]:
        return [self._id(w) for w in text.split()] if text else []

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        ids = list(ids)
        if tuple(ids) in self._outputs:
            return self._outputs[tuple(ids)]
        return " ".join(self._inv[i] for i in ids if i in self._inv and i not in _SPECIAL_IDS)

    @staticmethod
    def _content_text(message: dict) -> str:
        c = message.get("content")
        parts: list[str] = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            parts.append(reasoning)
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            parts.append(f"toolcall:{fn.get('name', '')}")
        return " ".join(p for p in parts if p)

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        out: list[int] = []
        for m in messages:
            role = m.get("role", "user")
            out.append(_ROLE_BEGIN.get(role, _ROLE_BEGIN["user"]))
            out.extend(self.encode(self._content_text(m)))
            out.append(_ROLE_END)
        if add_generation_prompt:
            out.append(_GEN)
        return out


class ScriptedTokenizer:
    """Pre-baked prompt-id queue + id->text decode, for adapter unit tests that
    assert exact token sequences. ``apply_chat_template`` ignores the messages
    and pops the next scripted prompt; ``decode`` maps an output-id tuple to its
    scripted text."""

    def __init__(self, prompts: list[list[int]], outputs: dict[tuple[int, ...], str]) -> None:
        self.prompts = [list(p) for p in prompts]
        self.outputs = dict(outputs)
        self.rendered: list[tuple[list[dict], list[dict] | None]] = []

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        self.rendered.append((list(messages), tools))
        assert self.prompts, "unexpected chat-template render (prompt queue exhausted)"
        return self.prompts.pop(0)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.outputs.get(tuple(ids), "")


# ---------------------------------------------------------------------------
# sglang /generate fakes
# ---------------------------------------------------------------------------


class FakeSGLangServer:
    """Real aiohttp ``/generate`` upstream returning scripted turns.

    Each turn is a list of ``(logprob, token_id)`` pairs (sglang's
    ``output_token_logprobs`` shape). Records every request body + routing key so
    tests can assert the adapter posted the right ``input_ids`` / sampling params.
    Use as an async context manager; ``.url`` is the base url to hand the adapter.
    """

    def __init__(self, turns: list[list[tuple[float, int]]], *, finish_reason: str = "stop") -> None:
        self.turns = [list(t) for t in turns]
        self.finish_reason = finish_reason
        self.requests: list[dict] = []
        self.routing_keys: list[str | None] = []
        self._server: web.Application | None = None
        self._runner = None

    async def _handle(self, request: web.Request) -> web.Response:
        self.routing_keys.append(request.headers.get("X-SMG-Routing-Key"))
        self.requests.append(await request.json())
        assert self.turns, "unexpected /generate call (turn script exhausted)"
        pairs = self.turns.pop(0)
        return web.json_response(
            {
                "meta_info": {
                    "output_token_logprobs": [[lp, tid] for lp, tid in pairs],
                    "finish_reason": {"type": self.finish_reason},
                }
            }
        )

    async def __aenter__(self) -> FakeSGLangServer:
        from aiohttp.test_utils import TestServer

        app = web.Application()
        app.router.add_post("/generate", self._handle)
        self._server = TestServer(app)
        await self._server.start_server()
        self.url = str(self._server.make_url("")).rstrip("/")
        return self

    async def __aexit__(self, *exc) -> None:
        if self._server is not None:
            await self._server.close()


def fake_call_sglang_generate(scripted: list[tuple[str, str, list[float] | None]], tokenizer: FakeTokenizer):
    """Build a drop-in for ``common.call_sglang_generate`` (for monkeypatch).

    ``scripted`` is a list of ``(response_text, finish_reason, logprobs)`` consumed
    one per turn. The response text is encoded with ``tokenizer`` so the adapter's
    ``decode(output_ids)`` round-trips it back, and the real
    ``parse_model_output`` runs on it. The returned coroutine matches the real
    signature ``(prompt_ids, session, body, *, adapter, session_id=None)``.
    """
    queue = list(scripted)

    async def _fake(prompt_ids, session, body, *, adapter, session_id=None) -> TurnRecord:
        assert queue, "unexpected sglang /generate call (response script exhausted)"
        text, finish, logprobs = queue.pop(0)
        output_ids = tokenizer.encode(text)
        lp = list(logprobs) if logprobs is not None else [0.0] * len(output_ids)
        assert len(lp) == len(output_ids), "scripted logprobs length must match encoded response"
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=lp,
        )

    return _fake


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

_POLL_RE = re.compile(r"test -f (\S+) && cat \1")


class FakeSandbox:
    """In-memory :class:`slime.agent.sandbox.Sandbox` for CPU tests.

    Records every ``exec`` (so harness tests can assert the right commands were
    issued) and keeps an in-memory file store for ``write_file`` / ``read_file``.
    It drives the detached-launch / poll-marker handshake of
    ``harness.common.run_command`` without any real process: when it sees the
    ``setsid`` launch command it awaits the injected ``on_launch(env)`` agent
    coroutine, then writes its exit code into the done-marker file so the next
    poll succeeds.

    Construct directly, or via :meth:`factory` to get a zero-arg callable that
    ``examples...generate.E2BSandbox`` / ``swe.E2BSandbox`` can be monkeypatched
    to (they call ``E2BSandbox(image)``).
    """

    def __init__(
        self,
        image: str = "fake-image",
        *,
        on_launch: Callable[[dict], Awaitable[int]] | None = None,
        responses: list[tuple[str, tuple[int, str, str]]] | None = None,
    ) -> None:
        self.image = image
        self.sandbox_id = f"fake-{image}"
        self.on_launch = on_launch
        # ordered (substring -> (exit, stdout, stderr)) overrides, first match wins.
        self.responses = list(responses or [])
        self.files: dict[str, str | bytes] = {}
        self.exec_log: list[tuple[str, str]] = []  # (cmd, user)

    @classmethod
    def factory(cls, **kwargs) -> Callable[..., FakeSandbox]:
        """Return ``E2BSandbox(image)``-compatible constructor with kwargs baked in."""

        def _make(image: str = "fake-image", **_ignored) -> FakeSandbox:
            return cls(image, **kwargs)

        return _make

    async def __aenter__(self) -> FakeSandbox:
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False):
        self.exec_log.append((cmd, user))

        # Detached launch (run_command): drive the fake agent, then drop the marker.
        if "setsid" in cmd and self.on_launch is not None:
            code = await self.on_launch(env or {})
            done = _done_path_from_launch(cmd)
            if done:
                self.files[done] = f"{code}\n"
            return 0, "", ""

        # Marker poll (run_command): succeed only once the marker file exists.
        m = _POLL_RE.search(cmd)
        if m:
            path = m.group(1)
            if path in self.files:
                return 0, _as_str(self.files[path]), ""
            return 1, "", ""

        for needle, result in self.responses:
            if needle in cmd:
                return result
        return 0, "", ""

    async def write_file(self, sandbox_path, content, *, user="root") -> None:
        self.files[sandbox_path] = content

    async def read_file(self, sandbox_path, *, user="root") -> str:
        return _as_str(self.files.get(sandbox_path, ""))


def _as_str(v: str | bytes) -> str:
    return v.decode() if isinstance(v, bytes) else v


def _done_path_from_launch(cmd: str) -> str | None:
    """The launcher script writes ``$PIPESTATUS`` into ``{workdir}/.harness/done``;
    recover that path from the ``setsid {launcher}`` command so the poll matches.
    ``run_command`` always names the marker ``.harness/done`` under the workdir."""
    m = re.search(r"(\S+/\.harness)/run\.sh", cmd)
    if m:
        return f"{m.group(1)}/done"
    return None
