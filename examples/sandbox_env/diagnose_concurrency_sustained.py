#!/usr/bin/env python3
"""High-fidelity sandbox communication diagnosis for SWE rollout.

This script intentionally keeps only the useful checks from the earlier
exploration:

  1. ``llm-real``: a high-fidelity simulation of the real Inspire ROCK
     ``anti_call_llm`` file-log path.  Each sandbox has one outstanding request
     at a time, matching the real rollout loop.
  2. ``commands``: a small optional baseline for sandbox creation and
     ``sandbox.commands.run`` stability.

The older synthetic localhost HTTP, synthetic log polling, and UDS phases were
removed because they are useful for transport exploration but do not represent
the real training pipeline closely enough.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import statistics
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

SITE_PACKAGES = os.environ.get(
    "INSPIRE_SANDBOX_SITE_PACKAGES",
    "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/.local/share/inspire_sandbox_site_packages",
)
if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

from inspire_sandbox import Sandbox as InspireSandbox


DEFAULT_TEMPLATE = "rebench-aio-libs-aiohttp-9047"
SANDBOX_TIMEOUT = 600
COMMAND_TIMEOUT = 30
DEFAULT_PHASES = ("llm-real",)

TURN_COMMANDS = (
    "printf 'sandbox-ok '; uname -m",
    "test -r /etc/os-release && head -5 /etc/os-release || true",
    "pwd; ls -ld /tmp",
)

PHASE_ALIASES = {
    "commands": "commands",
    "command": "commands",
    "cmd": "commands",
    "llm-real": "llm-real",
    "llm-fidelity": "llm-real",
    "high-fidelity": "llm-real",
    "anti-call": "llm-real",
    "anti_call": "llm-real",
    "real-helper": "llm-real",
}

PHASE_DISPLAY = {
    "commands": "External commands.run",
    "llm-real-setup": "Real anti-call setup",
    "llm-real-batch": "Real anti-call turns",
}

STATUS_PATTERNS = (
    re.compile(r"\bHTTP_STATUS=(\d{3})\b"),
    re.compile(r"\bstatus_code[=: ]+(\d{3})\b", re.IGNORECASE),
    re.compile(r"\bstatus[=: ]+(\d{3})\b", re.IGNORECASE),
    re.compile(r"\bcode[=: ]+(\d{3})\b", re.IGNORECASE),
    re.compile(r"\bHTTP/\d(?:\.\d)?\s+(\d{3})\b"),
)


def _dig_attr(obj, path: str):
    cur = obj
    for name in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, name, None)
    return cur


def _render_exc_details(exc: Exception) -> str:
    details: list[str] = []

    for key in ("status_code", "code", "error_code", "request_id"):
        val = getattr(exc, key, None)
        if val is not None:
            details.append(f"{key}={val}")

    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
        if status is not None:
            details.append(f"response.status_code={status}")
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text:
            details.append(f"response.text={text[:400]}")
        json_fn = getattr(resp, "json", None)
        if callable(json_fn):
            try:
                payload = json_fn()
            except Exception:
                payload = None
            if payload is not None:
                details.append(f"response.json={str(payload)[:400]}")

    for path in ("error.code", "error.type", "error.message", "body", "detail"):
        val = _dig_attr(exc, path)
        if val is not None:
            details.append(f"{path}={str(val)[:400]}")

    if exc.__cause__ is not None:
        details.append(
            f"cause={exc.__cause__.__class__.__name__}: {str(exc.__cause__)[:300]}"
        )
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        details.append(
            f"context={exc.__context__.__class__.__name__}: {str(exc.__context__)[:300]}"
        )

    unique = []
    seen = set()
    for item in details:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return " | ".join(unique)


def _format_exception(exc: Exception) -> str:
    base = f"{exc.__class__.__name__}: {exc}"
    extra = _render_exc_details(exc)
    return f"{base} | {extra}" if extra else base


class AsyncBarrier:
    """Python 3.10 compatible one-shot async barrier."""

    def __init__(self, parties: int):
        self._parties = parties
        self._count = 0
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            self._count += 1
            if self._count >= self._parties:
                self._event.set()
        await self._event.wait()


@dataclass
class OpTiming:
    sandbox_idx: int
    phase: str
    turn: int
    op_idx: int
    label: str
    start: float
    end: float
    error: str | None = None

    @property
    def latency(self) -> float:
        return self.end - self.start


@dataclass
class SandboxResult:
    index: int
    create_sec: float = 0.0
    kill_sec: float = 0.0
    op_timings: list[OpTiming] = field(default_factory=list)
    error: str | None = None


@dataclass
class LlmRealState:
    log_path: str
    request_payload_path: str
    helper_path: str
    simulator_path: str
    runtime_cmd_file: str
    handle: object


@dataclass
class HarnessConfig:
    prompt_bytes: int
    response_bytes: int
    poll_interval: float
    tail_chunk_bytes: int
    ready_retries: int
    timeout: float


def parse_phases(raw: str) -> list[str]:
    phases: list[str] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        normalized = PHASE_ALIASES.get(part)
        if normalized is None:
            valid = ", ".join(sorted(PHASE_ALIASES))
            raise ValueError(f"Unknown phase '{part}'. Valid phases: {valid}")
        if normalized not in phases:
            phases.append(normalized)
    return phases or list(DEFAULT_PHASES)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(len(values) * q)))
    return sorted(values)[idx]


def _extract_status_like_code(text: str) -> str | None:
    for pattern in STATUS_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _extract_exit_error(result_obj) -> str | None:
    exit_code = getattr(result_obj, "exit_code", 0)
    if exit_code == 0:
        return None
    stderr = str(getattr(result_obj, "stderr", "") or "").strip()
    stdout = str(getattr(result_obj, "stdout", "") or "").strip()
    extra = stderr or stdout
    status_code = _extract_status_like_code(f"{stderr}\n{stdout}")
    prefix = f"status={status_code} exit={exit_code}" if status_code else f"exit={exit_code}"
    return f"{prefix}: {extra[:300]}" if extra else prefix


async def _call_timed(
    sandbox_idx: int,
    phase: str,
    turn: int,
    op_idx: int,
    label: str,
    func,
    *args,
    **kwargs,
):
    timing = OpTiming(
        sandbox_idx=sandbox_idx,
        phase=phase,
        turn=turn,
        op_idx=op_idx,
        label=label,
        start=time.monotonic(),
        end=0.0,
    )
    try:
        value = await asyncio.to_thread(func, *args, **kwargs)
    except Exception as exc:
        timing.end = time.monotonic()
        timing.error = _format_exception(exc)
        return None, timing
    timing.end = time.monotonic()
    return value, timing


def _build_llm_real_agent_code() -> str:
    return textwrap.dedent(
        """\
        import json
        import os
        import sys
        import time

        REQUEST_START_MARKER = "LLM_REQUEST_START"
        REQUEST_END_MARKER = "LLM_REQUEST_END"
        RESPONSE_START_MARKER = "LLM_RESPONSE_START"
        RESPONSE_END_MARKER = "LLM_RESPONSE_END"
        SESSION_END_MARKER = "SESSION_END"

        log_file = sys.argv[1]
        request_payload_path = sys.argv[2]
        total_turns = int(sys.argv[3])
        poll_interval = float(sys.argv[4])


        def append_line(line: str) -> None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line)


        def write_request(index: int, request_payload: str) -> None:
            meta = json.dumps({"timestamp": int(time.time() * 1000), "index": index}, ensure_ascii=False)
            append_line(f"{REQUEST_START_MARKER}{request_payload}{REQUEST_END_MARKER}{meta}\\n")


        def wait_for_response(expected_index: int) -> bool:
            pos = 0
            while True:
                if os.path.exists(log_file):
                    with open(log_file, encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        lines = f.readlines()
                        pos = f.tell()
                    for line in lines:
                        if SESSION_END_MARKER in line:
                            return False
                        if RESPONSE_START_MARKER not in line or RESPONSE_END_MARKER not in line:
                            continue
                        try:
                            meta_raw = line.split(RESPONSE_END_MARKER, 1)[1].strip()
                            meta = json.loads(meta_raw) if meta_raw else {}
                        except Exception:
                            continue
                        if meta.get("index") == expected_index:
                            return True
                time.sleep(poll_interval)


        with open(request_payload_path, encoding="utf-8") as f:
            request_payload = f.read()

        if total_turns <= 0:
            append_line(SESSION_END_MARKER + "\\n")
            raise SystemExit(0)

        write_request(1, request_payload)
        for index in range(1, total_turns + 1):
            if not wait_for_response(index):
                break
            if index < total_turns:
                write_request(index + 1, request_payload)
            else:
                append_line(SESSION_END_MARKER + "\\n")
        """
    ).strip()


def _build_llm_real_perl_helper_code() -> str:
    return textwrap.dedent(
        """\
        use strict;
        use warnings;
        use JSON::PP;
        use Time::HiRes qw(time sleep);

        my $REQUEST_START_MARKER = "LLM_REQUEST_START";
        my $REQUEST_END_MARKER = "LLM_REQUEST_END";
        my $RESPONSE_START_MARKER = "LLM_RESPONSE_START";
        my $RESPONSE_END_MARKER = "LLM_RESPONSE_END";
        my $SESSION_END_MARKER = "SESSION_END";

        my $index = int($ARGV[0]);
        my $log_file = $ARGV[1];
        my $poll_interval = 0.0 + $ARGV[2];
        my $tail_chunk_bytes = int($ARGV[3]);
        $tail_chunk_bytes = 4096 if $tail_chunk_bytes < 4096;
        my $payload = undef;
        if (defined $ARGV[4]) {
          open my $pf, '<', $ARGV[4] or die $!;
          local $/ = undef;
          $payload = <$pf>;
          close $pf;
        }

        sub get_file_size {
          return -e $log_file ? (-s $log_file || 0) : 0;
        }

        sub read_last_line_with_marker {
          my ($marker) = @_;
          return undef unless -e $log_file;
          open my $fh, '<:raw', $log_file or die $!;
          seek($fh, 0, 2);
          my $pos = tell($fh);
          return undef if !$pos;
          my $buf = '';
          while ($pos > 0) {
            my $step = $tail_chunk_bytes < $pos ? $tail_chunk_bytes : $pos;
            $pos -= $step;
            seek($fh, $pos, 0);
            read($fh, my $chunk, $step);
            $buf = $chunk . $buf;
            my @lines = split(/(?<=\\n)/, $buf);
            if ($pos > 0) {
              $buf = shift @lines;
            } else {
              $buf = '';
            }
            for (my $i = $#lines; $i >= 0; $i--) {
              return $lines[$i] if index($lines[$i], $marker) >= 0;
            }
          }
          return $buf if $buf ne '' && index($buf, $marker) >= 0;
          return undef;
        }

        sub parse_request_line {
          my ($line) = @_;
          return ($SESSION_END_MARKER, {}) if index($line, $SESSION_END_MARKER) >= 0;
          my ($prefix, $meta_json) = split(/\\Q$REQUEST_END_MARKER\\E/, $line, 2);
          my (undef, $request_json) = split(/\\Q$REQUEST_START_MARKER\\E/, $prefix, 2);
          my $meta = decode_json($meta_json);
          return ($request_json, $meta);
        }

        sub parse_response_line {
          my ($line) = @_;
          my ($prefix, $meta_json) = split(/\\Q$RESPONSE_END_MARKER\\E/, $line, 2);
          my (undef, $response_json) = split(/\\Q$RESPONSE_START_MARKER\\E/, $prefix, 2);
          my $meta = decode_json($meta_json);
          return ($response_json, $meta);
        }

        sub append_response {
          my ($last_response, $response_index) = @_;
          my $meta = encode_json({
            timestamp => int(time() * 1000),
            index => $response_index,
          });
          open my $fh, '>>:encoding(UTF-8)', $log_file or die $!;
          print $fh $RESPONSE_START_MARKER . $last_response . $RESPONSE_END_MARKER . $meta . "\\n";
          close $fh;
        }

        sub wait_for_request {
          my ($request_index, $start_pos) = @_;
          my $pos = $start_pos > 0 ? $start_pos : 0;
          while (1) {
            if (-e $log_file) {
              open my $fh, '<:encoding(UTF-8)', $log_file or die $!;
              seek($fh, $pos, 0);
              while (my $line = <$fh>) {
                $pos = tell($fh);
                return $SESSION_END_MARKER if index($line, $SESSION_END_MARKER) >= 0;
                next if index($line, $REQUEST_START_MARKER) < 0 || index($line, $REQUEST_END_MARKER) < 0;
                my ($request_json, $meta) = parse_request_line($line);
                return $request_json if ($meta->{index} // -1) == $request_index;
              }
              close $fh;
            }
            sleep($poll_interval);
          }
        }

        my $request_index = $index + 1;
        my $start_pos = get_file_size();
        my $last_response_line = read_last_line_with_marker($RESPONSE_START_MARKER);
        my $last_request_line = read_last_line_with_marker($REQUEST_START_MARKER);

        if ($index == 0) {
          if (!defined $last_request_line) {
            print wait_for_request($request_index, 0);
            exit 0;
          }
          my ($last_request, $meta) = parse_request_line($last_request_line);
          if (($meta->{index} // -1) == $request_index) {
            print $last_request;
            exit 0;
          }
          print wait_for_request($request_index, 0);
          exit 0;
        }

        die "missing response payload for index $index" if !defined $payload;

        if (defined $last_response_line) {
          my ($last_response, $meta) = parse_response_line($last_response_line);
          if (($meta->{index} // -1) < $index) {
            append_response($payload, $index);
            $start_pos = get_file_size();
          }
        } else {
          append_response($payload, $index);
          $start_pos = get_file_size();
        }

        $last_request_line = read_last_line_with_marker($REQUEST_START_MARKER);
        if (defined $last_request_line) {
          my ($last_request, $meta) = parse_request_line($last_request_line);
          if (($meta->{index} // -1) == $request_index) {
            print $last_request;
            exit 0;
          }
        }

        print wait_for_request($request_index, $start_pos);
        """
    ).strip()


def _build_llm_real_perl_agent_code() -> str:
    return textwrap.dedent(
        """\
        use strict;
        use warnings;
        use JSON::PP;
        use Time::HiRes qw(time sleep);

        my $REQUEST_START_MARKER = "LLM_REQUEST_START";
        my $REQUEST_END_MARKER = "LLM_REQUEST_END";
        my $RESPONSE_START_MARKER = "LLM_RESPONSE_START";
        my $RESPONSE_END_MARKER = "LLM_RESPONSE_END";
        my $SESSION_END_MARKER = "SESSION_END";

        my ($log_file, $request_payload_path, $total_turns, $poll_interval) = @ARGV;
        $total_turns = int($total_turns);
        $poll_interval = 0.0 + $poll_interval;

        sub append_line {
          my ($line) = @_;
          open my $fh, '>>:encoding(UTF-8)', $log_file or die $!;
          print $fh $line;
          close $fh;
        }

        sub write_request {
          my ($index, $request_payload) = @_;
          my $meta = encode_json({
            timestamp => int(time() * 1000),
            index => $index,
          });
          append_line($REQUEST_START_MARKER . $request_payload . $REQUEST_END_MARKER . $meta . "\\n");
        }

        sub wait_for_response {
          my ($expected_index) = @_;
          my $pos = 0;
          while (1) {
            if (-e $log_file) {
              open my $fh, '<:encoding(UTF-8)', $log_file or die $!;
              seek($fh, $pos, 0);
              while (my $line = <$fh>) {
                $pos = tell($fh);
                return 0 if index($line, $SESSION_END_MARKER) >= 0;
                next if index($line, $RESPONSE_START_MARKER) < 0 || index($line, $RESPONSE_END_MARKER) < 0;
                my (undef, $meta_json) = split(/\\Q$RESPONSE_END_MARKER\\E/, $line, 2);
                my $meta = eval { decode_json($meta_json) };
                next if !$meta;
                return 1 if ($meta->{index} // -1) == $expected_index;
              }
              close $fh;
            }
            sleep($poll_interval);
          }
        }

        open my $pf, '<:encoding(UTF-8)', $request_payload_path or die $!;
        local $/ = undef;
        my $request_payload = <$pf>;
        close $pf;

        if ($total_turns <= 0) {
          append_line($SESSION_END_MARKER . "\\n");
          exit 0;
        }

        write_request(1, $request_payload);
        for my $index (1 .. $total_turns) {
          last unless wait_for_response($index);
          if ($index < $total_turns) {
            write_request($index + 1, $request_payload);
          } else {
            append_line($SESSION_END_MARKER . "\\n");
          }
        }
        """
    ).strip()


def _build_llm_real_prep_cmd(root_dir: str, prompt_bytes: int) -> str:
    request_payload_path = f"{root_dir}/request_payload.json"
    runtime_cmd_file = f"{root_dir}/runtime_cmd.txt"
    log_path = f"{root_dir}/LLMService.log"
    return textwrap.dedent(
        f"""\
        mkdir -p {shlex.quote(root_dir)}
        : > {shlex.quote(log_path)}
        runtime_kind=""
        runtime_cmd=""
        if command -v python3 >/dev/null 2>&1; then
          runtime_kind="python"
          runtime_cmd="$(command -v python3)"
        elif command -v python >/dev/null 2>&1; then
          runtime_kind="python"
          runtime_cmd="$(command -v python)"
        elif command -v perl >/dev/null 2>&1; then
          runtime_kind="perl"
          runtime_cmd="$(command -v perl)"
        else
          echo "no python/perl runtime available" >&2
          exit 1
        fi
        printf '%s\\n' "$runtime_cmd" > {shlex.quote(runtime_cmd_file)}
        echo "RUNTIME_KIND=$runtime_kind"
        if [ "$runtime_kind" = "python" ]; then
          "$runtime_cmd" - {shlex.quote(request_payload_path)} {prompt_bytes} <<'PY'
        import json
        import sys

        out = sys.argv[1]
        n = int(sys.argv[2])
        payload = {{
            "model": "diag-long-context",
            "stream": False,
            "messages": [{{"role": "user", "content": "A" * n}}],
        }}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        PY
        else
          perl -MJSON::PP -e '
            my ($out, $n) = @ARGV;
            my $payload = {{
              model => "diag-long-context",
              stream => JSON::PP::false,
              messages => [{{ role => "user", content => ("A" x $n) }}],
            }};
            open my $fh, ">", $out or die $!;
            print $fh JSON::PP::encode_json($payload);
            close $fh;
          ' {shlex.quote(request_payload_path)} {prompt_bytes}
        fi
        test -s {shlex.quote(request_payload_path)}
        """
    ).strip()


def _build_llm_real_simulator_cmd(
    runtime_cmd_file: str,
    simulator_path: str,
    log_path: str,
    request_payload_path: str,
    total_turns: int,
    poll_interval: float,
) -> str:
    return textwrap.dedent(
        f"""\
        runtime_cmd="$(tr -d '\\n' < {shlex.quote(runtime_cmd_file)})"
        [ -n "$runtime_cmd" ]
        "$runtime_cmd" {shlex.quote(simulator_path)} \
          {shlex.quote(log_path)} \
          {shlex.quote(request_payload_path)} \
          {total_turns} \
          {poll_interval}
        """
    ).strip()


def _build_llm_real_probe_cmd(log_path: str, helper_path: str, simulator_path: str) -> str:
    return textwrap.dedent(
        f"""\
        test -s {shlex.quote(helper_path)}
        test -s {shlex.quote(simulator_path)}
        grep -q 'LLM_REQUEST_START' {shlex.quote(log_path)}
        echo ready
        """
    ).strip()


def _build_llm_real_turn_cmd(
    runtime_cmd_file: str,
    helper_path: str,
    log_path: str,
    turn: int,
    poll_interval: float,
    tail_chunk_bytes: int,
    response_payload: str | None,
) -> str:
    pieces = [
        f"runtime_cmd=\"$(tr -d '\\n' < {shlex.quote(runtime_cmd_file)})\"",
        '[ -n "$runtime_cmd" ]',
        f'"$runtime_cmd" {shlex.quote(helper_path)} {turn} {shlex.quote(log_path)} {poll_interval} {tail_chunk_bytes}',
    ]
    if response_payload is not None:
        pieces[-1] += f" {shlex.quote(response_payload)}"
    return "\n".join(pieces)


@lru_cache(maxsize=None)
def _build_llm_response_payload(response_bytes: int) -> str:
    payload = {
        "id": "chatcmpl-diag",
        "object": "chat.completion",
        "created": 0,
        "model": "diag-long-context",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "B" * response_bytes},
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _run_llm_real_turn_sync(sandbox, state: LlmRealState, turn: int, cfg: HarnessConfig):
    response_payload = _build_llm_response_payload(cfg.response_bytes) if turn > 0 else None
    return sandbox.commands.run(
        _build_llm_real_turn_cmd(
            state.runtime_cmd_file,
            state.helper_path,
            state.log_path,
            turn,
            cfg.poll_interval,
            cfg.tail_chunk_bytes,
            response_payload,
        ),
        timeout=cfg.timeout,
        user="root",
    )


async def _safe_kill_handle(handle) -> None:
    if handle is None:
        return
    try:
        await asyncio.to_thread(handle.kill)
    except Exception:
        pass


async def _setup_llm_real(
    sandbox,
    sandbox_idx: int,
    cfg: HarnessConfig,
    num_turns: int,
) -> tuple[LlmRealState | None, OpTiming]:
    from rock_inspire_adapter import build_inspire_anti_call_llm_helper_code

    root_dir = f"/tmp/diag-llm-real-{sandbox_idx}"
    log_path = f"{root_dir}/LLMService.log"
    request_payload_path = f"{root_dir}/request_payload.json"
    helper_path = f"{root_dir}/anti_call_helper.py"
    simulator_path = f"{root_dir}/agent_simulator.py"
    runtime_cmd_file = f"{root_dir}/runtime_cmd.txt"

    timing = OpTiming(
        sandbox_idx=sandbox_idx,
        phase="llm-real-setup",
        turn=-1,
        op_idx=0,
        label=f"prompt={cfg.prompt_bytes}B resp={cfg.response_bytes}B turns={num_turns}",
        start=time.monotonic(),
        end=0.0,
    )

    handle = None
    last_probe_error = ""
    try:
        prep = await asyncio.to_thread(
            sandbox.commands.run,
            _build_llm_real_prep_cmd(root_dir, cfg.prompt_bytes),
            timeout=COMMAND_TIMEOUT,
            user="root",
        )
        prep_error = _extract_exit_error(prep)
        if prep_error:
            timing.error = f"prep: {prep_error}"
            return None, timing

        prep_stdout = str(getattr(prep, "stdout", "") or "")
        runtime_match = re.search(r"RUNTIME_KIND=(python|perl)\b", prep_stdout)
        runtime_kind = runtime_match.group(1) if runtime_match else "python"
        if runtime_kind == "perl":
            helper_code = _build_llm_real_perl_helper_code().encode("utf-8")
            simulator_code = _build_llm_real_perl_agent_code().encode("utf-8")
        else:
            helper_code = build_inspire_anti_call_llm_helper_code().encode("utf-8")
            simulator_code = _build_llm_real_agent_code().encode("utf-8")

        await asyncio.to_thread(
            sandbox.files.write,
            helper_path,
            helper_code,
            "root",
            cfg.timeout,
        )
        await asyncio.to_thread(
            sandbox.files.write,
            simulator_path,
            simulator_code,
            "root",
            cfg.timeout,
        )

        handle = await asyncio.to_thread(
            sandbox.commands.run,
            _build_llm_real_simulator_cmd(
                runtime_cmd_file,
                simulator_path,
                log_path,
                request_payload_path,
                num_turns,
                cfg.poll_interval,
            ),
            background=True,
            timeout=cfg.timeout,
            user="root",
        )

        for _ in range(cfg.ready_retries):
            probe, probe_timing = await _call_timed(
                sandbox_idx,
                "llm-real-setup",
                -1,
                1,
                "probe",
                sandbox.commands.run,
                _build_llm_real_probe_cmd(log_path, helper_path, simulator_path),
                timeout=10,
                user="root",
            )
            if probe_timing.error is None and probe is not None:
                probe_error = _extract_exit_error(probe)
                if probe_error is None:
                    timing.end = time.monotonic()
                    return (
                        LlmRealState(
                            log_path=log_path,
                            request_payload_path=request_payload_path,
                            helper_path=helper_path,
                            simulator_path=simulator_path,
                            runtime_cmd_file=runtime_cmd_file,
                            handle=handle,
                        ),
                        timing,
                    )
                last_probe_error = probe_error
            else:
                last_probe_error = probe_timing.error or "probe failed"
            await asyncio.sleep(1)

        timing.error = (
            f"llm-real simulator not ready after {cfg.ready_retries} probes: "
            f"{last_probe_error}"
        )
        await _safe_kill_handle(handle)
        return None, timing
    except Exception as exc:
        timing.error = _format_exception(exc)
        await _safe_kill_handle(handle)
        return None, timing
    finally:
        timing.end = time.monotonic()


async def run_sandbox_workload(
    index: int,
    template: str,
    num_turns: int,
    phases: list[str],
    cfg: HarnessConfig,
    run_tag: str,
    create_barrier: AsyncBarrier,
    phase_barrier: AsyncBarrier,
    create_semaphore: asyncio.Semaphore | None,
) -> SandboxResult:
    result = SandboxResult(index=index)
    sandbox = None
    llm_real_state: LlmRealState | None = None
    background_handles: list[object] = []

    try:
        t0 = time.monotonic()
        try:
            if create_semaphore is None:
                sandbox = await asyncio.to_thread(
                    InspireSandbox.create,
                    template=template,
                    timeout=SANDBOX_TIMEOUT,
                    metadata={
                        "purpose": "diagnose_concurrency_sustained",
                        "run_tag": run_tag,
                        "sandbox_idx": str(index),
                    },
                )
            else:
                async with create_semaphore:
                    sandbox = await asyncio.to_thread(
                        InspireSandbox.create,
                        template=template,
                        timeout=SANDBOX_TIMEOUT,
                        metadata={
                            "purpose": "diagnose_concurrency_sustained",
                            "run_tag": run_tag,
                            "sandbox_idx": str(index),
                        },
                    )
        except Exception as create_exc:
            result.create_sec = time.monotonic() - t0
            result.error = f"create: {_format_exception(create_exc)}"
        else:
            result.create_sec = time.monotonic() - t0

        await create_barrier.wait()

        if sandbox is not None and "llm-real" in phases:
            llm_real_state, setup_timing = await _setup_llm_real(
                sandbox, index, cfg, num_turns
            )
            result.op_timings.append(setup_timing)
            if llm_real_state is not None:
                background_handles.append(llm_real_state.handle)

        await phase_barrier.wait()

        if sandbox is None:
            return result

        for turn in range(num_turns):
            if "commands" in phases:
                for cmd_idx, cmd in enumerate(TURN_COMMANDS):
                    cmd_result, timing = await _call_timed(
                        index,
                        "commands",
                        turn,
                        cmd_idx,
                        f"cmd[{cmd_idx}]",
                        sandbox.commands.run,
                        cmd,
                        timeout=COMMAND_TIMEOUT,
                        user="root",
                    )
                    if timing.error is None and cmd_result is not None:
                        timing.error = _extract_exit_error(cmd_result)
                    result.op_timings.append(timing)

            if "llm-real" in phases and llm_real_state is not None:
                turn_result, timing = await _call_timed(
                    index,
                    "llm-real-batch",
                    turn,
                    0,
                    f"prompt={cfg.prompt_bytes}B resp={cfg.response_bytes}B timeout={cfg.timeout}s",
                    _run_llm_real_turn_sync,
                    sandbox,
                    llm_real_state,
                    turn,
                    cfg,
                )
                if timing.error is None and turn_result is not None:
                    timing.error = _extract_exit_error(turn_result)
                result.op_timings.append(timing)
    except Exception as exc:
        result.error = _format_exception(exc)
    finally:
        for handle in background_handles:
            await _safe_kill_handle(handle)
        if sandbox is not None:
            t0 = time.monotonic()
            try:
                await asyncio.to_thread(sandbox.kill)
            except Exception:
                pass
            result.kill_sec = time.monotonic() - t0
    return result


def _print_phase_summary(
    phase: str,
    all_ops: list[OpTiming],
    num_turns: int,
) -> None:
    timings = [op for op in all_ops if op.phase == phase]
    if not timings:
        return

    ok_ops = [op for op in timings if not op.error]
    err_ops = [op for op in timings if op.error]
    print(
        f"\n  {PHASE_DISPLAY.get(phase, phase)}: total={len(timings)} "
        f"ok={len(ok_ops)} err={len(err_ops)}"
    )

    if ok_ops:
        lats = [op.latency for op in ok_ops]
        print(
            f"  Latency:  min={min(lats):.3f}s  med={statistics.median(lats):.3f}s  "
            f"p90={percentile(lats, 0.90):.3f}s  p99={percentile(lats, 0.99):.3f}s  "
            f"max={max(lats):.3f}s"
        )
        if any(op.turn >= 0 for op in ok_ops):
            print("  Per-turn median latency:")
            for turn in range(num_turns):
                turn_lats = [op.latency for op in ok_ops if op.turn == turn]
                if turn_lats:
                    print(
                        f"    turn {turn}: med={statistics.median(turn_lats):.3f}s  "
                        f"max={max(turn_lats):.3f}s  n={len(turn_lats)}"
                    )

    if err_ops:
        print("  Errors (first 5):")
        for op in err_ops[:5]:
            turn_text = f"turn={op.turn} " if op.turn >= 0 else ""
            print(f"    [sbx={op.sandbox_idx} {turn_text}{op.label}] {op.error}")


async def run_test(
    concurrency: int,
    template: str,
    num_turns: int,
    phases: list[str],
    pool_size: int | None,
    create_batch_size: int | None,
    cfg: HarnessConfig,
    run_tag: str,
) -> None:
    loop = asyncio.get_event_loop()
    old_executor = loop._default_executor

    if pool_size is not None:
        executor = ThreadPoolExecutor(max_workers=pool_size)
        loop.set_default_executor(executor)
    else:
        executor = None

    pool_label = (
        f"{pool_size}" if pool_size else f"DEFAULT({min(32, (os.cpu_count() or 1) + 4)})"
    )
    print(
        f"Config: concurrency={concurrency}  turns={num_turns}  "
        f"phases={','.join(phases)}  pool={pool_label}  "
        f"create_batch={create_batch_size or 'unlimited'}"
    )
    print()

    create_barrier = AsyncBarrier(concurrency)
    phase_barrier = AsyncBarrier(concurrency)
    create_semaphore = (
        asyncio.Semaphore(create_batch_size)
        if create_batch_size and create_batch_size > 0
        else None
    )
    wall_start = time.monotonic()
    tasks = [
        asyncio.create_task(
            run_sandbox_workload(
                i,
                template,
                num_turns,
                phases,
                cfg,
                run_tag,
                create_barrier,
                phase_barrier,
                create_semaphore,
            )
        )
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    wall_sec = time.monotonic() - wall_start

    if executor is not None:
        loop._default_executor = old_executor
        executor.shutdown(wait=False)

    ok = [r for r in results if not r.error]
    errs = [r for r in results if r.error]
    all_ops = [op for r in ok for op in r.op_timings]

    print(f"{'=' * 70}")
    print(
        f"RESULTS: concurrency={concurrency}  pool={pool_label}  "
        f"phases={','.join(phases)}  create_batch={create_batch_size or 'unlimited'}  "
        f"wall={wall_sec:.1f}s"
    )
    print(f"{'=' * 70}")
    print(f"  Sandboxes: ok={len(ok)} errors={len(errs)}")

    if ok:
        creates = [r.create_sec for r in ok]
        kills = [r.kill_sec for r in ok]
        print(
            f"  Create:  min={min(creates):.2f}s  med={statistics.median(creates):.2f}s  "
            f"max={max(creates):.2f}s"
        )
        print(
            f"  Kill:    min={min(kills):.2f}s  med={statistics.median(kills):.2f}s  "
            f"max={max(kills):.2f}s"
        )

    for phase in ("commands", "llm-real-setup", "llm-real-batch"):
        _print_phase_summary(phase, all_ops, num_turns)

    if errs:
        print("\n  Sandbox errors:")
        for r in errs[:5]:
            print(f"    [{r.index}] {r.error}")
    print()


async def main(args: argparse.Namespace) -> None:
    phases = parse_phases(args.phases)
    cfg = HarnessConfig(
        prompt_bytes=args.prompt_bytes,
        response_bytes=args.response_bytes,
        poll_interval=args.poll_interval,
        tail_chunk_bytes=args.tail_chunk_bytes,
        ready_retries=args.ready_retries,
        timeout=args.timeout,
    )

    print("High-fidelity sandbox communication diagnosis")
    print(f"  template       : {args.template}")
    print(f"  turns          : {args.turns}")
    print(f"  phases         : {','.join(phases)}")
    run_tag = args.run_tag or f"diag-{int(time.time())}"
    print(f"  run_tag        : {run_tag}")
    if "llm-real" in phases:
        print(
            f"  llm-real       : prompt={cfg.prompt_bytes}B resp={cfg.response_bytes}B "
            f"poll={cfg.poll_interval}s timeout={cfg.timeout}s"
        )
    print()

    concurrency_levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    for c in concurrency_levels:
        pool = args.pool if args.pool else None
        create_batch = args.create_batch_size if args.create_batch_size else None
        await run_test(c, args.template, args.turns, phases, pool, create_batch, cfg, run_tag)
        if c != concurrency_levels[-1]:
            print("--- cooling down 5s ---\n")
            await asyncio.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="Inspire template alias. Default is a real SWE template with python available.",
    )
    parser.add_argument(
        "--concurrency",
        default="8,16",
        help="Comma-separated concurrency levels, e.g. 8,16,32",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Sequential anti-call turns per sandbox",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=0,
        help="Thread pool size; 0 uses Python default",
    )
    parser.add_argument(
        "--create-batch-size",
        type=int,
        default=0,
        help="Limit concurrent sandbox creation; 0 creates all at once",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Metadata tag attached to created sandboxes for debugging/cleanup",
    )
    parser.add_argument(
        "--phases",
        default="llm-real",
        help="Comma-separated phases. Useful values: llm-real, commands",
    )
    parser.add_argument(
        "--prompt-bytes",
        type=int,
        default=2 * 1024 * 1024,
        help="Approximate prompt/context bytes in each simulated request",
    )
    parser.add_argument(
        "--response-bytes",
        type=int,
        default=64 * 1024,
        help="Approximate assistant response bytes in each simulated response payload",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Polling interval for helper and simulator",
    )
    parser.add_argument(
        "--tail-chunk-bytes",
        type=int,
        default=65536,
        help="Tail chunk size used by the real helper when scanning from EOF",
    )
    parser.add_argument(
        "--ready-retries",
        type=int,
        default=10,
        help="Readiness probes before setup is considered failed",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Per-turn command/files.write timeout in seconds",
    )
    asyncio.run(main(parser.parse_args()))
