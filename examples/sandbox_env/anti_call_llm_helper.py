import json
import os
import sys
import time
import traceback

REQUEST_START_MARKER = "LLM_REQUEST_START"
REQUEST_END_MARKER = "LLM_REQUEST_END"
RESPONSE_START_MARKER = "LLM_RESPONSE_START"
RESPONSE_END_MARKER = "LLM_RESPONSE_END"
SESSION_END_MARKER = "SESSION_END"

index = int(sys.argv[1])
log_file = sys.argv[2]
poll_interval = float(sys.argv[3])
tail_chunk_bytes = max(4096, int(sys.argv[4]))
payload = None if len(sys.argv) < 6 else sys.argv[5]


def get_file_size():
    try:
        return os.path.getsize(log_file)
    except OSError:
        return 0


def read_last_line_with_marker(marker):
    if not os.path.exists(log_file):
        return None
    with open(log_file, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        if pos <= 0:
            return None
        buf = b""
        while pos > 0:
            step = min(tail_chunk_bytes, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            lines = buf.splitlines(keepends=True)
            if pos > 0:
                buf = lines[0]
                lines = lines[1:]
            else:
                buf = b""
            for line in reversed(lines):
                text = line.decode("utf-8", errors="replace")
                if marker in text:
                    return text
        if buf:
            text = buf.decode("utf-8", errors="replace")
            if marker in text:
                return text
    return None


def parse_request_line(line):
    if SESSION_END_MARKER in line:
        return SESSION_END_MARKER, {}
    meta_json = line.split(REQUEST_END_MARKER, 1)[1]
    request_json = line.split(REQUEST_END_MARKER, 1)[0].split(REQUEST_START_MARKER, 1)[1]
    return request_json, json.loads(meta_json)


def parse_response_line(line):
    meta_json = line.split(RESPONSE_END_MARKER, 1)[1]
    response_json = line.split(RESPONSE_END_MARKER, 1)[0].split(RESPONSE_START_MARKER, 1)[1]
    return response_json, json.loads(meta_json)


def append_response(last_response, response_index):
    meta = {"timestamp": int(time.time() * 1000), "index": response_index}
    content = (
        f"{RESPONSE_START_MARKER}{last_response}"
        f"{RESPONSE_END_MARKER}{json.dumps(meta, ensure_ascii=False)}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(content)


def wait_for_request(request_index, start_pos):
    pos = max(0, start_pos)
    while True:
        if os.path.exists(log_file):
            with open(log_file, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                lines = f.readlines()
                pos = f.tell()
            for line in lines:
                if SESSION_END_MARKER in line:
                    return SESSION_END_MARKER
                if REQUEST_START_MARKER not in line or REQUEST_END_MARKER not in line:
                    continue
                request_json, meta = parse_request_line(line)
                if meta.get("index") == request_index:
                    return request_json
        time.sleep(poll_interval)


def tail_preview(limit=10):
    if not os.path.exists(log_file):
        return ""
    with open(log_file, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0 and buf.count(b"\n") <= limit:
            step = min(tail_chunk_bytes, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    lines = buf.decode("utf-8", errors="replace").splitlines(True)
    return "".join(lines[-limit:])


try:
    if index == 0:
        if payload is not None:
            raise ValueError("last_response must be None when index is 0")
        result = wait_for_request(1, 0)
    else:
        if payload is None:
            raise ValueError("last_response must not be None when index is greater than 0")
        start_pos = get_file_size()
        last_response_line = read_last_line_with_marker(RESPONSE_START_MARKER)
        if last_response_line is None:
            append_response(payload, index)
        else:
            _response_json, meta = parse_response_line(last_response_line)
            last_response_index = int(meta.get("index"))
            if index < last_response_index:
                raise ValueError(
                    f"index {index} must not be smaller than last_response_index {last_response_index}"
                )
            if index > last_response_index:
                append_response(payload, index)

        # Try to satisfy the read from a request that's already on disk; otherwise
        # tail-poll the log starting from the response we just appended.
        result = None
        last_request_line = read_last_line_with_marker(REQUEST_START_MARKER)
        if last_request_line is not None:
            request_json, meta = parse_request_line(last_request_line)
            if meta.get("index") == index + 1:
                result = request_json
        if result is None:
            result = wait_for_request(index + 1, start_pos)
    if result is None:
        raise RuntimeError(f"anti_call_llm returned None at index={index}")
    if not isinstance(result, str):
        raise RuntimeError(
            f"anti_call_llm returned non-str {type(result).__name__} at index={index}: {result!r}"
        )
    if not result.strip():
        raise RuntimeError(
            f"anti_call_llm returned empty response at index={index}; "
            f"payload_present={payload is not None}; "
            f"payload_len={0 if payload is None else len(payload)}"
        )
    sys.stdout.write(result)
except Exception as exc:
    sys.stderr.write(
        f"[anti_call_llm_helper] index={index} "
        f"payload_present={payload is not None} "
        f"payload_len={0 if payload is None else len(payload)} "
        f"log_file={log_file} exists={os.path.exists(log_file)}\n"
    )
    sys.stderr.write(f"[anti_call_llm_helper] error={exc!r}\n")
    preview = tail_preview()
    if preview:
        sys.stderr.write("[anti_call_llm_helper] log_tail:\n")
        sys.stderr.write(preview)
        if not preview.endswith("\n"):
            sys.stderr.write("\n")
    traceback.print_exc(file=sys.stderr)
    raise
