"""Stateless Responses API adapter for current Codex CLI agent rollouts.

Codex sends the full input history with store=false. Function calls, custom
tools and namespaces are translated to the served model's tool schema. The
shared turn pipeline still captures the original SGLang tokens/logprobs;
protocol text is never retokenized to manufacture training targets.
"""

from __future__ import annotations

import json
import secrets
import time

from aiohttp import web

from slime.agent.adapters.common import Reply, flatten_content, manager_finish_reason, tool_call_dict
from slime.agent.adapters.openai import OpenAIAdapter, _arguments_as_dict


def _tool_specs(tools, namespace=None):
    for tool in tools or []:
        kind = tool.get("type", "function")
        if kind == "namespace":
            yield from _tool_specs(tool.get("tools"), tool["name"])
        elif kind in {"function", "custom"}:
            name = tool["name"]
            yield (f"{namespace}.{name}" if namespace else name), tool, namespace
        else:
            raise web.HTTPBadRequest(text=f"Unsupported Responses tool type: {kind}")


def _translate_tools(tools):
    result = []
    for name, tool, _ in _tool_specs(tools):
        description = tool.get("description", "")
        if tool.get("type") == "custom":
            description += "\nPass the tool's raw text as the input string."
            if tool.get("format"):
                description += "\nInput format: " + json.dumps(tool["format"], ensure_ascii=False)
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            }
        else:
            parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return result or None


def _translate_input(items, instructions=None):
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list):
        raise web.HTTPBadRequest(text="input must be a string or list")
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    assistant = None
    call_order = []
    results = {}

    def flush():
        nonlocal assistant, call_order, results
        if assistant is not None:
            messages.append(assistant)
        # Parallel tool results may arrive in completion order. Templates use
        # positional correspondence, so restore call order before dropping IDs.
        for call_id in call_order:
            if call_id in results:
                messages.append({"role": "tool", "content": results.pop(call_id)})
        if results:
            raise web.HTTPBadRequest(text="Tool output has no matching call_id")
        assistant, call_order, results = None, [], {}

    for item in items:
        kind = item.get("type", "message")
        if kind == "reasoning":
            # Wire reasoning summaries are not the model's sampled reasoning.
            continue
        if kind in {"function_call_output", "custom_tool_call_output"}:
            call_id = item["call_id"]
            if call_id in results or call_id not in call_order:
                raise web.HTTPBadRequest(text="Duplicate or unknown tool output call_id")
            results[call_id] = flatten_content(item.get("output"))
            continue
        if results:
            flush()
        if kind in {"function_call", "custom_tool_call"}:
            if assistant is None:
                assistant = {"role": "assistant", "content": ""}
            name = item["name"]
            if item.get("namespace"):
                name = f"{item['namespace']}.{name}"
            arguments = (
                {"input": item.get("input", "")}
                if kind == "custom_tool_call"
                else _arguments_as_dict(item.get("arguments"))
            )
            assistant.setdefault("tool_calls", []).append(tool_call_dict(name, arguments))
            call_order.append(item["call_id"])
        elif kind == "message":
            role = item.get("role")
            content = flatten_content(item.get("content"))
            if role == "assistant":
                if assistant is None:
                    assistant = {"role": "assistant", "content": ""}
                assistant["content"] += content
            elif role in {"system", "developer", "user"}:
                flush()
                if role in {"system", "developer"}:
                    if not messages:
                        messages.append({"role": "system", "content": content})
                    elif len(messages) == 1 and messages[0]["role"] == "system":
                        # Codex sends multiple developer messages. Qwen3.8's
                        # template permits exactly one initial system message.
                        messages[0]["content"] += "\n\n" + content
                    else:
                        messages.append({"role": "user", "content": f"<{role}_message>\n{content}\n</{role}_message>"})
                else:
                    messages.append({"role": role, "content": content})
            else:
                raise web.HTTPBadRequest(text=f"Unsupported Responses message role: {role}")
        else:
            raise web.HTTPBadRequest(text=f"Unsupported Responses input type: {kind}")
    flush()
    return messages


def _output_items(parsed, tools):
    specs = {name: (tool, ns) for name, tool, ns in _tool_specs(tools)}
    output = []
    if parsed.text or not parsed.tool_uses:
        output.append(
            {
                "type": "message",
                "id": f"msg_{secrets.token_hex(12)}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": parsed.text or "", "annotations": []}],
            }
        )
    for call in parsed.tool_uses:
        name = call["name"]
        tool, namespace = specs.get(name, ({"name": name, "type": "function"}, None))
        item = {"id": f"fc_{secrets.token_hex(12)}", "call_id": f"call_{secrets.token_hex(12)}", "name": tool["name"]}
        if namespace:
            item["namespace"] = namespace
        if tool.get("type") == "custom":
            item.update(type="custom_tool_call", input=call.get("input", {}).get("input", ""))
        else:
            item.update(
                type="function_call",
                status="completed",
                arguments=json.dumps(call.get("input") or {}, ensure_ascii=False),
            )
        output.append(item)
    return output


class ResponsesAdapter(OpenAIAdapter):
    """Codex's text/tool subset of /v1/responses, with JSON or SSE replies.

    Stored responses, previous_response_id, hosted tools and multimodal output
    are deliberately unsupported. A compaction request must supply its new
    history; the trajectory manager then captures it as a separate branch.
    """

    log_prefix = "responses_adapter"

    def _register_routes(self, app):
        app.router.add_post("/v1/responses", self._run_turn)

    def _translate(self, body):
        if body.get("previous_response_id") or body.get("store") is True:
            raise web.HTTPBadRequest(text="Send full input history with store=false; stored responses are unsupported")
        return _translate_input(body.get("input", []), body.get("instructions")), _translate_tools(body.get("tools"))

    def _build_reply(self, parsed, raw_finish, translated, tools_schema):
        message = {"role": "assistant", "content": parsed.text or ""}
        if parsed.tool_uses:
            message["tool_calls"] = [tool_call_dict(t["name"], t.get("input")) for t in parsed.tool_uses]
        return Reply(message, manager_finish_reason(parsed.tool_uses, raw_finish), (parsed, raw_finish))

    async def _respond(self, request, body, reply, in_tok, out_tok, stream):
        parsed, finish = reply.wire
        output = _output_items(parsed, body.get("tools"))
        response = {
            "id": f"resp_{secrets.token_hex(12)}",
            "object": "response",
            "created_at": int(time.time()),
            "model": body.get("model", "slime-actor"),
            "status": "incomplete" if finish == "length" else "completed",
            "output": output,
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if finish == "length" else None,
            "usage": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        if not stream:
            return web.json_response(response)
        result = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await result.prepare(request)
        seq = 0

        async def emit(kind, **data):
            nonlocal seq
            payload = {"type": kind, "sequence_number": seq, **data}
            seq += 1
            await result.write(f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())

        await emit("response.created", response={**response, "status": "in_progress", "output": [], "usage": None})
        for index, item in enumerate(output):
            common = {"item_id": item["id"], "output_index": index}
            field = "arguments" if item["type"] == "function_call" else "input"
            initial = {**item, "status": "in_progress"}
            initial["content" if item["type"] == "message" else field] = [] if item["type"] == "message" else ""
            await emit("response.output_item.added", output_index=index, item=initial)
            if item["type"] == "message":
                part = item["content"][0]
                await emit("response.content_part.added", **common, content_index=0, part={**part, "text": ""})
                await emit("response.output_text.delta", **common, content_index=0, delta=part["text"])
                await emit("response.output_text.done", **common, content_index=0, text=part["text"])
                await emit("response.content_part.done", **common, content_index=0, part=part)
            else:
                prefix = (
                    "response.function_call_arguments" if field == "arguments" else "response.custom_tool_call_input"
                )
                await emit(f"{prefix}.delta", **common, delta=item[field])
                await emit(f"{prefix}.done", **common, **{field: item[field]})
            await emit("response.output_item.done", output_index=index, item=item)
        await emit(f"response.{response['status']}", response=response)
        await result.write_eof()
        return result
