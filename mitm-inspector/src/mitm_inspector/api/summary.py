"""Content-derived flow summaries for the browser grid."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from mitm_inspector.api.bodies import body_content_encoding, decoded_body_bytes
from mitm_inspector.json_boundary import PlainJsonObject
from mitm_inspector.protocol import MAX_U64

_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_WHITESPACE = re.compile(r"\s+")


def flow_summary(metadata: Mapping[str, object]) -> PlainJsonObject:
    """Derive the additive protocol-v1 summary for one projected flow."""

    host = metadata.get("host")
    path = metadata.get("path")
    if not isinstance(host, str) or host.casefold() != "api.anthropic.com":
        return {"kind": "generic"}
    if path == "/v1/messages":
        summary: PlainJsonObject = {"kind": "anthropic_messages"}
        request = _json_body(metadata, "request")
        if request is not None:
            _request_fields(summary, request)
        response_objects = _response_objects(metadata)
        for event in response_objects:
            _response_fields(summary, event)
        return summary
    if path == "/v1/messages/count_tokens":
        summary = {"kind": "anthropic_count_tokens"}
        request = _json_body(metadata, "request")
        if request is not None:
            _request_fields(summary, request)
        for response_object in _response_objects(metadata):
            value = _decimal(response_object.get("input_tokens"))
            if value is not None:
                summary["count_tokens_result"] = value
        return summary
    return {"kind": "generic"}


def _json_body(metadata: Mapping[str, object], side: str) -> dict[str, object] | None:
    descriptor = metadata.get(f"{side}_body")
    encoding = body_content_encoding(metadata, side)
    body, _decoded = decoded_body_bytes(descriptor, encoding)
    if body is None:
        return None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _request_fields(summary: PlainJsonObject, request: Mapping[str, object]) -> None:
    model = request.get("model")
    if isinstance(model, str):
        summary["model"] = model
    messages = request.get("messages")
    if isinstance(messages, list):
        summary["message_count"] = str(len(messages))
        summary["preview"] = _preview(messages)
    stream = request.get("stream")
    if isinstance(stream, bool):
        summary["stream"] = stream


def _preview(messages: list[object]) -> PlainJsonObject:
    user_index: int | None = None
    user_message: Mapping[str, object] | None = None
    for index in range(len(messages) - 1, -1, -1):
        candidate = messages[index]
        if isinstance(candidate, Mapping) and candidate.get("role") == "user":
            user_index = index
            user_message = candidate
            break
    if user_message is None or user_index is None:
        return {"source": "none"}
    content = user_message.get("content")
    if isinstance(content, str):
        return _text_preview(content)
    if not isinstance(content, list):
        return {"source": "none"}
    for block in reversed(content):
        if isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                return _text_preview(text)
    tool_results = [
        block
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "tool_result"
    ]
    if not content or len(tool_results) != len(content):
        return {"source": "none"}
    preview: PlainJsonObject = {"source": "tool_result"}
    tool_ids = [
        value
        for block in reversed(tool_results)
        if isinstance((value := block.get("tool_use_id")), str)
    ]
    tool_name = _preceding_tool_name(messages, user_index, tool_ids)
    if tool_name is not None:
        preview["tool_name"] = tool_name
    return preview


def _text_preview(value: str) -> PlainJsonObject:
    stripped = _SYSTEM_REMINDER.sub(" ", value)
    collapsed = _WHITESPACE.sub(" ", stripped).strip()
    if not collapsed:
        return {"source": "none"}
    return {"source": "user_text", "text": collapsed[:140]}


def _preceding_tool_name(
    messages: list[object], user_index: int, tool_ids: Sequence[str]
) -> str | None:
    if not tool_ids:
        return None
    for index in range(user_index - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return None
        for tool_id in tool_ids:
            for block in reversed(content):
                if (
                    isinstance(block, Mapping)
                    and block.get("type") == "tool_use"
                    and block.get("id") == tool_id
                ):
                    name = block.get("name")
                    return name if isinstance(name, str) else None
        return None
    return None


def _response_objects(metadata: Mapping[str, object]) -> list[dict[str, object]]:
    descriptor = metadata.get("response_body")
    encoding = body_content_encoding(metadata, "response")
    body, _decoded = decoded_body_bytes(descriptor, encoding)
    if body is None:
        return []
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return [value]
    result: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            result.append(event)
    return result


def _response_fields(summary: PlainJsonObject, event: Mapping[str, object]) -> None:
    event_type = event.get("type")
    message = event.get("message") if event_type == "message_start" else event
    if not isinstance(message, Mapping):
        message = event
    stop_reason = message.get("stop_reason")
    delta = event.get("delta")
    if isinstance(delta, Mapping) and isinstance(delta.get("stop_reason"), str):
        stop_reason = delta["stop_reason"]
    if isinstance(stop_reason, str):
        summary["stop_reason"] = stop_reason
    usage = message.get("usage")
    if not isinstance(usage, Mapping):
        usage = event.get("usage")
    if not isinstance(usage, Mapping):
        return
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens"),
    ):
        value = _decimal(usage.get(source))
        if value is not None:
            summary[target] = value
    details = usage.get("output_tokens_details")
    if isinstance(details, Mapping):
        thinking = _decimal(details.get("thinking_tokens"))
        if thinking is not None:
            summary["thinking_tokens"] = thinking


def _decimal(value: object) -> str | None:
    if type(value) is int and 0 <= value <= MAX_U64:
        return str(value)
    return None


__all__ = ["flow_summary"]
