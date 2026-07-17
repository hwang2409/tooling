"""Documented mitmproxy-hook adapter for sanitized protocol-v1 capture."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from mitmproxy import http

from mitm_inspector.capture.redaction import sanitize_header, sanitize_path
from mitm_inspector.capture.sink import BoundedMessageSink
from mitm_inspector.protocol import ParsedMessage, ParsedMessageResult, parse_message

MessageEmitter = Callable[[ParsedMessage], None]
Clock = Callable[[], str]

MAX_BODY_PREFIX_BYTES = 1 * 1024 * 1024
MAX_IN_MEMORY_BYTES = 128 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class _BodyCapture:
    content_type: str | None = None
    total_bytes: int = 0
    prefix: bytearray = field(default_factory=bytearray)
    chunk_index: int = 0
    observed: bool = False
    lifecycle_emitted: bool = False
    ended: bool = False


@dataclass
class _FlowCapture:
    flow_id: str
    identity: dict[str, str] = field(default_factory=dict)
    request_headers: list[dict[str, str]] = field(default_factory=list)
    response_headers: list[dict[str, str]] | None = None
    request: _BodyCapture = field(default_factory=_BodyCapture)
    response: _BodyCapture = field(default_factory=_BodyCapture)
    lifecycle_states: set[str] = field(default_factory=set)
    completed: bool = False
    tombstone: bool = False


class CaptureAddon:
    """Capture HTTP flows using only the public mitmproxy 12.2 hook surface.

    ``requestheaders`` and ``responseheaders`` install the documented public
    ``Message.stream`` callback.  The callback observes bounded copies of body
    chunks and returns the original bytes unchanged, so forwarding never waits
    for or depends on the inspector sink.
    """

    PUBLIC_HOOKS = frozenset(
        {
            "requestheaders",
            "request",
            "responseheaders",
            "response",
            "error",
        }
    )

    def __init__(
        self,
        emit: MessageEmitter | None = None,
        *,
        sink: BoundedMessageSink | None = None,
        source_id: str = "mitm-inspector",
        max_body_prefix_bytes: int = MAX_BODY_PREFIX_BYTES,
        clock: Clock = _utc_now,
        max_pending_messages: int = 4_096,
    ) -> None:
        if emit is not None and sink is not None:
            raise ValueError("pass either emit or sink, not both")
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if max_body_prefix_bytes < 0:
            raise ValueError("max_body_prefix_bytes must not be negative")
        self.source_id = source_id
        self.max_body_prefix_bytes = max_body_prefix_bytes
        self._clock = clock
        self._emit_callback = emit
        self.sink = sink if sink is not None else BoundedMessageSink(max_pending_messages)
        self._flows: dict[str, _FlowCapture] = {}
        self._completed_ids: deque[str] = deque(maxlen=2_000)
        self._sequence = 0
        self._source_announced = False

    def requestheaders(self, flow: http.HTTPFlow) -> None:
        """Observe request start/headers and install the public body callback."""

        state = self._ensure_flow(flow)
        if state.tombstone:
            return
        self._announce_source()
        self._lifecycle(state, "request_started")
        if "request_headers" not in state.lifecycle_states:
            state.request_headers = _headers(flow.request.headers)
            state.request.content_type = _content_type(flow.request.headers)
            self._set_identity(state, flow)
            self._install_stream(flow.request, state, "request")
            self._lifecycle(state, "request_headers")
            self._metadata(state)

    def request(self, flow: http.HTTPFlow) -> None:
        """Observe the terminal request body and request end."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        self._finish_body(state, "request", flow.request.raw_content)
        if "request_end" not in state.lifecycle_states:
            self._lifecycle(state, "request_end")
            self._metadata(state)
        self._discard_if_complete(state)

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        """Observe response start/headers and install the public body callback."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        self._lifecycle(state, "response_started")
        if "response_headers" not in state.lifecycle_states:
            state.response_headers = _headers(flow.response.headers) if flow.response else []
            if flow.response:
                state.response.content_type = _content_type(flow.response.headers)
                self._install_stream(flow.response, state, "response")
            self._lifecycle(state, "response_headers")
            self._metadata(state)

    def response(self, flow: http.HTTPFlow) -> None:
        """Observe the terminal response body and complete the flow."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        response_content = flow.response.raw_content if flow.response else None
        self._finish_body(state, "response", response_content)
        if "response_end" not in state.lifecycle_states:
            self._lifecycle(state, "response_end")
            self._metadata(state)
        self._complete(state, wait_for_request=True)

    def error(self, flow: http.HTTPFlow) -> None:
        """Observe an HTTP error and complete the flow without exposing its text."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        if flow.request.raw_content is not None and "request_end" not in state.lifecycle_states:
            self._finish_body(state, "request", flow.request.raw_content)
            self._lifecycle(state, "request_end")
            self._metadata(state)
        self._lifecycle(state, "error")
        self._complete(state, wait_for_request=False)

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Drain messages when the addon owns its default bounded sink."""

        if self._emit_callback is not None:
            raise RuntimeError("drain is unavailable when using a callback emitter")
        return self.sink.drain(limit)

    def _announce_source(self) -> None:
        if self._source_announced:
            return
        self._source_announced = True
        self._send(
            {
                "protocol_version": "1",
                "type": "source.hello",
                "source_id": self.source_id,
                "occurred_at": self._clock(),
                "capabilities": {"body_chunks": True, "redaction": "headers-and-query"},
                "limits": {
                    "max_body_prefix_bytes": str(self.max_body_prefix_bytes),
                    "max_in_memory_bytes": str(MAX_IN_MEMORY_BYTES),
                },
            }
        )

    def _ensure_flow(self, flow: http.HTTPFlow) -> _FlowCapture:
        flow_id = str(flow.id)
        if not flow_id:
            raise ValueError("mitmproxy flow id must be non-empty")
        state = self._flows.get(flow_id)
        if state is None:
            if flow_id in self._completed_ids:
                return _FlowCapture(flow_id=flow_id, completed=True, tombstone=True)
            state = _FlowCapture(flow_id=flow_id)
            self._flows[flow_id] = state
        return state

    def _ensure_request(self, flow: http.HTTPFlow) -> _FlowCapture:
        state = self._ensure_flow(flow)
        self._announce_source()
        if "request_headers" not in state.lifecycle_states:
            self.requestheaders(flow)
        return state

    def _install_stream(
        self,
        message: http.Request | http.Response,
        state: _FlowCapture,
        side: str,
    ) -> None:
        def observe(chunk: bytes) -> bytes:
            self._observe_chunk(state, side, chunk)
            return chunk

        message.stream = observe

    def _observe_chunk(self, state: _FlowCapture, side: str, chunk: bytes) -> None:
        if side not in {"request", "response"}:
            raise AssertionError("invalid body side")
        body = state.request if side == "request" else state.response
        copied = bytes(chunk)
        offset = body.total_bytes
        body.total_bytes += len(copied)
        body.observed = True
        remaining = max(0, self.max_body_prefix_bytes - len(body.prefix))
        captured = copied[:remaining]
        body.prefix.extend(captured)
        if not body.lifecycle_emitted:
            body.lifecycle_emitted = True
            self._lifecycle(state, f"{side}_body")
        if captured:
            self._send(
                {
                    "protocol_version": "1",
                    "type": "body.chunk",
                    "flow_id": state.flow_id,
                    "body_side": side,
                    "chunk_index": str(body.chunk_index),
                    "offset_bytes": str(offset),
                    "data_base64": base64.b64encode(captured).decode("ascii"),
                }
            )
        body.chunk_index += 1

    def _finish_body(self, state: _FlowCapture, side: str, raw_content: bytes | None) -> None:
        body = state.request if side == "request" else state.response
        if body.ended:
            return
        if not body.observed and raw_content is not None:
            self._observe_chunk(state, side, raw_content)
        if not body.lifecycle_emitted:
            body.lifecycle_emitted = True
            self._lifecycle(state, f"{side}_body")
        body.ended = True
        descriptor = _body_descriptor(body)
        self._send(
            {
                "protocol_version": "1",
                "type": "body.end",
                "flow_id": state.flow_id,
                "body_side": side,
                "total_bytes": str(body.total_bytes),
                "body": descriptor,
            }
        )

    def _metadata(self, state: _FlowCapture) -> None:
        if not state.request_headers:
            return
        metadata: dict[str, object] = {
            "flow_id": state.flow_id,
            "method": "GET",
            "scheme": "https",
            "host": "unknown",
            "port": "443",
            "path": "/",
            "request_headers": state.request_headers,
            "request_body": _body_descriptor(state.request),
        }
        # Identity fields are copied on the first hook.  No Flow or Message
        # object crosses out of this adapter.
        metadata.update(state.identity)
        if state.response_headers is not None:
            metadata["response_headers"] = state.response_headers
            metadata["response_body"] = _body_descriptor(state.response)
        self._send({"protocol_version": "1", "type": "flow.metadata", "metadata": metadata})

    def _set_identity(self, state: _FlowCapture, flow: http.HTTPFlow) -> None:
        request = flow.request
        identity = {
            "method": str(request.method),
            "scheme": str(request.scheme),
            "host": str(request.host),
            "port": str(request.port),
            "path": sanitize_path(str(request.path)) or "/",
        }
        state.identity = identity

    def _lifecycle(self, state: _FlowCapture, lifecycle_state: str) -> None:
        if lifecycle_state in state.lifecycle_states:
            return
        state.lifecycle_states.add(lifecycle_state)
        sequence = self._sequence
        self._sequence += 1
        self._send(
            {
                "protocol_version": "1",
                "type": "flow.lifecycle",
                "source_id": self.source_id,
                "flow_id": state.flow_id,
                "event_id": f"{self.source_id}:{sequence}",
                "occurred_at": self._clock(),
                "sequence": str(sequence),
                "state": lifecycle_state,
            }
        )

    def _complete(self, state: _FlowCapture, *, wait_for_request: bool) -> None:
        if state.completed:
            return
        state.completed = True
        self._lifecycle(state, "flow_completed")
        if not wait_for_request or state.request.ended:
            self._discard_if_complete(state)

    def _discard_if_complete(self, state: _FlowCapture) -> None:
        if state.completed and state.request.ended:
            self._flows.pop(state.flow_id, None)
            self._completed_ids.append(state.flow_id)

    def _send(self, raw: dict[str, object]) -> None:
        parsed = parse_message(raw)
        if self._emit_callback is not None:
            self._emit_callback(parsed)
        else:
            self.sink.offer(parsed)


def _headers(headers: Mapping[str, str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    items = cast(Callable[..., Iterable[tuple[object, object]]], headers.items)
    try:
        header_items = items(multi=True)
    except TypeError:
        header_items = items()
    for name, value in header_items:
        sanitized_name, sanitized_value = sanitize_header(str(name), str(value))
        result.append({"name": sanitized_name, "value": sanitized_value})
    return result


def _content_type(headers: Mapping[str, str]) -> str | None:
    value = headers.get("content-type")
    return None if value is None else sanitize_header("content-type", str(value))[1]


def _body_descriptor(body: _BodyCapture) -> dict[str, str]:
    if not body.observed:
        return {"state": "missing"}
    if body.total_bytes == 0:
        descriptor: dict[str, str] = {"state": "empty", "size_bytes": "0"}
    else:
        descriptor = {
            "state": "captured" if len(body.prefix) == body.total_bytes else "truncated",
            "size_bytes": str(body.total_bytes),
            "encoding": "base64",
            "data": base64.b64encode(bytes(body.prefix)).decode("ascii"),
        }
        if descriptor["state"] == "truncated":
            descriptor["captured_bytes"] = str(len(body.prefix))
    if body.content_type is not None:
        descriptor["content_type"] = body.content_type
    return descriptor
