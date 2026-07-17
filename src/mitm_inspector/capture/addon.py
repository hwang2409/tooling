"""Documented mitmproxy-hook adapter for sanitized protocol-v1 capture."""

from __future__ import annotations

import base64
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from mitmproxy import http

from mitm_inspector.capture.config import (
    DEFAULT_MAX_BODY_PREFIX_BYTES,
    DEFAULT_MAX_IN_MEMORY_BYTES,
    DEFAULT_MAX_PENDING_MESSAGES,
    DEFAULT_SOURCE_ID,
    CaptureConfig,
)
from mitm_inspector.capture.redaction import sanitize_header, sanitize_path
from mitm_inspector.capture.sink import BoundedMessageSink
from mitm_inspector.protocol import MAX_U64, ParsedMessage, ParsedMessageResult, parse_message

MessageEmitter = Callable[[ParsedMessage], None]
Clock = Callable[[], str]

MAX_BODY_PREFIX_BYTES = DEFAULT_MAX_BODY_PREFIX_BYTES
MAX_IN_MEMORY_BYTES = DEFAULT_MAX_IN_MEMORY_BYTES
MAX_ACTIVE_FLOWS = 2_000
MAX_ACTIVE_AGE_SECONDS = 30 * 60


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
    stream_enabled: bool = True


@dataclass
class _FlowCapture:
    flow_id: str
    created_at: float
    identity: dict[str, str] = field(default_factory=dict)
    request_headers: list[dict[str, str]] = field(default_factory=list)
    request_headers_captured: bool = False
    response_headers: list[dict[str, str]] | None = None
    response_headers_captured: bool = False
    request: _BodyCapture = field(default_factory=_BodyCapture)
    response: _BodyCapture = field(default_factory=_BodyCapture)
    lifecycle_states: set[str] = field(default_factory=set)
    completed: bool = False
    completion_pending: bool = False
    terminal_observed: bool = False
    tombstone: bool = False
    discarded: bool = False
    retained_weight: int = 0


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
            "load",
        }
    )

    @classmethod
    def from_environment(cls) -> CaptureAddon:
        """Build a configured addon without opening the configured endpoint."""

        return cls(config=CaptureConfig.from_environment())

    def __init__(
        self,
        emit: MessageEmitter | None = None,
        *,
        sink: BoundedMessageSink | None = None,
        config: CaptureConfig | None = None,
        source_id: str = DEFAULT_SOURCE_ID,
        max_body_prefix_bytes: int = DEFAULT_MAX_BODY_PREFIX_BYTES,
        clock: Clock = _utc_now,
        max_pending_messages: int = DEFAULT_MAX_PENDING_MESSAGES,
        max_active_flows: int = MAX_ACTIVE_FLOWS,
        max_active_age_seconds: float = MAX_ACTIVE_AGE_SECONDS,
        active_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if emit is not None and sink is not None:
            raise ValueError("pass either emit or sink, not both")
        if config is None:
            config = CaptureConfig(
                source_id=source_id,
                max_body_prefix_bytes=max_body_prefix_bytes,
                max_pending_messages=max_pending_messages,
            )
        self.config = config
        self.source_id = config.source_id
        self.capture_socket = config.capture_socket
        self.max_body_prefix_bytes = config.max_body_prefix_bytes
        self.max_in_memory_bytes = config.max_in_memory_bytes
        if max_active_flows < 1:
            raise ValueError("max_active_flows must be positive")
        if max_active_age_seconds < 0:
            raise ValueError("max_active_age_seconds must not be negative")
        self.max_active_flows = max_active_flows
        self.max_active_age_seconds = max_active_age_seconds
        self._active_clock = active_clock
        self._clock = clock
        self._emit_callback = emit
        self._sink_injected = sink is not None
        self.sink = (
            sink
            if sink is not None
            else BoundedMessageSink(
                config.max_pending_messages,
                config.max_in_memory_bytes,
                max_memory_bytes=config.max_in_memory_bytes,
            )
        )
        self._flows: dict[str, _FlowCapture] = {}
        self._completed_ids: deque[str] = deque(maxlen=2_000)
        self._captured_prefix_bytes = 0
        self._active_metadata_bytes = 0
        self._capture_evicted_flows = 0
        self._sequence = 0
        self._source_announced = False

    def load(self, _loader: object) -> None:
        """Parse B1's environment at addon load without opening IPC."""

        self.configure(CaptureConfig.from_environment())

    def configure(self, config: CaptureConfig) -> None:
        """Apply parsed configuration before the first captured flow."""

        if self._source_announced or self._flows:
            raise RuntimeError("capture configuration cannot change after capture starts")
        self.config = config
        self.source_id = config.source_id
        self.capture_socket = config.capture_socket
        self.max_body_prefix_bytes = config.max_body_prefix_bytes
        self.max_in_memory_bytes = config.max_in_memory_bytes
        if not self._sink_injected:
            self.sink = BoundedMessageSink(
                config.max_pending_messages,
                config.max_in_memory_bytes,
                max_memory_bytes=config.max_in_memory_bytes,
            )

    def requestheaders(self, flow: http.HTTPFlow) -> None:
        """Observe request start/headers and install the public body callback."""

        state = self._ensure_flow(flow)
        if state.tombstone:
            return
        self._announce_source()
        self._lifecycle(state, "request_started")
        if "request_headers" not in state.lifecycle_states:
            state.request_headers = _headers(flow.request.headers)
            state.request_headers_captured = True
            state.request.content_type = _content_type(flow.request.headers)
            self._set_identity(state, flow)
            self._install_stream(flow.request, state, "request")
            self._lifecycle(state, "request_headers")
            self._metadata(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def request(self, flow: http.HTTPFlow) -> None:
        """Observe the terminal request body and request end."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        self._finish_body(state, "request", flow.request.raw_content)
        if "request_end" not in state.lifecycle_states:
            self._lifecycle(state, "request_end")
            self._metadata(state)
        self._finalize(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        """Observe response start/headers and install the public body callback."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        self._lifecycle(state, "response_started")
        if "response_headers" not in state.lifecycle_states:
            state.response_headers = _headers(flow.response.headers) if flow.response else []
            state.response_headers_captured = True
            if flow.response:
                state.response.content_type = _content_type(flow.response.headers)
                self._install_stream(flow.response, state, "response")
            self._lifecycle(state, "response_headers")
            self._metadata(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

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
        state.terminal_observed = True
        self._finalize(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def error(self, flow: http.HTTPFlow) -> None:
        """Observe an HTTP error and complete the flow without exposing its text."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        if "request_end" not in state.lifecycle_states:
            self._finish_body(state, "request", flow.request.raw_content)
            self._lifecycle(state, "request_end")
            self._metadata(state)
        if state.response_headers_captured and not state.response.ended:
            response_content = flow.response.raw_content if flow.response else None
            self._finish_body(state, "response", response_content)
            if "response_end" not in state.lifecycle_states:
                self._lifecycle(state, "response_end")
                self._metadata(state)
        self._lifecycle(state, "error")
        state.terminal_observed = True
        self._finalize(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Drain messages when the addon owns its default bounded sink."""

        self._purge_active()
        messages = self.sink.drain(limit)
        if self._emit_callback is not None:
            for message in messages:
                self._emit_callback(message)
        return messages

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
                    "max_in_memory_bytes": str(self.max_in_memory_bytes),
                },
            }
        )

    def _ensure_flow(self, flow: http.HTTPFlow) -> _FlowCapture:
        self._purge_active()
        flow_id = _safe_text(flow.id, label="flow id")
        if not flow_id:
            raise ValueError("mitmproxy flow id must be non-empty")
        state = self._flows.get(flow_id)
        if state is None:
            if flow_id in self._completed_ids:
                return _FlowCapture(
                    flow_id=flow_id,
                    created_at=self._active_clock(),
                    completed=True,
                    tombstone=True,
                )
            state = _FlowCapture(flow_id=flow_id, created_at=self._active_clock())
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
            body = state.request if side == "request" else state.response
            if state.completed or state.discarded or not body.stream_enabled:
                return chunk
            self._observe_chunk(state, side, chunk)
            return chunk

        message.stream = observe

    def _observe_chunk(self, state: _FlowCapture, side: str, chunk: bytes) -> None:
        if side not in {"request", "response"}:
            raise AssertionError("invalid body side")
        body = state.request if side == "request" else state.response
        if state.completed or state.discarded or body.ended or not body.stream_enabled:
            return
        copied = _safe_bytes(chunk, label="body chunk")
        offset = body.total_bytes
        body.total_bytes += len(copied)
        body.observed = True
        remaining = max(0, self.max_body_prefix_bytes - len(body.prefix))
        other_prefix_bytes = self._captured_prefix_bytes - len(body.prefix)
        global_remaining = max(0, self.max_in_memory_bytes - other_prefix_bytes - len(body.prefix))
        captured = copied[: min(remaining, global_remaining)]
        body.prefix.extend(captured)
        self._captured_prefix_bytes += len(captured)
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
        body.stream_enabled = False
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
        if not state.request_headers_captured:
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
        if state.response_headers_captured:
            metadata["response_headers"] = state.response_headers
            metadata["response_body"] = _body_descriptor(state.response)
        self._send({"protocol_version": "1", "type": "flow.metadata", "metadata": metadata})

    def _set_identity(self, state: _FlowCapture, flow: http.HTTPFlow) -> None:
        request = flow.request
        identity = {
            "method": _safe_text(request.method, label="method"),
            "scheme": _safe_text(request.scheme, label="scheme"),
            "host": _safe_text(request.host, label="host"),
            "port": _safe_uint_text(request.port, label="port"),
            "path": sanitize_path(_safe_text(request.path, label="path")) or "/",
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

    def _finalize(self, state: _FlowCapture) -> None:
        """Idempotently finish only after request terminal observation exists."""

        if state.completed or state.discarded:
            return
        if not state.terminal_observed:
            return
        if not state.request.ended:
            state.completion_pending = True
            return
        state.completed = True
        state.completion_pending = False
        self._disable_streams(state)
        self._lifecycle(state, "flow_completed")
        self._flows.pop(state.flow_id, None)
        self._completed_ids.append(state.flow_id)
        self._captured_prefix_bytes -= len(state.request.prefix) + len(state.response.prefix)
        self._active_metadata_bytes -= state.retained_weight
        assert self._captured_prefix_bytes >= 0
        assert self._active_metadata_bytes >= 0
        state.discarded = True

    def _disable_streams(self, state: _FlowCapture) -> None:
        state.request.stream_enabled = False
        state.response.stream_enabled = False

    def _refresh_state_weight(self, state: _FlowCapture) -> None:
        weight = _state_weight(state)
        self._active_metadata_bytes += weight - state.retained_weight
        state.retained_weight = weight

    def _purge_active(self) -> None:
        now = self._active_clock()
        for state in list(self._flows.values()):
            if now - state.created_at >= self.max_active_age_seconds:
                self._evict_active(state)

    def _enforce_active_bounds(self) -> None:
        while self._flows and (
            len(self._flows) > self.max_active_flows
            or self._active_metadata_bytes > self.max_in_memory_bytes
        ):
            oldest = min(self._flows.values(), key=lambda item: (item.created_at, item.flow_id))
            self._evict_active(oldest)

    def _evict_active(self, state: _FlowCapture) -> None:
        if state.discarded:
            return
        self._disable_streams(state)
        self._flows.pop(state.flow_id, None)
        self._completed_ids.append(state.flow_id)
        self._captured_prefix_bytes -= len(state.request.prefix) + len(state.response.prefix)
        self._active_metadata_bytes -= state.retained_weight
        assert self._captured_prefix_bytes >= 0
        assert self._active_metadata_bytes >= 0
        state.discarded = True
        self._capture_evicted_flows += 1
        self.sink.record_loss()

    def _send(self, raw: dict[str, object]) -> None:
        parsed = parse_message(raw)
        prepared = self.sink.prepare(parsed)
        self.sink.offer_prepared(prepared)

    @property
    def counters(self) -> dict[str, int]:
        """Expose active-bound and tombstone accounting for health checks."""

        self._purge_active()
        return {
            "active_flows": len(self._flows),
            "active_metadata_bytes": self._active_metadata_bytes,
            "active_prefix_bytes": self._captured_prefix_bytes,
            "evicted_flows": self._capture_evicted_flows,
            "tombstones": len(self._completed_ids),
        }


def _headers(headers: Mapping[str, str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    items = cast(Callable[..., Iterable[tuple[object, object]]], headers.items)
    try:
        header_items = items(multi=True)
    except TypeError:
        header_items = items()
    for name, value in header_items:
        sanitized_name, sanitized_value = sanitize_header(
            _safe_text(name, label="header name"), _safe_text(value, label="header value")
        )
        result.append({"name": sanitized_name, "value": sanitized_value})
    return result


def _content_type(headers: Mapping[str, str]) -> str | None:
    value = headers.get("content-type")
    return (
        None
        if value is None
        else sanitize_header("content-type", _safe_text(value, label="content type"))[1]
    )


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


def _safe_text(value: object, *, label: str) -> str:
    """Accept only builtin text/bytes before attacker-controlled coercion."""

    if type(value) is str:
        return value
    if type(value) is bytes:
        return value.decode("utf-8", "surrogateescape")
    raise ValueError(f"{label} must be an exact str or bytes")


def _safe_bytes(value: object, *, label: str) -> bytes:
    if type(value) is bytes:
        return value
    raise ValueError(f"{label} must be exact bytes")


def _safe_uint_text(value: object, *, label: str) -> str:
    if type(value) is not int or value < 0 or value > MAX_U64:
        raise ValueError(f"{label} must be an exact non-negative int")
    return str(value)


def _state_weight(state: _FlowCapture) -> int:
    """Bound copied metadata, including every header value."""

    weight = len(state.flow_id)
    weight += sum(len(key) + len(value) for key, value in state.identity.items())
    weight += sum(
        len(header["name"]) + len(header["value"])
        for header in state.request_headers
    )
    if state.response_headers is not None:
        weight += sum(
            len(header["name"]) + len(header["value"])
            for header in state.response_headers
        )
    return weight


def make_addon_from_environment() -> CaptureAddon:
    """Construct a configured addon without connecting to the IPC endpoint."""

    return CaptureAddon.from_environment()


# mitmdump -s imports this module and discovers the documented addon list.
# Construction is intentionally local-only; CaptureConfig performs no I/O.
addons = [CaptureAddon()]
