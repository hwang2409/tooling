"""Documented mitmproxy-hook adapter for sanitized protocol-v1 capture."""

from __future__ import annotations

import base64
import math
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
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
from mitm_inspector.capture.sequencer import DrainBatch
from mitm_inspector.capture.sink import BoundedMessageSink
from mitm_inspector.protocol import MAX_U64, ParsedMessageResult

MessageEmitter = Callable[[ParsedMessageResult], None]
Clock = Callable[[], str]

MAX_BODY_PREFIX_BYTES = DEFAULT_MAX_BODY_PREFIX_BYTES
MAX_IN_MEMORY_BYTES = DEFAULT_MAX_IN_MEMORY_BYTES
MAX_ACTIVE_FLOWS = 2_000
MAX_ACTIVE_AGE_SECONDS = 30 * 60


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_active_limits(max_flows: object, max_age: object) -> None:
    if type(max_flows) is not int or max_flows < 1 or max_flows > MAX_U64:
        raise ValueError("max_active_flows must be an exact bounded integer")
    if type(max_age) is int:
        valid_age = max_age >= 0
    elif type(max_age) is float:
        valid_age = math.isfinite(max_age) and max_age >= 0
    else:
        valid_age = False
    if not valid_age:
        raise ValueError("max_active_age_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class _PendingChunk:
    offset: int
    chunk_index: int
    captured_start: int
    captured_length: int
    source_object: bytes


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
    prefix_sealed: bool = False
    pending_chunk: _PendingChunk | None = None
    retry_chunk: _PendingChunk | None = None


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
        _validate_active_limits(max_active_flows, max_active_age_seconds)
        self.config = config
        self.source_id = config.source_id
        self.capture_socket = config.capture_socket
        self.max_body_prefix_bytes = config.max_body_prefix_bytes
        self.max_in_memory_bytes = config.max_in_memory_bytes
        self.max_active_flows = max_active_flows
        self.max_active_age_seconds = max_active_age_seconds
        self._active_clock = active_clock
        self._clock = clock
        self._emit_callback = emit
        self._dispatch_lock = Lock()
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

        self._apply_config(CaptureConfig.from_environment())

    def _apply_config(self, config: CaptureConfig) -> None:
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
        if not self._try_lifecycle(state, "request_started"):
            return
        if "request_headers" not in state.lifecycle_states:
            state.request_headers = _headers(flow.request.headers)
            state.request_headers_captured = True
            state.request.content_type = _content_type(flow.request.headers)
            self._set_identity(state, flow)
            self._install_stream(flow.request, state, "request")
            if not self._try_lifecycle(state, "request_headers"):
                return
            self._metadata(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def request(self, flow: http.HTTPFlow) -> None:
        """Observe the terminal request body and request end."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        if not self._finish_body(state, "request", flow.request.raw_content):
            return
        if "request_end" not in state.lifecycle_states:
            if not self._try_lifecycle(state, "request_end"):
                return
            self._metadata(state)
        self._finalize(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        """Observe response start/headers and install the public body callback."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        if not self._try_lifecycle(state, "response_started"):
            return
        if "response_headers" not in state.lifecycle_states:
            state.response_headers = _headers(flow.response.headers) if flow.response else []
            state.response_headers_captured = True
            if flow.response:
                state.response.content_type = _content_type(flow.response.headers)
                self._install_stream(flow.response, state, "response")
            if not self._try_lifecycle(state, "response_headers"):
                return
            self._metadata(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def response(self, flow: http.HTTPFlow) -> None:
        """Observe the terminal response body and complete the flow."""

        state = self._ensure_request(flow)
        if state.tombstone:
            return
        response_content = flow.response.raw_content if flow.response else None
        if not self._finish_body(state, "response", response_content):
            return
        if "response_end" not in state.lifecycle_states:
            if not self._try_lifecycle(state, "response_end"):
                return
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
            if not self._finish_body(state, "request", flow.request.raw_content):
                return
            if not self._try_lifecycle(state, "request_end"):
                return
            self._metadata(state)
        if state.response_headers_captured and not state.response.ended:
            response_content = flow.response.raw_content if flow.response else None
            if not self._finish_body(state, "response", response_content):
                return
            if "response_end" not in state.lifecycle_states:
                if not self._try_lifecycle(state, "response_end"):
                    return
                self._metadata(state)
        if not self._try_lifecycle(state, "error"):
            return
        state.terminal_observed = True
        self._finalize(state)
        self._refresh_state_weight(state)
        self._enforce_active_bounds()

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Drain and dispatch one single-flight batch.

        Reentrant or overlapping calls return an empty batch. Callback
        failures leave the same unacknowledged token available; a retry
        resumes at the first callback that did not return successfully.
        """

        _validate_drain_limit(limit)
        if not self._dispatch_lock.acquire(False):
            return []
        try:
            self._purge_active()
            batch = self.sink.drain(limit)
            messages = batch.messages
            callback_failure: BaseException | None = None
            start = batch._progress.next_index
            if self._emit_callback is not None:
                for index in range(start, len(messages)):
                    message = messages[index]
                    try:
                        self._emit_callback(message)
                    except BaseException as error:
                        if callback_failure is None:
                            callback_failure = error
                        break
                    try:
                        self._publish_callback_progress(batch, index + 1)
                    except BaseException:
                        # The callback has already returned successfully.
                        # Keep its receipt in the handed-off batch even when
                        # the progress publication itself is interrupted.
                        batch._progress.next_index = index + 1
                        raise
            if callback_failure is not None:
                raise callback_failure
            result = list(messages)
            try:
                self.sink.acknowledge(batch)
            except BaseException:
                # A post-ack cleanup/return fault cannot make an already
                # acknowledged batch disappear from this consumer.
                if not self.sink._sequencer.has_committed_batch():
                    return result
                raise
            return result
        finally:
            self._dispatch_lock.release()

    @staticmethod
    def _publish_callback_progress(batch: DrainBatch, next_index: int) -> None:
        """Publish the accepted index in the durable handed-off receipt."""

        batch._progress.next_index = next_index

    def _announce_source(self) -> None:
        if self._source_announced:
            return
        accepted = self._send(
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
        if accepted:
            self._source_announced = True

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
        state_ref = weakref.ref(state)

        def observe(chunk: bytes) -> bytes:
            current = state_ref()
            if current is None:
                return chunk
            body = current.request if side == "request" else current.response
            if current.completed or current.discarded or not body.stream_enabled:
                return chunk
            try:
                self._observe_chunk(current, side, chunk)
            except OverflowError:
                # Forwarding owns the return value.  Exhaustion is a capture
                # terminal state, never a reason to interrupt mitmproxy.
                self._discard_active(current, count_eviction=False)
            return chunk

        message.stream = observe

    def _observe_chunk(self, state: _FlowCapture, side: str, chunk: bytes) -> None:
        if side not in {"request", "response"}:
            raise AssertionError("invalid body side")
        body = state.request if side == "request" else state.response
        if state.completed or state.discarded or body.ended or not body.stream_enabled:
            return
        copied = _safe_bytes(chunk, label="body chunk")
        if body.retry_chunk is not None:
            retry = body.retry_chunk
            body.retry_chunk = None
            if retry.source_object is copied:
                return
        if body.pending_chunk is not None:
            pending = body.pending_chunk
            is_retry = pending.source_object is copied
            self._emit_body_lifecycle(state, side, body)
            self._emit_pending_chunk(state, side, body, pending)
            if is_retry:
                return
        if body.total_bytes > MAX_U64 - len(copied):
            raise OverflowError(f"{side} body byte offset exhausted")
        if body.chunk_index >= MAX_U64:
            raise OverflowError(f"{side} body chunk index exhausted")
        if not body.lifecycle_emitted:
            self._ensure_lifecycle_capacity()
        offset = body.total_bytes
        remaining = (
            0
            if body.prefix_sealed
            else max(0, self.max_body_prefix_bytes - len(body.prefix))
        )
        other_prefix_bytes = self._captured_prefix_bytes - len(body.prefix)
        global_remaining = max(
            0,
            self.max_in_memory_bytes
            - self._active_metadata_bytes
            - other_prefix_bytes
            - len(body.prefix),
        )
        captured = copied[: min(remaining, global_remaining)]
        captured_start = len(body.prefix)
        pending = _PendingChunk(
            offset,
            body.chunk_index,
            captured_start,
            len(captured),
            copied,
        )
        # Build every allocation that can fail before publishing any body or
        # global accounting.  A constructor/allocation fault is then a clean
        # retry of the same caller observation, not a half-accounted chunk.
        next_prefix = body.prefix + captured
        next_total = body.total_bytes + len(copied)
        next_sealed = body.prefix_sealed or len(captured) < len(copied)
        body.prefix = next_prefix
        body.total_bytes = next_total
        body.observed = True
        self._captured_prefix_bytes += len(captured)
        body.pending_chunk = pending
        body.prefix_sealed = next_sealed
        self._emit_body_lifecycle(state, side, body)
        self._emit_pending_chunk(state, side, body, body.pending_chunk)

    def _emit_body_lifecycle(
        self, state: _FlowCapture, side: str, body: _BodyCapture
    ) -> None:
        if body.lifecycle_emitted:
            return
        try:
            self._lifecycle(state, f"{side}_body")
        except OverflowError:
            self._discard_active(state, count_eviction=False)
            raise
        except BaseException as error:
            if getattr(error, "capture_committed", False):
                body.lifecycle_emitted = True
            raise
        try:
            self._publish_body_lifecycle(body)
        except BaseException:
            body.lifecycle_emitted = True
            raise

    @staticmethod
    def _publish_body_lifecycle(body: _BodyCapture) -> None:
        body.lifecycle_emitted = True

    def _emit_pending_chunk(
        self,
        state: _FlowCapture,
        side: str,
        body: _BodyCapture,
        pending: _PendingChunk | None,
    ) -> None:
        if pending is None:
            return
        committed_error: BaseException | None = None
        try:
            captured = bytes(
                body.prefix[
                    pending.captured_start : pending.captured_start + pending.captured_length
                ]
            )
            if captured:
                self._send(
                    {
                        "protocol_version": "1",
                        "type": "body.chunk",
                        "flow_id": state.flow_id,
                        "body_side": side,
                        "chunk_index": str(pending.chunk_index),
                        "offset_bytes": str(pending.offset),
                        "data_base64": base64.b64encode(captured).decode("ascii"),
                    }
                )
        except BaseException as error:
            if not getattr(error, "capture_committed", False):
                raise
            committed_error = error
        try:
            self._publish_body_chunk(body, pending, committed_error is not None)
        except BaseException:
            self._force_body_chunk_receipt(body, pending)
            raise
        if committed_error is not None:
            raise committed_error

    @staticmethod
    def _publish_body_chunk(
        body: _BodyCapture, pending: _PendingChunk, retryable: bool
    ) -> None:
        body.chunk_index = pending.chunk_index + 1
        body.pending_chunk = None
        body.retry_chunk = pending if retryable else None

    @staticmethod
    def _force_body_chunk_receipt(body: _BodyCapture, pending: _PendingChunk) -> None:
        body.chunk_index = pending.chunk_index + 1
        body.pending_chunk = None
        body.retry_chunk = pending

    def _finish_body(
        self, state: _FlowCapture, side: str, raw_content: bytes | None
    ) -> bool:
        body = state.request if side == "request" else state.response
        if body.ended:
            return True
        if not body.observed and raw_content is not None:
            try:
                self._observe_chunk(state, side, raw_content)
            except OverflowError:
                self._discard_active(state, count_eviction=False)
                return False
            if state.discarded:
                return False
        if body.retry_chunk is not None:
            body.retry_chunk = None
        if not body.lifecycle_emitted:
            try:
                self._emit_body_lifecycle(state, side, body)
            except OverflowError:
                self._discard_active(state, count_eviction=False)
                return False
        if body.pending_chunk is not None:
            self._emit_pending_chunk(state, side, body, body.pending_chunk)
        descriptor = _body_descriptor(body)
        try:
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
        except BaseException as error:
            if getattr(error, "capture_committed", False):
                body.lifecycle_emitted = True
                body.ended = True
                body.stream_enabled = False
            raise
        try:
            self._publish_body_end(body)
        except BaseException:
            self._force_body_end_receipt(body)
            raise
        return True

    @staticmethod
    def _publish_body_end(body: _BodyCapture) -> None:
        body.lifecycle_emitted = True
        body.ended = True
        body.stream_enabled = False

    @staticmethod
    def _force_body_end_receipt(body: _BodyCapture) -> None:
        body.lifecycle_emitted = True
        body.ended = True
        body.stream_enabled = False

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
        self._ensure_lifecycle_capacity()
        sequence = self._sequence
        try:
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
        except BaseException as error:
            if getattr(error, "capture_committed", False):
                state.lifecycle_states.add(lifecycle_state)
                self._sequence = sequence + 1
            raise
        state.lifecycle_states.add(lifecycle_state)
        self._sequence = sequence + 1

    def _ensure_lifecycle_capacity(self) -> None:
        if self._sequence >= MAX_U64:
            raise OverflowError("capture lifecycle sequence exhausted")

    def _try_lifecycle(self, state: _FlowCapture, lifecycle_state: str) -> bool:
        try:
            self._lifecycle(state, lifecycle_state)
        except OverflowError:
            self._discard_active(state, count_eviction=False)
            return False
        return True

    def _finalize(self, state: _FlowCapture) -> None:
        """Idempotently finish only after request terminal observation exists."""

        if state.completed or state.discarded:
            return
        if not state.terminal_observed:
            return
        if not state.request.ended:
            state.completion_pending = True
            return
        # Emit the terminal lifecycle before setting sticky completion state.
        # If its uint64 sequence is exhausted, discard coherently so a later
        # hook cannot observe a completed-but-live flow.
        try:
            self._lifecycle(state, "flow_completed")
        except OverflowError:
            self._discard_active(state, count_eviction=False)
            return
        state.completed = True
        state.completion_pending = False
        self._complete_active(state)

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
            or self._active_metadata_bytes + self._captured_prefix_bytes
            > self.max_in_memory_bytes
        ):
            oldest = min(self._flows.values(), key=lambda item: (item.created_at, item.flow_id))
            self._evict_active(oldest)

    def _evict_active(self, state: _FlowCapture) -> None:
        self._discard_active(state, count_eviction=True)

    def _complete_active(self, state: _FlowCapture) -> None:
        self._release_active(state, count_eviction=False)

    def _discard_active(self, state: _FlowCapture, *, count_eviction: bool) -> None:
        if state.discarded:
            return
        self._release_active(state, count_eviction=count_eviction)
        self.sink.record_loss()

    def _release_active(self, state: _FlowCapture, *, count_eviction: bool) -> None:
        if state.discarded:
            return
        self._disable_streams(state)
        if self._flows.get(state.flow_id) is state:
            self._flows.pop(state.flow_id, None)
        self._completed_ids.append(state.flow_id)
        self._captured_prefix_bytes -= len(state.request.prefix) + len(state.response.prefix)
        self._active_metadata_bytes -= state.retained_weight
        assert self._captured_prefix_bytes >= 0
        assert self._active_metadata_bytes >= 0
        self._scrub_released_state(state)
        state.discarded = True
        state.tombstone = True
        if count_eviction:
            self._capture_evicted_flows += 1

    @staticmethod
    def _scrub_released_state(state: _FlowCapture) -> None:
        """Drop secret-bearing state even if a caller retains the Flow."""

        state.identity.clear()
        state.request_headers.clear()
        state.response_headers = None
        state.lifecycle_states.clear()
        for body in (state.request, state.response):
            body.content_type = None
            body.total_bytes = 0
            body.prefix.clear()
            body.chunk_index = 0
            body.observed = False
            body.lifecycle_emitted = False
            body.ended = True
            body.stream_enabled = False
            body.prefix_sealed = True
            body.pending_chunk = None
            body.retry_chunk = None
        state.retained_weight = 0

    def _send(self, raw: dict[str, object]) -> bool:
        try:
            return self.sink.offer(raw)
        except BaseException as error:
            if (
                raw.get("type") == "source.hello"
                and getattr(error, "capture_committed", False)
            ):
                self._source_announced = True
            raise

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


def _validate_drain_limit(limit: object) -> None:
    if limit is not None and (
        type(limit) is not int or limit < 1 or limit > MAX_U64
    ):
        raise ValueError("limit must be an exact bounded positive integer")
