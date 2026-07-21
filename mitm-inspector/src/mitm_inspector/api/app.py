"""Capture ingestion and bounded flow-detail projection.

Browse reads are request-scoped HTTP queries.  There is deliberately no
server-held stream or shared browser cursor here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mitm_inspector.detail_limits import MAX_DURABLE_DETAIL_OUTPUT_BYTES
from mitm_inspector.json_boundary import PlainJsonObject
from mitm_inspector.protocol import (
    KnownParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    _trusted_parsed_message_to_plain_json,
    parse_message,
    parsed_message_to_plain_json,
    require_parsed_message,
)
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SQLiteFlowStorage


@dataclass(frozen=True, slots=True)
class IngestResult:
    parsed: ParsedMessageResult
    relayed: bool = False
    delta_emitted: bool = False


class FlowDetailTooLarge(RuntimeError):
    """Raised when bounded durable detail exceeds its wire ceiling."""


class ApiApplication:
    """Own capture retention and durable persistence; HTTP owns reads."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        source_id: str = "mitm-inspector",
        max_body_prefix_bytes: int = 0,
        max_in_memory_bytes: int = 128 * 1024 * 1024,
        storage: SQLiteFlowStorage | None = None,
        **_legacy: object,
    ) -> None:
        if type(source_id) is not str or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if type(max_body_prefix_bytes) is not int or max_body_prefix_bytes < 0:
            raise ValueError("max_body_prefix_bytes must be nonnegative")
        if type(max_in_memory_bytes) is not int or max_in_memory_bytes < 0:
            raise ValueError("max_in_memory_bytes must be nonnegative")
        self._store = store
        self._storage = storage
        self._ingested_messages = 0

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def storage(self) -> SQLiteFlowStorage | None:
        return self._storage

    @property
    def counters(self) -> dict[str, object]:
        counters: dict[str, object] = {
            "ingested_messages": self._ingested_messages,
            "store": dict(self._store.counters),
        }
        if self._storage is not None:
            counters["storage"] = self._storage.counters
        return counters

    def ingest(self, value: object) -> IngestResult:
        parsed = (
            require_parsed_message(value)
            if isinstance(value, ParsedMessage)
            else parse_message(value)
        )
        self._store.append(parsed)
        if self._storage is not None:
            self._storage.offer(parsed)
        self._ingested_messages += 1
        return IngestResult(parsed)

    def sweep(self) -> bool:
        # Reading the bounded store purges expired entries. Durable retention is
        # enforced synchronously by SQLiteFlowStorage.
        _ = self._store.counters
        return False

    def flow_detail_text(self, flow_id: str) -> str | None:
        if type(flow_id) is not str or not flow_id:
            return None
        messages = [
            _trusted_parsed_message_to_plain_json(parsed)
            for parsed in self._store.trusted_flow_messages(flow_id)
            if self._message_flow_id(self._payload_of(parsed)) == flow_id
        ]
        return self._detail_text(flow_id, messages) if messages else None

    def flow_detail_text_from_messages(
        self, flow_id: str, parsed_messages: Sequence[ParsedMessageResult]
    ) -> str | None:
        if type(flow_id) is not str or not flow_id:
            return None
        messages = [
            parsed_message_to_plain_json(parsed)
            for parsed in parsed_messages
            if self._message_flow_id(self._payload_of(parsed)) == flow_id
        ]
        return self._detail_text(flow_id, messages) if messages else None

    def durable_flow_detail_bytes(self, flow_id: str) -> bytes | None:
        if self._storage is None:
            return None
        messages = self._storage.flow_messages(flow_id)
        detail = self._detail_text(
            flow_id,
            [
                _trusted_parsed_message_to_plain_json(parsed)
                for parsed in messages
                if self._message_flow_id(self._payload_of(parsed)) == flow_id
            ],
        )
        if detail is None:
            return None
        encoded = detail.encode("utf-8")
        if len(encoded) > MAX_DURABLE_DETAIL_OUTPUT_BYTES:
            raise FlowDetailTooLarge
        return encoded

    @staticmethod
    def _detail_text(flow_id: str, messages: list[PlainJsonObject]) -> str | None:
        if not messages:
            return None
        return json.dumps(
            {"protocol_version": "1", "flow_id": flow_id, "messages": messages},
            separators=(",", ":"),
        )

    @staticmethod
    def _payload_of(parsed: ParsedMessageResult) -> Mapping[str, object]:
        return parsed.message if isinstance(parsed, KnownParsedMessage) else parsed.payload

    @staticmethod
    def _message_flow_id(payload: Mapping[str, object]) -> str | None:
        if payload.get("type") == "flow.metadata":
            metadata = payload.get("metadata")
            value = metadata.get("flow_id") if isinstance(metadata, Mapping) else None
        else:
            value = payload.get("flow_id")
        return value if isinstance(value, str) else None


__all__ = ["ApiApplication", "FlowDetailTooLarge", "IngestResult"]
