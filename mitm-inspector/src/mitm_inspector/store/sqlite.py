"""Asynchronous durable flow persistence backed by the stdlib sqlite3 module."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from mitm_inspector.detail_limits import (
    MAX_DURABLE_DETAIL_MESSAGES,
    MAX_DURABLE_DETAIL_RETAINED_BODY_BYTES,
)
from mitm_inspector.json_boundary import PlainJsonObject
from mitm_inspector.protocol import (
    KnownParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    is_rfc3339_utc,
    parse_message,
    require_parsed_message,
)

if TYPE_CHECKING:
    from mitm_inspector.store.memory import MemoryStore

DEFAULT_STORAGE_MAX_FLOWS = 10_000
DEFAULT_STORAGE_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_STORAGE_REPLAY = 500
DEFAULT_STORAGE_QUEUE_SIZE = 4_096
SESSION_FLOW_LIMIT = 2_000
_SENTINEL = object()
_WHITESPACE = re.compile(r"\s+")


class SearchMatch(TypedDict):
    flow_id: str
    field: str
    snippet: str
    flow: PlainJsonObject


class SearchCancelled(RuntimeError):
    """Raised when a superseding request cancels durable search work."""


def _normalize_storage_path(path: Path | str) -> Path:
    """Resolve storage paths and reject symlinked existing ancestors."""

    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise ValueError("storage path must not contain '..'")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    current = absolute
    ancestors = list(absolute.parents)
    for ancestor in reversed(ancestors):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError(f"storage path has a symlinked ancestor: {ancestor}")
    current_info: os.stat_result | None
    try:
        current_info = current.lstat()
    except FileNotFoundError:
        current_info = None
    if current_info is not None and stat.S_ISLNK(current_info.st_mode):
        raise PermissionError(f"storage path must not be a symlink: {current}")
    return absolute.resolve(strict=False)


def default_storage_path() -> Path:
    """Return the platform-independent local state path for flow history."""

    state_home = os.environ.get("XDG_STATE_HOME")
    root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return root / "mitm-inspector" / "flows.sqlite"


class SQLiteFlowStorage:
    """Persist flow messages without making capture ingestion wait on sqlite."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_flows: int = DEFAULT_STORAGE_MAX_FLOWS,
        max_bytes: int = DEFAULT_STORAGE_MAX_BYTES,
        queue_size: int = DEFAULT_STORAGE_QUEUE_SIZE,
        trace_sql: Callable[[str], None] | None = None,
    ) -> None:
        if type(max_flows) is not int or max_flows < 1:
            raise ValueError("max_flows must be a positive integer")
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        if os.fspath(path) == ":memory:":
            raise ValueError(":memory: disables storage and cannot create SQLiteFlowStorage")
        self.path = _normalize_storage_path(path)
        self.max_flows = max_flows
        self.max_bytes = max_bytes
        self._trace_sql = trace_sql
        self._minimum_storage_bytes = 0
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._closed = False
        self._dropped_messages = 0
        self._write_errors = 0
        self._next_created_order = 0
        self._thread = threading.Thread(
            target=self._run,
            name="mitm-inspector-storage",
            daemon=True,
        )
        self._prepare_database()
        if self.max_bytes < self._minimum_storage_bytes:
            raise ValueError(
                f"storage max_bytes {self.max_bytes} is below the SQLite overhead "
                f"of {self._minimum_storage_bytes} bytes"
            )
        self._thread.start()

    @property
    def counters(self) -> dict[str, int]:
        with self._lock:
            return {
                "dropped_messages": self._dropped_messages,
                "write_errors": self._write_errors,
                "queued_messages": self._queue.qsize(),
            }

    def offer(self, message: ParsedMessage | Mapping[str, object]) -> bool:
        """Queue one validated message, dropping the oldest queued item if full."""

        parsed = (
            require_parsed_message(message)
            if isinstance(message, ParsedMessage)
            else parse_message(message)
        )
        with self._lock:
            if self._closed:
                return False
            while True:
                try:
                    self._queue.put_nowait(parsed)
                    return True
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:  # pragma: no cover - producer won a race.
                        continue
                    self._queue.task_done()
                    self._dropped_messages += 1

    def flush(self) -> None:
        """Wait until every message queued so far has been committed."""

        self._queue.join()

    def close(self) -> None:
        """Flush and stop the writer thread; safe to call more than once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.flush()
        self._queue.put(_SENTINEL)
        self._thread.join()

    def replay(self, limit: int = DEFAULT_STORAGE_REPLAY) -> list[ParsedMessageResult]:
        """Read the newest flows as protocol messages, oldest lifecycle first."""

        if type(limit) is not int or limit < 0:
            raise ValueError("replay limit must be a nonnegative integer")
        self.flush()
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT flows.flow_id, flows.source_id, flows.method, flows.scheme,
                       flows.host, flows.port, flows.path, flows.response_status,
                       flows.request_content_type, flows.response_content_type,
                       flows.started_at, flows.ended_at, flows.request_body,
                       flows.response_body, flows.request_body_state,
                       flows.response_body_state, flows.request_body_size,
                       flows.response_body_size, flows.request_headers_json,
                       flows.response_headers_json, flows.session_id
                FROM flows
                WHERE method <> ''
                ORDER BY CASE WHEN flows.started_at_sort IS NULL THEN 1 ELSE 0 END,
                         flows.started_at_sort DESC,
                         flows.created_order DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return self._messages_from_rows(connection, rows)

    def flow_messages(self, flow_id: str) -> list[ParsedMessageResult]:
        """Read one durable flow as validated oldest-first detail messages."""

        if type(flow_id) is not str or not flow_id:
            return []
        self.flush()
        with sqlite3.connect(self.path) as connection:
            lengths = connection.execute(
                """
                SELECT COALESCE(length(request_body), 0),
                       COALESCE(length(response_body), 0)
                FROM flows WHERE flow_id = ? AND method <> ''
                """,
                (flow_id,),
            ).fetchone()
            if lengths is None:
                return []
            request_bytes, response_bytes = _detail_body_allocations(
                int(lengths[0]),
                int(lengths[1]),
                MAX_DURABLE_DETAIL_RETAINED_BODY_BYTES,
            )
            row = connection.execute(
                """
                SELECT flow_id, source_id, method, scheme, host, port, path,
                       response_status, request_content_type,
                       response_content_type, started_at, ended_at,
                       substr(request_body, 1, ?), substr(response_body, 1, ?),
                       request_body_state,
                       response_body_state, request_body_size,
                       response_body_size, request_headers_json,
                       response_headers_json, session_id
                FROM flows
                WHERE flow_id = ? AND method <> ''
                """,
                (request_bytes, response_bytes, flow_id),
            ).fetchone()
            if row is None:
                return []
            bounded_row = list(row)
            for state_index, body_index, stored_length in (
                (14, 12, int(lengths[0])),
                (15, 13, int(lengths[1])),
            ):
                body = bounded_row[body_index]
                retained_length = len(body) if isinstance(body, bytes | bytearray) else 0
                if retained_length < stored_length and bounded_row[state_index] in {
                    "captured",
                    "truncated",
                }:
                    bounded_row[state_index] = "truncated"
            retained_body_bytes = request_bytes + response_bytes
            return self._messages_from_rows(
                connection,
                [tuple(bounded_row)],
                max_messages=MAX_DURABLE_DETAIL_MESSAGES,
                max_chunk_bytes=(MAX_DURABLE_DETAIL_RETAINED_BODY_BYTES - retained_body_bytes),
                chunks_only_for_missing_bodies=True,
            )

    def session_summaries(
        self, limit: int = 100, before: str | None = None
    ) -> list[PlainJsonObject]:
        """Return redacted session metadata in newest-first order."""

        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("session limit must be from 1 through 200")
        if before is not None and (type(before) is not str or not before):
            raise ValueError("session before must be a non-empty timestamp")
        self.flush()
        with sqlite3.connect(self.path) as connection:
            rows = self._session_rows(
                connection,
                where=("AND started_at_sort < ?" if before is not None else ""),
                parameters=(_timestamp_sort_key(before),) if before is not None else (),
                limit=limit * SESSION_FLOW_LIMIT,
                order="DESC",
            )
        groups: dict[str | None, list[PlainJsonObject]] = {}
        for row in rows:
            metadata = self._metadata_from_session_row(row)["metadata"]
            if not isinstance(metadata, Mapping):
                continue
            projected = self._redacted_grid_flow(metadata)
            session_id = metadata.get("session_id")
            key = session_id if isinstance(session_id, str) else None
            groups.setdefault(key, []).append(projected)
        summaries = [self._session_summary(key, flows) for key, flows in groups.items()]
        summaries.sort(key=self._session_sort_key, reverse=True)
        return summaries[:limit]

    def session_detail(self, session_id: str) -> PlainJsonObject | None:
        """Return one session's redacted flows in chronological order."""

        if type(session_id) is not str or not session_id:
            return None
        self.flush()
        where = "AND session_id IS NULL" if session_id == "unassigned" else "AND session_id = ?"
        parameters: tuple[object, ...] = () if session_id == "unassigned" else (session_id,)
        with sqlite3.connect(self.path) as connection:
            rows = self._session_rows(
                connection,
                where=where,
                parameters=parameters,
                limit=SESSION_FLOW_LIMIT,
                order="ASC",
            )
        flows: list[PlainJsonObject] = []
        actual_id: str | None = None
        for row in rows:
            metadata = self._metadata_from_session_row(row)["metadata"]
            if not isinstance(metadata, Mapping):
                continue
            if actual_id is None and isinstance(metadata.get("session_id"), str):
                actual_id = metadata["session_id"]
            flows.append(self._redacted_grid_flow(metadata))
        if not flows:
            return None
        return {
            "session_id": actual_id,
            "flow_count": len(flows),
            "flows": flows,
        }

    @staticmethod
    def _session_rows(
        connection: sqlite3.Connection,
        *,
        where: str,
        parameters: tuple[object, ...],
        limit: int,
        order: str,
    ) -> list[tuple[object, ...]]:
        if order not in {"ASC", "DESC"}:
            raise ValueError("session order must be ASC or DESC")
        return connection.execute(
            f"""
            SELECT flow_id, source_id, method, scheme, host, port, path,
                   response_status, request_content_type,
                   response_content_type, started_at, ended_at,
                   request_body_state, response_body_state,
                   request_body_size, response_body_size, session_id
            FROM flows
            WHERE method <> '' {where}
            ORDER BY CASE WHEN started_at_sort IS NULL THEN 1 ELSE 0 END,
                     started_at_sort {order}, created_order {order}
            LIMIT ?
            """,
            (*parameters, limit),
        ).fetchall()

    @staticmethod
    def _metadata_from_session_row(row: tuple[object, ...]) -> dict[str, object]:
        metadata: dict[str, object] = {
            "flow_id": row[0],
            "session_id": row[16],
            "method": row[2],
            "scheme": row[3],
            "host": row[4],
            "port": row[5],
            "path": row[6],
            # Headers and body bytes are intentionally absent from the session
            # projection. The selected flow endpoint owns those reads.
            "request_headers": [],
            "request_body": _descriptor_from_row(None, row[12], row[14], row[8]),
        }
        if row[7] is not None:
            metadata["response_status"] = row[7]
        if row[13] != "missing":
            metadata["response_body"] = _descriptor_from_row(
                None, row[13], row[15], row[9]
            )
        if isinstance(row[10], str) and is_rfc3339_utc(row[10]):
            metadata["started_at"] = row[10]
        if isinstance(row[11], str) and is_rfc3339_utc(row[11]):
            metadata["ended_at"] = row[11]
        if row[12] != "missing":
            metadata["request_body_size"] = str(row[14])
        if row[13] != "missing":
            metadata["response_body_size"] = str(row[15])
        if row[8] is not None:
            metadata["request_content_type"] = row[8]
        if row[9] is not None:
            metadata["response_content_type"] = row[9]
        from mitm_inspector.api.projection import canonical_grid_flow

        return {
            "protocol_version": "1",
            "type": "flow.metadata",
            "metadata": canonical_grid_flow(metadata),
        }

    @staticmethod
    def _redacted_grid_flow(metadata: Mapping[str, object]) -> PlainJsonObject:
        from mitm_inspector.api.projection import grid_flow

        return grid_flow(metadata)

    @classmethod
    def _session_summary(
        cls, session_id: str | None, flows: list[PlainJsonObject]
    ) -> PlainJsonObject:
        chronological = list(reversed(flows))
        models: list[str] = []
        first_query: str | None = None
        started: str | None = None
        last_activity: str | None = None
        has_error = False
        for flow in chronological:
            value = flow.get("started_at")
            if started is None and isinstance(value, str):
                started = value
            ended = flow.get("ended_at")
            candidate_activity = ended if isinstance(ended, str) else value
            if isinstance(candidate_activity, str):
                last_activity = candidate_activity
            status = flow.get("response_status")
            if isinstance(status, str) and status.isdigit() and int(status) >= 400:
                has_error = True
            summary = flow.get("summary")
            if not isinstance(summary, Mapping):
                continue
            model = summary.get("model")
            if isinstance(model, str) and model not in models:
                models.append(model)
            if first_query is None:
                preview = summary.get("preview")
                if isinstance(preview, Mapping) and preview.get("source") == "user_text":
                    text = preview.get("text")
                    if isinstance(text, str):
                        first_query = text
        return {
            "session_id": session_id,
            "first_query": first_query,
            "flow_count": len(flows),
            "started_at": started,
            "last_activity": last_activity,
            "models": models,
            "has_error": has_error,
        }

    @staticmethod
    def _session_sort_key(summary: PlainJsonObject) -> tuple[int, str]:
        value = summary.get("started_at")
        return (1 if isinstance(value, str) else 0, value if isinstance(value, str) else "")

    def _messages_from_rows(
        self,
        connection: sqlite3.Connection,
        rows: list[tuple[object, ...]],
        *,
        max_messages: int | None = None,
        max_chunk_bytes: int | None = None,
        chunks_only_for_missing_bodies: bool = False,
    ) -> list[ParsedMessageResult]:
        result: list[ParsedMessageResult] = []
        remaining_chunk_bytes = max_chunk_bytes
        selected_flow_ids = {str(row[0]) for row in rows}
        for row in rows:
            if max_messages is not None and len(result) >= max_messages:
                break
            flow_id = str(row[0])
            result.append(parse_message(self._metadata_from_row(row)))
            for side, body, state, size, content_type in (
                ("request", row[12], row[14], row[16], row[8]),
                ("response", row[13], row[15], row[17], row[9]),
            ):
                if not chunks_only_for_missing_bodies or state == "missing":
                    if remaining_chunk_bytes is None:
                        chunk_rows = connection.execute(
                            """
                            SELECT chunk_index, offset_bytes, data
                            FROM body_chunks
                            WHERE flow_id = ? AND body_side = ?
                            ORDER BY LENGTH(offset_bytes), offset_bytes, chunk_index
                            """,
                            (flow_id, side),
                        )
                    else:
                        chunk_rows = connection.execute(
                            """
                            SELECT chunk_index, offset_bytes,
                                   CASE WHEN length(data) <= ? THEN data END
                            FROM body_chunks
                            WHERE flow_id = ? AND body_side = ?
                            ORDER BY LENGTH(offset_bytes), offset_bytes, chunk_index
                            """,
                            (remaining_chunk_bytes, flow_id, side),
                        )
                    for chunk_index, offset_bytes, chunk_value in chunk_rows:
                        if max_messages is not None and len(result) >= max_messages:
                            break
                        if chunk_value is None:
                            break
                        chunk_data = bytes(chunk_value)
                        if (
                            remaining_chunk_bytes is not None
                            and len(chunk_data) > remaining_chunk_bytes
                        ):
                            break
                        result.append(
                            parse_message(
                                {
                                    "protocol_version": "1",
                                    "type": "body.chunk",
                                    "flow_id": flow_id,
                                    "body_side": side,
                                    "chunk_index": chunk_index,
                                    "offset_bytes": offset_bytes,
                                    "data_base64": base64.b64encode(chunk_data).decode("ascii"),
                                }
                            )
                        )
                        if remaining_chunk_bytes is not None:
                            remaining_chunk_bytes -= len(chunk_data)
                if state != "missing":
                    if max_messages is not None and len(result) >= max_messages:
                        break
                    result.append(
                        parse_message(
                            {
                                "protocol_version": "1",
                                "type": "body.end",
                                "flow_id": flow_id,
                                "body_side": side,
                                "total_bytes": str(size),
                                "body": _descriptor_from_row(body, state, size, content_type),
                            }
                        )
                    )
        if selected_flow_ids:
            placeholders = ",".join("?" for _ in selected_flow_ids)
            lifecycle_rows = connection.execute(
                f"""
                SELECT source_id, event_id, occurred_at, sequence, state, flow_id
                FROM lifecycle
                WHERE flow_id IN ({placeholders})
                ORDER BY source_id, LENGTH(sequence), sequence, event_id
                """,
                tuple(selected_flow_ids),
            )
            for lifecycle in lifecycle_rows:
                if max_messages is not None and len(result) >= max_messages:
                    break
                result.append(
                    parse_message(
                        {
                            "protocol_version": "1",
                            "type": "flow.lifecycle",
                            "source_id": lifecycle[0],
                            "flow_id": lifecycle[5],
                            "event_id": lifecycle[1],
                            "occurred_at": lifecycle[2],
                            "sequence": lifecycle[3],
                            "state": lifecycle[4],
                        }
                    )
                )
        return result

    def replay_into(self, store: MemoryStore, limit: int = DEFAULT_STORAGE_REPLAY) -> None:
        """Materialize persisted history into a regular bounded memory store."""

        groups: dict[str, list[ParsedMessageResult]] = {}
        lifecycle_messages: list[ParsedMessageResult] = []
        for message in self.replay(limit):
            payload = (
                message.message if isinstance(message, KnownParsedMessage) else message.payload
            )
            flow_id = payload.get("flow_id")
            if payload.get("type") == "flow.metadata":
                metadata = payload.get("metadata")
                flow_id = metadata.get("flow_id") if isinstance(metadata, Mapping) else None
            if payload.get("type") == "flow.lifecycle":
                lifecycle_messages.append(message)
            elif isinstance(flow_id, str):
                groups.setdefault(flow_id, []).append(message)
        for messages in reversed(list(groups.values())):
            for message in messages:
                store.append(message)
        for message in lifecycle_messages:
            store.append(message)

    def search(
        self,
        query: str,
        limit: int,
        cancel: threading.Event | None = None,
    ) -> tuple[list[SearchMatch], bool]:
        """Search the durable text projection with bounded result materialization."""

        from mitm_inspector.api.projection import grid_flow

        if type(query) is not str or not query:
            raise ValueError("query must be a non-empty string")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be from 1 through 200")
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise SearchCancelled
        self.flush()
        folded_query = query.casefold()
        escaped_query = _escape_like(folded_query)
        with sqlite3.connect(self.path) as connection:
            connection.set_progress_handler(lambda: int(cancel.is_set()), 1_000)
            try:
                rows = connection.execute(
                    """
                    WITH candidates AS MATERIALIZED (
                        SELECT body_search.rowid, body_search.flow_id, body_search.field
                        FROM body_search_fts
                        JOIN body_search ON body_search.rowid = body_search_fts.rowid
                        JOIN flows ON flows.flow_id = body_search.flow_id
                        WHERE body_search_fts.folded_text LIKE ? ESCAPE '\\'
                        ORDER BY
                            CASE WHEN flows.started_at_sort IS NULL THEN 1 ELSE 0 END,
                            flows.started_at_sort DESC,
                            flows.created_order DESC,
                            CASE body_search.field WHEN 'request_body' THEN 0 ELSE 1 END
                        LIMIT ?
                    )
                    SELECT candidates.flow_id, candidates.field,
                           substr(
                               body_search.body_text,
                               max(1, instr(body_search.folded_text, ?) - 160),
                               320
                           ) AS body_window
                    FROM candidates
                    JOIN body_search ON body_search.rowid = candidates.rowid
                    JOIN flows ON flows.flow_id = candidates.flow_id
                    ORDER BY
                        CASE WHEN flows.started_at_sort IS NULL THEN 1 ELSE 0 END,
                        flows.started_at_sort DESC,
                        flows.created_order DESC,
                        CASE candidates.field WHEN 'request_body' THEN 0 ELSE 1 END
                    """,
                    (f"%{escaped_query}%", limit + 1, folded_query),
                ).fetchall()
            except sqlite3.OperationalError as error:
                if cancel.is_set():
                    raise SearchCancelled from error
                raise
            finally:
                connection.set_progress_handler(None, 0)
            truncated = len(rows) > limit
            projection_cache: dict[str, PlainJsonObject] = {}
            matches: list[SearchMatch] = []
            for flow_id_value, field_value, body_window in rows[:limit]:
                if cancel.is_set():
                    raise SearchCancelled
                flow_id = str(flow_id_value)
                flow = projection_cache.get(flow_id)
                if flow is None:
                    metadata_row = connection.execute(
                        """
                        SELECT flow_id, source_id, method, scheme, host, port, path,
                               response_status, request_content_type,
                               response_content_type, started_at, ended_at,
                               request_body, response_body, request_body_state,
                               response_body_state, request_body_size,
                               response_body_size, request_headers_json,
                               response_headers_json, session_id
                        FROM flows WHERE flow_id = ?
                        """,
                        (flow_id,),
                    ).fetchone()
                    if metadata_row is None:
                        continue
                    message = self._metadata_from_row(metadata_row)
                    metadata = message.get("metadata")
                    if not isinstance(metadata, Mapping):
                        continue
                    projected_flow = grid_flow(metadata)
                    projection_cache[flow_id] = projected_flow
                    flow = projected_flow
                if flow is None:  # pragma: no cover - guarded by construction.
                    continue
                window = str(body_window)
                snippet = _snippet_from_window(window, query)
                matches.append(
                    {
                        "flow_id": flow_id,
                        "field": str(field_value),
                        "snippet": snippet,
                        "flow": flow,
                    }
                )
        return matches, truncated

    def _backfill_started_sort_keys(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT flow_id, started_at FROM flows WHERE started_at IS NOT NULL"
        ).fetchall()
        connection.executemany(
            "UPDATE flows SET started_at_sort = ? WHERE flow_id = ?",
            [(_timestamp_sort_key(str(started_at)), str(flow_id)) for flow_id, started_at in rows],
        )

    def _rebuild_search_index(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM body_search")
        connection.execute("INSERT INTO body_search_fts(body_search_fts) VALUES ('rebuild')")
        flow_ids = connection.execute(
            "SELECT flow_id FROM flows WHERE method <> '' ORDER BY created_order"
        ).fetchall()
        for (flow_id,) in flow_ids:
            self._refresh_search_flow(connection, str(flow_id))

    def _refresh_search_flow(self, connection: sqlite3.Connection, flow_id: str) -> None:
        from mitm_inspector.api.bodies import decoded_body_bytes, header_value

        row = connection.execute(
            """
            SELECT request_body, response_body, request_body_state,
                   response_body_state, request_body_size, response_body_size,
                   request_content_type, response_content_type,
                   request_headers_json, response_headers_json
            FROM flows WHERE flow_id = ? AND method <> ''
            """,
            (flow_id,),
        ).fetchone()
        connection.execute("DELETE FROM body_search WHERE flow_id = ?", (flow_id,))
        if row is None:
            return
        request_headers = json.loads(str(row[8]))
        response_headers = json.loads(str(row[9])) if row[9] is not None else []
        for field, body, state, size, content_type, headers in (
            ("request_body", row[0], row[2], row[4], row[6], request_headers),
            ("response_body", row[1], row[3], row[5], row[7], response_headers),
        ):
            descriptor = _descriptor_from_row(body, state, size, content_type)
            encoding = header_value(headers, "content-encoding")
            normalized_encoding = encoding.strip().casefold() if isinstance(encoding, str) else None
            decoded, _was_decoded = decoded_body_bytes(descriptor, normalized_encoding)
            if decoded is None:
                continue
            try:
                text = decoded.decode("utf-8")
            except UnicodeDecodeError:
                continue
            connection.execute(
                """
                INSERT INTO body_search(flow_id, field, body_text, folded_text)
                VALUES (?, ?, ?, ?)
                """,
                (flow_id, field, text, text.casefold()),
            )

    def _prepare_database(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _validate_private_mode(self.path.parent, stat.S_IFDIR, 0o700)
        _prepare_private_file(self.path)
        _secure_existing_files(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS flows (
                    flow_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL DEFAULT '',
                    method TEXT NOT NULL DEFAULT '',
                    scheme TEXT NOT NULL DEFAULT 'https',
                    host TEXT NOT NULL DEFAULT '',
                    port TEXT NOT NULL DEFAULT '443',
                    path TEXT NOT NULL DEFAULT '/',
                    response_status TEXT,
                    request_content_type TEXT,
                    response_content_type TEXT,
                    started_at TEXT,
                    started_at_sort TEXT,
                    ended_at TEXT,
                    request_body BLOB,
                    response_body BLOB,
                    request_body_state TEXT NOT NULL DEFAULT 'missing',
                    response_body_state TEXT NOT NULL DEFAULT 'missing',
                    request_body_size INTEGER NOT NULL DEFAULT 0,
                    response_body_size INTEGER NOT NULL DEFAULT 0,
                    request_headers_json TEXT NOT NULL DEFAULT '[]',
                    response_headers_json TEXT,
                    created_order INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS lifecycle (
                    flow_id TEXT NOT NULL,
                    event_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    FOREIGN KEY (flow_id) REFERENCES flows(flow_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS body_chunks (
                    flow_id TEXT NOT NULL,
                    body_side TEXT NOT NULL,
                    chunk_index TEXT NOT NULL,
                    offset_bytes TEXT NOT NULL,
                    data BLOB NOT NULL,
                    PRIMARY KEY (flow_id, body_side, chunk_index),
                    FOREIGN KEY (flow_id) REFERENCES flows(flow_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS body_search (
                    flow_id TEXT NOT NULL,
                    field TEXT NOT NULL,
                    body_text TEXT NOT NULL,
                    folded_text TEXT NOT NULL,
                    PRIMARY KEY (flow_id, field),
                    FOREIGN KEY (flow_id) REFERENCES flows(flow_id) ON DELETE CASCADE
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS body_search_fts USING fts5(
                    folded_text,
                    content='body_search',
                    content_rowid='rowid',
                    tokenize='trigram'
                );
                CREATE TRIGGER IF NOT EXISTS body_search_ai AFTER INSERT ON body_search BEGIN
                    INSERT INTO body_search_fts(rowid, folded_text)
                    VALUES (new.rowid, new.folded_text);
                END;
                CREATE TRIGGER IF NOT EXISTS body_search_ad AFTER DELETE ON body_search BEGIN
                    INSERT INTO body_search_fts(body_search_fts, rowid, folded_text)
                    VALUES ('delete', old.rowid, old.folded_text);
                END;
                CREATE TRIGGER IF NOT EXISTS body_search_au AFTER UPDATE ON body_search BEGIN
                    INSERT INTO body_search_fts(body_search_fts, rowid, folded_text)
                    VALUES ('delete', old.rowid, old.folded_text);
                    INSERT INTO body_search_fts(rowid, folded_text)
                    VALUES (new.rowid, new.folded_text);
                END;
                CREATE TABLE IF NOT EXISTS storage_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS lifecycle_flow_sequence
                    ON lifecycle(flow_id, sequence);
                CREATE INDEX IF NOT EXISTS flows_created_order
                    ON flows(created_order);
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(flows)").fetchall()
            }
            if "session_id" not in columns:
                connection.execute("ALTER TABLE flows ADD COLUMN session_id TEXT")
            if "started_at_sort" not in columns:
                connection.execute("ALTER TABLE flows ADD COLUMN started_at_sort TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS flows_started_sort_desc ON flows(started_at_sort DESC)"
            )
            self._backfill_started_sort_keys(connection)
            version = connection.execute(
                "SELECT value FROM storage_meta WHERE key = 'body_search_version'"
            ).fetchone()
            if version != ("1",):
                self._rebuild_search_index(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO storage_meta(key, value) VALUES (?, ?)",
                    ("body_search_version", "1"),
                )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")
            _checkpoint(connection)
            connection.execute("SELECT COUNT(*) FROM flows").fetchone()
            self._minimum_storage_bytes = self._measure_empty_live_overhead(connection)
            if self.max_bytes < self._minimum_storage_bytes:
                raise ValueError(
                    f"storage max_bytes {self.max_bytes} is below the SQLite overhead "
                    f"of {self._minimum_storage_bytes} bytes"
                )
            self._enforce_retention(connection, None)
            _checkpoint(connection)
            connection.execute("SELECT COUNT(*) FROM flows").fetchone()
        _secure_existing_files(self.path)

    def _measure_empty_live_overhead(self, source: sqlite3.Connection) -> int:
        descriptor, raw_path = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.baseline-",
            suffix=".sqlite",
        )
        os.close(descriptor)
        baseline_path = Path(raw_path)
        try:
            with sqlite3.connect(baseline_path) as baseline:
                baseline.execute("PRAGMA foreign_keys = ON")
                baseline.execute("PRAGMA journal_mode = WAL")
                schema = source.execute(
                    """
                    SELECT sql FROM sqlite_master
                    WHERE sql IS NOT NULL
                      AND name NOT LIKE 'sqlite_%'
                      AND name NOT LIKE 'body_search_fts_%'
                    ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
                    """
                ).fetchall()
                baseline.executescript("\n".join(f"{statement[0]};" for statement in schema))
                baseline.commit()
                baseline.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                baseline.execute("VACUUM")
                baseline.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                baseline.execute("SELECT COUNT(*) FROM flows").fetchone()
                return _storage_size(baseline_path)
        finally:
            baseline_path.unlink(missing_ok=True)
            for sidecar in _sidecar_paths(baseline_path):
                sidecar.unlink(missing_ok=True)

    def _run(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            if self._trace_sql is not None:
                connection.set_trace_callback(self._trace_sql)
            row = connection.execute("SELECT COALESCE(MAX(created_order), 0) FROM flows").fetchone()
            self._next_created_order = int(row[0]) if row is not None else 0
            while True:
                item = self._queue.get()
                try:
                    if item is _SENTINEL:
                        return
                    self._write(connection, item)
                except Exception:
                    with self._lock:
                        self._write_errors += 1
                finally:
                    self._queue.task_done()

    def _write(self, connection: sqlite3.Connection, item: object) -> None:
        if not isinstance(item, ParsedMessage):  # pragma: no cover - internal invariant.
            raise TypeError("storage queue item is not a parsed message")
        parsed = require_parsed_message(item)
        payload = parsed.message if isinstance(parsed, KnownParsedMessage) else parsed.payload
        message_type = payload.get("type")
        with connection:
            if message_type == "flow.metadata":
                metadata = payload.get("metadata")
                if isinstance(metadata, Mapping):
                    self._write_metadata(connection, metadata)
            elif message_type == "flow.lifecycle":
                self._write_lifecycle(connection, payload)
            elif message_type == "body.chunk":
                self._write_body_chunk(connection, payload)
            elif message_type == "body.end":
                self._write_body_end(connection, payload)
            protected_flow_id = _message_flow_id(payload)
            self._enforce_retention(connection, protected_flow_id)

    def _write_metadata(
        self, connection: sqlite3.Connection, metadata: Mapping[str, object]
    ) -> None:
        flow_id = str(metadata["flow_id"])
        request = _body_values(metadata.get("request_body"))
        response = _body_values(metadata.get("response_body"))
        request_headers = json.dumps(
            _plain_json(metadata["request_headers"]), separators=(",", ":")
        )
        response_headers = (
            json.dumps(_plain_json(metadata["response_headers"]), separators=(",", ":"))
            if "response_headers" in metadata
            else None
        )
        metadata_started_at = metadata.get("started_at")
        if not isinstance(metadata_started_at, str):
            metadata_started_at = None
        metadata_ended_at = metadata.get("ended_at")
        if not isinstance(metadata_ended_at, str):
            metadata_ended_at = None
        connection.execute(
            """
            INSERT INTO flows (
                flow_id, method, scheme, host, port, path, response_status,
                request_content_type, response_content_type, request_body,
                response_body, request_body_state, response_body_state,
                request_body_size, response_body_size, request_headers_json,
                response_headers_json, started_at, started_at_sort, ended_at,
                created_order, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(flow_id) DO UPDATE SET
                method = excluded.method,
                scheme = excluded.scheme,
                host = excluded.host,
                port = excluded.port,
                path = excluded.path,
                response_status = COALESCE(excluded.response_status, flows.response_status),
                request_content_type = excluded.request_content_type,
                response_content_type = COALESCE(
                    excluded.response_content_type, flows.response_content_type
                ),
                request_body = excluded.request_body,
                response_body = CASE
                    WHEN excluded.response_body_state = 'missing' THEN flows.response_body
                    ELSE excluded.response_body
                END,
                request_body_state = excluded.request_body_state,
                response_body_state = CASE
                    WHEN excluded.response_body_state = 'missing' THEN flows.response_body_state
                    ELSE excluded.response_body_state
                END,
                request_body_size = excluded.request_body_size,
                response_body_size = CASE
                    WHEN excluded.response_body_state = 'missing' THEN flows.response_body_size
                    ELSE excluded.response_body_size
                END,
                request_headers_json = excluded.request_headers_json,
                response_headers_json = COALESCE(
                    excluded.response_headers_json, flows.response_headers_json
                ),
                started_at = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM lifecycle
                        WHERE flow_id = excluded.flow_id AND state = 'request_started'
                    ) THEN flows.started_at
                    ELSE excluded.started_at
                END,
                started_at_sort = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM lifecycle
                        WHERE flow_id = excluded.flow_id AND state = 'request_started'
                    ) THEN flows.started_at_sort
                    ELSE excluded.started_at_sort
                END,
                ended_at = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM lifecycle
                        WHERE flow_id = excluded.flow_id
                          AND state IN ('flow_completed', 'error')
                    ) THEN flows.ended_at
                    ELSE excluded.ended_at
                END,
                session_id = COALESCE(excluded.session_id, flows.session_id),
                created_order = CASE
                    WHEN flows.created_order = 0 THEN excluded.created_order
                    ELSE flows.created_order
                END
            """,
            (
                flow_id,
                metadata["method"],
                metadata["scheme"],
                metadata["host"],
                metadata["port"],
                metadata["path"],
                metadata.get("response_status"),
                request[3],
                response[3],
                sqlite3.Binary(request[0]) if request[0] is not None else None,
                sqlite3.Binary(response[0]) if response[0] is not None else None,
                request[1],
                response[1],
                request[2],
                response[2],
                request_headers,
                response_headers,
                metadata_started_at,
                _timestamp_sort_key(metadata_started_at)
                if metadata_started_at is not None
                else None,
                metadata_ended_at,
                self._next_order(),
                metadata.get("session_id"),
            ),
        )
        self._refresh_search_flow(connection, flow_id)

    def _write_lifecycle(
        self, connection: sqlite3.Connection, payload: Mapping[str, object]
    ) -> None:
        flow_id = str(payload["flow_id"])
        self._ensure_flow(connection, flow_id)
        connection.execute(
            """
            INSERT OR IGNORE INTO lifecycle
                (flow_id, event_id, source_id, state, occurred_at, sequence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                payload["event_id"],
                payload["source_id"],
                payload["state"],
                payload["occurred_at"],
                payload["sequence"],
            ),
        )
        connection.execute(
            """
            UPDATE flows SET source_id = CASE WHEN source_id = '' THEN ? ELSE source_id END
            WHERE flow_id = ?
            """,
            (payload["source_id"], flow_id),
        )
        timing_rows = connection.execute(
            """
            SELECT
                (
                    SELECT occurred_at FROM lifecycle
                    WHERE flow_id = ? AND state = 'request_started'
                    ORDER BY LENGTH(sequence), sequence,
                             occurred_at DESC, source_id DESC, event_id DESC
                    LIMIT 1
                ),
                (
                    SELECT occurred_at FROM lifecycle
                    WHERE flow_id = ? AND state IN ('flow_completed', 'error')
                    ORDER BY LENGTH(sequence) DESC, sequence DESC,
                             occurred_at DESC, source_id DESC, event_id DESC
                    LIMIT 1
                )
            """,
            (flow_id, flow_id),
        ).fetchone()
        started_at = timing_rows[0] if timing_rows is not None else None
        ended_at = timing_rows[1] if timing_rows is not None else None
        connection.execute(
            """
            UPDATE flows SET started_at = COALESCE(?, started_at),
                             started_at_sort = COALESCE(?, started_at_sort),
                             ended_at = COALESCE(?, ended_at)
            WHERE flow_id = ?
            """,
            (
                started_at,
                _timestamp_sort_key(started_at) if isinstance(started_at, str) else None,
                ended_at,
                flow_id,
            ),
        )

    def _write_body_chunk(
        self, connection: sqlite3.Connection, payload: Mapping[str, object]
    ) -> None:
        flow_id = str(payload["flow_id"])
        self._ensure_flow(connection, flow_id)
        data = base64.b64decode(str(payload["data_base64"]), validate=True)
        connection.execute(
            """
            INSERT OR REPLACE INTO body_chunks
                (flow_id, body_side, chunk_index, offset_bytes, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                flow_id,
                payload["body_side"],
                payload["chunk_index"],
                payload["offset_bytes"],
                sqlite3.Binary(data),
            ),
        )

    def _write_body_end(
        self, connection: sqlite3.Connection, payload: Mapping[str, object]
    ) -> None:
        flow_id = str(payload["flow_id"])
        self._ensure_flow(connection, flow_id)
        values = _body_values(payload["body"])
        side = str(payload["body_side"])
        column = "request" if side == "request" else "response"
        connection.execute(
            f"""
            UPDATE flows SET {column}_body = ?, {column}_body_state = ?,
                {column}_body_size = ?, {column}_content_type = ?
            WHERE flow_id = ?
            """,
            (
                sqlite3.Binary(values[0]) if values[0] is not None else None,
                values[1],
                int(str(payload["total_bytes"])),
                values[3],
                flow_id,
            ),
        )
        self._refresh_search_flow(connection, flow_id)

    @staticmethod
    def _ensure_flow(connection: sqlite3.Connection, flow_id: str) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO flows (flow_id, created_order) VALUES (?, 0)",
            (flow_id,),
        )

    def _next_order(self) -> int:
        self._next_created_order += 1
        return self._next_created_order

    def _enforce_retention(
        self, connection: sqlite3.Connection, protected_flow_id: str | None
    ) -> None:
        connection.commit()
        while True:
            if protected_flow_id is None:
                overflow_offset = self.max_flows
                where = ""
                parameters: tuple[object, ...] = ()
            else:
                overflow_offset = max(0, self.max_flows - 1)
                where = "WHERE flow_id <> ?"
                parameters = (protected_flow_id,)
            too_many = connection.execute(
                f"""
                SELECT flow_id FROM flows
                {where}
                ORDER BY CASE WHEN started_at_sort IS NULL THEN 1 ELSE 0 END,
                         started_at_sort, created_order
                LIMIT 1 OFFSET ?
                """,
                (*parameters, overflow_offset),
            ).fetchone()
            over_bytes = _storage_size(self.path) > self.max_bytes
            if too_many is None and not over_bytes:
                return
            oldest = connection.execute(
                f"""
                SELECT flow_id FROM flows
                {where}
                ORDER BY CASE WHEN started_at_sort IS NULL THEN 1 ELSE 0 END,
                         started_at_sort, created_order
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if oldest is None:
                if protected_flow_id is not None and over_bytes:
                    connection.execute("DELETE FROM flows WHERE flow_id = ?", (protected_flow_id,))
                    self._compact_search_index(connection)
                    connection.commit()
                    _checkpoint(connection)
                    connection.execute("VACUUM")
                    _checkpoint(connection)
                    _secure_existing_files(self.path)
                    protected_flow_id = None
                    continue
                return
            connection.execute("DELETE FROM flows WHERE flow_id = ?", (oldest[0],))
            self._compact_search_index(connection)
            connection.commit()
            _checkpoint(connection)
            connection.execute("VACUUM")
            _checkpoint(connection)
            _secure_existing_files(self.path)

    @staticmethod
    def _compact_search_index(connection: sqlite3.Connection) -> None:
        remaining = connection.execute("SELECT 1 FROM body_search LIMIT 1").fetchone()
        command = "rebuild" if remaining is None else "optimize"
        connection.execute("INSERT INTO body_search_fts(body_search_fts) VALUES (?)", (command,))

    @staticmethod
    def _metadata_from_row(row: tuple[object, ...]) -> dict[str, object]:
        request_headers = json.loads(str(row[18]))
        response_headers = json.loads(str(row[19])) if row[19] is not None else None
        metadata: dict[str, object] = {
            "flow_id": row[0],
            "session_id": row[20],
            "method": row[2],
            "scheme": row[3],
            "host": row[4],
            "port": row[5],
            "path": row[6],
            "request_headers": request_headers,
            "request_body": _descriptor_from_row(row[12], row[14], row[16], row[8]),
        }
        if row[19] is not None:
            metadata["response_headers"] = response_headers
        if row[7] is not None:
            metadata["response_status"] = row[7]
        if row[15] != "missing" or row[13] is not None or row[19] is not None:
            metadata["response_body"] = _descriptor_from_row(row[13], row[15], row[17], row[9])
        if isinstance(row[10], str) and is_rfc3339_utc(row[10]):
            metadata["started_at"] = row[10]
        if isinstance(row[11], str) and is_rfc3339_utc(row[11]):
            metadata["ended_at"] = row[11]
        if row[14] != "missing":
            metadata["request_body_size"] = str(row[16])
        if row[15] != "missing":
            metadata["response_body_size"] = str(row[17])
        if row[8] is not None:
            metadata["request_content_type"] = row[8]
        if row[9] is not None:
            metadata["response_content_type"] = row[9]
        from mitm_inspector.api.projection import canonical_grid_flow

        canonical_metadata = canonical_grid_flow(metadata)
        return {
            "protocol_version": "1",
            "type": "flow.metadata",
            "metadata": canonical_metadata,
        }


def _detail_body_allocations(
    request_length: int,
    response_length: int,
    limit: int,
) -> tuple[int, int]:
    """Split one retained-byte ceiling fairly, then reuse unneeded capacity."""

    half = limit // 2
    request_bytes = min(request_length, half)
    response_bytes = min(response_length, limit - half)
    remaining = limit - request_bytes - response_bytes
    request_extra = min(request_length - request_bytes, remaining)
    request_bytes += request_extra
    remaining -= request_extra
    response_bytes += min(response_length - response_bytes, remaining)
    return request_bytes, response_bytes


def _body_values(value: object) -> tuple[bytes | None, str, int, str | None]:
    if not isinstance(value, Mapping):
        return None, "missing", 0, None
    state = str(value["state"])
    size = int(str(value.get("size_bytes", "0")))
    content_type = value.get("content_type")
    data = (
        base64.b64decode(str(value["data"]), validate=True)
        if state in {"captured", "truncated"}
        else None
    )
    return data, state, size, str(content_type) if content_type is not None else None


def _message_flow_id(payload: Mapping[str, object]) -> str | None:
    if payload.get("type") == "flow.metadata":
        metadata = payload.get("metadata")
        flow_id = metadata.get("flow_id") if isinstance(metadata, Mapping) else None
    else:
        flow_id = payload.get("flow_id")
    return flow_id if isinstance(flow_id, str) else None


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_json(item) for item in value]
    return value


def _descriptor_from_row(
    body: object,
    state: object,
    size: object,
    content_type: object,
) -> dict[str, object]:
    body_state = str(state)
    descriptor: dict[str, object] = {"state": body_state}
    if content_type is not None:
        descriptor["content_type"] = str(content_type)
    if body_state == "missing":
        return descriptor
    descriptor["size_bytes"] = str(size)
    if body_state == "empty":
        return descriptor
    data = bytes(body) if isinstance(body, bytes | bytearray) else b""
    descriptor["encoding"] = "base64"
    descriptor["data"] = base64.b64encode(data).decode("ascii")
    if body_state == "truncated":
        descriptor["captured_bytes"] = str(len(data))
    return descriptor


def _sidecar_paths(path: Path) -> tuple[Path, Path]:
    return (Path(f"{path}-wal"), Path(f"{path}-shm"))


def _storage_size(path: Path) -> int:
    total = 0
    for candidate in (path, *_sidecar_paths(path)):
        try:
            total += candidate.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _centered_snippet(text: str, match_start: int, match_length: int) -> str:
    if len(text) <= 160:
        return text
    match_end = min(len(text), match_start + match_length)
    spare = max(0, 160 - (match_end - match_start))
    start = max(0, match_start - spare // 2)
    end = min(len(text), start + 160)
    start = max(0, end - 160)
    return text[start:end]


def _snippet_from_window(window: str, query: str) -> str:
    collapsed = _WHITESPACE.sub(" ", window).strip()
    collapsed_query = _WHITESPACE.sub(" ", query).strip()
    match_start = collapsed.casefold().find(collapsed_query.casefold())
    if match_start < 0:
        match_start = min(len(collapsed), 160)
    return _centered_snippet(collapsed, match_start, len(collapsed_query))


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _timestamp_sort_key(value: str) -> str | None:
    if not is_rfc3339_utc(value):
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    time_part = value.split("T", 1)[1]
    fraction = ""
    if "." in time_part:
        fraction = time_part.split(".", 1)[1].split("Z", 1)[0].split("+", 1)[0]
    return f"{parsed:%Y%m%d%H%M%S}|{fraction.rstrip('0')}"


def _checkpoint(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _validate_private_mode(path: Path, expected_type: int, mode: int) -> None:
    info = path.lstat()
    expected = (
        stat.S_ISDIR(info.st_mode) if expected_type == stat.S_IFDIR else stat.S_ISREG(info.st_mode)
    )
    if stat.S_ISLNK(info.st_mode) or not expected:
        raise PermissionError(f"storage path has an unsafe type: {path}")
    if stat.S_IMODE(info.st_mode) != mode:
        raise PermissionError(f"storage path has unsafe permissions: {path}")


def _prepare_private_file(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        descriptor = os.open(path, flags, 0o600)
        os.close(descriptor)


def _secure_existing_files(path: Path) -> None:
    _validate_private_mode(path, stat.S_IFREG, 0o600)
    for sidecar in _sidecar_paths(path):
        if sidecar.exists():
            _validate_private_mode(sidecar, stat.S_IFREG, 0o600)


__all__ = [
    "DEFAULT_STORAGE_MAX_BYTES",
    "DEFAULT_STORAGE_MAX_FLOWS",
    "DEFAULT_STORAGE_QUEUE_SIZE",
    "DEFAULT_STORAGE_REPLAY",
    "SearchCancelled",
    "SearchMatch",
    "SQLiteFlowStorage",
    "default_storage_path",
]
