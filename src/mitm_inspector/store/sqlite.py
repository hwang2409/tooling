"""Asynchronous durable flow persistence backed by the stdlib sqlite3 module."""

from __future__ import annotations

import base64
import json
import os
import queue
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mitm_inspector.protocol import (
    KnownParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    parse_message,
    require_parsed_message,
)

if TYPE_CHECKING:
    from mitm_inspector.store.memory import MemoryStore

DEFAULT_STORAGE_MAX_FLOWS = 10_000
DEFAULT_STORAGE_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_STORAGE_REPLAY = 500
DEFAULT_STORAGE_QUEUE_SIZE = 4_096
_SENTINEL = object()


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
                       flows.response_headers_json
                FROM flows
                WHERE method <> ''
                ORDER BY CASE WHEN flows.started_at IS NULL THEN 1 ELSE 0 END,
                         LENGTH(COALESCE(flows.started_at, '')) DESC,
                         COALESCE(flows.started_at, '') DESC,
                         flows.created_order DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            result: list[ParsedMessageResult] = []
            selected_flow_ids = {str(row[0]) for row in rows}
            for row in rows:
                flow_id = str(row[0])
                result.append(parse_message(self._metadata_from_row(row)))
                for side, body, state, size, content_type in (
                    ("request", row[12], row[14], row[16], row[8]),
                    ("response", row[13], row[15], row[17], row[9]),
                ):
                    chunk_rows = connection.execute(
                        """
                        SELECT chunk_index, offset_bytes, data
                        FROM body_chunks
                        WHERE flow_id = ? AND body_side = ?
                        ORDER BY LENGTH(offset_bytes), offset_bytes, chunk_index
                        """,
                        (flow_id, side),
                    ).fetchall()
                    for chunk_index, offset_bytes, chunk_data in chunk_rows:
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
                    if state != "missing":
                        result.append(
                            parse_message(
                                {
                                    "protocol_version": "1",
                                    "type": "body.end",
                                    "flow_id": flow_id,
                                    "body_side": side,
                                    "total_bytes": str(size),
                                    "body": _descriptor_from_row(
                                        body, state, size, content_type
                                    ),
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
                ).fetchall()
                for lifecycle in lifecycle_rows:
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
                    ended_at TEXT,
                    request_body BLOB,
                    response_body BLOB,
                    request_body_state TEXT NOT NULL DEFAULT 'missing',
                    response_body_state TEXT NOT NULL DEFAULT 'missing',
                    request_body_size INTEGER NOT NULL DEFAULT 0,
                    response_body_size INTEGER NOT NULL DEFAULT 0,
                    request_headers_json TEXT NOT NULL DEFAULT '[]',
                    response_headers_json TEXT,
                    created_order INTEGER NOT NULL DEFAULT 0
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
                CREATE INDEX IF NOT EXISTS flows_started_at_desc
                    ON flows(started_at DESC);
                CREATE INDEX IF NOT EXISTS lifecycle_flow_sequence
                    ON lifecycle(flow_id, sequence);
                CREATE INDEX IF NOT EXISTS flows_created_order
                    ON flows(created_order);
                """
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")
            _checkpoint(connection)
            self._enforce_retention(connection, None)
            _checkpoint(connection)
            connection.execute("SELECT COUNT(*) FROM flows").fetchone()
            self._minimum_storage_bytes = self._measure_empty_live_overhead(connection)
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
                    WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
                    ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
                    """
                ).fetchall()
                baseline.executescript(
                    "\n".join(f"{statement[0]};" for statement in schema)
                )
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
            row = connection.execute(
                "SELECT COALESCE(MAX(created_order), 0) FROM flows"
            ).fetchone()
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
            json.dumps(
                _plain_json(metadata["response_headers"]), separators=(",", ":")
            )
            if "response_headers" in metadata
            else None
        )
        connection.execute(
            """
            INSERT INTO flows (
                flow_id, method, scheme, host, port, path, response_status,
                request_content_type, response_content_type, request_body,
                response_body, request_body_state, response_body_state,
                request_body_size, response_body_size, request_headers_json,
                response_headers_json, created_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                self._next_order(),
            ),
        )

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
            UPDATE flows
            SET source_id = CASE WHEN source_id = '' THEN ? ELSE source_id END,
                started_at = CASE WHEN started_at IS NULL THEN ? ELSE started_at END,
                ended_at = CASE WHEN ? IN ('flow_completed', 'error') THEN ? ELSE ended_at END
            WHERE flow_id = ?
            """,
            (
                payload["source_id"],
                payload["occurred_at"],
                payload["state"],
                payload["occurred_at"],
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
                ORDER BY CASE WHEN started_at IS NULL THEN 1 ELSE 0 END,
                         COALESCE(started_at, ''), created_order
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
                ORDER BY CASE WHEN started_at IS NULL THEN 1 ELSE 0 END,
                         COALESCE(started_at, ''), created_order
                LIMIT 1
                """, parameters
            ).fetchone()
            if oldest is None:
                if protected_flow_id is not None and over_bytes:
                    connection.execute(
                        "DELETE FROM flows WHERE flow_id = ?", (protected_flow_id,)
                    )
                    connection.commit()
                    _checkpoint(connection)
                    connection.execute("VACUUM")
                    _checkpoint(connection)
                    _secure_existing_files(self.path)
                    protected_flow_id = None
                    continue
                return
            connection.execute("DELETE FROM flows WHERE flow_id = ?", (oldest[0],))
            connection.commit()
            _checkpoint(connection)
            connection.execute("VACUUM")
            _checkpoint(connection)
            _secure_existing_files(self.path)

    @staticmethod
    def _metadata_from_row(row: tuple[object, ...]) -> dict[str, object]:
        request_headers = json.loads(str(row[18]))
        response_headers = json.loads(str(row[19])) if row[19] is not None else None
        metadata: dict[str, object] = {
            "flow_id": row[0],
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
            metadata["response_body"] = _descriptor_from_row(
                row[13], row[15], row[17], row[9]
            )
        return {"protocol_version": "1", "type": "flow.metadata", "metadata": metadata}


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


def _checkpoint(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _validate_private_mode(path: Path, expected_type: int, mode: int) -> None:
    info = path.lstat()
    expected = (
        stat.S_ISDIR(info.st_mode)
        if expected_type == stat.S_IFDIR
        else stat.S_ISREG(info.st_mode)
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
    "SQLiteFlowStorage",
    "default_storage_path",
]
