"""Loopback asyncio HTTP server and capture ingest listener.

``python -m mitm_inspector.api.server`` accepts exactly the argv contract
reserved by the runtime boundary.  The HTTP surface binds a loopback TCP
address; the capture ingest surface binds the private Unix socket allocated
by the runtime.  All protocol logic lives in :mod:`mitm_inspector.api.app`.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import re
import signal
import stat
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from mitm_inspector.api.app import ApiApplication, FlowDetailTooLarge
from mitm_inspector.api.httpwire import (
    MAX_REQUEST_HEAD_BYTES,
    HttpWireError,
    is_loopback_host_header,
    parse_request_head,
)
from mitm_inspector.api.limits import (
    MAX_INGEST_BODY_PREFIX_BYTES,
    MAX_INGEST_LINE_BYTES,
)
from mitm_inspector.protocol import MAX_U64
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import (
    DEFAULT_STORAGE_MAX_BYTES,
    DEFAULT_STORAGE_MAX_FLOWS,
    DEFAULT_STORAGE_REPLAY,
    SearchCancelled,
    SearchMatch,
    SQLiteFlowStorage,
    default_storage_path,
)

API_VERSION_PREFIX = "/api/v1"
HEALTH_PATH = f"{API_VERSION_PREFIX}/health"
COUNTERS_PATH = f"{API_VERSION_PREFIX}/counters"
SESSIONS_PATH = f"{API_VERSION_PREFIX}/sessions"
SESSION_PATH_PREFIX = f"{SESSIONS_PATH}/"
SEARCH_PATH = f"{API_VERSION_PREFIX}/search"
FLOW_PATH_PREFIX = f"{API_VERSION_PREFIX}/flows/"
HEAD_READ_TIMEOUT_SECONDS = 10.0
DEFAULT_SWEEP_INTERVAL_SECONDS = 5.0

_FLOW_ID_SEGMENT = re.compile(r"^[A-Za-z0-9._:@-]{1,256}$")
_LOOPBACK_NAMES = frozenset({"localhost"})

_INDEX_BODY = (
    b"<!doctype html><title>mitm-inspector</title>"
    b"<p>mitm-inspector local API. The browser UI reads captured sessions "
    b"over HTTP.</p>"
)


class ApiServerError(RuntimeError):
    """Raised when the API server cannot be configured or started."""


def _validate_loopback_host(value: str) -> str:
    if type(value) is not str or not value:
        raise ApiServerError("host must be a loopback address")
    if value.lower() in _LOOPBACK_NAMES:
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise ApiServerError("host must be a loopback address") from error
    if not address.is_loopback:
        raise ApiServerError("host must be a loopback address")
    return value


def _validate_u64(value: int, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum or value > MAX_U64:
        raise ApiServerError(f"{name} must be a bounded unsigned 64-bit integer")
    return value


@dataclass(frozen=True, slots=True)
class ApiServerConfig:
    """Validated app-server settings mirroring the reserved argv contract."""

    host: str = "127.0.0.1"
    port: int = 8000
    proxy_port: int | None = None
    source_id: str = "mitm-inspector"
    max_retained_flows: int = 2_000
    retention_seconds: int = 1_800
    max_body_bytes: int = 128 * 1024 * 1024
    # Default to the full wire ceiling so bodies are not truncated in the UI
    # unless the operator explicitly asks for a smaller cap. F5 shipped a 1
    # MiB default that was quietly clipping typical Anthropic responses.
    max_body_prefix_bytes: int = MAX_INGEST_BODY_PREFIX_BYTES
    max_pending_messages: int = 4096
    capture_socket: Path | None = None
    sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS
    storage_path: Path | str | None = None
    no_storage: bool = False
    storage_max_flows: int = DEFAULT_STORAGE_MAX_FLOWS
    storage_max_bytes: int = DEFAULT_STORAGE_MAX_BYTES
    storage_replay: int = DEFAULT_STORAGE_REPLAY

    def __post_init__(self) -> None:
        _validate_loopback_host(self.host)
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise ApiServerError("port must be an integer from 0 through 65535")
        if self.proxy_port is not None and (
            type(self.proxy_port) is not int or not 1 <= self.proxy_port <= 65535
        ):
            raise ApiServerError("proxy port must be an integer from 1 through 65535")
        if type(self.source_id) is not str or not self.source_id:
            raise ApiServerError("source id must be a non-empty string")
        _validate_u64(self.max_retained_flows, "max_retained_flows")
        _validate_u64(self.retention_seconds, "retention_seconds")
        _validate_u64(self.max_body_bytes, "max_body_bytes")
        _validate_u64(self.max_body_prefix_bytes, "max_body_prefix_bytes", minimum=0)
        _validate_u64(self.max_pending_messages, "max_pending_messages")
        if self.max_body_prefix_bytes > self.max_body_bytes:
            raise ApiServerError("max_body_prefix_bytes cannot exceed max_body_bytes")
        if self.max_body_prefix_bytes > MAX_INGEST_BODY_PREFIX_BYTES:
            raise ApiServerError("max_body_prefix_bytes exceeds the bounded ingest line capacity")
        if self.capture_socket is not None:
            path = self.capture_socket
            if not isinstance(path, Path) or not path.is_absolute():
                raise ApiServerError("capture socket must be an absolute path")
        interval = self.sweep_interval_seconds
        if type(interval) not in {int, float} or not interval > 0:
            raise ApiServerError("sweep interval must be a positive number")
        if self.storage_path is not None and os.fspath(self.storage_path) != ":memory:":
            path = Path(self.storage_path).expanduser()
            if not path.is_absolute():
                raise ApiServerError("storage path must be absolute")
            object.__setattr__(self, "storage_path", path)
        if type(self.no_storage) is not bool:
            raise ApiServerError("no_storage must be a boolean")
        _validate_u64(self.storage_max_flows, "storage_max_flows")
        if type(self.storage_max_bytes) is not int or self.storage_max_bytes < 0:
            raise ApiServerError("storage_max_bytes must be a nonnegative integer")
        if type(self.storage_replay) is not int or self.storage_replay < 0:
            raise ApiServerError("storage_replay must be a nonnegative integer")


def _strict_json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON numbers are not accepted")


def parse_ingest_line(line: bytes) -> object:
    """Decode one strict JSONL ingest record."""

    text = line.decode("utf-8")
    return json.loads(
        text,
        object_pairs_hook=_strict_json_object_pairs,
        parse_constant=_reject_json_constant,
    )


class ApiServer:
    """Own the store, application, and the two asyncio listeners."""

    def __init__(
        self,
        config: ApiServerConfig,
        *,
        application: ApiApplication | None = None,
    ) -> None:
        self.config = config
        if application is None:
            store = MemoryStore(
                config.max_retained_flows,
                max_age_seconds=config.retention_seconds,
                max_body_bytes=config.max_body_bytes,
                max_memory_bytes=config.max_body_bytes,
            )
            storage = None
            if (
                not config.no_storage
                and config.storage_path is not None
                and os.fspath(config.storage_path) != ":memory:"
            ):
                storage = SQLiteFlowStorage(
                    config.storage_path,
                    max_flows=config.storage_max_flows,
                    max_bytes=config.storage_max_bytes,
                )
                storage.replay_into(store, config.storage_replay)
            application = ApiApplication(
                store,
                source_id=config.source_id,
                max_body_prefix_bytes=config.max_body_prefix_bytes,
                max_in_memory_bytes=config.max_body_bytes,
                storage=storage,
            )
        self.application = application
        self._storage = application.storage
        self._http_server: asyncio.Server | None = None
        self._ingest_server: asyncio.Server | None = None
        self._sweep_task: asyncio.Task[None] | None = None
        self._connections: set[asyncio.StreamWriter] = set()
        self._closing = False
        self._rejected_ingest_lines = 0
        self._ingest_connections = 0
        self._http_requests = 0
        self._sweep_failures = 0
        self._search_guard = asyncio.Lock()
        self._search_cancel: threading.Event | None = None
        self._durable_detail_guard = asyncio.Lock()

    @property
    def counters(self) -> dict[str, object]:
        counters = self.application.counters
        counters["rejected_ingest_lines"] = self._rejected_ingest_lines
        counters["ingest_connections"] = self._ingest_connections
        counters["http_requests"] = self._http_requests
        counters["sweep_failures"] = self._sweep_failures
        return counters

    @property
    def bound_port(self) -> int:
        if self._http_server is None or not self._http_server.sockets:
            raise ApiServerError("the HTTP server is not listening")
        port = self._http_server.sockets[0].getsockname()[1]
        if type(port) is not int:  # pragma: no cover - loopback TCP always has a port.
            raise ApiServerError("the HTTP server has no TCP port")
        return port

    async def start(self) -> None:
        self._closing = False
        if self._http_server is not None:
            raise ApiServerError("the API server is already started")
        self._http_server = await asyncio.start_server(
            self._handle_http_connection,
            host=self.config.host,
            port=self.config.port,
            limit=MAX_REQUEST_HEAD_BYTES,
        )
        if self.config.capture_socket is not None:
            path = self.config.capture_socket
            if os.path.lexists(path):
                info = os.lstat(path)
                if not stat.S_ISSOCK(info.st_mode):
                    await self._close_http()
                    raise ApiServerError(f"capture endpoint exists and is not a socket: {path}")
                path.unlink()
            try:
                self._ingest_server = await asyncio.start_unix_server(
                    self._handle_ingest_connection,
                    path=str(path),
                    limit=MAX_INGEST_LINE_BYTES,
                )
                os.chmod(path, 0o600)
            except BaseException:
                if self._ingest_server is not None:
                    self._ingest_server.close()
                    await self._ingest_server.wait_closed()
                    self._ingest_server = None
                await self._close_http()
                raise
        self._sweep_task = asyncio.get_running_loop().create_task(self._sweep_forever())

    async def serve_forever(self) -> None:
        if self._http_server is None:
            raise ApiServerError("the API server is not started")
        await self._http_server.serve_forever()

    async def close(self) -> None:
        self._closing = True
        if self._search_cancel is not None:
            self._search_cancel.set()
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None
        servers = [
            server for server in (self._ingest_server, self._http_server) if server is not None
        ]
        for server in servers:
            server.close()
        # Server.wait_closed() waits for every active connection handler on
        # Python 3.12.1+, so a connected HTTP or ingest producer
        # would stall shutdown forever.  Abort live transports and re-check:
        # a connection accepted in the same tick may have a handler that has
        # not run yet, so it is not in the snapshot; handlers observe
        # ``_closing`` on entry, and this loop aborts late registrations
        # until every handler has finished.
        for server in servers:
            while True:
                self._abort_connections()
                try:
                    await asyncio.wait_for(server.wait_closed(), 0.25)
                    break
                except TimeoutError:
                    continue
        self._ingest_server = None
        self._http_server = None
        if self._storage is not None:
            self._storage.close()
            self._storage = None

    def _abort_connections(self) -> None:
        for writer in list(self._connections):
            try:
                writer.transport.abort()
            except (ConnectionError, RuntimeError):
                pass

    async def _close_http(self) -> None:
        if self._http_server is not None:
            self._http_server.close()
            await self._http_server.wait_closed()
            self._http_server = None

    async def _sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(self.config.sweep_interval_seconds)
            try:
                self.application.sweep()
            except Exception:
                # A projection fault must not silently end idle-expiry
                # sweeps; the next tick retries and the counter records it.
                self._sweep_failures += 1

    # -- capture ingest ---------------------------------------------------

    async def _handle_ingest_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._connections.add(writer)
        self._ingest_connections += 1
        try:
            if self._closing:
                # This handler can start one tick after close() snapshotted
                # the connection set; never outlive an announced shutdown.
                return
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    self._rejected_ingest_lines += 1
                    break
                if not line:
                    break
                if not line.endswith(b"\n"):
                    self._rejected_ingest_lines += 1
                    break
                try:
                    value = parse_ingest_line(line)
                    self.application.ingest(value)
                except ValueError:
                    # Malformed input is a desync; fail closed by dropping
                    # the producer connection instead of resynchronizing on
                    # attacker-controlled bytes.
                    self._rejected_ingest_lines += 1
                    break
        finally:
            self._connections.discard(writer)
            await _close_writer(writer)

    # -- HTTP ---------------------------------------------------------------

    async def _handle_http_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._connections.add(writer)
        self._http_requests += 1
        try:
            if self._closing:
                # See the ingest handler: an accept can race close() by one
                # event-loop tick.
                return
            try:
                raw_head = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=HEAD_READ_TIMEOUT_SECONDS
                )
                head = parse_request_head(raw_head)
            except (
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                TimeoutError,
                HttpWireError,
                ValueError,
            ):
                await self._send_simple(writer, 400, b"malformed request")
                return
            try:
                host = head.header_values("host")
                if len(host) != 1 or not is_loopback_host_header(host[0]):
                    await self._send_simple(writer, 400, b"host must be loopback")
                    return
                if head.single_header("content-length") not in {
                    None,
                    "0",
                } or head.header_values("transfer-encoding"):
                    await self._send_simple(writer, 400, b"request bodies are not accepted")
                    return
                if head.method != "GET":
                    await self._send_simple(writer, 405, b"only GET is supported")
                    return
            except HttpWireError:
                await self._send_simple(writer, 400, b"malformed request")
                return
            await self._handle_plain_get(head.target, writer)
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            self._connections.discard(writer)
            await _close_writer(writer)

    async def _handle_plain_get(self, target: str, writer: asyncio.StreamWriter) -> None:
        split_target = urlsplit(target)
        path = split_target.path
        if path == "/":
            await self._send_response(writer, 200, "text/html; charset=utf-8", _INDEX_BODY)
            return
        if path == HEALTH_PATH:
            body = json.dumps(
                {"status": "ok", "protocol_version": "1", "counters": self.counters},
                separators=(",", ":"),
            ).encode("utf-8")
            await self._send_response(writer, 200, "application/json", body)
            return
        if path == COUNTERS_PATH:
            body = json.dumps(self.counters, separators=(",", ":")).encode("utf-8")
            await self._send_response(writer, 200, "application/json", body)
            return
        if path == SESSIONS_PATH:
            await self._handle_sessions(split_target.query, writer)
            return
        if path.startswith(SESSION_PATH_PREFIX):
            session_id = path[len(SESSION_PATH_PREFIX) :]
            if not _FLOW_ID_SEGMENT.fullmatch(session_id):
                await self._send_simple(writer, 404, b"unknown session")
                return
            await self._handle_session_detail(session_id, writer)
            return
        if path == SEARCH_PATH:
            try:
                parameters = parse_qs(split_target.query, keep_blank_values=True)
                query_values = parameters.get("q", [])
                limit_values = parameters.get("limit", [])
                if len(query_values) != 1 or not query_values[0] or len(limit_values) > 1:
                    raise ValueError
                limit = 50 if not limit_values else int(limit_values[0])
                if limit < 1:
                    raise ValueError
                limit = min(limit, 200)
            except ValueError:
                await self._send_simple(writer, 400, b"q and limit must be valid")
                return
            if self._storage is None:
                matches: list[SearchMatch] = []
                truncated = False
            else:
                cancel = threading.Event()
                previous = self._search_cancel
                self._search_cancel = cancel
                if previous is not None:
                    previous.set()
                async with self._search_guard:
                    if cancel.is_set():
                        return
                    try:
                        matches, truncated = await asyncio.to_thread(
                            self._storage.search, query_values[0], limit, cancel
                        )
                    except SearchCancelled:
                        return
                    finally:
                        if self._search_cancel is cancel:
                            self._search_cancel = None
            body = json.dumps(
                {"matches": matches, "truncated": truncated}, separators=(",", ":")
            ).encode("utf-8")
            await self._send_response(writer, 200, "application/json", body)
            return
        if path.startswith(FLOW_PATH_PREFIX):
            flow_id = path[len(FLOW_PATH_PREFIX) :]
            if not _FLOW_ID_SEGMENT.fullmatch(flow_id):
                await self._send_simple(writer, 404, b"unknown flow")
                return
            detail = self.application.flow_detail_text(flow_id)
            if detail is None and self._storage is not None:
                try:
                    async with self._durable_detail_guard:
                        durable_detail = await asyncio.to_thread(
                            self.application.durable_flow_detail_bytes, flow_id
                        )
                except FlowDetailTooLarge:
                    await self._send_simple(writer, 413, b"flow detail exceeds limit")
                    return
                if durable_detail is None:
                    await self._send_simple(writer, 404, b"unknown flow")
                    return
                await self._send_response(writer, 200, "application/json", durable_detail)
                return
            if detail is None:
                await self._send_simple(writer, 404, b"unknown flow")
                return
            await self._send_response(writer, 200, "application/json", detail.encode("utf-8"))
            return
        await self._send_simple(writer, 404, b"unknown path")

    async def _handle_sessions(self, query: str, writer: asyncio.StreamWriter) -> None:
        try:
            parameters = parse_qs(query, keep_blank_values=True)
            limit_values = parameters.get("limit", [])
            before_values = parameters.get("before", [])
            if len(limit_values) > 1 or len(before_values) > 1:
                raise ValueError
            limit = 100 if not limit_values else int(limit_values[0])
            before = before_values[0] if before_values else None
            if not 1 <= limit <= 200:
                raise ValueError
        except ValueError:
            await self._send_simple(writer, 400, b"limit and before must be valid")
            return
        if self._storage is None:
            payload = {"sessions": []}
        else:
            payload = {
                "sessions": await asyncio.to_thread(self._storage.session_summaries, limit, before)
            }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await self._send_response(writer, 200, "application/json", body)

    async def _handle_session_detail(self, session_id: str, writer: asyncio.StreamWriter) -> None:
        detail = (
            None
            if self._storage is None
            else await asyncio.to_thread(self._storage.session_detail, session_id)
        )
        if detail is None:
            await self._send_simple(writer, 404, b"unknown session")
            return
        body = json.dumps(detail, separators=(",", ":")).encode("utf-8")
        await self._send_response(writer, 200, "application/json", body)

    async def _send_simple(self, writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        await self._send_response(writer, status, "text/plain; charset=utf-8", body)

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        content_type: str,
        body: bytes,
    ) -> None:
        reasons = {
            200: "OK",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            413: "Payload Too Large",
        }
        reason = reasons.get(status, "Error")
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        try:
            writer.write(head + body)
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, RuntimeError):
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mitm-inspector-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--proxy-port", type=int, default=None)
    parser.add_argument("--max-retained-flows", type=int, default=2_000)
    parser.add_argument("--retention-seconds", type=int, default=1_800)
    parser.add_argument("--max-body-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument(
        "--max-body-prefix-bytes",
        type=int,
        default=MAX_INGEST_BODY_PREFIX_BYTES,
        help="body prefix retained per side; defaults to the wire ceiling",
    )
    parser.add_argument("--capture-socket", type=Path, default=None)
    parser.add_argument("--capture-source-id", default="mitm-inspector")
    parser.add_argument("--capture-max-body-prefix-bytes", type=int, default=None)
    parser.add_argument("--capture-max-in-memory-bytes", type=int, default=None)
    parser.add_argument("--capture-max-pending-messages", type=int, default=4096)
    parser.add_argument("--storage-path", default=str(default_storage_path()))
    parser.add_argument("--no-storage", action="store_true")
    parser.add_argument("--storage-max-flows", type=int, default=DEFAULT_STORAGE_MAX_FLOWS)
    parser.add_argument("--storage-max-bytes", type=int, default=DEFAULT_STORAGE_MAX_BYTES)
    parser.add_argument("--storage-replay", type=int, default=DEFAULT_STORAGE_REPLAY)
    return parser


def config_from_argv(argv: Sequence[str] | None = None) -> ApiServerConfig:
    args = _parser().parse_args(argv)
    max_body_prefix = (
        args.capture_max_body_prefix_bytes
        if args.capture_max_body_prefix_bytes is not None
        else args.max_body_prefix_bytes
    )
    max_body_bytes = (
        args.capture_max_in_memory_bytes
        if args.capture_max_in_memory_bytes is not None
        else args.max_body_bytes
    )
    return ApiServerConfig(
        host=args.host,
        port=args.port,
        proxy_port=args.proxy_port,
        source_id=args.capture_source_id,
        max_retained_flows=args.max_retained_flows,
        retention_seconds=args.retention_seconds,
        max_body_bytes=max_body_bytes,
        max_body_prefix_bytes=max_body_prefix,
        max_pending_messages=args.capture_max_pending_messages,
        capture_socket=args.capture_socket,
        storage_path=args.storage_path,
        no_storage=args.no_storage,
        storage_max_flows=args.storage_max_flows,
        storage_max_bytes=args.storage_max_bytes,
        storage_replay=args.storage_replay,
    )


async def _run(config: ApiServerConfig) -> int:
    server = ApiServer(config)
    await server.start()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    status = {"code": 0}

    def request_stop(code: int) -> None:
        status["code"] = code
        stop.set()

    installed: list[signal.Signals] = []
    for signum, code in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
        try:
            loop.add_signal_handler(signum, request_stop, code)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - non-POSIX loop.
            continue
        installed.append(signum)
    try:
        await stop.wait()
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
        await server.close()
    return status["code"]


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = config_from_argv(argv)
    except ApiServerError as error:
        print(f"mitm-inspector api: {error}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_run(config))
    except KeyboardInterrupt:
        return 130
    except (ApiServerError, OSError) as error:
        print(f"mitm-inspector api: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
