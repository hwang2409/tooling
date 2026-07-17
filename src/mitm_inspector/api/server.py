"""Loopback asyncio HTTP/WebSocket server and capture ingest listener.

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mitm_inspector.api.app import ApiApplication, Subscriber
from mitm_inspector.api.httpwire import (
    MAX_REQUEST_HEAD_BYTES,
    WEBSOCKET_VERSION,
    Close,
    FrameDecoder,
    HttpWireError,
    Ping,
    Pong,
    RequestHead,
    TextMessage,
    WebSocketWireError,
    encode_close_frame,
    encode_pong_frame,
    encode_text_frame,
    is_loopback_host_header,
    is_loopback_origin,
    parse_request_head,
    websocket_accept_key,
)
from mitm_inspector.protocol import MAX_U64
from mitm_inspector.store.memory import MemoryStore

API_VERSION_PREFIX = "/api/v1"
HEALTH_PATH = f"{API_VERSION_PREFIX}/health"
SNAPSHOT_PATH = f"{API_VERSION_PREFIX}/snapshot"
STREAM_PATH = f"{API_VERSION_PREFIX}/stream"
FLOW_PATH_PREFIX = f"{API_VERSION_PREFIX}/flows/"
MAX_INGEST_LINE_BYTES = 8 * 1024 * 1024
MAX_CLIENT_MESSAGE_BYTES = 64 * 1024
SUBSCRIBER_QUEUE_FRAMES = 256
HEAD_READ_TIMEOUT_SECONDS = 10.0
DEFAULT_SWEEP_INTERVAL_SECONDS = 5.0

_FLOW_ID_SEGMENT = re.compile(r"^[A-Za-z0-9._:@-]{1,256}$")
_LOOPBACK_NAMES = frozenset({"localhost"})

_INDEX_BODY = (
    b"<!doctype html><title>mitm-inspector</title>"
    b"<p>mitm-inspector local API. The browser UI connects to "
    b"<code>/api/v1/stream</code>.</p>"
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
    max_body_prefix_bytes: int = 1024 * 1024
    max_pending_messages: int = 4096
    capture_socket: Path | None = None
    sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS

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
        if self.capture_socket is not None:
            path = self.capture_socket
            if not isinstance(path, Path) or not path.is_absolute():
                raise ApiServerError("capture socket must be an absolute path")
        interval = self.sweep_interval_seconds
        if type(interval) not in {int, float} or not interval > 0:
            raise ApiServerError("sweep interval must be a positive number")


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
            application = ApiApplication(
                store,
                source_id=config.source_id,
                max_body_prefix_bytes=config.max_body_prefix_bytes,
                max_in_memory_bytes=config.max_body_bytes,
            )
        self.application = application
        self._http_server: asyncio.Server | None = None
        self._ingest_server: asyncio.Server | None = None
        self._sweep_task: asyncio.Task[None] | None = None
        self._rejected_ingest_lines = 0
        self._ingest_connections = 0
        self._http_requests = 0
        self._websocket_connections = 0
        self._sweep_failures = 0

    @property
    def counters(self) -> dict[str, object]:
        counters = self.application.counters
        counters["rejected_ingest_lines"] = self._rejected_ingest_lines
        counters["ingest_connections"] = self._ingest_connections
        counters["http_requests"] = self._http_requests
        counters["websocket_connections"] = self._websocket_connections
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
                    raise ApiServerError(
                        f"capture endpoint exists and is not a socket: {path}"
                    )
                path.unlink()
            try:
                self._ingest_server = await asyncio.start_unix_server(
                    self._handle_ingest_connection,
                    path=str(path),
                    limit=MAX_INGEST_LINE_BYTES,
                )
                os.chmod(path, 0o600)
            except BaseException:
                await self._close_http()
                raise
        self._sweep_task = asyncio.get_running_loop().create_task(self._sweep_forever())

    async def serve_forever(self) -> None:
        if self._http_server is None:
            raise ApiServerError("the API server is not started")
        await self._http_server.serve_forever()

    async def close(self) -> None:
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None
        for server in (self._ingest_server, self._http_server):
            if server is None:
                continue
            server.close()
            await server.wait_closed()
        self._ingest_server = None
        await self._close_http()

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
        self._ingest_connections += 1
        try:
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
            await _close_writer(writer)

    # -- HTTP and WebSocket ------------------------------------------------

    async def _handle_http_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._http_requests += 1
        try:
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
                target = head.target.split("?", 1)[0]
                if target == STREAM_PATH:
                    await self._handle_websocket(head, reader, writer)
                    return
            except HttpWireError:
                await self._send_simple(writer, 400, b"malformed request")
                return
            await self._handle_plain_get(target, writer)
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            await _close_writer(writer)

    async def _handle_plain_get(self, target: str, writer: asyncio.StreamWriter) -> None:
        if target == "/":
            await self._send_response(writer, 200, "text/html; charset=utf-8", _INDEX_BODY)
            return
        if target == HEALTH_PATH:
            body = json.dumps(
                {"status": "ok", "protocol_version": "1", "counters": self.counters},
                separators=(",", ":"),
            ).encode("utf-8")
            await self._send_response(writer, 200, "application/json", body)
            return
        if target == SNAPSHOT_PATH:
            body = self.application.snapshot_text().encode("utf-8")
            await self._send_response(writer, 200, "application/json", body)
            return
        if target.startswith(FLOW_PATH_PREFIX):
            flow_id = target[len(FLOW_PATH_PREFIX) :]
            if not _FLOW_ID_SEGMENT.fullmatch(flow_id):
                await self._send_simple(writer, 404, b"unknown flow")
                return
            detail = self.application.flow_detail_text(flow_id)
            if detail is None:
                await self._send_simple(writer, 404, b"unknown flow")
                return
            await self._send_response(writer, 200, "application/json", detail.encode("utf-8"))
            return
        await self._send_simple(writer, 404, b"unknown path")

    async def _handle_websocket(
        self,
        head: RequestHead,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            upgrade = head.single_header("upgrade")
            version = head.single_header("sec-websocket-version")
            key = head.single_header("sec-websocket-key")
            origin = head.single_header("origin")
        except HttpWireError:
            await self._send_simple(writer, 400, b"malformed websocket handshake")
            return
        if (
            upgrade is None
            or upgrade.lower() != "websocket"
            or "upgrade" not in head.token_list("connection")
            or version != WEBSOCKET_VERSION
            or key is None
        ):
            await self._send_simple(writer, 400, b"malformed websocket handshake")
            return
        if origin is not None and not is_loopback_origin(origin):
            await self._send_simple(writer, 403, b"origin is not loopback")
            return
        try:
            accept = websocket_accept_key(key)
        except WebSocketWireError:
            await self._send_simple(writer, 400, b"malformed websocket key")
            return
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept.encode("ascii") + b"\r\n\r\n"
        )
        await writer.drain()
        self._websocket_connections += 1
        await self._run_websocket_session(reader, writer)

    async def _run_websocket_session(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_FRAMES)

        def deliver(text: str) -> bool:
            try:
                queue.put_nowait(encode_text_frame(text))
            except asyncio.QueueFull:
                return False
            return True

        def on_drop() -> None:
            # Waking the transport closes both session tasks; the client
            # reconnects and receives a coherent fresh snapshot.
            writer.close()

        subscriber = self.application.subscribe(deliver, on_drop=on_drop)
        if subscriber.closed:
            await _close_writer(writer)
            return
        pump = asyncio.get_running_loop().create_task(self._pump_frames(queue, writer))
        close_frame = encode_close_frame(1000)
        try:
            close_frame = await self._read_websocket(reader, queue, subscriber)
        finally:
            self.application.unsubscribe(subscriber)
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
            try:
                writer.write(close_frame)
                await writer.drain()
            except (ConnectionError, RuntimeError):
                pass

    async def _read_websocket(
        self,
        reader: asyncio.StreamReader,
        queue: asyncio.Queue[bytes],
        subscriber: Subscriber,
    ) -> bytes:
        decoder = FrameDecoder(max_message_bytes=MAX_CLIENT_MESSAGE_BYTES)
        while True:
            data = await reader.read(4096)
            if not data:
                return encode_close_frame(1001)
            try:
                events = decoder.feed(data)
            except WebSocketWireError:
                return encode_close_frame(1002)
            for event in events:
                if isinstance(event, Close):
                    return encode_close_frame(1000)
                if isinstance(event, Ping):
                    try:
                        queue.put_nowait(encode_pong_frame(event.payload))
                    except asyncio.QueueFull:
                        return encode_close_frame(1013)
                    continue
                if isinstance(event, Pong):
                    continue
                if isinstance(event, TextMessage):
                    try:
                        value = parse_ingest_line(event.text.encode("utf-8"))
                        frames = self.application.handle_client_message(value)
                    except ValueError:
                        return encode_close_frame(1002)
                    for frame in frames:
                        if not deliverable(subscriber, queue, frame):
                            return encode_close_frame(1013)

    async def _pump_frames(
        self, queue: asyncio.Queue[bytes], writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                frame = await queue.get()
                writer.write(frame)
                await writer.drain()
        except (ConnectionError, RuntimeError):
            return

    async def _send_simple(
        self, writer: asyncio.StreamWriter, status: int, body: bytes
    ) -> None:
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


def deliverable(subscriber: Subscriber, queue: asyncio.Queue[bytes], text: str) -> bool:
    """Enqueue one session-scoped frame while respecting the shared bound."""

    if subscriber.closed:
        return False
    try:
        queue.put_nowait(encode_text_frame(text))
    except asyncio.QueueFull:
        return False
    return True


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
    parser.add_argument("--max-body-prefix-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--capture-socket", type=Path, default=None)
    parser.add_argument("--capture-source-id", default="mitm-inspector")
    parser.add_argument("--capture-max-body-prefix-bytes", type=int, default=None)
    parser.add_argument("--capture-max-in-memory-bytes", type=int, default=None)
    parser.add_argument("--capture-max-pending-messages", type=int, default=4096)
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
