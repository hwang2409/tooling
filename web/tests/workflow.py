"""Single end-to-end proof of the retained workflow.

Mirrors the production body-delivery boundary: the browser snapshot carries
only a stripped zero-byte body descriptor (the projection layer's shape), so
the JSON tree can only render after the click triggers a real fetch to
``GET /api/v1/flows/{id}`` for the full body.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import time
from pathlib import Path

from playwright.sync_api import Page, Route, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 4173
BASE_URL = f"http://127.0.0.1:{PORT}"

FLOW_ID = "workflow-flow-a"
REQUEST_PAYLOAD = {"model": "claude-example", "stream": True}
REQUEST_BODY = json.dumps(REQUEST_PAYLOAD).encode("utf-8")

# What the projection layer puts in the browser stream: a schema-valid
# descriptor with the true size but zero captured bytes.
STRIPPED_REQUEST_BODY = {
    "state": "truncated",
    "size_bytes": str(len(REQUEST_BODY)),
    "captured_bytes": "0",
    "encoding": "base64",
    "data": "",
}

FULL_REQUEST_BODY = {
    "state": "captured",
    "size_bytes": str(len(REQUEST_BODY)),
    "encoding": "base64",
    "data": base64.b64encode(REQUEST_BODY).decode("ascii"),
}

FLOW = {
    "flow_id": FLOW_ID,
    "session_id": "client-session-a",
    "method": "POST",
    "scheme": "https",
    "host": "api.example.test",
    "port": "443",
    "path": "/v1/messages",
    "request_headers": [{"name": "content-type", "value": "application/json"}],
    "request_body": STRIPPED_REQUEST_BODY,
    "response_status": "200",
}

DETAIL_RESPONSE = {
    "flow_id": FLOW_ID,
    "messages": [
        {
            "protocol_version": "1",
            "type": "body.end",
            "flow_id": FLOW_ID,
            "body_side": "request",
            "total_bytes": str(len(REQUEST_BODY)),
            "body": FULL_REQUEST_BODY,
        }
    ],
}


def wait_for_server(server: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.poll() is not None:
            output, _ = server.communicate()
            raise RuntimeError(f"preview server exited before ready:\n{output.strip()}")
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"preview server did not start on port {PORT}")


def ensure_port_available() -> None:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            pass
    except OSError:
        return
    raise RuntimeError(f"preview port {PORT} is already in use")


def install_fake_stream(page: Page) -> None:
    page.add_init_script(
        """
        (() => {
          const hello = {
            protocol_version: "1",
            type: "source.hello",
            source_id: "workflow-test-source",
            occurred_at: "2026-01-01T00:00:00Z",
            capabilities: { body_chunks: true, redaction: "headers-and-query" },
            limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
          };
          const snapshot = {
            protocol_version: "1",
            type: "browser.snapshot",
            snapshot_id: "workflow-test-snapshot",
            cursor: "1",
            flows: [__FLOW_JSON__],
          };
          class FakeWebSocket {
            static OPEN = 1;
            readyState = 0;
            onopen = null;
            onmessage = null;
            onerror = null;
            onclose = null;
            constructor(url) {
              this.url = url;
              setTimeout(() => {
                if (this.readyState !== 0) return;
                this.readyState = FakeWebSocket.OPEN;
                this.onopen?.();
                for (const message of [hello, snapshot]) {
                  setTimeout(() => this.onmessage?.({ data: JSON.stringify(message) }), 0);
                }
              }, 0);
            }
            send() {}
            close() {
              this.readyState = 3;
              this.onclose?.({ code: 1000, reason: "", wasClean: true });
            }
          }
          window.WebSocket = FakeWebSocket;
        })();
        """.replace("__FLOW_JSON__", json.dumps(FLOW))
    )


def run_workflow(page: Page) -> None:
    detail_requests: list[str] = []

    def serve_detail(route: Route) -> None:
        detail_requests.append(route.request.url)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(DETAIL_RESPONSE),
        )

    page.route("**/api/v1/flows/*", serve_detail)

    page.goto(BASE_URL)
    row = page.locator(".packet-row")
    expect(row).to_have_count(1)
    expect(row).to_contain_text("POST")
    expect(row).to_contain_text("api.example.test")
    expect(row).to_contain_text("/v1/messages")
    expect(row).to_contain_text("200")
    # The snapshot carries no body bytes, so nothing may render before the
    # click-triggered detail fetch.
    assert detail_requests == [], detail_requests

    row.click()
    tree = page.locator(".packet-detail .json-tree")
    expect(tree).to_be_visible()
    expect(tree).to_contain_text("model")
    expect(tree).to_contain_text("claude-example")
    assert len(detail_requests) == 1, detail_requests
    assert detail_requests[0].endswith(f"/api/v1/flows/{FLOW_ID}"), detail_requests


def main() -> None:
    ensure_port_available()
    server = subprocess.Popen(
        ["npm", "run", "preview", "--", "--host", "127.0.0.1", "--port", str(PORT), "--strictPort"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_server(server)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            install_fake_stream(page)
            run_workflow(page)
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
