from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 4173
BASE_URL = f"http://127.0.0.1:{PORT}"
WIDTHS = (600, 800, 1024, 1440)

FLOW_ONE = {
    "flow_id": "browser-responsive-flow-a",
    "session_id": "client-session-a",
    "method": "POST",
    "scheme": "https",
    "host": "api.example.test",
    "port": "443",
    "path": "/stream/a",
    "request_headers": [{"name": "content-type", "value": "application/json"}],
    "request_body": {"state": "empty", "size_bytes": "0"},
}
FLOW_TWO = {
    **FLOW_ONE,
    "flow_id": "browser-responsive-flow-b",
    "session_id": "client-session-b",
    "path": "/stream/b",
}
FLOWS = [FLOW_ONE, FLOW_TWO]


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
            source_id: "browser-test-source",
            occurred_at: "2026-01-01T00:00:00Z",
            capabilities: { body_chunks: true, redaction: "headers-and-query" },
            limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
          };
          const snapshot = {
            protocol_version: "1",
            type: "browser.snapshot",
            snapshot_id: "browser-test-snapshot",
            cursor: "1",
            flows: __FLOW_JSON__,
          };
          const lifecycles = [
            {
              protocol_version: "1",
              type: "flow.lifecycle",
              source_id: "browser-test-source",
              flow_id: "browser-responsive-flow-a",
              event_id: "browser-test-lifecycle-a",
              occurred_at: "2026-01-01T00:00:01Z",
              sequence: "1",
              state: "request_started",
            },
            {
              protocol_version: "1",
              type: "flow.lifecycle",
              source_id: "browser-test-source",
              flow_id: "browser-responsive-flow-b",
              event_id: "browser-test-lifecycle-b",
              occurred_at: "2026-01-01T00:00:02Z",
              sequence: "2",
              state: "request_started",
            },
          ];
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
                for (const message of [hello, snapshot, ...lifecycles]) {
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
        """.replace("__FLOW_JSON__", FLOW_JSON)
    )


FLOW_JSON = json.dumps(FLOWS)


def run_session_journey(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.locator(".flow-grid-row").first.wait_for()
    group_by = page.locator("select[aria-label='Group flows by']")
    group_by.select_option("session")
    headers = page.locator(".flow-grid-session")
    assert headers.count() == 2
    assert "session client-" in headers.nth(0).inner_text()
    assert "session client-" in headers.nth(1).inner_text()

    page.reload()
    page.wait_for_load_state("networkidle")
    page.locator(".flow-grid-row").first.wait_for()
    assert page.locator("select[aria-label='Group flows by']").input_value() == "session"
    headers = page.locator(".flow-grid-session")
    assert headers.count() == 2
    first_header_button = headers.nth(0).locator("button")
    first_header_button.click()
    assert page.locator(".flow-grid-row").filter(has_text="/stream/a").count() == 0
    first_header_button.click()
    assert page.locator(".flow-grid-row").filter(has_text="/stream/a").count() == 1


def check_width(page: Page, width: int) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.evaluate("localStorage.clear()")
    page.reload()
    page.wait_for_load_state("networkidle")
    page.locator(".flow-grid-row").first.wait_for()
    page.locator(".flow-grid-row").first.click()
    page.locator('[data-testid="paired-inspector"]').wait_for()
    result = page.evaluate(
        """
        () => {
          const inspector = document.querySelector('[data-testid="paired-inspector"]');
          const inspectorSide = document.querySelector('.flow-inspector-side');
          const grid = document.querySelector('.flow-grid');
          const flowMain = document.querySelector('.flow-main');
          const row = document.querySelector('.flow-grid-row');
          if (!inspector || !inspectorSide || !grid || !flowMain || !row) {
            throw new Error('responsive fixture did not mount');
          }
          const rect = (el) => el.getBoundingClientRect();
          return {
            inspectorLeft: rect(inspector).left,
            inspectorRight: rect(inspector).right,
            inspectorTop: rect(inspector).top,
            inspectorScrollWidth: inspector.scrollWidth,
            inspectorScrollRight: rect(inspector).left + inspector.scrollWidth,
            inspectorSideLeft: rect(inspectorSide).left,
            gridRight: rect(grid).right,
            gridBottom: rect(grid).bottom,
            flowMainDirection: getComputedStyle(flowMain).flexDirection,
            viewportWidth: window.innerWidth,
            rowHeight: rect(row).height,
          };
        }
        """
    )
    assert result["inspectorRight"] <= result["viewportWidth"], result
    assert result["inspectorScrollRight"] <= result["viewportWidth"], result
    assert result["rowHeight"] == 28, result
    if width >= 1024:
        # Wide: inspector docks to the RIGHT of the grid, flow-main is a row.
        assert result["flowMainDirection"] == "row", result
        assert result["inspectorSideLeft"] >= result["gridRight"] - 1, result
    else:
        # Narrow: layout stacks, inspector sits below the grid.
        assert result["flowMainDirection"] == "column", result
        assert result["inspectorTop"] >= result["gridBottom"] - 1, result
    print(width, result)


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
            run_session_journey(page)
            for width in WIDTHS:
                check_width(page, width)
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
