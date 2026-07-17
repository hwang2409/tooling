import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { formatBytes, Workbench } from "./App";
import type { ConnectionStatus } from "./features/connection/connectionClient";
import type { ConnectionViewModel } from "./features/connection/useConnection";
import { browserReducer, initialBrowserState } from "./state/browserState";
import type { BrowserState } from "./state/browserState";
import { parseProtocolMessage } from "./protocol";

function flow(flowId: string) {
  return {
    flow_id: flowId,
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: "/v1/messages",
    request_headers: [],
    request_body: { state: "empty", size_bytes: "0" },
  } as const;
}

function browser(value: unknown): BrowserState {
  return browserReducer(initialBrowserState, { type: "protocol", envelope: parseProtocolMessage(value) });
}

function makeView(overrides: Partial<ConnectionViewModel> = {}): ConnectionViewModel {
  const status: ConnectionStatus = {
    state: "live",
    attempt: 0,
    error: null,
    lastMessageAt: 1,
    requestResyncAvailable: true,
  };
  return {
    browser: initialBrowserState,
    latestBrowser: initialBrowserState,
    status,
    followLive: true,
    connect: vi.fn(),
    disconnect: vi.fn(),
    retry: vi.fn(),
    pauseLive: vi.fn(),
    resumeLive: vi.fn(),
    requestResync: vi.fn(() => ({ ok: true as const })),
    ...overrides,
  };
}

describe("Workbench component", () => {
  it("uses exact BigInt formatting for uint64 retention limits", () => {
    expect(formatBytes("18446744063223267327")).toBe("17592186034415 MiB");
  });

  it("retains reconnect failure context and uses reconnect-specific controls", () => {
    const html = renderToStaticMarkup(<Workbench view={makeView({
      status: { state: "reconnecting", attempt: 3, error: "socket reset", lastMessageAt: 1, requestResyncAvailable: false },
    })} />);

    expect(html).toContain("Reconnecting to local capture source");
    expect(html).toContain("Last failure: socket reset");
    expect(html).toContain("Stop reconnecting");
    expect(html).not.toContain("Waiting for a local source");
  });

  it("renders the complete paused snapshot cursor and retained count", () => {
    const paused = browser({
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "1", flows: [flow("old")],
    });
    const html = renderToStaticMarkup(<Workbench view={makeView({
      browser: paused,
      latestBrowser: paused,
      followLive: false,
    })} />);

    expect(html).toContain("Live paused");
    expect(html).toContain("paused at cursor 1");
    expect(html).toContain('aria-label="1 retained flows"');
  });

  it("makes resync capability availability visible in the gap action", () => {
    const snapshot = browser({
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "1", flows: [],
    });
    const gap = browserReducer(snapshot, { type: "protocol", envelope: parseProtocolMessage({
      protocol_version: "1", type: "browser.delta", cursor: "3", changes: [],
    }) });
    const unavailable = renderToStaticMarkup(<Workbench view={makeView({
      browser: gap,
      latestBrowser: gap,
      status: { state: "live", attempt: 0, error: null, lastMessageAt: 1, requestResyncAvailable: false },
    })} />);
    const available = renderToStaticMarkup(<Workbench view={makeView({
      browser: gap,
      latestBrowser: gap,
      status: { state: "live", attempt: 0, error: null, lastMessageAt: 1, requestResyncAvailable: true },
    })} />);

    expect(unavailable).toContain("Snapshot request unavailable");
    expect(available).toContain("Request snapshot");
  });

  it("keeps the skip target focusable and the workspace heading as its landmark label", () => {
    const html = renderToStaticMarkup(<Workbench view={makeView()} />);

    expect(html).toContain("Skip to workspace");
    expect(html).toMatch(/<section class="workspace-frame" aria-labelledby="[^"]+">/);
    expect(html).toMatch(/<h2 id="[^"]+" tabindex="-1">Live flows<\/h2>/);
    expect(html).toContain('aria-current="page"');
  });
});
