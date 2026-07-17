// @vitest-environment jsdom

import { act, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { TransportFactory, TransportHandlers } from "./features/connection/connectionClient";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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
  };
}

function hello(sourceId: string) {
  return {
    protocol_version: "1", type: "source.hello", source_id: sourceId,
    occurred_at: "2026-01-01T00:00:00Z",
    capabilities: { body_chunks: true, redaction: "headers-and-query" },
    limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
  };
}

function snapshot(cursor: string, snapshotId: string, flows: ReturnType<typeof flow>[]) {
  return { protocol_version: "1", type: "browser.snapshot", snapshot_id: snapshotId, cursor, flows };
}

function gap(cursor: string) {
  return { protocol_version: "1", type: "browser.delta", cursor, changes: [] };
}

interface MountedConnection {
  readonly close: ReturnType<typeof vi.fn>;
  readonly requestResync?: ReturnType<typeof vi.fn>;
}

interface MountedHarness {
  readonly handlers: TransportHandlers[];
  readonly connections: MountedConnection[];
  readonly factory: TransportFactory;
}

function mountedHarness(withResync = true): MountedHarness {
  const handlers: TransportHandlers[] = [];
  const connections: MountedConnection[] = [];
  const factory: TransportFactory = (nextHandlers) => {
    handlers.push(nextHandlers);
    const connection = { close: vi.fn(), requestResync: vi.fn() };
    connections.push(withResync ? connection : { close: connection.close });
    return withResync ? connection : { close: connection.close };
  };
  return { handlers, connections, factory };
}

function buttons(): HTMLButtonElement[] {
  return Array.from(document.querySelectorAll<HTMLButtonElement>("button"));
}

function buttonWithText(text: string): HTMLButtonElement {
  const button = buttons().find((candidate) => candidate.textContent?.includes(text));
  if (!button) throw new Error(`button not found: ${text}`);
  return button;
}

function buttonWithAttribute(name: string, value: string): HTMLButtonElement {
  const button = buttons().find((candidate) => candidate.getAttribute(name) === value);
  if (!button) throw new Error(`button not found: ${name}=${value}`);
  return button;
}

async function click(button: HTMLButtonElement): Promise<void> {
  await act(async () => button.click());
}

describe("mounted App connection behavior", () => {
  let root: ReturnType<typeof createRoot> | null = null;

  afterEach(() => {
    root?.unmount();
    root = null;
    vi.useRealTimers();
  });

  it("mounts under StrictMode, pauses A, reconnects to B, and resyncs only B", async () => {
    vi.useFakeTimers();
    const harness = mountedHarness();
    const container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    await act(async () => {
      root?.render(<StrictMode><App transportFactory={harness.factory} /></StrictMode>);
    });
    expect(harness.handlers).toHaveLength(2);
    expect(harness.connections[0].close).toHaveBeenCalledOnce();
    const connectionA = harness.connections[1];

    await act(async () => {
      harness.handlers[1].onOpen();
      harness.handlers[1].onMessage(hello("source-a"));
      harness.handlers[1].onMessage(snapshot("1", "a", [flow("old-flow")]));
      harness.handlers[1].onMessage(gap("3"));
    });
    await click(buttonWithAttribute("aria-pressed", "true"));
    expect(document.body.textContent).toContain("Live paused");
    expect(document.body.textContent).toContain("paused at cursor 1");
    expect(document.body.textContent).toContain("source-a");

    await act(async () => harness.handlers[1].onClose("source A closed"));
    expect(document.body.textContent).toContain("Reconnecting");
    expect(document.body.textContent).toContain("Last failure: source A closed");
    expect(document.body.textContent).toContain("Stop reconnecting");
    await act(async () => vi.advanceTimersByTime(5_000));
    expect(harness.handlers).toHaveLength(3);
    const connectionB = harness.connections[2];

    await act(async () => {
      harness.handlers[2].onOpen();
      harness.handlers[2].onMessage(hello("source-b"));
      harness.handlers[2].onMessage(snapshot("0", "b", [flow("new-flow")]));
    });
    expect(document.body.textContent).toContain("source-a");
    expect(document.body.textContent).not.toContain("source-b");
    const staleRequest = buttonWithText("Snapshot request unavailable");
    expect(staleRequest.disabled).toBe(true);
    await click(staleRequest);
    expect(connectionB.requestResync).not.toHaveBeenCalled();

    await act(async () => harness.handlers[2].onMessage(gap("2")));
    await click(buttonWithText("Request snapshot"));
    expect(connectionB.requestResync).toHaveBeenCalledWith(expect.objectContaining({ requested_cursor: "0" }));
    expect(connectionA.requestResync).not.toHaveBeenCalled();

    await click(buttonWithText("Live paused"));
    expect(document.body.textContent).toContain("source-b");
    expect(document.body.textContent).toContain("cursor 2");
    await act(async () => root?.unmount());
    root = null;
    expect(connectionB.close).toHaveBeenCalledOnce();
  });

  it("renders the flow grid for retained flows and pauses follow-live when a row opens the inspector", async () => {
    const harness = mountedHarness();
    const container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => root?.render(<App transportFactory={harness.factory} />));
    await act(async () => {
      harness.handlers[0].onOpen();
      harness.handlers[0].onMessage(hello("source-a"));
      harness.handlers[0].onMessage(snapshot("1", "a", [flow("flow-one"), flow("flow-two")]));
      harness.handlers[0].onMessage({
        protocol_version: "1", type: "flow.lifecycle", source_id: "source-a", flow_id: "flow-one",
        event_id: "e1", occurred_at: "2026-01-01T00:00:01Z", sequence: "1", state: "response_started",
      });
    });

    const grid = document.querySelector('[role="grid"]');
    expect(grid?.getAttribute("aria-rowcount")).toBe("3");
    expect(document.body.textContent).toContain("Following live");

    const rows = Array.from(document.querySelectorAll<HTMLElement>(".flow-grid-row"));
    expect(rows).toHaveLength(2);
    await act(async () => rows[0].click());

    expect(document.body.textContent).toContain("Live paused");
    expect(document.querySelector('[data-testid="paired-inspector"]')).not.toBeNull();
    expect(document.body.textContent).toContain("response opened");

    await act(async () => harness.handlers[0].onMessage(gap("2")));
    expect(document.body.textContent).toContain("paused at cursor 1");
    await act(async () => root?.unmount());
    root = null;
  });

  it("preserves the seen-flow marker across a reconnect that unmounts the flow workspace", async () => {
    vi.useFakeTimers();
    const harness = mountedHarness();
    const container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => root?.render(<App transportFactory={harness.factory} />));
    await act(async () => {
      harness.handlers[0].onOpen();
      harness.handlers[0].onMessage(hello("source-a"));
      harness.handlers[0].onMessage(snapshot("1", "a", [flow("kept-flow")]));
    });
    const beforeRow = document.querySelector<HTMLElement>('.flow-grid-row');
    expect(beforeRow?.classList.contains("is-unseen")).toBe(true);
    await act(async () => beforeRow?.click());
    expect(document.querySelector<HTMLElement>('.flow-grid-row')?.classList.contains("is-unseen")).toBe(false);
    // Resume follow-live so the source-reset that the reconnect issues
    // wipes the displayed flows and the empty-state branch unmounts the
    // FlowWorkspace. This is the failure mode the F1 reviewer probed.
    await click(buttonWithText("Live paused"));

    await act(async () => harness.handlers[0].onClose("dropped"));
    expect(document.body.textContent).toContain("Reconnecting");
    await act(async () => vi.advanceTimersByTime(5_000));
    expect(document.querySelector<HTMLElement>('.flow-grid-row')).toBeNull();
    await act(async () => {
      harness.handlers[1].onOpen();
      harness.handlers[1].onMessage(hello("source-b"));
      // Same retained flow-id survives the reconnect; the seen marker must
      // still apply after the workspace unmount/remount cycle.
      harness.handlers[1].onMessage(snapshot("1", "b", [flow("kept-flow")]));
    });
    const afterRow = document.querySelector<HTMLElement>('.flow-grid-row');
    expect(afterRow).not.toBeNull();
    expect(afterRow?.classList.contains("is-unseen")).toBe(false);
    await act(async () => root?.unmount());
    root = null;
  });

  it("mounts an unsupported transport as an explicit unavailable resync action", async () => {
    const harness = mountedHarness(false);
    const container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => root?.render(<App transportFactory={harness.factory} />));
    await act(async () => {
      harness.handlers[0].onOpen();
      harness.handlers[0].onMessage(snapshot("1", "one", []));
      harness.handlers[0].onMessage(gap("3"));
    });
    const unavailable = buttonWithText("Snapshot request unavailable");
    expect(unavailable.disabled).toBe(true);
    await act(async () => root?.unmount());
    root = null;
    expect(harness.connections[0].close).toHaveBeenCalledOnce();
  });

  it("renders a transport resync error and cleans up on unmount", async () => {
    const harness = mountedHarness();
    const container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => root?.render(<App transportFactory={harness.factory} />));
    await act(async () => {
      harness.handlers[0].onOpen();
      harness.handlers[0].onMessage(snapshot("1", "one", []));
      harness.handlers[0].onMessage(gap("3"));
    });
    expect(buttonWithText("Request snapshot").disabled).toBe(false);
    harness.connections[0].requestResync?.mockImplementation(() => { throw new Error("send failed"); });
    await click(buttonWithText("Request snapshot"));
    expect(document.body.textContent).toContain("Snapshot request could not be sent.");
    await act(async () => root?.unmount());
    root = null;
    expect(harness.connections[0].close).toHaveBeenCalledOnce();
  });
});
