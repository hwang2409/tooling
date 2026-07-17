// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseProtocolMessage } from "../../protocol";
import { browserReducer, initialBrowserState } from "../../state/browserState";
import type { BrowserState } from "../../state/browserState";
import { FlowWorkspace } from "./FlowWorkspace";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function flow(flowId: string, overrides: Record<string, unknown> = {}) {
  return {
    flow_id: flowId,
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: `/v1/${flowId}`,
    request_headers: [{ name: "content-type", value: "application/json" }],
    request_body: { state: "empty", size_bytes: "0" },
    ...overrides,
  };
}

function reduceAll(messages: unknown[]): BrowserState {
  return messages.reduce<BrowserState>(
    (state, message) => browserReducer(state, { type: "protocol", envelope: parseProtocolMessage(message) }),
    initialBrowserState,
  );
}

function hello(sourceId = "source-a") {
  return {
    protocol_version: "1", type: "source.hello", source_id: sourceId,
    occurred_at: "2026-01-01T00:00:00Z",
    capabilities: { body_chunks: true, redaction: "headers-and-query" },
    limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
  };
}

function snapshot(cursor: string, flows: unknown[]) {
  return { protocol_version: "1", type: "browser.snapshot", snapshot_id: `snap-${cursor}`, cursor, flows };
}

function lifecycleEvent(flowId: string, state: string, sequence: string) {
  return {
    protocol_version: "1", type: "flow.lifecycle", source_id: "source-a", flow_id: flowId,
    event_id: `${flowId}-${sequence}`, occurred_at: "2026-01-01T00:00:01Z", sequence, state,
  };
}

interface Mounted {
  root: ReturnType<typeof createRoot>;
  container: HTMLDivElement;
}

const mounts: Mounted[] = [];

async function mountWorkspace(browser: BrowserState, options: { followLive?: boolean; pauseLive?: () => void } = {}): Promise<Mounted> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(
      <FlowWorkspace
        browser={browser}
        followLive={options.followLive ?? true}
        pauseLive={options.pauseLive ?? (() => {})}
        gridViewportHeight={280}
      />,
    );
  });
  const mounted = { root, container };
  mounts.push(mounted);
  return mounted;
}

async function rerender(mounted: Mounted, browser: BrowserState, options: { followLive?: boolean; pauseLive?: () => void } = {}): Promise<void> {
  await act(async () => {
    mounted.root.render(
      <FlowWorkspace
        browser={browser}
        followLive={options.followLive ?? true}
        pauseLive={options.pauseLive ?? (() => {})}
        gridViewportHeight={280}
      />,
    );
  });
}

function rowByText(container: HTMLElement, text: string): HTMLElement {
  const row = Array.from(container.querySelectorAll<HTMLElement>(".flow-grid-row"))
    .find((candidate) => candidate.textContent?.includes(text));
  if (!row) throw new Error(`row not found: ${text}`);
  return row;
}

function filterInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>("input.flow-filter-input");
  if (!input) throw new Error("filter input not mounted");
  return input;
}

async function typeFilter(mounted: Mounted, value: string): Promise<void> {
  const input = filterInput(mounted.container);
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  await act(async () => {
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("mounted FlowWorkspace", () => {
  it("renders one 28px grid row per retained flow with lifecycle-derived phases", async () => {
    const browser = reduceAll([
      hello(),
      snapshot("1", [flow("alpha"), flow("bravo"), flow("charlie")]),
      lifecycleEvent("alpha", "response_started", "1"),
      lifecycleEvent("bravo", "error", "2"),
      lifecycleEvent("charlie", "flow_completed", "3"),
    ]);
    const mounted = await mountWorkspace(browser);
    expect(mounted.container.textContent).toContain("3/3");
    expect(rowByText(mounted.container, "/v1/alpha").textContent).toContain("resp");
    expect(rowByText(mounted.container, "/v1/bravo").textContent).toContain("error");
    expect(rowByText(mounted.container, "/v1/charlie").textContent).toContain("done");
  });

  it("filters rows with mitmproxy syntax and reports the shown count", async () => {
    const browser = reduceAll([
      hello(),
      snapshot("1", [flow("alpha", { method: "GET" }), flow("bravo"), flow("charlie", { host: "other.test" })]),
    ]);
    const mounted = await mountWorkspace(browser);
    await typeFilter(mounted, "~m post ~d example");
    expect(mounted.container.textContent).toContain("1/3");
    expect(mounted.container.textContent).toContain("/v1/bravo");
    expect(mounted.container.textContent).not.toContain("/v1/alpha");
    expect(mounted.container.textContent).not.toContain("/v1/charlie");
  });

  it("surfaces filter errors as an alert and keeps every row visible", async () => {
    const browser = reduceAll([hello(), snapshot("1", [flow("alpha"), flow("bravo")])]);
    const mounted = await mountWorkspace(browser);
    await typeFilter(mounted, "~c 200");
    const alert = mounted.container.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain("status code");
    expect(filterInput(mounted.container).getAttribute("aria-invalid")).toBe("true");
    expect(mounted.container.textContent).toContain("2/2");
    expect(mounted.container.textContent).toContain("/v1/alpha");
  });

  it("opens the paired inspector on row selection and pauses follow-live", async () => {
    const pauseLive = vi.fn();
    const browser = reduceAll([
      hello(),
      snapshot("1", [flow("alpha")]),
      lifecycleEvent("alpha", "response_started", "1"),
      lifecycleEvent("alpha", "request_end", "2"),
    ]);
    const mounted = await mountWorkspace(browser, { pauseLive });
    expect(mounted.container.textContent).toContain("Select a flow row");
    await act(async () => rowByText(mounted.container, "/v1/alpha").click());
    expect(pauseLive).toHaveBeenCalledOnce();
    expect(mounted.container.textContent).toContain("Request / response inspector");
    expect(mounted.container.textContent).toContain("response opened");
    expect(mounted.container.textContent).toContain("request ended");
  });

  it("does not re-pause when selection happens while already paused", async () => {
    const pauseLive = vi.fn();
    const browser = reduceAll([hello(), snapshot("1", [flow("alpha")])]);
    const mounted = await mountWorkspace(browser, { pauseLive, followLive: false });
    await act(async () => rowByText(mounted.container, "/v1/alpha").click());
    expect(pauseLive).not.toHaveBeenCalled();
    expect(mounted.container.textContent).toContain("Request / response inspector");
  });

  it("closes the inspector with the close control and with Escape", async () => {
    const browser = reduceAll([hello(), snapshot("1", [flow("alpha")])]);
    const mounted = await mountWorkspace(browser, { followLive: false });
    await act(async () => rowByText(mounted.container, "/v1/alpha").click());
    const close = Array.from(mounted.container.querySelectorAll("button")).find((button) => button.textContent === "Close inspector");
    await act(async () => close?.click());
    expect(mounted.container.textContent).toContain("Select a flow row");

    await act(async () => rowByText(mounted.container, "/v1/alpha").click());
    expect(mounted.container.textContent).toContain("Request / response inspector");
    await act(async () => {
      rowByText(mounted.container, "/v1/alpha").dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
    });
    expect(mounted.container.textContent).toContain("Select a flow row");
  });

  it("reports when the selected flow leaves bounded retention", async () => {
    const messages = [hello(), snapshot("1", [flow("alpha"), flow("bravo")])];
    const browser = reduceAll(messages);
    const mounted = await mountWorkspace(browser, { followLive: false });
    await act(async () => rowByText(mounted.container, "/v1/alpha").click());
    expect(mounted.container.textContent).toContain("Request / response inspector");

    const evicted = reduceAll([...messages, {
      protocol_version: "1", type: "browser.delta", cursor: "2",
      changes: [{ op: "remove", flow_id: "alpha" }],
    }]);
    await rerender(mounted, evicted, { followLive: false });
    expect(mounted.container.textContent).toContain("left the bounded retention window");
    const dismiss = Array.from(mounted.container.querySelectorAll("button")).find((button) => button.textContent === "Dismiss");
    await act(async () => dismiss?.click());
    expect(mounted.container.textContent).toContain("Select a flow row");
  });
});
