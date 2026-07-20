// @vitest-environment jsdom

import { Buffer } from "node:buffer";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { parseProtocolMessage } from "../../protocol";
import { browserReducer, initialBrowserState } from "../../state/browserState";
import type { BrowserState } from "../../state/browserState";
import type { FlowDetailLoader } from "../inspector/flowDetail";
import { PacketList } from "./PacketList";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const encoded = (value: string) => Buffer.from(value, "utf8").toString("base64");

function captured(text: string) {
  return { state: "captured" as const, size_bytes: String(Buffer.byteLength(text, "utf8")), encoding: "base64" as const, data: encoded(text) };
}

const unavailableLoader: FlowDetailLoader = async () => ({ status: "unavailable" });

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

function stateOf(flows: unknown[]): BrowserState {
  const messages = [
    {
      protocol_version: "1", type: "source.hello", source_id: "source-a",
      occurred_at: "2026-01-01T00:00:00Z",
      capabilities: { body_chunks: true, redaction: "headers-and-query" },
      limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
    },
    { protocol_version: "1", type: "browser.snapshot", snapshot_id: "snap-1", cursor: "1", flows },
  ];
  return messages.reduce<BrowserState>(
    (state, message) => browserReducer(state, { type: "protocol", envelope: parseProtocolMessage(message) }),
    initialBrowserState,
  );
}

interface Mounted {
  root: ReturnType<typeof createRoot>;
  container: HTMLDivElement;
}

const mounts: Mounted[] = [];

async function mountList(browser: BrowserState, loader: FlowDetailLoader = unavailableLoader): Promise<Mounted> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => root.render(<PacketList browser={browser} loadFlowDetail={loader} />));
  const mounted = { root, container };
  mounts.push(mounted);
  return mounted;
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

function rowByPath(container: HTMLElement, path: string): HTMLButtonElement {
  const row = Array.from(container.querySelectorAll<HTMLButtonElement>(".packet-row"))
    .find((candidate) => candidate.textContent?.includes(path));
  if (!row) throw new Error(`row not found: ${path}`);
  return row;
}

async function click(element: HTMLElement): Promise<void> {
  await act(async () => element.dispatchEvent(new MouseEvent("click", { bubbles: true })));
}

describe("PacketList", () => {
  it("renders one dense row per captured flow with method, host, path, and status", async () => {
    const browser = stateOf([
      flow("flow-a", { response_status: "200" }),
      flow("flow-b", { method: "GET" }),
    ]);
    const { container } = await mountList(browser);
    const rows = container.querySelectorAll(".packet-row");
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("POST");
    expect(rows[0].textContent).toContain("api.example.test");
    expect(rows[0].textContent).toContain("/v1/flow-a");
    expect(rows[0].textContent).toContain("200");
    // No response yet: status renders a placeholder, not a blank cell.
    expect(rows[1].textContent).toContain("—");
    expect(container.querySelector(".packet-detail")).toBeNull();
  });

  it("shows a quiet line when no packets are captured", async () => {
    const { container } = await mountList(stateOf([]));
    expect(container.querySelector(".packet-empty")?.textContent).toBe("no packets captured");
  });

  it("opens the request JSON tree when a row is clicked", async () => {
    const payload = { model: "claude", stream: true };
    const browser = stateOf([
      flow("flow-a", {
        request_body: captured(JSON.stringify(payload)),
      }),
    ]);
    const { container } = await mountList(browser);
    const row = rowByPath(container, "/v1/flow-a");
    expect(row.getAttribute("aria-expanded")).toBe("false");
    await click(row);
    expect(row.getAttribute("aria-expanded")).toBe("true");
    const tree = container.querySelector(".packet-detail .json-tree");
    expect(tree).not.toBeNull();
    expect(tree?.textContent).toContain("model");
    expect(tree?.textContent).toContain("claude");
  });

  it("replaces the open panel when another row is clicked and closes on re-click", async () => {
    const bodyFor = (name: string) => captured(JSON.stringify({ name }));
    const browser = stateOf([
      flow("flow-a", { request_body: bodyFor("first") }),
      flow("flow-b", { request_body: bodyFor("second") }),
    ]);
    const { container } = await mountList(browser);
    await click(rowByPath(container, "/v1/flow-a"));
    expect(container.querySelectorAll(".packet-detail")).toHaveLength(1);
    expect(container.querySelector(".packet-detail")?.textContent).toContain("first");

    await click(rowByPath(container, "/v1/flow-b"));
    expect(container.querySelectorAll(".packet-detail")).toHaveLength(1);
    expect(container.querySelector(".packet-detail")?.textContent).toContain("second");
    expect(rowByPath(container, "/v1/flow-a").getAttribute("aria-expanded")).toBe("false");

    await click(rowByPath(container, "/v1/flow-b"));
    expect(container.querySelector(".packet-detail")).toBeNull();
  });

  it("shows an italic placeholder line when the request body is empty", async () => {
    const browser = stateOf([flow("flow-a")]);
    const { container } = await mountList(browser);
    await click(rowByPath(container, "/v1/flow-a"));
    expect(container.querySelector(".packet-nobody")?.textContent).toBe("no request body");
    expect(container.querySelector(".json-tree")).toBeNull();
  });

  it("shows the placeholder, not an empty pane, for the projection-stripped zero-byte truncated body", async () => {
    // The browser-stream projection strips every body to this shape, so it is
    // the default observed descriptor whenever detail has not (yet) loaded.
    const browser = stateOf([
      flow("flow-a", {
        request_body: { state: "truncated", size_bytes: "100", captured_bytes: "0", encoding: "base64", data: "" },
      }),
    ]);
    const { container } = await mountList(browser);
    await click(rowByPath(container, "/v1/flow-a"));
    const placeholder = container.querySelector(".packet-detail .packet-nobody");
    expect(placeholder?.textContent).toBe("no request body");
    expect(container.querySelector(".packet-detail pre")).toBeNull();
    expect(container.querySelector(".json-tree")).toBeNull();
  });

  it("falls back to plain text when the request body is not JSON", async () => {
    const browser = stateOf([
      flow("flow-a", {
        request_body: captured("plain payload"),
      }),
    ]);
    const { container } = await mountList(browser);
    await click(rowByPath(container, "/v1/flow-a"));
    expect(container.querySelector(".json-tree")).toBeNull();
    expect(container.querySelector(".packet-text")?.textContent).toBe("plain payload");
  });

  it("renders the response JSON as a second panel below the request panel", async () => {
    const browser = stateOf([
      flow("flow-a", {
        request_body: captured(JSON.stringify({ ask: true })),
        response_status: "200",
        response_headers: [{ name: "content-type", value: "application/json" }],
        response_body: captured(JSON.stringify({ answer: 42 })),
      }),
    ]);
    const { container } = await mountList(browser);
    await click(rowByPath(container, "/v1/flow-a"));
    const response = container.querySelector(".packet-response");
    expect(response).not.toBeNull();
    expect(response?.querySelector(".json-tree")?.textContent).toContain("answer");
    const detail = container.querySelector(".packet-detail");
    expect(detail?.textContent?.indexOf("ask")).toBeLessThan(detail?.textContent?.indexOf("answer") ?? -1);
  });

  it("omits the response panel entirely when no response body exists", async () => {
    const browser = stateOf([
      flow("flow-a", {
        request_body: captured(JSON.stringify({ ask: true })),
      }),
    ]);
    const { container } = await mountList(browser);
    await click(rowByPath(container, "/v1/flow-a"));
    expect(container.querySelector(".packet-response")).toBeNull();
  });

  it("prefers the fetched flow detail body over grid metadata", async () => {
    const browser = stateOf([
      flow("flow-a", {
        request_body: { state: "truncated", size_bytes: "64", captured_bytes: "7", encoding: "base64", data: encoded("{\"trunc") },
      }),
    ]);
    const loader: FlowDetailLoader = async () => ({
      status: "loaded",
      overrides: {
        request_body: captured(JSON.stringify({ full: "body" })),
      },
    });
    const { container } = await mountList(browser, loader);
    await click(rowByPath(container, "/v1/flow-a"));
    expect(container.querySelector(".json-tree")?.textContent).toContain("full");
  });

  it("groups consecutive same-session flows under one session header with a count", async () => {
    const browser = stateOf([
      flow("flow-a", { session_id: "aaaaaaaa-1111-2222-3333-444444444444" }),
      flow("flow-b", { session_id: "aaaaaaaa-1111-2222-3333-444444444444" }),
      flow("flow-c", { session_id: "bbbbbbbb-9999-8888-7777-666666666666" }),
    ]);
    const { container } = await mountList(browser);
    const headers = container.querySelectorAll(".session-header");
    expect(headers).toHaveLength(2);
    expect(headers[0].textContent).toContain("session aaaaaaaa");
    expect(headers[0].textContent).toContain("2 flows");
    expect(headers[1].textContent).toContain("session bbbbbbbb");
    expect(headers[1].textContent).toContain("1 flow");
    // Rows preserve arrival order across headers.
    const rows = container.querySelectorAll(".packet-row");
    expect(rows).toHaveLength(3);
    expect(rows[0].textContent).toContain("/v1/flow-a");
    expect(rows[2].textContent).toContain("/v1/flow-c");
    // Header is not selectable and is hidden from AT.
    expect(headers[0].getAttribute("aria-hidden")).toBe("true");
    expect(headers[0].querySelector("button")).toBeNull();
  });

  it("splits interleaved sessions into distinct headers preserving arrival order", async () => {
    // Session A appears, then B, then A again — three headers, not two, so the
    // user can see when sessions interleaved rather than assuming coalescence.
    const browser = stateOf([
      flow("flow-a1", { session_id: "aaaa-1" }),
      flow("flow-b1", { session_id: "bbbb-1" }),
      flow("flow-a2", { session_id: "aaaa-1" }),
    ]);
    const { container } = await mountList(browser);
    const headers = container.querySelectorAll(".session-header");
    expect(headers).toHaveLength(3);
    expect(headers[0].textContent).toContain("aaaa-1");
    expect(headers[1].textContent).toContain("bbbb-1");
    expect(headers[2].textContent).toContain("aaaa-1");
  });

  it("labels a run of null session_id flows as unassigned", async () => {
    const browser = stateOf([
      flow("flow-a", { session_id: null }),
      flow("flow-b", { session_id: null }),
      flow("flow-c", { session_id: "cccc-1" }),
    ]);
    const { container } = await mountList(browser);
    const headers = container.querySelectorAll(".session-header");
    expect(headers).toHaveLength(2);
    expect(headers[0].textContent).toContain("unassigned");
    expect(headers[0].textContent).toContain("2 flows");
    expect(headers[1].textContent).toContain("session cccc-1");
  });

  it("still renders a header when every flow shares a single session", async () => {
    const browser = stateOf([
      flow("flow-a", { session_id: "solo-1" }),
      flow("flow-b", { session_id: "solo-1" }),
    ]);
    const { container } = await mountList(browser);
    const headers = container.querySelectorAll(".session-header");
    expect(headers).toHaveLength(1);
    expect(headers[0].textContent).toContain("2 flows");
  });
});
