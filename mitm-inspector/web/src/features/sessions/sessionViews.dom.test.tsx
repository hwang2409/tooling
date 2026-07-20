// @vitest-environment jsdom

import { Buffer } from "node:buffer";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { Workspace } from "../../App";
import { parseProtocolMessage } from "../../protocol";
import { browserReducer, initialBrowserState } from "../../state/browserState";
import type { BrowserState } from "../../state/browserState";
import type { FlowDetailLoader } from "../inspector/flowDetail";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const encoded = (value: string) => Buffer.from(value, "utf8").toString("base64");

function captured(text: string) {
  return {
    state: "captured" as const,
    size_bytes: String(Buffer.byteLength(text, "utf8")),
    encoding: "base64" as const,
    data: encoded(text),
  };
}

const MAIN_REQUEST = JSON.stringify({
  model: "claude-opus-4",
  messages: [
    { role: "user", content: "fix the bug" },
    { role: "assistant", content: "looking at it" },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "tests pass" }] },
  ],
});

const SUGGESTION_REQUEST = JSON.stringify({
  model: "claude-opus-4",
  messages: [
    { role: "user", content: "fix the bug" },
    { role: "assistant", content: "looking at it" },
    { role: "user", content: [{ type: "text", text: "[SUGGESTION MODE: propose follow-ups]" }] },
  ],
});

const MAIN_RESPONSE = JSON.stringify({
  model: "claude-opus-4",
  role: "assistant",
  content: [{ type: "text", text: "final assistant turn" }],
  stop_reason: "end_turn",
});

function sessionFlow(flowId: string, overrides: Record<string, unknown> = {}) {
  return {
    flow_id: flowId,
    session_id: "aaaa1111-2222-3333-4444-555555555555",
    method: "POST",
    scheme: "https",
    host: "api.anthropic.test",
    port: "443",
    path: `/v1/messages`,
    request_headers: [{ name: "content-type", value: "application/json" }],
    response_headers: [{ name: "content-type", value: "application/json" }],
    request_body: { state: "truncated", size_bytes: "64", captured_bytes: "0", encoding: "base64", data: "" },
    response_status: "200",
    ...overrides,
  };
}

// Grid order is newest-first. The suggestion side-call carries a HIGHER
// message_count than the main thread and a tool_result preview, so the
// metadata heuristic alone ranks it first — only the body check catches it.
const FLOWS = [
  sessionFlow("aux-suggestion", {
    started_at: "2026-01-01T00:02:00Z",
    ended_at: "2026-01-01T00:02:05Z",
    summary: {
      kind: "anthropic_messages",
      model: "claude-opus-4",
      message_count: "3",
      preview: { source: "tool_result", tool_name: "Bash" },
    },
  }),
  sessionFlow("main-flow", {
    started_at: "2026-01-01T00:01:00Z",
    ended_at: "2026-01-01T00:01:30Z",
    response_status: "500",
    summary: {
      kind: "anthropic_messages",
      model: "claude-opus-4",
      message_count: "3",
      preview: { source: "user_text", text: "fix the bug" },
    },
  }),
  sessionFlow("util-flow", {
    started_at: "2026-01-01T00:00:00Z",
    ended_at: "2026-01-01T00:00:01Z",
    summary: { kind: "anthropic_count_tokens", model: "claude-haiku-4", count_tokens_result: "12" },
  }),
  sessionFlow("loose-flow", {
    session_id: null,
    host: "other.example.test",
    path: "/loose",
    started_at: "2025-12-31T23:59:00Z",
    summary: { kind: "generic" },
  }),
];

const DETAIL_BODIES: Record<string, { request: string; response?: string }> = {
  "aux-suggestion": { request: SUGGESTION_REQUEST },
  "main-flow": { request: MAIN_REQUEST, response: MAIN_RESPONSE },
};

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

const mounts: Array<{ root: ReturnType<typeof createRoot>; container: HTMLDivElement }> = [];

type DetailBodies = Record<string, { request: string; response?: string }>;

async function mountWorkspace(browser: BrowserState, bodies: DetailBodies | (() => DetailBodies) = DETAIL_BODIES) {
  const requested: string[] = [];
  const loadFlowDetail: FlowDetailLoader = async (flowId) => {
    requested.push(flowId);
    const current = typeof bodies === "function" ? bodies() : bodies;
    const entry = current[flowId];
    if (entry === undefined) return { status: "unavailable" };
    return {
      status: "loaded",
      overrides: {
        request_body: captured(entry.request),
        ...(entry.response !== undefined ? { response_body: captured(entry.response) } : {}),
      },
    };
  };
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mounts.push({ root, container });
  const render = async (nextBrowser: BrowserState) => {
    await act(async () => root.render(<Workspace browser={nextBrowser} loadFlowDetail={loadFlowDetail} />));
  };
  await render(browser);
  return { container, requested, render };
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

async function click(element: Element | null | undefined): Promise<void> {
  if (!element) throw new Error("element to click not found");
  await act(async () => element.dispatchEvent(new MouseEvent("click", { bubbles: true })));
}

function sessionRows(container: HTMLElement): HTMLButtonElement[] {
  return Array.from(container.querySelectorAll<HTMLButtonElement>(".session-row"));
}

async function settle(): Promise<void> {
  // Let queued detail fetches + candidate re-selection effects flush.
  for (let index = 0; index < 4; index += 1) await act(async () => {});
}

describe("Workspace session-first navigation", () => {
  it("lands on the session list: one metadata-only row per session, newest first", async () => {
    const { container, requested } = await mountWorkspace(stateOf(FLOWS));
    const rows = sessionRows(container);
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("aaaa1111");
    expect(rows[0].textContent).toContain("3f");
    expect(rows[0].textContent).toContain("“fix the bug”");
    expect(rows[0].textContent).toContain("opus-4");
    expect(rows[0].textContent).toContain("err");
    expect(rows[0].querySelector(".session-status-error")).not.toBeNull();
    expect(rows[1].textContent).toContain("unassigned");
    // The home list renders from the grid projection only — no detail fetches.
    expect(requested).toEqual([]);
  });

  it("drills into a session, rejects the suggestion-mode sibling, and renders the chat with the response as the final turn", async () => {
    const { container, requested } = await mountWorkspace(stateOf(FLOWS));
    await click(sessionRows(container)[0]);
    await settle();

    // Metadata ranking tried the suggestion sibling first; the body check
    // rejected it and fell through to the main thread.
    expect(requested).toEqual(["aux-suggestion", "main-flow"]);
    const conversation = container.querySelector("[data-testid='conversation-view']");
    expect(conversation).not.toBeNull();
    expect(conversation?.textContent).toContain("fix the bug");
    expect(conversation?.textContent).not.toContain("[SUGGESTION MODE:");
    // Response appended as the last assistant turn.
    expect(conversation?.textContent).toContain("final assistant turn");
    const text = conversation?.textContent ?? "";
    expect(text.indexOf("final assistant turn")).toBeGreaterThan(text.indexOf("tests pass"));
    // Raw JSON stays one toggle away.
    expect(container.querySelector(".packet-detail-modes")).not.toBeNull();

    // Auxiliary side-calls are grouped, not dropped.
    const aux = Array.from(container.querySelectorAll<HTMLButtonElement>(".session-aux .conv-collapse-head"))
      .find((button) => button.textContent?.includes("auxiliary calls (2)"));
    expect(aux).toBeDefined();
    await click(aux);
    expect(container.querySelectorAll(".session-aux .packet-row").length).toBe(2);
  });

  it("keeps the per-session flow grid one toggle away and navigates back home", async () => {
    const { container } = await mountWorkspace(stateOf(FLOWS));
    await click(sessionRows(container)[0]);
    await settle();
    const flowsToggle = Array.from(container.querySelectorAll<HTMLButtonElement>(".session-detail-modes .packet-mode"))
      .find((button) => button.textContent === "flows");
    await click(flowsToggle);
    expect(container.querySelectorAll(".packet-row")).toHaveLength(3);
    expect(container.querySelector("[data-testid='conversation-view']")).toBeNull();

    await click(container.querySelector(".session-back"));
    expect(sessionRows(container)).toHaveLength(2);
  });

  it("drills the unassigned bucket straight into the flow-grid experience", async () => {
    const { container, requested } = await mountWorkspace(stateOf(FLOWS));
    await click(sessionRows(container)[1]);
    await settle();
    expect(container.querySelectorAll(".packet-row")).toHaveLength(1);
    expect(container.querySelector(".packet-row")?.textContent).toContain("other.example.test");
    expect(requested).toEqual([]);
  });

  it("switches to the global flow grid (with search) and back via the nav", async () => {
    const { container } = await mountWorkspace(stateOf(FLOWS));
    expect(container.querySelector(".search-input")).toBeNull();
    const flowsNav = Array.from(container.querySelectorAll<HTMLButtonElement>(".app-nav .packet-mode"))
      .find((button) => button.textContent === "flows");
    await click(flowsNav);
    expect(container.querySelectorAll(".packet-row")).toHaveLength(4);
    expect(container.querySelector(".search-input")).not.toBeNull();
    const sessionsNav = Array.from(container.querySelectorAll<HTMLButtonElement>(".app-nav .packet-mode"))
      .find((button) => button.textContent === "sessions");
    await click(sessionsNav);
    expect(sessionRows(container)).toHaveLength(2);
  });

  it("shows a quiet retention notice when the selected session has been pruned", async () => {
    const { container, render } = await mountWorkspace(stateOf(FLOWS));
    await click(sessionRows(container)[0]);
    await settle();
    // Simulate retention pruning the whole session away.
    await render(stateOf([FLOWS[3]]));
    expect(container.textContent).toContain("session no longer retained");
    await click(container.querySelector(".session-back"));
    expect(sessionRows(container)).toHaveLength(1);
  });

  it("prefix-dedupe regression: a newer equal-count utility branch never becomes the conversation", async () => {
    const session = "cccc9999-0000-1111-2222-333333333333";
    const branchRequest = JSON.stringify({
      model: "claude-opus-4",
      messages: [
        { role: "user", content: "fix the bug" },
        { role: "assistant", content: "checking quota" },
        { role: "user", content: "what quota remains?" },
      ],
    });
    const oldMainRequest = JSON.stringify({
      model: "claude-opus-4",
      messages: [{ role: "user", content: "fix the bug" }],
    });
    const flows = [
      // Newest, same message_count as the main thread, no suggestion marker:
      // the retired count/recency heuristic selected THIS flow.
      sessionFlow("branch-flow", {
        session_id: session,
        started_at: "2026-01-01T00:03:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "user_text", text: "what quota remains?" } },
      }),
      sessionFlow("main2-flow", {
        session_id: session,
        started_at: "2026-01-01T00:02:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "tool_result", tool_name: "Bash" } },
      }),
      sessionFlow("old-main-flow", {
        session_id: session,
        started_at: "2026-01-01T00:01:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "1", preview: { source: "user_text", text: "fix the bug" } },
      }),
    ];
    const { container, requested } = await mountWorkspace(stateOf(flows), {
      "branch-flow": { request: branchRequest },
      "main2-flow": { request: MAIN_REQUEST, response: MAIN_RESPONSE },
      "old-main-flow": { request: oldMainRequest },
    });
    await click(sessionRows(container)[0]);
    await settle();
    expect(new Set(requested)).toEqual(new Set(["branch-flow", "main2-flow", "old-main-flow"]));
    const conversation = container.querySelector("[data-testid='conversation-view']");
    // Canonical = tip of the dominant prefix chain {old-main -> main2}, so
    // the chat shows the main thread and its response as the final turn.
    expect(conversation?.textContent).toContain("tests pass");
    expect(conversation?.textContent).toContain("final assistant turn");
    expect(conversation?.textContent).not.toContain("quota");
    // The off-chain branch stays reachable in the auxiliary group.
    const aux = Array.from(container.querySelectorAll<HTMLButtonElement>(".session-aux .conv-collapse-head"))
      .find((button) => button.textContent?.includes("auxiliary calls (2)"));
    expect(aux).toBeDefined();
  });

  it("refetches the same flow when it transitions incomplete-to-complete and shows the final turn", async () => {
    const session = "dddd8888-0000-1111-2222-333333333333";
    const liveRequest = JSON.stringify({ model: "claude-opus-4", messages: [{ role: "user", content: "hello" }] });
    const liveResponse = JSON.stringify({
      model: "claude-opus-4",
      role: "assistant",
      content: [{ type: "text", text: "late final answer" }],
      stop_reason: "end_turn",
    });
    const inFlight = {
      flow_id: "live-flow",
      session_id: session,
      method: "POST",
      scheme: "https",
      host: "api.anthropic.test",
      port: "443",
      path: "/v1/messages",
      request_headers: [{ name: "content-type", value: "application/json" }],
      request_body: { state: "truncated", size_bytes: "64", captured_bytes: "0", encoding: "base64", data: "" },
      started_at: "2026-01-01T00:00:00Z",
      summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "1", preview: { source: "user_text", text: "hello" } },
    };
    const completed = {
      ...inFlight,
      response_headers: [{ name: "content-type", value: "application/json" }],
      response_status: "200",
      response_body: { state: "truncated", size_bytes: "128", captured_bytes: "0", encoding: "base64", data: "" },
      ended_at: "2026-01-01T00:00:09Z",
    };
    let complete = false;
    const { container, requested, render } = await mountWorkspace(
      stateOf([inFlight]),
      () => ({ "live-flow": complete ? { request: liveRequest, response: liveResponse } : { request: liveRequest } }),
    );
    await click(sessionRows(container)[0]);
    await settle();
    expect(requested).toEqual(["live-flow"]);
    expect(container.querySelector("[data-testid='conversation-view']")?.textContent).toContain("hello");
    expect(container.textContent).not.toContain("late final answer");

    // The flow completes: same id, new terminal fields. The detail cache must
    // invalidate and refetch instead of pinning the response-less body.
    complete = true;
    await render(stateOf([completed]));
    await settle();
    expect(requested).toEqual(["live-flow", "live-flow"]);
    expect(container.querySelector("[data-testid='conversation-view']")?.textContent).toContain("late final answer");
  });
});
