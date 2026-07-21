// @vitest-environment jsdom

import { Buffer } from "node:buffer";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { Workspace } from "../../App";
import { parseProtocolMessage } from "../../protocol";
import { browserReducer, initialBrowserState } from "../../state/browserState";
import type { BrowserState, ImmutableFlowMetadata } from "../../state/browserState";
import type { FlowDetailLoader } from "../inspector/flowDetail";
import { createCanonicalIndex } from "./canonical";
import type { CanonicalIndex } from "./canonical";
import { createCandidateParser } from "./SessionDetail";
import type { CandidateParser } from "./SessionDetail";
import { createSessionIndex } from "./sessionSummary";
import type { SessionIndex } from "./sessionSummary";

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

interface MountInstruments {
  sessionIndex?: SessionIndex;
  candidateParser?: CandidateParser;
  canonicalIndex?: CanonicalIndex;
}

async function mountWorkspace(
  browser: BrowserState,
  bodies: DetailBodies | (() => DetailBodies) = DETAIL_BODIES,
  { sessionIndex, candidateParser, canonicalIndex }: MountInstruments = {},
) {
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
    await act(async () => root.render(
      <Workspace
        browser={nextBrowser}
        loadFlowDetail={loadFlowDetail}
        sessionIndex={sessionIndex}
        candidateParser={candidateParser}
        canonicalIndex={canonicalIndex}
      />,
    ));
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

  it("lists canonical request first, then newest candidates, and manually renders a utility flow", async () => {
    const utilityRequest = JSON.stringify({
      model: "claude-haiku-4",
      messages: [
        { role: "user", content: "fix the bug" },
        { role: "assistant", content: "checking quota" },
      ],
    });
    const utilityResponse = JSON.stringify({
      model: "claude-haiku-4",
      role: "assistant",
      content: [{ type: "text", text: "utility answer" }],
      stop_reason: "end_turn",
    });
    const utility = sessionFlow("utility-flow", {
      started_at: "2026-01-01T00:00:30Z",
      ended_at: "2026-01-01T00:00:40Z",
      summary: {
        kind: "anthropic_messages",
        model: "claude-haiku-4",
        message_count: "2",
        preview: { source: "user_text", text: "checking quota" },
      },
    });
    const { container } = await mountWorkspace(
      stateOf([FLOWS[0], FLOWS[1], utility]),
      { ...DETAIL_BODIES, "utility-flow": { request: utilityRequest, response: utilityResponse } },
    );
    await click(sessionRows(container)[0]);
    await settle();

    const source = container.querySelector<HTMLButtonElement>("[data-testid='rendered-source']");
    expect(source?.textContent).toContain("rendered from request main-flo");
    expect(source?.textContent).toContain("auto");
    await click(source);
    const options = Array.from(container.querySelectorAll<HTMLButtonElement>("[data-testid='session-source-option']"));
    expect(options.map((option) => option.dataset.flowId)).toEqual(["main-flow", "aux-suggestion", "utility-flow"]);
    expect(options[0].textContent).toContain("3m ·");
    expect(options[0].textContent).toContain("opus-4");
    expect(options[1].textContent).toContain("suggestion");
    expect(options[2].textContent).toContain("2m ·");
    expect(options[2].textContent).toContain("haiku-4");
    expect(options[2].textContent).toContain("utility");

    await click(options[2]);
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("utility-");
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("manual");
    expect(container.querySelector("[data-testid='conversation-view']")?.textContent).toContain("utility answer");
    const raw = Array.from(container.querySelectorAll<HTMLButtonElement>(".packet-detail-modes .packet-mode"))
      .find((button) => button.textContent === "raw");
    await click(raw);
    expect(container.querySelector(".packet-detail .json-tree")).not.toBeNull();
  });

  it("resets manual source choice across source reconnect while SessionDetail stays mounted", async () => {
    const initial = stateOf(FLOWS);
    const { container, render } = await mountWorkspace(initial);
    await click(sessionRows(container)[0]);
    await settle();
    await click(container.querySelector<HTMLButtonElement>("[data-testid='rendered-source']"));
    const options = Array.from(container.querySelectorAll<HTMLButtonElement>("[data-testid='session-source-option']"));
    await click(options[1]);
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("manual");

    // Same workspace view and same session key; only source incarnation changes.
    await render({ ...initial, sourceEpoch: initial.sourceEpoch + 1 });
    await settle();
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("main-flo");
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("auto");
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).not.toContain("manual");
  });

  it("does not restore a picked flow after prune and reinsert", async () => {
    const initial = stateOf(FLOWS);
    const { container, render } = await mountWorkspace(initial);
    await click(sessionRows(container)[0]);
    await settle();
    await click(container.querySelector<HTMLButtonElement>("[data-testid='rendered-source']"));
    const options = Array.from(container.querySelectorAll<HTMLButtonElement>("[data-testid='session-source-option']"));
    await click(options[1]);
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("manual");

    await render(stateOf([FLOWS[1]]));
    await settle();
    await render(initial);
    await settle();
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("main-flo");
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).toContain("auto");
    expect(container.querySelector("[data-testid='rendered-source']")?.textContent).not.toContain("manual");
  });

  it("uses a labeled native-button dialog with Escape and focus return", async () => {
    const { container } = await mountWorkspace(stateOf(FLOWS));
    await click(sessionRows(container)[0]);
    await settle();
    const source = container.querySelector<HTMLButtonElement>("[data-testid='rendered-source']");
    await click(source);
    const picker = container.querySelector("#session-source-picker");
    expect(picker?.getAttribute("role")).toBe("dialog");
    expect(picker?.getAttribute("aria-labelledby")).toBe("session-source-picker-label");
    expect(container.querySelector("[role='listbox']")).toBeNull();
    const firstOption = container.querySelector<HTMLButtonElement>("[data-testid='session-source-option']");
    expect(document.activeElement).toBe(firstOption);

    await act(async () => document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(container.querySelector("#session-source-picker")).toBeNull();
    expect(document.activeElement).toBe(source);
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

  it("prefix regression: an older equal-count branch never steals the conversation from the live thread", async () => {
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
      sessionFlow("main2-flow", {
        session_id: session,
        started_at: "2026-01-01T00:03:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "tool_result", tool_name: "Bash" } },
      }),
      // Older equal-count divergent branch sharing the root: the retired
      // greedy partition let it steal the shared root and dominate by
      // member count.
      sessionFlow("branch-flow", {
        session_id: session,
        started_at: "2026-01-01T00:02:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "user_text", text: "what quota remains?" } },
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
    // Canonical = the maximal chain with the newest tip ({old-main -> main2}),
    // so the chat shows the main thread and its response as the final turn.
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

  it("refetches detail when a reconnected source reuses a flow id (epoch is part of cache identity)", async () => {
    const session = "ffff6666-0000-1111-2222-333333333333";
    const reused = sessionFlow("reused-flow", {
      session_id: session,
      started_at: "2026-01-01T00:00:00Z",
      ended_at: "2026-01-01T00:00:05Z",
      summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "1", preview: { source: "user_text", text: "hello" } },
    });
    const state1 = stateOf([reused]);
    // A NEW source reuses the exact same flow id and body descriptors, so
    // flow id + detail version alone cannot distinguish the captures.
    const state2 = [
      {
        protocol_version: "1", type: "source.hello", source_id: "source-b",
        occurred_at: "2026-01-01T01:00:00Z",
        capabilities: { body_chunks: true, redaction: "headers-and-query" },
        limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
      },
      { protocol_version: "1", type: "browser.snapshot", snapshot_id: "snap-2", cursor: "1", flows: [reused] },
    ].reduce(
      (current, message) => browserReducer(current, { type: "protocol", envelope: parseProtocolMessage(message) }),
      state1,
    );
    const requestFor = (marker: string) => JSON.stringify({
      model: "claude-opus-4",
      messages: [{ role: "user", content: `hello from ${marker}` }],
    });
    let source = "source A";
    const { container, requested, render } = await mountWorkspace(
      state1,
      () => ({ "reused-flow": { request: requestFor(source) } }),
    );
    await click(sessionRows(container)[0]);
    await settle();
    expect(requested).toEqual(["reused-flow"]);
    expect(container.querySelector("[data-testid='conversation-view']")?.textContent).toContain("hello from source A");

    source = "source B";
    await render(state2);
    await settle();
    // Same id, same version — but a different capture incarnation MUST load
    // fresh, never serve the previous source's body as current.
    expect(requested).toEqual(["reused-flow", "reused-flow"]);
    const conversation = container.querySelector("[data-testid='conversation-view']");
    expect(conversation?.textContent).toContain("hello from source B");
    expect(conversation?.textContent).not.toContain("hello from source A");
  });

  it("renders the auxiliary/flow view, not a fake chat, when every candidate is suggestion-mode", async () => {
    const session = "eeee7777-0000-1111-2222-333333333333";
    const flows = [
      sessionFlow("sugg-2", {
        session_id: session,
        started_at: "2026-01-01T00:02:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "tool_result", tool_name: "Bash" } },
      }),
      sessionFlow("sugg-1", {
        session_id: session,
        started_at: "2026-01-01T00:01:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "user_text", text: "[SUGGESTION MODE: propose] x" } },
      }),
    ];
    const { container } = await mountWorkspace(stateOf(flows), {
      "sugg-2": { request: SUGGESTION_REQUEST },
      "sugg-1": { request: SUGGESTION_REQUEST },
    });
    await click(sessionRows(container)[0]);
    await settle();
    // No verifiable main thread: never promote a rejected suggestion call
    // into a fake conversation.
    expect(container.querySelector("[data-testid='conversation-view']")).toBeNull();
    expect(container.textContent).toContain("auxiliary calls (2)");
    // The auxiliary group arrives expanded so the flows are immediately visible.
    expect(container.querySelectorAll(".session-aux .packet-row")).toHaveLength(2);
  });

  it("context regression: an identical-history different-model side-call never steals the conversation", async () => {
    const session = "abab5555-0000-1111-2222-333333333333";
    // The haiku side-call resends the main thread's EXACT message history
    // under its own model/system and is the newest flow in the session.
    const cloneRequest = JSON.stringify({
      model: "claude-haiku-4",
      system: "you generate titles",
      messages: JSON.parse(MAIN_REQUEST).messages,
    });
    const cloneResponse = JSON.stringify({
      model: "claude-haiku-4",
      role: "assistant",
      content: [{ type: "text", text: "borrowed-history side answer" }],
      stop_reason: "end_turn",
    });
    const oldMainRequest = JSON.stringify({
      model: "claude-opus-4",
      messages: [{ role: "user", content: "fix the bug" }],
    });
    const flows = [
      sessionFlow("haiku-clone", {
        session_id: session,
        started_at: "2026-01-01T00:03:00Z",
        summary: { kind: "anthropic_messages", model: "claude-haiku-4", message_count: "3", preview: { source: "tool_result", tool_name: "Bash" } },
      }),
      sessionFlow("main-flow", {
        session_id: session,
        started_at: "2026-01-01T00:02:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "3", preview: { source: "user_text", text: "fix the bug" } },
      }),
      sessionFlow("old-main-flow", {
        session_id: session,
        started_at: "2026-01-01T00:01:00Z",
        summary: { kind: "anthropic_messages", model: "claude-opus-4", message_count: "1", preview: { source: "user_text", text: "fix the bug" } },
      }),
    ];
    const { container } = await mountWorkspace(stateOf(flows), {
      "haiku-clone": { request: cloneRequest, response: cloneResponse },
      "main-flow": { request: MAIN_REQUEST, response: MAIN_RESPONSE },
      "old-main-flow": { request: oldMainRequest },
    });
    await click(sessionRows(container)[0]);
    await settle();
    const conversation = container.querySelector("[data-testid='conversation-view']");
    // The main thread renders with ITS response — a different-context call
    // with an equal history must not win purely by being newer.
    expect(conversation?.textContent).toContain("final assistant turn");
    expect(conversation?.textContent).not.toContain("borrowed-history side answer");
    expect(container.textContent).toContain("auxiliary calls (2)");
  });

  it("work pin: a delta touching the open session reparses and re-relates only the changed candidate", async () => {
    const session = "cdcd4444-0000-1111-2222-333333333333";
    const chainMessages = (length: number) => Array.from({ length }, (_, index) =>
      ({ role: index % 2 === 0 ? "user" : "assistant", content: `turn ${index}` }));
    const chainRequest = (length: number) =>
      JSON.stringify({ model: "claude-opus-4", messages: chainMessages(length) });
    const chainFlow = (length: number) => sessionFlow(`chain-${length}`, {
      session_id: session,
      started_at: `2026-01-01T00:0${length}:00Z`,
      summary: {
        kind: "anthropic_messages",
        model: "claude-opus-4",
        message_count: String(length),
        preview: { source: "user_text", text: "turn 0" },
      },
    });
    const initialLengths = [4, 3, 2, 1];
    const bodies: DetailBodies = Object.fromEntries(
      [5, ...initialLengths].map((length) => [`chain-${length}`, { request: chainRequest(length) }]),
    );
    const parser = createCandidateParser();
    const index = createCanonicalIndex();
    const state1 = stateOf(initialLengths.map(chainFlow));
    const { container, requested, render } = await mountWorkspace(state1, bodies, {
      candidateParser: parser,
      canonicalIndex: index,
    });
    await click(sessionRows(container)[0]);
    await settle();
    expect(container.querySelector("[data-testid='conversation-view']")?.textContent).toContain("turn 3");
    const baseParses = parser.stats().parses;
    const basePrefix = index.stats().prefixEvaluations;
    const baseNormalizations = index.stats().normalizations;
    expect(baseNormalizations).toBe(initialLengths.length);

    // One delta extends the conversation: a single new flow prepends, every
    // grid order shifts, nothing else changes.
    const deltaEnvelope = parseProtocolMessage({
      protocol_version: "1",
      type: "browser.delta",
      cursor: "2",
      changes: [{ op: "upsert", flow: chainFlow(5) }],
    });
    await render(browserReducer(state1, { type: "protocol", envelope: deltaEnvelope }));
    await settle();
    expect(container.querySelector("[data-testid='conversation-view']")?.textContent).toContain("turn 4");
    // Detail fetch: only the new flow.
    expect(requested.filter((flowId) => flowId === "chain-5")).toHaveLength(1);
    expect(requested).toHaveLength(initialLengths.length + 1);
    // Parse cache: the new flow parses (metadata-only, then loaded); the
    // four unchanged candidates are cache hits. A full-reparse regression
    // reprocesses every candidate on every render and fails this bound.
    expect(parser.stats().parses - baseParses).toBeLessThanOrEqual(2);
    // Canonical index: one new candidate normalized, related against the
    // existing four — never a whole-session rescan (which would add
    // O(candidates^2) evaluations across the delta's renders).
    expect(index.stats().normalizations - baseNormalizations).toBe(1);
    const prefixDelta = index.stats().prefixEvaluations - basePrefix;
    expect(prefixDelta).toBeGreaterThanOrEqual(1);
    expect(prefixDelta).toBeLessThanOrEqual(2 * (initialLengths.length + 1));
  });

  it("pins parser-cache retention across add, prune, and source reset", () => {
    const parser = createCandidateParser();
    const flowA = FLOWS[0] as unknown as ImmutableFlowMetadata;
    const flowB = FLOWS[1] as unknown as ImmutableFlowMetadata;
    const order = (flows: readonly ImmutableFlowMetadata[]) =>
      new Map(flows.map((flow, index) => [flow.flow_id, index]));

    parser.parseAll([flowA], order([flowA]), new Map(), 1);
    expect(parser.stats().retainedEntries).toBe(1);
    parser.parseAll([flowA, flowB], order([flowA, flowB]), new Map(), 1);
    expect(parser.stats().retainedEntries).toBe(2);

    parser.parseAll([flowB], order([flowB]), new Map(), 1);
    expect(parser.stats().retainedEntries).toBe(1);
    // New source epoch replaces old identity instead of retaining stale cache.
    parser.parseAll([flowB], order([flowB]), new Map(), 2);
    expect(parser.stats().retainedEntries).toBe(1);
    parser.parseAll([], new Map(), new Map(), 2);
    expect(parser.stats().retainedEntries).toBe(0);
  });

  it("keeps one incremental session index across deltas (guards against fresh-index-per-render)", async () => {
    const inner = createSessionIndex();
    const index: SessionIndex = {
      update: (flows, flowsUpdate) => inner.update(flows, flowsUpdate),
      stats: () => inner.stats(),
    };
    const state1 = stateOf(FLOWS);
    const deltaEnvelope = parseProtocolMessage({
      protocol_version: "1",
      type: "browser.delta",
      cursor: "2",
      changes: [{
        op: "upsert",
        flow: sessionFlow("aux-new", {
          started_at: "2026-01-01T00:04:00Z",
          summary: { kind: "anthropic_count_tokens", model: "claude-haiku-4", count_tokens_result: "9" },
        }),
      }],
    });
    const state2 = browserReducer(state1, { type: "protocol", envelope: deltaEnvelope });

    const { container, render } = await mountWorkspace(state1, DETAIL_BODIES, { sessionIndex: index });
    expect(inner.stats().fullRebuilds).toBe(1);
    await render(state2);
    // The delta must flow through the SAME index incrementally; a fresh
    // index per render would register another full rebuild (or bypass this
    // index entirely) and fail here.
    expect(inner.stats().fullRebuilds).toBe(1);
    expect(inner.stats().incrementalUpdates).toBe(1);
    expect(sessionRows(container)[0].textContent).toContain("4f");
  });
});
