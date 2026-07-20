import { describe, expect, it } from "vitest";

import type { ImmutableFlowMetadata } from "../../state/browserState";
import { parseAnthropicRequest } from "../inspector/anthropic";
import { conversationCandidates, deriveSessions, isSuggestionRequest, looksLikeSuggestionFlow } from "./sessionSummary";

interface FlowOptions {
  session?: string | null;
  status?: string;
  startedAt?: string;
  endedAt?: string;
  model?: string;
  userText?: string;
  kind?: "anthropic_messages" | "anthropic_count_tokens" | "generic";
  messageCount?: string;
}

function flow(flowId: string, options: FlowOptions = {}): ImmutableFlowMetadata {
  const summary: Record<string, unknown> = { kind: options.kind ?? "anthropic_messages" };
  if (options.model !== undefined) summary.model = options.model;
  if (options.messageCount !== undefined) summary.message_count = options.messageCount;
  if (options.userText !== undefined) summary.preview = { source: "user_text", text: options.userText };
  return {
    flow_id: flowId,
    session_id: options.session === undefined ? null : options.session,
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: `/v1/${flowId}`,
    request_headers: [],
    request_body: { state: "missing" },
    ...(options.status !== undefined ? { response_status: options.status } : {}),
    ...(options.startedAt !== undefined ? { started_at: options.startedAt } : {}),
    ...(options.endedAt !== undefined ? { ended_at: options.endedAt } : {}),
    summary,
  } as unknown as ImmutableFlowMetadata;
}

describe("deriveSessions", () => {
  it("groups flows per session keyed by session_id, newest session first", () => {
    // Grid order is newest-first: b2 arrived last.
    const sessions = deriveSessions([
      flow("b2", { session: "bbbb" }),
      flow("a2", { session: "aaaa" }),
      flow("b1", { session: "bbbb" }),
      flow("a1", { session: "aaaa" }),
    ]);
    expect(sessions.map((session) => session.key)).toEqual(["bbbb", "aaaa"]);
    expect(sessions[0].flowCount).toBe(2);
    expect(sessions[0].flows.map((entry) => entry.flow_id)).toEqual(["b2", "b1"]);
    expect(sessions[1].flows.map((entry) => entry.flow_id)).toEqual(["a2", "a1"]);
  });

  it("merges interleaved runs of one session into a single summary", () => {
    const sessions = deriveSessions([
      flow("a2", { session: "aaaa" }),
      flow("b1", { session: "bbbb" }),
      flow("a1", { session: "aaaa" }),
    ]);
    expect(sessions.map((session) => session.key)).toEqual(["aaaa", "bbbb"]);
    expect(sessions[0].flowCount).toBe(2);
  });

  it("collapses null session ids into one unassigned bucket", () => {
    const sessions = deriveSessions([
      flow("n2", { session: null }),
      flow("a1", { session: "aaaa" }),
      flow("n1", { session: null }),
    ]);
    expect(sessions.map((session) => session.key)).toEqual([null, "aaaa"]);
    expect(sessions[0].flowCount).toBe(2);
  });

  it("takes the oldest user_text preview as the session's first query", () => {
    const sessions = deriveSessions([
      flow("a3", { session: "aaaa", userText: "latest follow-up" }),
      flow("a2", { session: "aaaa" }),
      flow("a1", { session: "aaaa", userText: "initial ask" }),
    ]);
    expect(sessions[0].firstQuery).toBe("initial ask");
  });

  it("collects distinct models oldest-first and flags 4xx/5xx errors", () => {
    const sessions = deriveSessions([
      flow("a3", { session: "aaaa", model: "claude-opus-4", status: "529" }),
      flow("a2", { session: "aaaa", model: "claude-haiku-4", status: "200" }),
      flow("a1", { session: "aaaa", model: "claude-opus-4", status: "200" }),
    ]);
    expect(sessions[0].models).toEqual(["claude-opus-4", "claude-haiku-4"]);
    expect(sessions[0].hasError).toBe(true);
    const clean = deriveSessions([flow("b1", { session: "bbbb", status: "200" })]);
    expect(clean[0].hasError).toBe(false);
  });

  it("derives start, last activity, and tolerates missing timings", () => {
    const sessions = deriveSessions([
      flow("a3", { session: "aaaa", startedAt: "2026-01-01T00:02:00Z" }),
      flow("a2", { session: "aaaa", startedAt: "2026-01-01T00:01:00Z", endedAt: "2026-01-01T00:03:00Z" }),
      flow("a1", { session: "aaaa", startedAt: "2026-01-01T00:00:00Z", endedAt: "2026-01-01T00:00:30Z" }),
    ]);
    expect(sessions[0].startedAt).toBe("2026-01-01T00:00:00Z");
    expect(sessions[0].lastActivity).toBe("2026-01-01T00:03:00Z");
    const bare = deriveSessions([flow("b1", { session: "bbbb" })]);
    expect(bare[0].startedAt).toBeUndefined();
    expect(bare[0].lastActivity).toBeUndefined();
  });

  it("reflects pruning: sessions derive only from the retained flows", () => {
    const retained = [flow("a2", { session: "aaaa" }), flow("b1", { session: "bbbb" })];
    const before = deriveSessions([...retained, flow("a1", { session: "aaaa" })]);
    expect(before.map((session) => session.flowCount)).toEqual([2, 1]);
    const after = deriveSessions(retained);
    expect(after.map((session) => session.key)).toEqual(["aaaa", "bbbb"]);
    expect(after.map((session) => session.flowCount)).toEqual([1, 1]);
  });
});

describe("conversationCandidates", () => {
  it("ranks the highest message_count anthropic_messages flow first", () => {
    const ranked = conversationCandidates([
      flow("side", { session: "aaaa", messageCount: "2" }),
      flow("main", { session: "aaaa", messageCount: "40" }),
      flow("count", { session: "aaaa", kind: "anthropic_count_tokens", messageCount: "99" }),
    ]);
    expect(ranked.map((entry) => entry.flow_id)).toEqual(["main", "side"]);
  });

  it("prefers the newest flow on ties and is empty without conversation flows", () => {
    const ranked = conversationCandidates([
      flow("newer", { session: "aaaa", messageCount: "3" }),
      flow("older", { session: "aaaa", messageCount: "3" }),
    ]);
    expect(ranked.map((entry) => entry.flow_id)).toEqual(["newer", "older"]);
    expect(conversationCandidates([flow("g1", { session: "aaaa", kind: "generic" })])).toEqual([]);
  });

  it("ranks a suggestion-mode sibling behind the main thread even with equal message_count", () => {
    // Suggestion-mode calls resend the full history, so message_count alone
    // cannot separate them from the main thread; the preview marker must.
    const ranked = conversationCandidates([
      flow("suggestion", { session: "aaaa", messageCount: "40", userText: "[SUGGESTION MODE: propose next steps] please" }),
      flow("main", { session: "aaaa", messageCount: "40", userText: "run the tests" }),
    ]);
    expect(ranked.map((entry) => entry.flow_id)).toEqual(["main", "suggestion"]);
    expect(looksLikeSuggestionFlow(ranked[1])).toBe(true);
    expect(looksLikeSuggestionFlow(ranked[0])).toBe(false);
  });
});

describe("isSuggestionRequest", () => {
  it("detects the injected suggestion prompt on the last user message", () => {
    const request = parseAnthropicRequest({
      messages: [
        { role: "user", content: "real question" },
        { role: "assistant", content: "answer" },
        { role: "user", content: [{ type: "text", text: "[SUGGESTION MODE: suggest things]" }] },
      ],
    });
    expect(request).not.toBeNull();
    expect(isSuggestionRequest(request!)).toBe(true);
  });

  it("accepts a genuine main-thread request whose earlier turns mention the marker", () => {
    const request = parseAnthropicRequest({
      messages: [
        { role: "user", content: "[SUGGESTION MODE: old aside]" },
        { role: "assistant", content: "answer" },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "ok" }] },
      ],
    });
    expect(request).not.toBeNull();
    expect(isSuggestionRequest(request!)).toBe(false);
  });
});
