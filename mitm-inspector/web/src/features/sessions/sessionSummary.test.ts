import { describe, expect, it } from "vitest";

import type { FlowsUpdate, ImmutableFlowMetadata } from "../../state/browserState";
import { parseAnthropicRequest } from "../inspector/anthropic";
import { collectionOf, conversationCandidates, createSessionIndex, deriveSessions, isSuggestionRequest, looksLikeSuggestionFlow } from "./sessionSummary";

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

describe("createSessionIndex", () => {
  const delta = (revision: number, changedFlowIds: readonly string[]): FlowsUpdate =>
    ({ revision, kind: "delta", changedFlowIds });
  const snapshot = (revision: number): FlowsUpdate => ({ revision, kind: "snapshot", changedFlowIds: [] });

  it("touches only the sessions owning the changed flow ids", () => {
    const a1 = flow("a1", { session: "aaaa" });
    const b1 = flow("b1", { session: "bbbb" });
    const index = createSessionIndex();
    const first = index.update(collectionOf([a1, b1]), snapshot(1));
    expect(index.stats().fullRebuilds).toBe(1);
    const recomputedAfterRebuild = index.stats().sessionsRecomputed;

    const second = index.update(collectionOf([flow("b2", { session: "bbbb" }), a1, b1]), delta(2, ["b2"]));
    expect(index.stats().incrementalUpdates).toBe(1);
    expect(index.stats().fullRebuilds).toBe(1);
    // Work bound: exactly ONE session resummarized for the delta; the
    // untouched session is not regrouped and keeps its summary identity.
    expect(index.stats().sessionsRecomputed).toBe(recomputedAfterRebuild + 1);
    expect(second.find((session) => session.key === "aaaa"))
      .toBe(first.find((session) => session.key === "aaaa"));
    expect(second.map((session) => session.key)).toEqual(["bbbb", "aaaa"]);
    expect(second.find((session) => session.key === "bbbb")?.flowCount).toBe(2);
  });

  it("caches repeated revisions and full-rebuilds on a revision gap", () => {
    const a1 = flow("a1", { session: "aaaa" });
    const index = createSessionIndex();
    const collection = collectionOf([a1]);
    const first = index.update(collection, snapshot(1));
    expect(index.update(collection, snapshot(1))).toBe(first);
    expect(index.stats().fullRebuilds).toBe(1);
    // Skipped revision (coalesced renders, paused view resuming): the delta's
    // changed ids no longer describe the full difference — full rebuild.
    const b1 = flow("b1", { session: "bbbb" });
    const rebuilt = index.update(collectionOf([b1, a1]), delta(3, ["b1"]));
    expect(index.stats().fullRebuilds).toBe(2);
    expect(index.stats().incrementalUpdates).toBe(0);
    expect(rebuilt.map((session) => session.key)).toEqual(["bbbb", "aaaa"]);
  });

  it("handles incremental removals including whole-session pruning", () => {
    const a1 = flow("a1", { session: "aaaa" });
    const b1 = flow("b1", { session: "bbbb" });
    const b2 = flow("b2", { session: "bbbb" });
    const index = createSessionIndex();
    index.update(collectionOf([b2, a1, b1]), snapshot(1));
    const afterOne = index.update(collectionOf([b2, a1]), delta(2, ["b1"]));
    expect(afterOne.find((session) => session.key === "bbbb")?.flowCount).toBe(1);
    const afterSession = index.update(collectionOf([a1]), delta(3, ["b2"]));
    expect(afterSession.map((session) => session.key)).toEqual(["aaaa"]);
  });

  it("matches from-scratch derivation across randomized delta sequences", () => {
    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const pick = <Item,>(items: readonly Item[]): Item => items[Math.floor(rand() * items.length)];
    const sessionPool: Array<string | null> = ["s1", "s2", "s3", null];
    let entries: ImmutableFlowMetadata[] = [];
    let revision = 1;
    let nextId = 0;
    const index = createSessionIndex();
    index.update(collectionOf(entries), { revision, kind: "snapshot", changedFlowIds: [] });

    for (let step = 0; step < 120; step += 1) {
      const changed: string[] = [];
      const operation = rand();
      if (operation < 0.5 || entries.length === 0) {
        // Batch of brand-new flows, block-prepended in batch order.
        const fresh: ImmutableFlowMetadata[] = [];
        const count = 1 + Math.floor(rand() * 3);
        for (let item = 0; item < count; item += 1) {
          const created = flow(`f${nextId}`, {
            session: pick(sessionPool),
            status: rand() < 0.2 ? "500" : "200",
            model: rand() < 0.5 ? "claude-opus-4" : "claude-haiku-4",
          });
          nextId += 1;
          fresh.push(created);
          changed.push(created.flow_id);
        }
        entries = [...fresh, ...entries];
      } else if (operation < 0.8) {
        // In-place upsert of an existing flow.
        const target = pick(entries);
        const updated = { ...target, response_status: "201" } as ImmutableFlowMetadata;
        entries = entries.map((candidate) => (candidate === target ? updated : candidate));
        changed.push(target.flow_id);
      } else {
        const target = pick(entries);
        entries = entries.filter((candidate) => candidate !== target);
        changed.push(target.flow_id);
      }
      revision += 1;
      const incremental = index.update(collectionOf(entries), { revision, kind: "delta", changedFlowIds: changed });
      const scratch = deriveSessions(entries);
      expect(incremental.map((session) => session.key)).toEqual(scratch.map((session) => session.key));
      incremental.forEach((session, position) => {
        const expected = scratch[position];
        expect(session.flows.map((entry) => entry.flow_id)).toEqual(expected.flows.map((entry) => entry.flow_id));
        expect(session.flowCount).toBe(expected.flowCount);
        expect(session.hasError).toBe(expected.hasError);
        expect(session.models).toEqual(expected.models);
        expect(session.firstQuery).toBe(expected.firstQuery);
      });
    }
    expect(index.stats().incrementalUpdates).toBe(120);
    expect(index.stats().fullRebuilds).toBe(1);
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
