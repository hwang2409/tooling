import { describe, expect, it } from "vitest";

import { parseProtocolMessage } from "../../protocol";
import { browserReducer, initialBrowserState } from "../../state/browserState";
import type { BrowserState, FlowsUpdate, ImmutableFlowMetadata } from "../../state/browserState";
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

  it("firstQuery regression: skips suggestion-mode flows even when they are the oldest retained", () => {
    // After pruning, the oldest RETAINED flow can be an auxiliary
    // suggestion-mode call; its preview is the injected internal prompt and
    // must never surface as the session's opening query on the home row.
    const sessions = deriveSessions([
      flow("a3", { session: "aaaa", userText: "real follow-up" }),
      flow("a2", { session: "aaaa", userText: "[SUGGESTION MODE: propose next] internals" }),
    ]);
    expect(sessions[0].firstQuery).toBe("real follow-up");
    // All-suggestion sessions fall through to no query at all.
    const onlySuggestions = deriveSessions([
      flow("s1", { session: "bbbb", userText: "[SUGGESTION MODE: propose next] internals" }),
    ]);
    expect(onlySuggestions[0].firstQuery).toBeUndefined();
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
  const delta = (revision: number, changedFlowIds: readonly string[], prependedCount = 0): FlowsUpdate =>
    ({ revision, kind: "delta", changedFlowIds, prependedCount });
  const snapshot = (revision: number): FlowsUpdate =>
    ({ revision, kind: "snapshot", changedFlowIds: [], prependedCount: 0 });

  it("touches only the sessions owning the changed flow ids, order work included", () => {
    const sessionCount = 20;
    const base = Array.from({ length: sessionCount }, (_, position) =>
      flow(`f${position}`, { session: `sess-${position}` }));
    const index = createSessionIndex();
    const first = index.update(collectionOf(base), snapshot(1));
    const afterRebuild = index.stats();
    expect(afterRebuild.fullRebuilds).toBe(1);
    // The full rebuild orders by first appearance — no incremental order work.
    expect(afterRebuild.orderVisits).toBe(0);

    const extra = flow("extra", { session: "sess-7" });
    const second = index.update(collectionOf([extra, ...base]), delta(2, ["extra"], 1));
    const afterDelta = index.stats();
    expect(afterDelta.incrementalUpdates).toBe(1);
    expect(afterDelta.fullRebuilds).toBe(1);
    // Exactly ONE session resummarized; untouched sessions keep identity.
    expect(afterDelta.sessionsRecomputed - afterRebuild.sessionsRecomputed).toBe(1);
    expect(second.find((session) => session.key === "sess-3"))
      .toBe(first.find((session) => session.key === "sess-3"));
    // Order maintenance is two binary searches repositioning ONE key. The
    // POSITIVE lower bound proves the instrumented incremental path actually
    // ran (a full-sort revert records zero visits and fails it); the upper
    // bound proves the DECISION work stays O(log sessions).
    const orderWork = afterDelta.orderVisits - afterRebuild.orderVisits;
    expect(orderWork).toBeGreaterThanOrEqual(2 + 2 * Math.floor(Math.log2(sessionCount)));
    expect(orderWork).toBeLessThanOrEqual(2 * (Math.ceil(Math.log2(sessionCount)) + 2));
    // Honest linear accounting: the splices behind ONE reposition shift up
    // to O(sessions) pointers, and materializing the fresh result array
    // copies exactly one pointer per session — measured, bounded, and free
    // of per-session recomputation.
    const shiftWork = afterDelta.orderShifts - afterRebuild.orderShifts;
    expect(shiftWork).toBeGreaterThanOrEqual(1);
    expect(shiftWork).toBeLessThanOrEqual(2 * sessionCount);
    expect(afterDelta.resultCopies - afterRebuild.resultCopies).toBe(sessionCount);
    expect(second[0].key).toBe("sess-7");
    expect(second[0].flowCount).toBe(2);
    expect(second).toHaveLength(sessionCount);
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
    const rebuilt = index.update(collectionOf([b1, a1]), delta(3, ["b1"], 1));
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

  it("pins deduplicated final-grid-order changed ids: upsert A, upsert B, remove A, upsert A", () => {
    const boot = [
      {
        protocol_version: "1", type: "source.hello", source_id: "source-a",
        occurred_at: "2026-01-01T00:00:00Z",
        capabilities: { body_chunks: true, redaction: "headers-and-query" },
        limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
      },
      { protocol_version: "1", type: "browser.snapshot", snapshot_id: "snap-1", cursor: "1", flows: [] },
    ].reduce<BrowserState>(
      (current, message) => browserReducer(current, { type: "protocol", envelope: parseProtocolMessage(message) }),
      initialBrowserState,
    );
    const index = createSessionIndex();
    index.update(boot.flows, boot.flowsUpdate);

    const message = {
      protocol_version: "1", type: "browser.delta", cursor: "2",
      changes: [
        { op: "upsert", flow: flow("a-flow", { session: "aaaa" }) },
        { op: "upsert", flow: flow("b-flow", { session: "bbbb" }) },
        { op: "remove", flow_id: "a-flow" },
        { op: "upsert", flow: flow("a-flow", { session: "aaaa" }) },
      ],
    };
    const state = browserReducer(boot, { type: "protocol", envelope: parseProtocolMessage(message) });
    // The grid ends as [B, A]: A's first upsert was removed mid-delta and A
    // was reinserted after B. Raw message.changes ids would report the
    // duplicated [A, B, A] — the published contract is deduplicated ids in
    // FINAL grid order.
    expect(state.flows.ids).toEqual(["b-flow", "a-flow"]);
    expect(state.flowsUpdate.changedFlowIds).toEqual(["b-flow", "a-flow"]);
    expect(state.flowsUpdate.prependedCount).toBe(2);

    const incremental = index.update(state.flows, state.flowsUpdate);
    const scratch = deriveSessions(state.flows.entries);
    expect(incremental.map((session) => session.key)).toEqual(["bbbb", "aaaa"]);
    expect(incremental.map((session) => session.key)).toEqual(scratch.map((session) => session.key));
  });

  it("matches from-scratch derivation across randomized reducer-driven deltas", () => {
    // Drive real protocol messages through the reducer so the published
    // FlowsUpdate (dedupe, final grid order, prepended count) is the exact
    // production contract — including repeated upserts of one id in a
    // single delta and remove-then-reinsert of an existing id.
    let seed = 1337;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const pick = <Item,>(items: readonly Item[]): Item => items[Math.floor(rand() * items.length)];
    const sessionPool: Array<string | null> = ["s1", "s2", "s3", null];
    const makeFlow = (flowId: string): ImmutableFlowMetadata => flow(flowId, {
      session: pick(sessionPool),
      status: rand() < 0.2 ? "500" : "200",
      model: rand() < 0.5 ? "claude-opus-4" : "claude-haiku-4",
    });

    let state: BrowserState = [
      {
        protocol_version: "1", type: "source.hello", source_id: "source-a",
        occurred_at: "2026-01-01T00:00:00Z",
        capabilities: { body_chunks: true, redaction: "headers-and-query" },
        limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
      },
      { protocol_version: "1", type: "browser.snapshot", snapshot_id: "snap-1", cursor: "1", flows: [] },
    ].reduce((current, message) => browserReducer(current, { type: "protocol", envelope: parseProtocolMessage(message) }), initialBrowserState);

    const index = createSessionIndex();
    index.update(state.flows, state.flowsUpdate);
    let cursor = 1;
    let nextId = 0;

    for (let step = 0; step < 150; step += 1) {
      const changes: Array<Record<string, unknown>> = [];
      const operation = rand();
      const existing = state.flows.entries;
      if (operation < 0.35 || existing.length === 0) {
        const count = 1 + Math.floor(rand() * 3);
        for (let item = 0; item < count; item += 1) {
          changes.push({ op: "upsert", flow: makeFlow(`f${nextId}`) });
          nextId += 1;
        }
      } else if (operation < 0.5) {
        // Repeated upsert of the SAME new id within one delta.
        const flowId = `f${nextId}`;
        nextId += 1;
        changes.push({ op: "upsert", flow: makeFlow(flowId) });
        changes.push({ op: "upsert", flow: makeFlow(flowId) });
      } else if (operation < 0.7) {
        const target = pick(existing);
        changes.push({ op: "upsert", flow: { ...target, response_status: "201" } });
      } else if (operation < 0.85) {
        changes.push({ op: "remove", flow_id: pick(existing).flow_id });
      } else {
        // Remove then reinsert the same existing id in one delta: the flow
        // jumps to the top of the grid.
        const target = pick(existing);
        changes.push({ op: "remove", flow_id: target.flow_id });
        changes.push({ op: "upsert", flow: makeFlow(target.flow_id) });
      }
      cursor += 1;
      const message = { protocol_version: "1", type: "browser.delta", cursor: String(cursor), changes };
      state = browserReducer(state, { type: "protocol", envelope: parseProtocolMessage(message) });

      const incremental = index.update(state.flows, state.flowsUpdate);
      const scratch = deriveSessions(state.flows.entries);
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
    expect(index.stats().incrementalUpdates).toBe(150);
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
