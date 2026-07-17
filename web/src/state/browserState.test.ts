import { describe, expect, it } from "vitest";

import { parseProtocolMessage } from "../protocol";
import {
  browserReducer,
  browserViewReducer,
  initialBrowserState,
  initialBrowserViewState,
} from "./browserState";
import type { BrowserState, BrowserViewState } from "./browserState";

function flow(flowId: string) {
  return {
    flow_id: flowId,
    method: "POST",
    scheme: "https",
    host: `${flowId}.example.test`,
    port: "443",
    path: "/v1/messages",
    request_headers: [],
    request_body: { state: "empty", size_bytes: "0" },
  } as const;
}

function hello(sourceId: string) {
  return {
    protocol_version: "1",
    type: "source.hello",
    source_id: sourceId,
    occurred_at: "2026-01-01T00:00:00Z",
    capabilities: { body_chunks: true, redaction: "headers-and-query" },
    limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
  };
}

function message(value: unknown) {
  return parseProtocolMessage(value);
}

function reduce(state: BrowserState, value: unknown): BrowserState {
  return browserReducer(state, { type: "protocol", envelope: message(value) });
}

function viewReduce(state: BrowserViewState, value: unknown): BrowserViewState {
  return browserViewReducer(state, { type: "protocol", envelope: message(value) });
}

describe("browser state reducer", () => {
  it("replaces a snapshot while preserving an explicit, prototype-safe order", () => {
    const next = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "7",
      flows: [flow("__proto__"), flow("10"), flow("2")],
    });

    expect(next.flows.ids).toEqual(["__proto__", "10", "2"]);
    expect(next.flows.entries.map((item) => item.flow_id)).toEqual(["__proto__", "10", "2"]);
    expect(next.flows.get("__proto__")?.host).toBe("__proto__.example.test");
    expect(Object.isFrozen(next.flows)).toBe(true);
    expect(Object.isFrozen(next.flows.ids)).toBe(true);
    expect(next.cursor).toBe("7");
    expect(next.snapshotId).toBe("one");
  });

  it("upserts in place, appends new IDs, and removes without numeric reordering", () => {
    const snapshot = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "4",
      flows: [flow("__proto__"), flow("10"), flow("2")],
    });
    const next = reduce(snapshot, {
      protocol_version: "1", type: "browser.delta", cursor: "5", changes: [
        { op: "upsert", flow: flow("10") }, { op: "remove", flow_id: "2" }, { op: "upsert", flow: flow("1") },
      ],
    });

    expect(next.flows.ids).toEqual(["__proto__", "10", "1"]);
    expect(next.flows.get("10")?.path).toBe("/v1/messages");
    expect(next.flows.get("2")).toBeUndefined();
    expect(next.counters.appliedChanges).toBe(6);
  });

  it("ignores stale snapshots and deltas without regressing the cursor", () => {
    const snapshot = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "4", flows: [flow("kept")],
    });
    const staleSnapshot = reduce(snapshot, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "old", cursor: "3", flows: [flow("stale")],
    });
    const staleDelta = reduce(staleSnapshot, {
      protocol_version: "1", type: "browser.delta", cursor: "4", changes: [{ op: "remove", flow_id: "kept" }],
    });

    expect(staleDelta.flows.ids).toEqual(["kept"]);
    expect(staleDelta.cursor).toBe("4");
    expect(staleDelta.counters.staleMessages).toBe(2);
  });

  it("keeps browser cursor gaps separate from stream sequence gaps at high cursors", () => {
    const snapshot = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "100", flows: [flow("kept")],
    });
    const gap = reduce(snapshot, {
      protocol_version: "1", type: "stream.gap", expected_sequence: "5", actual_sequence: "8",
    });

    expect(gap.cursor).toBe("100");
    expect(gap.gap).toBeNull();
    expect(gap.streamGap).toEqual({ expected: "5", actual: "8", droppedCount: null });
    expect(gap.counters.droppedMessages).toBe("0");
  });

  it("preserves omitted versus explicit zero and uint64 dropped counts", () => {
    const omitted = reduce(initialBrowserState, {
      protocol_version: "1", type: "stream.gap", expected_sequence: "5", actual_sequence: "8",
    });
    const zero = reduce(omitted, {
      protocol_version: "1", type: "stream.gap", expected_sequence: "9", actual_sequence: "10", dropped_count: "0",
    });
    const max = reduce(zero, {
      protocol_version: "1", type: "stream.gap", expected_sequence: "0", actual_sequence: "18446744073709551615",
      dropped_count: "18446744073709551614",
    });

    expect(omitted.streamGap?.droppedCount).toBeNull();
    expect(zero.streamGap?.droppedCount).toBe("0");
    expect(max.counters.droppedMessages).toBe("18446744073709551614");
  });

  it("tracks lifecycle delivery sequence independently from browser cursor", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const snapshot = reduce(source, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "100", flows: [],
    });
    const lifecycle = reduce(snapshot, {
      protocol_version: "1", type: "flow.lifecycle", source_id: "source-a", flow_id: "f", event_id: "e",
      occurred_at: "2026-01-01T00:00:00Z", sequence: "2", state: "response_started",
    });
    const next = reduce(lifecycle, {
      protocol_version: "1", type: "browser.delta", cursor: "101", changes: [],
    });

    expect(next.streamSequence).toBe("2");
    expect(next.cursor).toBe("101");
    expect(next.streamGap).toBeNull();
  });

  it("keys lifecycle sequence by source epoch and ignores late A after switching to B", () => {
    const sourceA = reduce(initialBrowserState, hello("source-a"));
    const sequenceA = reduce(sourceA, {
      protocol_version: "1", type: "flow.lifecycle", source_id: "source-a", flow_id: "f", event_id: "a100",
      occurred_at: "2026-01-01T00:00:00Z", sequence: "100", state: "response_started",
    });
    const sourceB = reduce(sequenceA, hello("source-b"));
    const lateA = reduce(sourceB, {
      protocol_version: "1", type: "flow.lifecycle", source_id: "source-a", flow_id: "f", event_id: "late",
      occurred_at: "2026-01-01T00:00:01Z", sequence: "100", state: "response_end",
    });
    const sequenceB = reduce(lateA, {
      protocol_version: "1", type: "flow.lifecycle", source_id: "source-b", flow_id: "g", event_id: "b1",
      occurred_at: "2026-01-01T00:00:02Z", sequence: "1", state: "request_started",
    });

    expect(lateA.streamSequence).toBeNull();
    expect(sequenceB.streamSequence).toBe("1");
    expect(sequenceB.sourceId).toBe("source-b");
  });

  it("resets flows and both cursor domains at a new source epoch", () => {
    const sourceA = reduce(initialBrowserState, hello("source-a"));
    const withFlow = reduce(sourceA, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "a", cursor: "10", flows: [flow("old")],
    });
    const sourceB = reduce(withFlow, hello("source-b"));
    const fresh = reduce(sourceB, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "b", cursor: "0", flows: [flow("new")],
    });

    expect(sourceB.sourceEpoch).toBe(2);
    expect(sourceB.sourceId).toBe("source-b");
    expect(sourceB.cursor).toBe("0");
    expect(sourceB.streamSequence).toBeNull();
    expect(sourceB.flows.ids).toEqual([]);
    expect(fresh.flows.ids).toEqual(["new"]);
    expect(fresh.cursor).toBe("0");
  });

  it("holds a complete immutable displayed snapshot while paused and atomically resumes", () => {
    const withSnapshot = viewReduce(initialBrowserViewState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "1", flows: [flow("old")],
    });
    const paused = browserViewReducer(withSnapshot, { type: "pause" });
    const whilePaused = viewReduce(paused, {
      protocol_version: "1", type: "browser.delta", cursor: "2", changes: [{ op: "upsert", flow: flow("new") }],
    });
    const resumed = browserViewReducer(whilePaused, { type: "resume" });

    expect(whilePaused.followLive).toBe(false);
    expect(whilePaused.displayed).toBe(paused.displayed);
    expect(whilePaused.displayed.flows.ids).toEqual(["old"]);
    expect(whilePaused.displayed.counters.receivedMessages).toBe(paused.displayed.counters.receivedMessages);
    expect(whilePaused.latest.flows.ids).toEqual(["old", "new"]);
    expect(resumed.followLive).toBe(true);
    expect(resumed.displayed).toBe(resumed.latest);
    expect(resumed.displayed.cursor).toBe("2");
  });

  it("keeps the paused display through a reconnect source reset until resume", () => {
    const sourceA = viewReduce(initialBrowserViewState, hello("source-a"));
    const withSnapshot = viewReduce(sourceA, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "a", cursor: "1", flows: [flow("old")],
    });
    const paused = browserViewReducer(withSnapshot, { type: "pause" });
    const reset = browserViewReducer(paused, { type: "source-reset" });
    const sourceB = viewReduce(reset, hello("source-b"));
    const resumed = browserViewReducer(sourceB, { type: "resume" });

    expect(reset.followLive).toBe(false);
    expect(reset.latest.cursor).toBe("0");
    expect(reset.latest.sourceEpoch).toBeGreaterThan(sourceA.latest.sourceEpoch);
    expect(reset.latest.flows.ids).toEqual([]);
    expect(reset.displayed).toBe(paused.displayed);
    expect(reset.displayed.cursor).toBe("1");
    expect(reset.displayed.flows.ids).toEqual(["old"]);
    expect(sourceB.displayed).toBe(paused.displayed);
    expect(sourceB.latest.sourceEpoch).toBeGreaterThan(sourceA.latest.sourceEpoch);
    expect(resumed.displayed).toBe(resumed.latest);
    expect(resumed.displayed.sourceId).toBe("source-b");
    expect(resumed.displayed.cursor).toBe("0");
  });

  it("deep-freezes public state, nested flow data, counters, and gap metadata", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const state = reduce(source, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "1", flows: [flow("f")],
    });
    const gap = reduce(state, {
      protocol_version: "1", type: "stream.gap", expected_sequence: "2", actual_sequence: "4", dropped_count: "1",
    });
    const retained = gap.flows.get("f");

    expect(Object.isFrozen(gap)).toBe(true);
    expect(Object.isFrozen(gap.counters)).toBe(true);
    expect(Object.isFrozen(gap.streamGap)).toBe(true);
    expect(Object.isFrozen(gap.sourceLimits)).toBe(true);
    expect(Object.isFrozen(retained)).toBe(true);
    expect(Object.isFrozen(retained?.request_headers)).toBe(true);
    expect(() => { (gap.counters as { receivedMessages: number }).receivedMessages = 0; }).toThrow(TypeError);
    expect(() => { (retained as { host: string }).host = "mutated"; }).toThrow(TypeError);
    expect(gap.counters.receivedMessages).toBe(3);
    expect(retained?.host).toBe("f.example.test");
  });

  it("records server resync signals and unknown messages without failing", () => {
    const resync = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.resync", reason: "history_evicted", requested_cursor: "9",
    });
    const unknown = reduce(resync, {
      protocol_version: "1", type: "future.message", sequence: "18446744073709551615", safe: true,
    });

    expect(resync.resyncRequested).toBe("9");
    expect(unknown.lastUnknownType).toBe("future.message");
    expect(unknown.counters.unknownMessages).toBe(1);
  });

  it("ignores stale resync acknowledgements without regressing a current gap", () => {
    const current = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "current", cursor: "12", flows: [],
    });
    const staleAck = reduce(current, {
      protocol_version: "1", type: "browser.resync", reason: "history_evicted", requested_cursor: "5",
    });
    const staleSnapshot = reduce(staleAck, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "old", cursor: "5", flows: [flow("old")],
    });
    const pending = reduce(current, {
      protocol_version: "1", type: "browser.delta", cursor: "14", changes: [],
    });
    const stalePendingAck = reduce(pending, {
      protocol_version: "1", type: "browser.resync", reason: "cursor_gap", requested_cursor: "5",
    });
    const validAck = reduce(pending, {
      protocol_version: "1", type: "browser.resync", reason: "cursor_gap", requested_cursor: "12",
    });

    expect(staleAck.cursor).toBe("12");
    expect(staleAck.resyncRequested).toBeNull();
    expect(staleAck.gap).toBeNull();
    expect(staleSnapshot.cursor).toBe("12");
    expect(staleSnapshot.resyncRequested).toBeNull();
    expect(stalePendingAck.resyncRequested).toBe("12");
    expect(stalePendingAck.gap).toEqual(pending.gap);
    expect(validAck.resyncRequested).toBe("12");
    expect(validAck.gap).toEqual(pending.gap);
  });
});
