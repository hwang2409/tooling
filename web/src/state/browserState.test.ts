import { describe, expect, it } from "vitest";

import { parseProtocolMessage } from "../protocol";
import {
  browserReducer,
  browserViewReducer,
  initialBrowserState,
  initialBrowserViewState,
  LIFECYCLE_EVENTS_PER_FLOW,
  LIFECYCLE_FLOW_LIMIT,
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
      protocol_version: "1", type: "browser.resync", reason: "initial_connect", requested_cursor: "9",
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

  it("accepts only current-epoch initial and explicitly pending resync acknowledgements", () => {
    const sourceA = reduce(initialBrowserState, hello("source-a"));
    const initial = reduce(sourceA, {
      protocol_version: "1", type: "browser.resync", reason: "initial_connect", requested_cursor: "0",
    });
    const initialAck = reduce(initial, {
      protocol_version: "1", type: "browser.resync", reason: "history_evicted", requested_cursor: "0",
    });
    const withGap = reduce(sourceA, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "a", cursor: "10", flows: [],
    });
    const pending = reduce(withGap, {
      protocol_version: "1", type: "browser.delta", cursor: "12", changes: [],
    });
    const sourceB = reduce(pending, hello("source-b"));
    const lateSourceAAck = reduce(sourceB, {
      protocol_version: "1", type: "browser.resync", reason: "cursor_gap", requested_cursor: "10",
    });

    expect(initial.initialConnectPending).toBe(false);
    expect(initial.resyncEpoch).toBe(initial.sourceEpoch);
    expect(initialAck.resyncRequested).toBe("0");
    expect(lateSourceAAck.sourceId).toBe("source-b");
    expect(lateSourceAAck.resyncRequested).toBeNull();
    expect(lateSourceAAck.gap).toBeNull();
    expect(lateSourceAAck.counters.staleMessages).toBe(1);
  });

  it("ignores a late cursor-zero acknowledgement after a snapshot resolves the gap", () => {
    const snapshot = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "1", flows: [],
    });
    const pending = reduce(snapshot, {
      protocol_version: "1", type: "browser.delta", cursor: "3", changes: [],
    });
    const resolved = reduce(pending, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "two", cursor: "12", flows: [],
    });
    const late = reduce(resolved, {
      protocol_version: "1", type: "browser.resync", reason: "history_evicted", requested_cursor: "0",
    });

    expect(resolved.gap).toBeNull();
    expect(resolved.resyncRequested).toBeNull();
    expect(late.cursor).toBe("12");
    expect(late.gap).toBeNull();
    expect(late.resyncRequested).toBeNull();
    expect(late.counters.staleMessages).toBe(1);
  });
});

describe("per-flow lifecycle retention", () => {
  function lifecycleMessage(flowId: string, sequence: string, state = "request_started", sourceId = "source-a") {
    return {
      protocol_version: "1", type: "flow.lifecycle", source_id: sourceId, flow_id: flowId,
      event_id: `${flowId}-${sequence}`, occurred_at: "2026-01-01T00:00:00Z", sequence, state,
    };
  }

  it("retains accepted lifecycle events per flow in arrival order without assuming state order", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const first = reduce(source, lifecycleMessage("f", "5", "response_started"));
    const second = reduce(first, lifecycleMessage("f", "6", "request_end"));
    const other = reduce(second, lifecycleMessage("g", "7", "request_started"));

    expect(other.lifecycles.get("f")?.map((event) => event.state)).toEqual(["response_started", "request_end"]);
    expect(other.lifecycles.get("g")?.map((event) => event.sequence)).toEqual(["7"]);
    expect(other.lifecycles.flowIds).toEqual(["f", "g"]);
  });

  it("does not retain events rejected as stale or from a foreign source", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const accepted = reduce(source, lifecycleMessage("f", "5"));
    const duplicate = reduce(accepted, lifecycleMessage("f", "5", "response_started"));
    const foreign = reduce(duplicate, lifecycleMessage("f", "6", "response_started", "source-x"));

    expect(foreign.lifecycles.get("f")).toHaveLength(1);
    expect(foreign.counters.staleMessages).toBe(2);
  });

  it("caps retained events per flow, keeping the newest observations", () => {
    let state = reduce(initialBrowserState, hello("source-a"));
    for (let sequence = 1; sequence <= LIFECYCLE_EVENTS_PER_FLOW + 8; sequence += 1) {
      state = reduce(state, lifecycleMessage("f", String(sequence)));
    }

    const events = state.lifecycles.get("f");
    expect(events).toHaveLength(LIFECYCLE_EVENTS_PER_FLOW);
    expect(events?.[0].sequence).toBe("9");
    expect(events?.[events.length - 1].sequence).toBe(String(LIFECYCLE_EVENTS_PER_FLOW + 8));
  });

  it("evicts exactly one oldest flow at the flow-limit boundary", () => {
    // Pins the `>=` limit comparison: with a `>` mutant the collection
    // would hold LIFECYCLE_FLOW_LIMIT + 1 flows after one insert past the cap.
    let state = reduce(initialBrowserState, hello("source-a"));
    for (let index = 0; index < LIFECYCLE_FLOW_LIMIT; index += 1) {
      state = reduce(state, lifecycleMessage(`flow-${index}`, String(index + 1)));
    }
    expect(state.lifecycles.size).toBe(LIFECYCLE_FLOW_LIMIT);
    expect(state.lifecycles.get("flow-0")).toHaveLength(1);

    const overflowed = reduce(state, lifecycleMessage("flow-overflow", String(LIFECYCLE_FLOW_LIMIT + 1)));
    expect(overflowed.lifecycles.size).toBe(LIFECYCLE_FLOW_LIMIT);
    expect(overflowed.lifecycles.get("flow-0")).toBeUndefined();
    expect(overflowed.lifecycles.get("flow-1")).toHaveLength(1);
    expect(overflowed.lifecycles.get("flow-overflow")).toHaveLength(1);
  });

  it("reuses untouched per-flow event lists instead of re-copying retained state", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const first = reduce(source, lifecycleMessage("f", "1"));
    const second = reduce(first, lifecycleMessage("g", "2"));
    // Recording flow g must not rebuild flow f's retained (frozen) events.
    expect(second.lifecycles.get("f")).toBe(first.lifecycles.get("f"));
    const third = reduce(second, lifecycleMessage("f", "3"));
    expect(third.lifecycles.get("g")).toBe(second.lifecycles.get("g"));
    expect(third.lifecycles.get("f")).not.toBe(second.lifecycles.get("f"));
    expect(third.lifecycles.get("f")).toHaveLength(2);
  });

  it("caps the number of tracked flows by evicting the oldest tracked flow", () => {
    let state = reduce(initialBrowserState, hello("source-a"));
    for (let index = 0; index < LIFECYCLE_FLOW_LIMIT + 2; index += 1) {
      state = reduce(state, lifecycleMessage(`flow-${index}`, String(index + 1)));
    }

    expect(state.lifecycles.size).toBe(LIFECYCLE_FLOW_LIMIT);
    expect(state.lifecycles.get("flow-0")).toBeUndefined();
    expect(state.lifecycles.get("flow-1")).toBeUndefined();
    expect(state.lifecycles.get("flow-2")).toHaveLength(1);
    expect(state.lifecycles.get(`flow-${LIFECYCLE_FLOW_LIMIT + 1}`)).toHaveLength(1);
  });

  it("clears lifecycle retention at a new source epoch", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const tracked = reduce(source, lifecycleMessage("f", "5"));
    const sourceB = reduce(tracked, hello("source-b"));

    expect(tracked.lifecycles.get("f")).toHaveLength(1);
    expect(sourceB.lifecycles.size).toBe(0);
  });

  it("prunes lifecycle retention when a delta removes the flow", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const snapshot = reduce(source, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "1", flows: [flow("f"), flow("g")],
    });
    const tracked = reduce(reduce(snapshot, lifecycleMessage("f", "1")), lifecycleMessage("g", "2"));
    const removed = reduce(tracked, {
      protocol_version: "1", type: "browser.delta", cursor: "2", changes: [{ op: "remove", flow_id: "f" }],
    });

    expect(removed.lifecycles.get("f")).toBeUndefined();
    expect(removed.lifecycles.get("g")).toHaveLength(1);
    expect(removed.lifecycles.flowIds).toEqual(["g"]);
  });

  it("freezes retained lifecycle events and their containers", () => {
    const source = reduce(initialBrowserState, hello("source-a"));
    const tracked = reduce(source, lifecycleMessage("f", "5"));
    const events = tracked.lifecycles.get("f");

    expect(Object.isFrozen(tracked.lifecycles)).toBe(true);
    expect(Object.isFrozen(tracked.lifecycles.flowIds)).toBe(true);
    expect(Object.isFrozen(events)).toBe(true);
    expect(Object.isFrozen(events?.[0])).toBe(true);
    expect(() => { (events?.[0] as { state: string }).state = "error"; }).toThrow(TypeError);
  });
});
