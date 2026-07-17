import { describe, expect, it } from "vitest";

import { parseProtocolMessage } from "../protocol";
import { browserReducer, initialBrowserState } from "./browserState";

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

function message(value: unknown) {
  return parseProtocolMessage(value);
}

function reduce(state: typeof initialBrowserState, value: unknown) {
  return browserReducer(state, { type: "protocol", envelope: message(value) });
}

describe("browser state reducer", () => {
  it("replaces the retained flow set on a fresh snapshot", () => {
    const first = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "4", flows: [flow("old")],
    });
    const next = reduce(first, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "two", cursor: "7", flows: [flow("new")],
    });

    expect(Object.keys(next.flows)).toEqual(["new"]);
    expect(next.cursor).toBe("7");
    expect(next.snapshotId).toBe("two");
  });

  it("upserts and removes delta entries while preserving cursor order", () => {
    const snapshot = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "4", flows: [flow("old")],
    });
    const next = reduce(snapshot, {
      protocol_version: "1", type: "browser.delta", cursor: "5", changes: [
        { op: "upsert", flow: flow("new") }, { op: "remove", flow_id: "old" },
      ],
    });

    expect(Object.keys(next.flows)).toEqual(["new"]);
    expect(next.cursor).toBe("5");
    expect(next.counters.appliedChanges).toBe(3);
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

    expect(Object.keys(staleDelta.flows)).toEqual(["kept"]);
    expect(staleDelta.cursor).toBe("4");
    expect(staleDelta.counters.staleMessages).toBe(2);
  });

  it("marks a cursor gap and keeps the last coherent view for resync", () => {
    const snapshot = reduce(initialBrowserState, {
      protocol_version: "1", type: "browser.snapshot", snapshot_id: "one", cursor: "4", flows: [flow("kept")],
    });
    const next = reduce(snapshot, {
      protocol_version: "1", type: "browser.delta", cursor: "8", changes: [{ op: "remove", flow_id: "kept" }],
    });

    expect(Object.keys(next.flows)).toEqual(["kept"]);
    expect(next.gap).toEqual({ expected: "5", received: "8", requested: "4", reason: "cursor_gap" });
    expect(next.resyncRequested).toBe("4");
    expect(next.counters.droppedMessages).toBe(1);
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
});
