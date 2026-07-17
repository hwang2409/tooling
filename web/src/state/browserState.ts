/* eslint-disable no-unused-vars */

import type {
  BrowserDelta,
  BrowserResync,
  BrowserSnapshot,
  FlowLifecycle,
  FlowMetadata,
  ParsedMessage,
  SourceHello,
  StreamGap,
} from "../protocol";

export interface CursorGap {
  expected: string;
  received: string;
  requested: string;
  reason: "cursor_gap" | "history_evicted";
}

export interface StreamGapState {
  expected: string;
  actual: string;
  droppedCount: string | null;
}

export interface BrowserCounters {
  receivedMessages: number;
  appliedChanges: number;
  staleMessages: number;
  unknownMessages: number;
  droppedMessages: string;
}

export interface ImmutableFlowCollection {
  readonly ids: readonly string[];
  readonly entries: readonly FlowMetadata[];
  readonly size: number;
  get: (flowId: string) => FlowMetadata | undefined;
}

export interface BrowserState {
  flows: ImmutableFlowCollection;
  cursor: string;
  streamSequence: string | null;
  sourceEpoch: number;
  sourceId: string | null;
  sourceLimits: SourceHello["limits"] | null;
  sourceCapabilities: SourceHello["capabilities"] | null;
  snapshotId: string | null;
  gap: CursorGap | null;
  streamGap: StreamGapState | null;
  resyncRequested: string | null;
  lastUnknownType: string | null;
  counters: BrowserCounters;
}

export interface BrowserViewState {
  latest: BrowserState;
  displayed: BrowserState;
  followLive: boolean;
}

export const emptyFlowCollection = createFlowCollection([]);

export const initialBrowserState: BrowserState = {
  flows: emptyFlowCollection,
  cursor: "0",
  streamSequence: null,
  sourceEpoch: 0,
  sourceId: null,
  sourceLimits: null,
  sourceCapabilities: null,
  snapshotId: null,
  gap: null,
  streamGap: null,
  resyncRequested: null,
  lastUnknownType: null,
  counters: {
    receivedMessages: 0,
    appliedChanges: 0,
    staleMessages: 0,
    unknownMessages: 0,
    droppedMessages: "0",
  },
};

export const initialBrowserViewState: BrowserViewState = {
  latest: initialBrowserState,
  displayed: initialBrowserState,
  followLive: true,
};

export type BrowserAction =
  | { type: "protocol"; envelope: ParsedMessage }
  | { type: "reset" };

export type BrowserViewAction =
  | { type: "protocol"; envelope: ParsedMessage }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "reset" };

function cursor(value: string): bigint {
  return BigInt(value);
}

function createFlowCollection(entries: readonly FlowMetadata[]): ImmutableFlowCollection {
  const byId = new Map<string, FlowMetadata>();
  const ids: string[] = [];
  for (const flow of entries) {
    if (!byId.has(flow.flow_id)) ids.push(flow.flow_id);
    byId.set(flow.flow_id, flow);
  }
  const frozenIds = Object.freeze(ids);
  const frozenEntries = Object.freeze(frozenIds.map((flowId) => byId.get(flowId)!));
  return Object.freeze({
    ids: frozenIds,
    entries: frozenEntries,
    size: frozenEntries.length,
    get: (flowId: string) => byId.get(flowId),
  });
}

function withCounter(state: BrowserState, key: keyof Omit<BrowserCounters, "droppedMessages">, amount = 1): BrowserState {
  return {
    ...state,
    counters: { ...state.counters, [key]: state.counters[key] + amount },
  };
}

function withDroppedCount(state: BrowserState, droppedCount: string): BrowserState {
  return {
    ...state,
    counters: {
      ...state.counters,
      droppedMessages: (BigInt(state.counters.droppedMessages) + BigInt(droppedCount)).toString(),
    },
  };
}

function stale(state: BrowserState): BrowserState {
  return withCounter(state, "staleMessages");
}

function resetForSource(state: BrowserState, message: SourceHello): BrowserState {
  return {
    ...state,
    flows: emptyFlowCollection,
    cursor: "0",
    streamSequence: null,
    sourceEpoch: state.sourceEpoch + 1,
    sourceId: message.source_id,
    sourceLimits: message.limits,
    sourceCapabilities: message.capabilities,
    snapshotId: null,
    gap: null,
    streamGap: null,
    resyncRequested: null,
    counters: {
      receivedMessages: state.counters.receivedMessages,
      appliedChanges: 0,
      staleMessages: 0,
      unknownMessages: 0,
      droppedMessages: "0",
    },
  };
}

function applySnapshot(state: BrowserState, message: BrowserSnapshot): BrowserState {
  if (cursor(message.cursor) < cursor(state.cursor)) return stale(state);
  return {
    ...state,
    flows: createFlowCollection(message.flows),
    cursor: message.cursor,
    snapshotId: message.snapshot_id,
    gap: null,
    resyncRequested: null,
    counters: {
      ...state.counters,
      appliedChanges: state.counters.appliedChanges + message.flows.length,
    },
  };
}

function applyDelta(state: BrowserState, message: BrowserDelta): BrowserState {
  const nextCursor = cursor(message.cursor);
  const currentCursor = cursor(state.cursor);
  if (nextCursor <= currentCursor) return stale(state);

  if (nextCursor > currentCursor + 1n) {
    return {
      ...state,
      gap: {
        expected: (currentCursor + 1n).toString(),
        received: message.cursor,
        requested: state.cursor,
        reason: "cursor_gap",
      },
      resyncRequested: state.cursor,
    };
  }

  const entries = [...state.flows.entries];
  for (const change of message.changes) {
    const flowId = change.op === "upsert" ? change.flow.flow_id : change.flow_id;
    const index = entries.findIndex((flow) => flow.flow_id === flowId);
    if (change.op === "upsert") {
      if (index === -1) entries.push(change.flow);
      else entries[index] = change.flow;
    } else if (index !== -1) {
      entries.splice(index, 1);
    }
  }
  return {
    ...state,
    flows: createFlowCollection(entries),
    cursor: message.cursor,
    gap: null,
    resyncRequested: null,
    counters: {
      ...state.counters,
      appliedChanges: state.counters.appliedChanges + message.changes.length,
    },
  };
}

function applyResync(state: BrowserState, message: BrowserResync): BrowserState {
  if (message.reason === "initial_connect") {
    return { ...state, resyncRequested: message.requested_cursor };
  }
  return {
    ...state,
    gap: state.gap ?? {
      expected: state.cursor,
      received: message.requested_cursor,
      requested: message.requested_cursor,
      reason: message.reason,
    },
    resyncRequested: message.requested_cursor,
  };
}

function applyStreamGap(state: BrowserState, message: StreamGap): BrowserState {
  const next = {
    ...state,
    streamGap: {
      expected: message.expected_sequence,
      actual: message.actual_sequence,
      droppedCount: message.dropped_count === undefined ? null : message.dropped_count,
    },
  };
  return message.dropped_count === undefined ? next : withDroppedCount(next, message.dropped_count);
}

function applyLifecycle(state: BrowserState, message: FlowLifecycle): BrowserState {
  if (state.streamSequence !== null && cursor(message.sequence) <= cursor(state.streamSequence)) return stale(state);
  return { ...state, streamSequence: message.sequence };
}

export function browserReducer(state: BrowserState, action: BrowserAction): BrowserState {
  if (action.type === "reset") return initialBrowserState;
  const counted = withCounter(state, "receivedMessages");
  const { envelope } = action;

  if (envelope.kind === "unknown") {
    return {
      ...withCounter(counted, "unknownMessages"),
      lastUnknownType: envelope.original_type,
    };
  }

  const message = envelope.message;
  switch (message.type) {
    case "source.hello":
      return resetForSource(counted, message as unknown as SourceHello);
    case "browser.snapshot":
      return applySnapshot(counted, message as unknown as BrowserSnapshot);
    case "browser.delta":
      return applyDelta(counted, message as unknown as BrowserDelta);
    case "browser.resync":
      return applyResync(counted, message as unknown as BrowserResync);
    case "stream.gap":
      return applyStreamGap(counted, message as unknown as StreamGap);
    case "flow.lifecycle":
      return applyLifecycle(counted, message as unknown as FlowLifecycle);
    default:
      return counted;
  }
}

export function browserViewReducer(state: BrowserViewState, action: BrowserViewAction): BrowserViewState {
  if (action.type === "pause") {
    if (!state.followLive) return state;
    return { latest: state.latest, displayed: state.latest, followLive: false };
  }
  if (action.type === "resume") {
    return { latest: state.latest, displayed: state.latest, followLive: true };
  }
  if (action.type === "reset") return initialBrowserViewState;

  const latest = browserReducer(state.latest, { type: "protocol", envelope: action.envelope });
  return { latest, displayed: state.followLive ? latest : state.displayed, followLive: state.followLive };
}
