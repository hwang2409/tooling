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
  readonly sourceEpoch: number;
  readonly expected: string;
  readonly received: string;
  readonly requested: string;
  readonly reason: "cursor_gap" | "history_evicted";
}

export interface StreamGapState {
  readonly expected: string;
  readonly actual: string;
  readonly droppedCount: string | null;
}

export interface BrowserCounters {
  readonly receivedMessages: number;
  readonly appliedChanges: number;
  readonly staleMessages: number;
  readonly unknownMessages: number;
  readonly droppedMessages: string;
}

type DeepReadonly<T> = T extends ReadonlyArray<infer Item>
  ? ReadonlyArray<DeepReadonly<Item>>
  : T extends object
    ? { readonly [Key in keyof T]: DeepReadonly<T[Key]> }
    : T;

export type ImmutableFlowMetadata = DeepReadonly<FlowMetadata>;

export interface ImmutableFlowCollection {
  readonly ids: readonly string[];
  readonly entries: readonly ImmutableFlowMetadata[];
  readonly size: number;
  readonly get: (flowId: string) => ImmutableFlowMetadata | undefined;
}

export interface BrowserState {
  readonly flows: ImmutableFlowCollection;
  readonly cursor: string;
  readonly streamSequence: string | null;
  readonly sourceEpoch: number;
  readonly sourceId: string | null;
  readonly sourceLimits: DeepReadonly<SourceHello["limits"]> | null;
  readonly sourceCapabilities: DeepReadonly<SourceHello["capabilities"]> | null;
  readonly snapshotId: string | null;
  readonly gap: CursorGap | null;
  readonly streamGap: StreamGapState | null;
  readonly resyncRequested: string | null;
  readonly lastUnknownType: string | null;
  readonly counters: BrowserCounters;
}

export interface BrowserViewState {
  readonly latest: BrowserState;
  readonly displayed: BrowserState;
  readonly followLive: boolean;
}

export const emptyFlowCollection = createFlowCollection([]);

const initialBrowserStateValue: BrowserState = {
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

export const initialBrowserState = freezeBrowserState(initialBrowserStateValue);

export const initialBrowserViewState = freezeBrowserViewState({
  latest: initialBrowserState,
  displayed: initialBrowserState,
  followLive: true,
});

export type BrowserAction =
  | { type: "protocol"; envelope: ParsedMessage }
  | { type: "reset" };

export type BrowserViewAction =
  | { type: "protocol"; envelope: ParsedMessage }
  | { type: "source-reset" }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "reset" };

function cursor(value: string): bigint {
  return BigInt(value);
}

function freezeCopy<T>(value: T): T {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return Object.freeze(value.map((item) => freezeCopy(item))) as T;
  const copy = Object.create(null) as Record<string, unknown>;
  for (const [key, item] of Object.entries(value)) copy[key] = freezeCopy(item);
  return Object.freeze(copy) as T;
}

function freezeBrowserState(state: BrowserState): BrowserState {
  return Object.freeze({
    ...state,
    sourceLimits: state.sourceLimits === null ? null : freezeCopy(state.sourceLimits),
    sourceCapabilities: state.sourceCapabilities === null ? null : freezeCopy(state.sourceCapabilities),
    gap: state.gap === null ? null : Object.freeze({ ...state.gap }),
    streamGap: state.streamGap === null ? null : Object.freeze({ ...state.streamGap }),
    counters: Object.freeze({ ...state.counters }),
  });
}

function freezeBrowserViewState(state: BrowserViewState): BrowserViewState {
  return Object.freeze({ ...state });
}

function createFlowCollection(entries: readonly ImmutableFlowMetadata[]): ImmutableFlowCollection {
  const byId = new Map<string, ImmutableFlowMetadata>();
  const ids: string[] = [];
  for (const flow of entries) {
    if (!byId.has(flow.flow_id)) ids.push(flow.flow_id);
    byId.set(flow.flow_id, freezeCopy(flow));
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

function resetForTransport(state: BrowserState): BrowserState {
  return {
    ...initialBrowserState,
    sourceEpoch: state.sourceEpoch + 1,
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
        sourceEpoch: state.sourceEpoch,
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
  const requestedCursor = cursor(message.requested_cursor);
  const currentCursor = cursor(state.cursor);

  const hasCurrentPendingRequest = state.gap !== null
    && state.gap.sourceEpoch === state.sourceEpoch
    && state.resyncRequested !== null;
  if (hasCurrentPendingRequest) {
    if (message.requested_cursor !== state.resyncRequested) return stale(state);
    return { ...state, resyncRequested: state.resyncRequested };
  }

  if (requestedCursor < currentCursor) return stale(state);
  if (state.cursor !== "0") return stale(state);
  if (message.reason === "initial_connect") {
    return { ...state, resyncRequested: message.requested_cursor };
  }
  return {
    ...state,
    gap: state.gap ?? {
      sourceEpoch: state.sourceEpoch,
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
  if (state.sourceId === null || message.source_id !== state.sourceId) return stale(state);
  if (state.streamSequence !== null && cursor(message.sequence) <= cursor(state.streamSequence)) return stale(state);
  return { ...state, streamSequence: message.sequence };
}

export function browserReducer(state: BrowserState, action: BrowserAction): BrowserState {
  if (action.type === "reset") return initialBrowserState;
  const counted = withCounter(state, "receivedMessages");
  const { envelope } = action;

  if (envelope.kind === "unknown") {
    return freezeBrowserState({
      ...withCounter(counted, "unknownMessages"),
      lastUnknownType: envelope.original_type,
    });
  }

  const message = envelope.message;
  let next: BrowserState;
  switch (message.type) {
    case "source.hello":
      next = resetForSource(counted, message as unknown as SourceHello);
      break;
    case "browser.snapshot":
      next = applySnapshot(counted, message as unknown as BrowserSnapshot);
      break;
    case "browser.delta":
      next = applyDelta(counted, message as unknown as BrowserDelta);
      break;
    case "browser.resync":
      next = applyResync(counted, message as unknown as BrowserResync);
      break;
    case "stream.gap":
      next = applyStreamGap(counted, message as unknown as StreamGap);
      break;
    case "flow.lifecycle":
      next = applyLifecycle(counted, message as unknown as FlowLifecycle);
      break;
    default:
      next = counted;
      break;
  }
  return freezeBrowserState(next);
}

export function browserViewReducer(state: BrowserViewState, action: BrowserViewAction): BrowserViewState {
  if (action.type === "source-reset") {
    const latest = freezeBrowserState(resetForTransport(state.latest));
    return freezeBrowserViewState({ latest, displayed: state.followLive ? latest : state.displayed, followLive: state.followLive });
  }
  if (action.type === "pause") {
    if (!state.followLive) return state;
    return freezeBrowserViewState({ latest: state.latest, displayed: state.latest, followLive: false });
  }
  if (action.type === "resume") {
    return freezeBrowserViewState({ latest: state.latest, displayed: state.latest, followLive: true });
  }
  if (action.type === "reset") return initialBrowserViewState;

  const latest = browserReducer(state.latest, { type: "protocol", envelope: action.envelope });
  return freezeBrowserViewState({ latest, displayed: state.followLive ? latest : state.displayed, followLive: state.followLive });
}
