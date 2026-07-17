/* eslint-disable no-unused-vars */

import type { DeepReadonly } from "../immutable";
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

export type ImmutableFlowMetadata = DeepReadonly<FlowMetadata>;
export type ImmutableFlowLifecycle = DeepReadonly<FlowLifecycle>;

export interface ImmutableFlowCollection {
  readonly ids: readonly string[];
  readonly entries: readonly ImmutableFlowMetadata[];
  readonly size: number;
  readonly get: (flowId: string) => ImmutableFlowMetadata | undefined;
}

export const LIFECYCLE_FLOW_LIMIT = 512;
export const LIFECYCLE_EVENTS_PER_FLOW = 32;

export interface ImmutableLifecycleCollection {
  readonly flowIds: readonly string[];
  readonly size: number;
  readonly truncatedFlowIds: readonly string[];
  readonly isTruncated: (flowId: string) => boolean;
  readonly get: (flowId: string) => readonly ImmutableFlowLifecycle[] | undefined;
}

export interface BrowserState {
  readonly flows: ImmutableFlowCollection;
  readonly lifecycles: ImmutableLifecycleCollection;
  readonly cursor: string;
  readonly streamSequence: string | null;
  readonly sourceEpoch: number;
  readonly initialConnectPending: boolean;
  readonly sourceId: string | null;
  readonly sourceLimits: DeepReadonly<SourceHello["limits"]> | null;
  readonly sourceCapabilities: DeepReadonly<SourceHello["capabilities"]> | null;
  readonly snapshotId: string | null;
  readonly gap: CursorGap | null;
  readonly streamGap: StreamGapState | null;
  readonly resyncRequested: string | null;
  readonly resyncEpoch: number | null;
  readonly lastUnknownType: string | null;
  readonly counters: BrowserCounters;
}

export interface BrowserViewState {
  readonly latest: BrowserState;
  readonly displayed: BrowserState;
  readonly followLive: boolean;
}

export const emptyFlowCollection = createFlowCollection([]);
export const emptyLifecycleCollection = createLifecycleCollection([]);

const initialBrowserStateValue: BrowserState = {
  flows: emptyFlowCollection,
  lifecycles: emptyLifecycleCollection,
  cursor: "0",
  streamSequence: null,
  sourceEpoch: 0,
  initialConnectPending: true,
  sourceId: null,
  sourceLimits: null,
  sourceCapabilities: null,
  snapshotId: null,
  gap: null,
  streamGap: null,
  resyncRequested: null,
  resyncEpoch: null,
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
  // Every frozen object reaching this module is deep-frozen: protocol.ts
  // recursively freezes parsed messages and this module only freezes values
  // it deep-copied.  Reusing them keeps retention incremental instead of
  // re-copying all retained state on every message.
  if (Object.isFrozen(value)) return value;
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

type LifecycleEntries = readonly (readonly [string, readonly ImmutableFlowLifecycle[]])[];

function createLifecycleCollection(
  entries: LifecycleEntries,
  truncatedFlowIds: readonly string[] = [],
): ImmutableLifecycleCollection {
  const byId = new Map<string, readonly ImmutableFlowLifecycle[]>();
  for (const [flowId, events] of entries) {
    // Retained event lists are already frozen; only new lists pay a copy.
    byId.set(flowId, Object.isFrozen(events) ? events : Object.freeze(events.map((event) => freezeCopy(event))));
  }
  const flowIds = Object.freeze([...byId.keys()]);
  const retained = new Set(flowIds);
  const truncated = Object.freeze([...new Set(truncatedFlowIds)].filter((flowId) => retained.has(flowId)));
  const truncatedSet = new Set(truncated);
  return Object.freeze({
    flowIds,
    size: flowIds.length,
    truncatedFlowIds: truncated,
    isTruncated: (flowId: string) => truncatedSet.has(flowId),
    get: (flowId: string) => byId.get(flowId),
  });
}

function recordLifecycle(collection: ImmutableLifecycleCollection, message: FlowLifecycle): ImmutableLifecycleCollection {
  const existing = collection.get(message.flow_id);
  const events = Object.freeze(
    [...(existing ?? []), freezeCopy(message) as ImmutableFlowLifecycle].slice(-LIFECYCLE_EVENTS_PER_FLOW),
  );
  const truncated = new Set(collection.truncatedFlowIds);
  if (existing !== undefined && existing.length >= LIFECYCLE_EVENTS_PER_FLOW) {
    truncated.add(message.flow_id);
  }
  let flowIds = collection.flowIds;
  if (existing === undefined && flowIds.length >= LIFECYCLE_FLOW_LIMIT) {
    truncated.delete(flowIds[0]);
    flowIds = flowIds.slice(flowIds.length - LIFECYCLE_FLOW_LIMIT + 1);
  }
  const entries: Array<readonly [string, readonly ImmutableFlowLifecycle[]]> = [];
  for (const flowId of flowIds) {
    if (flowId === message.flow_id) continue;
    entries.push([flowId, collection.get(flowId) ?? []]);
  }
  entries.push([message.flow_id, events]);
  return createLifecycleCollection(entries, [...truncated]);
}

function pruneLifecycle(collection: ImmutableLifecycleCollection, removedFlowIds: readonly string[]): ImmutableLifecycleCollection {
  const removed = new Set(removedFlowIds);
  if (!collection.flowIds.some((flowId) => removed.has(flowId))) return collection;
  const entries: Array<readonly [string, readonly ImmutableFlowLifecycle[]]> = [];
  for (const flowId of collection.flowIds) {
    if (!removed.has(flowId)) entries.push([flowId, collection.get(flowId) ?? []]);
  }
  return createLifecycleCollection(entries, collection.truncatedFlowIds);
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
    lifecycles: emptyLifecycleCollection,
    cursor: "0",
    streamSequence: null,
    sourceEpoch: state.sourceEpoch + 1,
    initialConnectPending: true,
    sourceId: message.source_id,
    sourceLimits: message.limits,
    sourceCapabilities: message.capabilities,
    snapshotId: null,
    gap: null,
    streamGap: null,
    resyncRequested: null,
    resyncEpoch: null,
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
    initialConnectPending: false,
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
    initialConnectPending: false,
    resyncEpoch: null,
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
      initialConnectPending: false,
      resyncEpoch: state.sourceEpoch,
    };
  }

  const entries = [...state.flows.entries];
  const removedFlowIds: string[] = [];
  for (const change of message.changes) {
    const flowId = change.op === "upsert" ? change.flow.flow_id : change.flow_id;
    const index = entries.findIndex((flow) => flow.flow_id === flowId);
    if (change.op === "upsert") {
      if (index === -1) entries.push(change.flow);
      else entries[index] = change.flow;
    } else if (index !== -1) {
      entries.splice(index, 1);
      removedFlowIds.push(flowId);
    }
  }
  return {
    ...state,
    flows: createFlowCollection(entries),
    lifecycles: pruneLifecycle(state.lifecycles, removedFlowIds),
    cursor: message.cursor,
    gap: null,
    resyncRequested: null,
    initialConnectPending: false,
    resyncEpoch: null,
    counters: {
      ...state.counters,
      appliedChanges: state.counters.appliedChanges + message.changes.length,
    },
  };
}

function applyResync(state: BrowserState, message: BrowserResync): BrowserState {
  if (message.reason === "initial_connect") {
    if (!state.initialConnectPending || state.resyncRequested !== null) return stale(state);
    return {
      ...state,
      initialConnectPending: false,
      resyncRequested: message.requested_cursor,
      resyncEpoch: state.sourceEpoch,
    };
  }
  const hasCurrentPendingRequest = state.resyncRequested !== null
    && state.resyncEpoch === state.sourceEpoch
    && (state.gap === null || state.gap.sourceEpoch === state.sourceEpoch);
  if (!hasCurrentPendingRequest || message.requested_cursor !== state.resyncRequested) return stale(state);
  return state;
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
  return {
    ...state,
    streamSequence: message.sequence,
    lifecycles: recordLifecycle(state.lifecycles, message),
  };
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
