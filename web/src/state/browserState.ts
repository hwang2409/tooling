import type {
  BrowserDelta,
  BrowserResync,
  BrowserSnapshot,
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

export interface BrowserCounters {
  receivedMessages: number;
  appliedChanges: number;
  staleMessages: number;
  unknownMessages: number;
  droppedMessages: number;
}

export interface BrowserState {
  flows: Readonly<Record<string, FlowMetadata>>;
  cursor: string;
  sourceId: string | null;
  sourceLimits: SourceHello["limits"] | null;
  sourceCapabilities: SourceHello["capabilities"] | null;
  snapshotId: string | null;
  gap: CursorGap | null;
  resyncRequested: string | null;
  lastUnknownType: string | null;
  counters: BrowserCounters;
}

export const initialBrowserState: BrowserState = {
  flows: Object.freeze(Object.create(null) as Record<string, FlowMetadata>),
  cursor: "0",
  sourceId: null,
  sourceLimits: null,
  sourceCapabilities: null,
  snapshotId: null,
  gap: null,
  resyncRequested: null,
  lastUnknownType: null,
  counters: {
    receivedMessages: 0,
    appliedChanges: 0,
    staleMessages: 0,
    unknownMessages: 0,
    droppedMessages: 0,
  },
};

export type BrowserAction =
  | { type: "protocol"; envelope: ParsedMessage }
  | { type: "reset" };

function cursor(value: string): bigint {
  return BigInt(value);
}

function withCounter(state: BrowserState, key: keyof BrowserCounters, amount = 1): BrowserState {
  return {
    ...state,
    counters: { ...state.counters, [key]: state.counters[key] + amount },
  };
}

function stale(state: BrowserState): BrowserState {
  return withCounter(state, "staleMessages");
}

function applySnapshot(state: BrowserState, message: BrowserSnapshot): BrowserState {
  if (cursor(message.cursor) < cursor(state.cursor)) return stale(state);

  const flows: Record<string, FlowMetadata> = Object.create(null) as Record<string, FlowMetadata>;
  for (const flow of message.flows) flows[flow.flow_id] = flow;
  return {
    ...state,
    flows: Object.freeze(flows),
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
      ...withCounter(state, "droppedMessages"),
      gap: {
        expected: (currentCursor + 1n).toString(),
        received: message.cursor,
        requested: state.cursor,
        reason: "cursor_gap",
      },
      resyncRequested: state.cursor,
    };
  }

  const flows: Record<string, FlowMetadata> = { ...state.flows };
  for (const change of message.changes) {
    if (change.op === "upsert") flows[change.flow.flow_id] = change.flow;
    else delete flows[change.flow_id];
  }
  return {
    ...state,
    flows: Object.freeze(flows),
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
  const expected = cursor(message.expected_sequence);
  const received = message.actual_sequence;
  if (expected <= cursor(state.cursor) && cursor(received) <= cursor(state.cursor)) return stale(state);
  return {
    ...withCounter(state, "droppedMessages", Number(message.dropped_count ?? "0")),
    gap: {
      expected: message.expected_sequence,
      received,
      requested: state.cursor,
      reason: "cursor_gap",
    },
    resyncRequested: state.cursor,
  };
}

export function browserReducer(state: BrowserState, action: BrowserAction): BrowserState {
  if (action.type === "reset") return initialBrowserState;
  const { envelope } = action;
  const counted = withCounter(state, "receivedMessages");

  if (envelope.kind === "unknown") {
    return {
      ...withCounter(counted, "unknownMessages"),
      lastUnknownType: envelope.original_type,
    };
  }

  const message = envelope.message;
  switch (message.type) {
    case "source.hello":
      return {
        ...counted,
        sourceId: message.source_id,
        sourceLimits: message.limits,
        sourceCapabilities: message.capabilities,
      };
    case "browser.snapshot":
      return applySnapshot(counted, message as unknown as BrowserSnapshot);
    case "browser.delta":
      return applyDelta(counted, message as unknown as BrowserDelta);
    case "browser.resync":
      return applyResync(counted, message as unknown as BrowserResync);
    case "stream.gap":
      return applyStreamGap(counted, message as unknown as StreamGap);
    default:
      return counted;
  }
}
