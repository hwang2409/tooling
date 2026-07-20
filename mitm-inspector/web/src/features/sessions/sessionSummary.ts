/* eslint-disable no-unused-vars */

import { parseFlowExtras } from "../../protocol";
import type { FlowsUpdate, ImmutableFlowCollection, ImmutableFlowMetadata } from "../../state/browserState";
import type { AnthropicRequest } from "../inspector/anthropic";

/**
 * Session aggregation over the grid projection ONLY (`FlowMetadata` +
 * enrichment extras). Rendering the home list must never require a per-flow
 * detail fetch; detail loads happen on drill-in.
 */

export type SessionKey = string | null;

export interface SessionSummary {
  readonly key: SessionKey;
  /** Newest-first, same relative order as the grid projection. */
  readonly flows: readonly ImmutableFlowMetadata[];
  readonly flowCount: number;
  /** Earliest started_at across the session's flows. */
  readonly startedAt?: string;
  /** Latest ended_at (falling back to started_at) across the session's flows. */
  readonly lastActivity?: string;
  /** Oldest user_text preview — the query that opened the session. */
  readonly firstQuery?: string;
  /** Distinct models, oldest-first. */
  readonly models: readonly string[];
  /** True when any flow in the session carries a 4xx/5xx status. */
  readonly hasError: boolean;
}

interface FlowFacts {
  readonly sessionKey: SessionKey;
  readonly startedAt?: string;
  readonly endedAt?: string;
  readonly model?: string;
  readonly userText?: string;
  readonly isError: boolean;
  readonly isConversation: boolean;
  readonly messageCount: number;
}

// Flow metadata objects are deep-frozen and reused across deltas by the
// browser reducer, so per-flow projections computed once stay valid for the
// flow's whole retention lifetime and are dropped automatically when the
// flow is pruned. This keeps each derive pass to cheap cache lookups plus
// O(flows) grouping — the same order of work the reducer itself already does
// per delta.
const factsCache = new WeakMap<ImmutableFlowMetadata, FlowFacts>();

function deriveFlowFacts(metadata: ImmutableFlowMetadata): FlowFacts {
  const cached = factsCache.get(metadata);
  if (cached !== undefined) return cached;
  const extras = parseFlowExtras(metadata);
  const summary = extras.summary;
  const preview = summary?.preview;
  const facts: FlowFacts = {
    sessionKey: metadata.session_id ?? null,
    startedAt: extras.started_at,
    endedAt: extras.ended_at,
    model: summary?.model,
    userText: preview?.source === "user_text" ? preview.text : undefined,
    isError: typeof metadata.response_status === "string" && Number(metadata.response_status) >= 400,
    isConversation: summary?.kind === "anthropic_messages",
    messageCount: summary?.message_count === undefined ? 0 : Number(summary.message_count),
  };
  factsCache.set(metadata, facts);
  return facts;
}

function parsedTime(value: string | undefined): number | null {
  if (value === undefined) return null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

export interface SessionIndexStats {
  readonly fullRebuilds: number;
  readonly incrementalUpdates: number;
  /** summarise() invocations — one per session actually recomputed. */
  readonly sessionsRecomputed: number;
  /**
   * Session-order work on the incremental path: one visit per binary-search
   * comparison or dirty-key reposition. Stays O(dirty x log sessions) per
   * delta — never a scan of all sessions.
   */
  readonly orderVisits: number;
}

export interface SessionIndex {
  /** Project the flow collection onto session summaries, incrementally when the provenance allows. */
  readonly update: (flows: ImmutableFlowCollection, flowsUpdate?: FlowsUpdate) => readonly SessionSummary[];
  readonly stats: () => SessionIndexStats;
}

/**
 * Delta-driven session aggregation. Consecutive "delta" revisions update
 * only the sessions owning the changed flow ids: everything else keeps its
 * summary object, its group, and is never regrouped or rescanned. A
 * snapshot, source reset, or a skipped revision (e.g. coalesced renders or
 * a paused view resuming) falls back to one full rebuild.
 *
 * Session order is newest-first by each session's newest flow. Internally
 * that is a monotonic arrival sequence per flow — new flows are prepended by
 * the delta contract, so descending sequence within a batch mirrors the
 * entries order exactly, in-place upserts keep their position, and removals
 * preserve relative order.
 */
export function createSessionIndex(): SessionIndex {
  let lastRevision: number | null = null;
  let lastEntries: readonly ImmutableFlowMetadata[] | null = null;
  let result: readonly SessionSummary[] = [];
  const flowById = new Map<string, ImmutableFlowMetadata>();
  const sessionOf = new Map<string, SessionKey>();
  const seqOf = new Map<string, number>();
  const groups = new Map<SessionKey, ImmutableFlowMetadata[]>();
  const maxSeqOf = new Map<SessionKey, number>();
  const summaries = new Map<SessionKey, SessionSummary>();
  // Result ordering, maintained sorted by maxSeq descending. On the
  // incremental path only dirty keys are repositioned (binary search), so
  // per-delta order work is O(dirty x log sessions), not a full scan; the
  // returned array is a pointer copy with no per-session recomputation.
  const orderedKeys: SessionKey[] = [];
  const orderedSummaries: SessionSummary[] = [];
  let seqCounter = 0;
  let fullRebuilds = 0;
  let incrementalUpdates = 0;
  let sessionsRecomputed = 0;
  let orderVisits = 0;

  const summariseSession = (key: SessionKey, sessionFlows: readonly ImmutableFlowMetadata[]): SessionSummary => {
    sessionsRecomputed += 1;
    return summarise(key, [...sessionFlows]);
  };

  /** First position whose maxSeq is <= max in the descending-ordered list. */
  const locate = (max: number): number => {
    let low = 0;
    let high = orderedKeys.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      orderVisits += 1;
      if ((maxSeqOf.get(orderedKeys[mid]) ?? -1) > max) low = mid + 1;
      else high = mid;
    }
    return low;
  };

  const removeOrdered = (key: SessionKey, max: number): void => {
    orderVisits += 1;
    let index = locate(max);
    if (orderedKeys[index] !== key) index = orderedKeys.indexOf(key);
    if (index === -1) return;
    orderedKeys.splice(index, 1);
    orderedSummaries.splice(index, 1);
  };

  const insertOrdered = (key: SessionKey, summary: SessionSummary, max: number): void => {
    orderVisits += 1;
    const index = locate(max);
    orderedKeys.splice(index, 0, key);
    orderedSummaries.splice(index, 0, summary);
  };

  const finalize = (dirty: ReadonlySet<SessionKey>): void => {
    for (const key of dirty) {
      const previousMax = maxSeqOf.get(key);
      if (previousMax !== undefined) removeOrdered(key, previousMax);
      const group = groups.get(key);
      if (group === undefined || group.length === 0) {
        groups.delete(key);
        maxSeqOf.delete(key);
        summaries.delete(key);
        continue;
      }
      group.sort((left, right) => (seqOf.get(right.flow_id) ?? 0) - (seqOf.get(left.flow_id) ?? 0));
      const newMax = seqOf.get(group[0].flow_id) ?? 0;
      maxSeqOf.set(key, newMax);
      const summary = summariseSession(key, group);
      summaries.set(key, summary);
      insertOrdered(key, summary, newMax);
    }
    result = [...orderedSummaries];
  };

  const fullRebuild = (entries: readonly ImmutableFlowMetadata[]): void => {
    fullRebuilds += 1;
    flowById.clear();
    sessionOf.clear();
    seqOf.clear();
    groups.clear();
    maxSeqOf.clear();
    summaries.clear();
    orderedKeys.length = 0;
    orderedSummaries.length = 0;
    const base = seqCounter + entries.length;
    seqCounter = base;
    entries.forEach((flow, index) => {
      const key = deriveFlowFacts(flow).sessionKey;
      flowById.set(flow.flow_id, flow);
      sessionOf.set(flow.flow_id, key);
      seqOf.set(flow.flow_id, base - index);
      const group = groups.get(key);
      if (group === undefined) {
        groups.set(key, [flow]);
        // First appearance in newest-first entries defines the session order.
        orderedKeys.push(key);
      } else {
        group.push(flow);
      }
    });
    for (const key of orderedKeys) {
      const group = groups.get(key)!;
      maxSeqOf.set(key, seqOf.get(group[0].flow_id) ?? 0);
      const summary = summariseSession(key, group);
      summaries.set(key, summary);
      orderedSummaries.push(summary);
    }
    result = [...orderedSummaries];
  };

  const applyDelta = (flows: ImmutableFlowCollection, changedFlowIds: readonly string[], prependedCount: number): void => {
    incrementalUpdates += 1;
    const dirty = new Set<SessionKey>();
    // The leading changedFlowIds were block-prepended (brand-new or
    // reinserted) in final grid order: first id sits closest to the top of
    // the grid, so it takes the highest sequence — including a reinserted id
    // whose stale sequence must be replaced.
    const prepended = changedFlowIds.slice(0, prependedCount);
    const base = seqCounter + prepended.length;
    seqCounter = base;
    prepended.forEach((flowId, index) => seqOf.set(flowId, base - index));

    for (const flowId of changedFlowIds) {
      const metadata = flows.get(flowId);
      if (metadata === undefined) {
        const key = sessionOf.get(flowId);
        if (key === undefined) continue;
        groups.set(key, (groups.get(key) ?? []).filter((flow) => flow.flow_id !== flowId));
        flowById.delete(flowId);
        sessionOf.delete(flowId);
        seqOf.delete(flowId);
        dirty.add(key);
        continue;
      }
      if (!seqOf.has(flowId)) {
        seqCounter += 1;
        seqOf.set(flowId, seqCounter);
      }
      const key = deriveFlowFacts(metadata).sessionKey;
      const previousKey = sessionOf.get(flowId);
      if (previousKey === undefined) {
        const group = groups.get(key);
        if (group === undefined) groups.set(key, [metadata]);
        else group.push(metadata);
      } else if (previousKey === key) {
        const group = groups.get(key) ?? [];
        const index = group.findIndex((flow) => flow.flow_id === flowId);
        if (index === -1) group.push(metadata);
        else group[index] = metadata;
        groups.set(key, group);
      } else {
        groups.set(previousKey, (groups.get(previousKey) ?? []).filter((flow) => flow.flow_id !== flowId));
        dirty.add(previousKey);
        const group = groups.get(key);
        if (group === undefined) groups.set(key, [metadata]);
        else group.push(metadata);
      }
      flowById.set(flowId, metadata);
      sessionOf.set(flowId, key);
      dirty.add(key);
    }
    finalize(dirty);
  };

  return {
    update(flows, flowsUpdate) {
      if (flowsUpdate !== undefined) {
        if (lastRevision === flowsUpdate.revision && lastEntries === flows.entries) return result;
        if (lastRevision !== null && flowsUpdate.revision === lastRevision + 1 && flowsUpdate.kind === "delta") {
          applyDelta(flows, flowsUpdate.changedFlowIds, flowsUpdate.prependedCount);
        } else {
          fullRebuild(flows.entries);
        }
        lastRevision = flowsUpdate.revision;
      } else {
        if (lastEntries === flows.entries) return result;
        fullRebuild(flows.entries);
        lastRevision = null;
      }
      lastEntries = flows.entries;
      return result;
    },
    stats: () => ({ fullRebuilds, incrementalUpdates, sessionsRecomputed, orderVisits }),
  };
}

/** Wrap a plain entries array as a flow collection (tests, one-shot derives). */
export function collectionOf(entries: readonly ImmutableFlowMetadata[]): ImmutableFlowCollection {
  const byId = new Map(entries.map((flow) => [flow.flow_id, flow]));
  return {
    ids: entries.map((flow) => flow.flow_id),
    entries,
    size: entries.length,
    get: (flowId: string) => byId.get(flowId),
  };
}

/**
 * One-shot grouping of the newest-first grid flows into session summaries.
 * Session order follows each session's newest flow; null session ids
 * collapse into a single "unassigned" bucket.
 */
export function deriveSessions(flows: readonly ImmutableFlowMetadata[]): readonly SessionSummary[] {
  return createSessionIndex().update(collectionOf(flows));
}

function summarise(key: SessionKey, flows: readonly ImmutableFlowMetadata[]): SessionSummary {
  let startedAt: string | undefined;
  let startedTime = Infinity;
  let lastActivity: string | undefined;
  let lastTime = -Infinity;
  let firstQuery: string | undefined;
  const models: string[] = [];
  let hasError = false;
  // Iterate oldest-first so firstQuery and model order reflect session time.
  for (let index = flows.length - 1; index >= 0; index -= 1) {
    const facts = deriveFlowFacts(flows[index]);
    const started = parsedTime(facts.startedAt);
    if (started !== null && started < startedTime) {
      startedTime = started;
      startedAt = facts.startedAt;
    }
    const ended = parsedTime(facts.endedAt) ?? started;
    if (ended !== null && ended > lastTime) {
      lastTime = ended;
      lastActivity = facts.endedAt ?? facts.startedAt;
    }
    if (firstQuery === undefined && facts.userText !== undefined) firstQuery = facts.userText;
    if (facts.model !== undefined && !models.includes(facts.model)) models.push(facts.model);
    if (facts.isError) hasError = true;
  }
  const summary: SessionSummary = {
    key,
    flows,
    flowCount: flows.length,
    models,
    hasError,
  };
  return {
    ...summary,
    ...(startedAt !== undefined ? { startedAt } : {}),
    ...(lastActivity !== undefined ? { lastActivity } : {}),
    ...(firstQuery !== undefined ? { firstQuery } : {}),
  };
}

export const SUGGESTION_PREFIX = "[SUGGESTION MODE:";

/**
 * Metadata-level heuristic for side-channel suggestion calls: they resend the
 * full history with the suggestion prompt injected on the last user message,
 * so the grid preview surfaces the "[SUGGESTION MODE:" marker. Definitive
 * classification needs the request body, so callers must re-verify after the
 * detail fetch (see isSuggestionRequest in anthropic.ts).
 */
export function looksLikeSuggestionFlow(metadata: ImmutableFlowMetadata): boolean {
  return deriveFlowFacts(metadata).userText?.startsWith(SUGGESTION_PREFIX) === true;
}

/**
 * Rank the session's conversation-shaped flows by how likely each one is the
 * canonical main-thread request. Every main-thread request resends the full
 * history (consecutive requests are prefixes of each other), so the latest
 * main-thread flow's messages array IS the whole conversation. Suggestion-mode
 * siblings can carry the SAME message_count as the main thread, so metadata
 * suggestion flows sort behind; within each class higher message_count wins,
 * then grid recency (newest first). Callers verify the winner against its
 * fetched body and fall through to the next candidate on a mis-classification.
 */
export function conversationCandidates(flows: readonly ImmutableFlowMetadata[]): readonly ImmutableFlowMetadata[] {
  const gridIndex = new Map(flows.map((flow, index) => [flow.flow_id, index]));
  return flows
    .filter((flow) => deriveFlowFacts(flow).isConversation)
    .sort((left, right) => {
      const suggestion = Number(looksLikeSuggestionFlow(left)) - Number(looksLikeSuggestionFlow(right));
      if (suggestion !== 0) return suggestion;
      const count = deriveFlowFacts(right).messageCount - deriveFlowFacts(left).messageCount;
      if (count !== 0) return count;
      return (gridIndex.get(left.flow_id) ?? 0) - (gridIndex.get(right.flow_id) ?? 0);
    });
}

/**
 * Definitive suggestion-mode check against the fetched request body: the
 * suggestion prompt is injected as a text block on the last user message.
 */
export function isSuggestionRequest(request: AnthropicRequest): boolean {
  for (let index = request.messages.length - 1; index >= 0; index -= 1) {
    const message = request.messages[index];
    if (message.role !== "user") continue;
    return message.blocks.some((block) => block.kind === "text" && block.text.startsWith(SUGGESTION_PREFIX));
  }
  return false;
}
