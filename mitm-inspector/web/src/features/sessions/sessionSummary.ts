import { parseFlowExtras } from "../../protocol";
import type { ImmutableFlowMetadata } from "../../state/browserState";
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

/**
 * Group the newest-first grid flows into one summary per session. Session
 * order follows each session's newest flow, so the list is newest-first too.
 * Null session ids collapse into a single "unassigned" bucket.
 */
export function deriveSessions(flows: readonly ImmutableFlowMetadata[]): readonly SessionSummary[] {
  const groups = new Map<SessionKey, ImmutableFlowMetadata[]>();
  for (const flow of flows) {
    const key = deriveFlowFacts(flow).sessionKey;
    const existing = groups.get(key);
    if (existing === undefined) groups.set(key, [flow]);
    else existing.push(flow);
  }
  return Array.from(groups, ([key, sessionFlows]) => summarise(key, sessionFlows));
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
