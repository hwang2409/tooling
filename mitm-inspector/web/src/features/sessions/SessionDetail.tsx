/* eslint-disable no-unused-vars */

import { useMemo, useRef, useState } from "react";

import { PacketDetail, PacketList } from "../flows/PacketList";
import { PLACEHOLDER, durationBetween, shortModel } from "../flows/rowSummary";
import { parseAnthropicRequest } from "../inspector/anthropic";
import type { AnthropicRequest } from "../inspector/anthropic";
import { Collapse } from "../inspector/ConversationView";
import { bodyText } from "../inspector/decoders";
import { useFlowDetails } from "../inspector/flowDetail";
import type { FlowDetailLoader, FlowDetailResult } from "../inspector/flowDetail";
import { safeParseJson } from "../inspector/jsonTree";
import type { JsonValue } from "../inspector/jsonTree";
import type { ImmutableFlowMetadata } from "../../state/browserState";
import { createCanonicalIndex, requestContextKey } from "./canonical";
import type { CanonicalIndex } from "./canonical";
import { sessionLabel } from "./SessionList";
import { conversationCandidates, isSuggestionRequest } from "./sessionSummary";
import type { SessionSummary } from "./sessionSummary";

export interface SessionDetailProps {
  summary: SessionSummary;
  onBack: () => void;
  loadFlowDetail?: FlowDetailLoader;
  /** Capture incarnation — part of detail-cache identity across reconnects. */
  sourceEpoch?: number;
  /** Instrumented instances for tests; production creates its own. */
  candidateParser?: CandidateParser;
  canonicalIndex?: CanonicalIndex;
}

type SessionMode = "conversation" | "flows";

const NO_FLOWS: readonly ImmutableFlowMetadata[] = [];
const NO_PARSED: readonly ParsedCandidate[] = [];

interface ParsedCandidate {
  readonly flow: ImmutableFlowMetadata;
  readonly order: number;
  readonly detail: FlowDetailResult | null;
  readonly request: AnthropicRequest | null;
  readonly rawMessages: readonly JsonValue[] | null;
  /** Body-verified suggestion-mode request; false until the body loads. */
  readonly suggestion: boolean;
  readonly contextKey?: string;
}

function parseCandidate(
  flow: ImmutableFlowMetadata,
  order: number,
  detail: FlowDetailResult | undefined,
): ParsedCandidate {
  const base = { flow, order, detail: detail ?? null, request: null, rawMessages: null, suggestion: false };
  if (detail?.status !== "loaded") return base;
  const requestBody = detail.overrides.request_body ?? flow.request_body;
  const decoded = bodyText(requestBody);
  if (decoded.kind !== "text") return base;
  const parsed = safeParseJson(decoded.text);
  if (!parsed.ok) return base;
  const request = parseAnthropicRequest(parsed.value);
  if (request === null) return base;
  const object = parsed.value as { messages?: JsonValue };
  const rawMessages = Array.isArray(object.messages) ? object.messages : null;
  return {
    ...base,
    request,
    rawMessages,
    suggestion: isSuggestionRequest(request),
    contextKey: requestContextKey(parsed.value),
  };
}

export interface CandidateParserStats {
  /** parseCandidate executions — cache misses only. */
  readonly parses: number;
}

export interface CandidateParser {
  readonly parseAll: (
    flows: readonly ImmutableFlowMetadata[],
    orderOf: ReadonlyMap<string, number>,
    details: ReadonlyMap<string, FlowDetailResult>,
    sourceEpoch: number | undefined,
  ) => readonly ParsedCandidate[];
  readonly stats: () => CandidateParserStats;
}

/**
 * Per-flow body-projection cache. The reducer deep-freezes and reuses flow
 * metadata objects across deltas and useFlowDetails keeps result identity
 * per flow id + detail version + epoch, so (flow, detail, epoch) identity is
 * exactly "flow version + source epoch": a delta touching other flows reuses
 * every cached projection; only the changed flow reparses. Grid-order shifts
 * rewrap the cached projection without touching the body.
 */
export function createCandidateParser(): CandidateParser {
  interface CacheEntry {
    readonly flow: ImmutableFlowMetadata;
    readonly detail: FlowDetailResult | null;
    readonly epoch: number | undefined;
    readonly value: ParsedCandidate;
  }
  const cache = new Map<string, CacheEntry>();
  let parses = 0;
  return {
    parseAll(flows, orderOf, details, sourceEpoch) {
      const live = new Set<string>();
      const result = flows.map((flow) => {
        live.add(flow.flow_id);
        const order = orderOf.get(flow.flow_id) ?? 0;
        const detail = details.get(flow.flow_id) ?? null;
        const cached = cache.get(flow.flow_id);
        if (cached !== undefined && cached.flow === flow && cached.detail === detail && cached.epoch === sourceEpoch) {
          if (cached.value.order === order) return cached.value;
          const value = { ...cached.value, order };
          cache.set(flow.flow_id, { flow, detail, epoch: sourceEpoch, value });
          return value;
        }
        parses += 1;
        const value = parseCandidate(flow, order, detail ?? undefined);
        cache.set(flow.flow_id, { flow, detail, epoch: sourceEpoch, value });
        return value;
      });
      for (const flowId of [...cache.keys()]) {
        if (!live.has(flowId)) cache.delete(flowId);
      }
      return result;
    },
    stats: () => ({ parses }),
  };
}

/**
 * One session's drill-in view. Conversation mode fetches every
 * conversation-shaped flow in the session (drill-in detail fetches are
 * allowed; only the home list is metadata-only), builds prefix chains over
 * the verified bodies, and renders the tip of the dominant chain — the
 * latest main-thread request, whose messages array is the whole conversation
 * and whose response is the final assistant turn. Suggestion-mode and
 * off-chain utility side-calls stay reachable in a collapsed auxiliary
 * group, and the full flow grid is one toggle away.
 */
export function SessionDetail({ summary, onBack, loadFlowDetail, sourceEpoch, candidateParser, canonicalIndex }: SessionDetailProps) {
  const candidates = useMemo(() => conversationCandidates(summary.flows), [summary.flows]);
  // The unassigned bucket keeps the grid experience as its default drill-in.
  const [mode, setMode] = useState<SessionMode>(summary.key === null ? "flows" : "conversation");
  const effectiveMode: SessionMode = candidates.length === 0 ? "flows" : mode;

  const details = useFlowDetails(effectiveMode === "conversation" ? candidates : NO_FLOWS, loadFlowDetail, sourceEpoch);
  // Created order for the tip-recency policy is the GRID position of the
  // flow (0 = newest), not the metadata-ranking position of the candidate.
  const gridOrder = useMemo(
    () => new Map(summary.flows.map((flow, index) => [flow.flow_id, index])),
    [summary.flows],
  );
  // One stateful parser + canonical index per drill-in: consecutive deltas
  // reparse and re-relate only the changed candidates (see
  // createCandidateParser / createCanonicalIndex); fresh instances per render
  // would degrade every delta to a whole-session reparse and rescan.
  const parserRef = useRef<CandidateParser | null>(null);
  if (parserRef.current === null) parserRef.current = candidateParser ?? createCandidateParser();
  const indexRef = useRef<CanonicalIndex | null>(null);
  if (indexRef.current === null) indexRef.current = canonicalIndex ?? createCanonicalIndex();
  const parsed = useMemo(
    () => (effectiveMode === "conversation"
      ? parserRef.current!.parseAll(candidates, gridOrder, details, sourceEpoch)
      : NO_PARSED),
    [effectiveMode, candidates, gridOrder, details, sourceEpoch],
  );
  const settled = effectiveMode === "conversation" && parsed.every((candidate) => candidate.detail !== null);

  // Body-verified selection only. When nothing verifiable and non-suggestion
  // remains, canonical stays null and the drill-in shows the auxiliary/flow
  // view instead of promoting a rejected flow into a fake chat.
  const canonical = useMemo(() => {
    if (!settled) return null;
    const usable = parsed
      .filter((candidate) => candidate.request !== null && candidate.rawMessages !== null)
      .map((candidate) => ({
        flowId: candidate.flow.flow_id,
        messages: candidate.rawMessages!,
        suggestion: candidate.suggestion,
        order: candidate.order,
        ...(candidate.contextKey !== undefined ? { contextKey: candidate.contextKey } : {}),
      }));
    const selected = indexRef.current!.update(usable).canonicalId;
    return selected === null ? null : parsed.find((candidate) => candidate.flow.flow_id === selected) ?? null;
  }, [settled, parsed]);

  const auxiliary = useMemo(
    () => (canonical === null ? summary.flows : summary.flows.filter((flow) => flow.flow_id !== canonical.flow.flow_id)),
    [summary.flows, canonical],
  );
  const models = summary.models.map(shortModel).join(" · ");
  const duration = durationBetween(summary.startedAt, summary.lastActivity);

  return (
    <section className="session-detail" data-testid="session-detail">
      <div className="session-detail-bar">
        <button type="button" className="session-back" onClick={onBack}>← sessions</button>
        <span className="session-detail-title">{sessionLabel(summary.key)}</span>
        <span className="session-detail-meta">
          {summary.flowCount} {summary.flowCount === 1 ? "flow" : "flows"}
          {models.length > 0 ? ` · ${models}` : ""}
          {duration === PLACEHOLDER ? "" : ` · ${duration}`}
          {summary.hasError ? " · " : ""}
          {summary.hasError ? <span className="session-status-error">err</span> : null}
        </span>
        <div className="packet-detail-modes session-detail-modes">
          <button
            type="button"
            className="packet-mode"
            aria-pressed={effectiveMode === "conversation"}
            disabled={candidates.length === 0}
            onClick={() => setMode("conversation")}
          >conversation</button>
          <button
            type="button"
            className="packet-mode"
            aria-pressed={effectiveMode === "flows"}
            onClick={() => setMode("flows")}
          >flows</button>
        </div>
      </div>
      {effectiveMode === "conversation" ? (
        <div className="session-conversation">
          {!settled ? (
            <p className="packet-empty">loading conversation…</p>
          ) : canonical === null ? (
            <>
              <p className="packet-empty">no main-thread conversation in this session — auxiliary calls only</p>
              <Collapse
                className="session-aux"
                label={`auxiliary calls (${auxiliary.length})`}
                meta={<span className="conv-section-meta">suggestion + utility side-calls</span>}
                defaultOpen
              >
                <PacketList flows={auxiliary} sourceEpoch={sourceEpoch} showSearch={false} loadFlowDetail={loadFlowDetail} />
              </Collapse>
            </>
          ) : (
            <>
              <PacketDetail metadata={canonical.flow} detail={canonical.detail} />
              {auxiliary.length > 0 ? (
                <Collapse
                  className="session-aux"
                  label={`auxiliary calls (${auxiliary.length})`}
                  meta={<span className="conv-section-meta">suggestion + utility side-calls</span>}
                >
                  <PacketList flows={auxiliary} sourceEpoch={sourceEpoch} showSearch={false} loadFlowDetail={loadFlowDetail} />
                </Collapse>
              ) : null}
            </>
          )}
        </div>
      ) : (
        <PacketList flows={summary.flows} sourceEpoch={sourceEpoch} showSearch={false} loadFlowDetail={loadFlowDetail} />
      )}
    </section>
  );
}
