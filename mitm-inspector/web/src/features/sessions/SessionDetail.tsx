import { useMemo, useState } from "react";

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
import { selectCanonicalFlow } from "./canonical";
import { sessionLabel } from "./SessionList";
import { conversationCandidates, isSuggestionRequest } from "./sessionSummary";
import type { SessionSummary } from "./sessionSummary";

export interface SessionDetailProps {
  summary: SessionSummary;
  onBack: () => void;
  loadFlowDetail?: FlowDetailLoader;
}

type SessionMode = "conversation" | "flows";

const NO_FLOWS: readonly ImmutableFlowMetadata[] = [];

interface ParsedCandidate {
  readonly flow: ImmutableFlowMetadata;
  readonly order: number;
  readonly detail: FlowDetailResult | null;
  readonly request: AnthropicRequest | null;
  readonly rawMessages: readonly JsonValue[] | null;
}

function parseCandidate(
  flow: ImmutableFlowMetadata,
  order: number,
  detail: FlowDetailResult | undefined,
): ParsedCandidate {
  const base = { flow, order, detail: detail ?? null, request: null, rawMessages: null };
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
  return { ...base, request, rawMessages };
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
export function SessionDetail({ summary, onBack, loadFlowDetail }: SessionDetailProps) {
  const candidates = useMemo(() => conversationCandidates(summary.flows), [summary.flows]);
  // The unassigned bucket keeps the grid experience as its default drill-in.
  const [mode, setMode] = useState<SessionMode>(summary.key === null ? "flows" : "conversation");
  const effectiveMode: SessionMode = candidates.length === 0 ? "flows" : mode;

  const details = useFlowDetails(effectiveMode === "conversation" ? candidates : NO_FLOWS, loadFlowDetail);
  const parsed = useMemo(
    () => candidates.map((flow, order) => parseCandidate(flow, order, details.get(flow.flow_id))),
    [candidates, details],
  );
  const settled = effectiveMode === "conversation" && parsed.every((candidate) => candidate.detail !== null);

  const canonical = useMemo(() => {
    if (!settled) return null;
    const usable = parsed
      .filter((candidate) => candidate.request !== null && candidate.rawMessages !== null)
      .map((candidate) => ({
        flowId: candidate.flow.flow_id,
        messages: candidate.rawMessages!,
        suggestion: isSuggestionRequest(candidate.request!),
        order: candidate.order,
      }));
    const selected = selectCanonicalFlow(usable).canonicalId;
    // No verifiable body anywhere (all fetches unavailable/errored): fall
    // back to the metadata ranking so the drill-in still shows something.
    const fallback = parsed.find((candidate) => candidate.request === null || !isSuggestionRequest(candidate.request));
    const chosenId = selected ?? fallback?.flow.flow_id ?? parsed[0]?.flow.flow_id ?? null;
    return parsed.find((candidate) => candidate.flow.flow_id === chosenId) ?? null;
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
          {canonical === null
            ? <p className="packet-empty">loading conversation…</p>
            : <PacketDetail metadata={canonical.flow} detail={canonical.detail} />}
          {canonical !== null && auxiliary.length > 0 ? (
            <Collapse
              className="session-aux"
              label={`auxiliary calls (${auxiliary.length})`}
              meta={<span className="conv-section-meta">suggestion + utility side-calls</span>}
            >
              <PacketList flows={auxiliary} showSearch={false} loadFlowDetail={loadFlowDetail} />
            </Collapse>
          ) : null}
        </div>
      ) : (
        <PacketList flows={summary.flows} showSearch={false} loadFlowDetail={loadFlowDetail} />
      )}
    </section>
  );
}
