import { useEffect, useMemo, useState } from "react";

import { PacketDetail, PacketList } from "../flows/PacketList";
import { PLACEHOLDER, durationBetween, shortModel } from "../flows/rowSummary";
import { parseAnthropicRequest } from "../inspector/anthropic";
import { Collapse } from "../inspector/ConversationView";
import { bodyText } from "../inspector/decoders";
import { useFlowDetail } from "../inspector/flowDetail";
import type { FlowDetailLoader } from "../inspector/flowDetail";
import { safeParseJson } from "../inspector/jsonTree";
import { sessionLabel } from "./SessionList";
import { conversationCandidates, isSuggestionRequest } from "./sessionSummary";
import type { SessionSummary } from "./sessionSummary";

export interface SessionDetailProps {
  summary: SessionSummary;
  onBack: () => void;
  loadFlowDetail?: FlowDetailLoader;
}

type SessionMode = "conversation" | "flows";

/**
 * One session's drill-in view. Conversation mode renders the canonical
 * main-thread flow — every main-thread request resends the full history, so
 * its messages array IS the whole conversation and the flow's response is the
 * final assistant turn. Everything else in the session (suggestion-mode and
 * utility side-calls) stays reachable in a collapsed auxiliary group, and the
 * full flow grid is one toggle away.
 */
export function SessionDetail({ summary, onBack, loadFlowDetail }: SessionDetailProps) {
  const candidates = useMemo(() => conversationCandidates(summary.flows), [summary.flows]);
  // The unassigned bucket keeps the grid experience as its default drill-in.
  const [mode, setMode] = useState<SessionMode>(summary.key === null ? "flows" : "conversation");
  const effectiveMode: SessionMode = candidates.length === 0 ? "flows" : mode;

  // Suggestion-mode side-calls can carry the same message_count as the main
  // thread; the metadata ranking is only a heuristic. Verify the fetched body
  // and fall through to the next candidate on a mis-classification.
  const [rejectedIds, setRejectedIds] = useState<readonly string[]>([]);
  const canonical = candidates.find((flow) => !rejectedIds.includes(flow.flow_id)) ?? candidates[0] ?? null;
  const detail = useFlowDetail(effectiveMode === "conversation" ? (canonical?.flow_id ?? null) : null, loadFlowDetail);

  const verifiedRequest = useMemo(() => {
    if (detail?.status !== "loaded") return null;
    const requestBody = detail.overrides.request_body ?? canonical?.request_body;
    if (requestBody === undefined) return null;
    const decoded = bodyText(requestBody);
    if (decoded.kind !== "text") return null;
    const parsed = safeParseJson(decoded.text);
    return parsed.ok ? parseAnthropicRequest(parsed.value) : null;
  }, [detail, canonical]);

  useEffect(() => {
    if (canonical === null || verifiedRequest === null) return;
    if (!isSuggestionRequest(verifiedRequest)) return;
    const alternative = candidates.some(
      (flow) => flow.flow_id !== canonical.flow_id && !rejectedIds.includes(flow.flow_id),
    );
    if (!alternative) return;
    setRejectedIds((previous) => [...previous, canonical.flow_id]);
  }, [canonical, verifiedRequest, candidates, rejectedIds]);

  const auxiliary = useMemo(
    () => (canonical === null ? summary.flows : summary.flows.filter((flow) => flow.flow_id !== canonical.flow_id)),
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
      {effectiveMode === "conversation" && canonical !== null ? (
        <div className="session-conversation">
          {detail === null
            ? <p className="packet-empty">loading conversation…</p>
            : <PacketDetail metadata={canonical} detail={detail} />}
          {auxiliary.length > 0 ? (
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
