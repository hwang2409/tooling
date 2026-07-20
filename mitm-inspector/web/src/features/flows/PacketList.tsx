/* eslint-disable no-unused-vars */

import { useEffect, useMemo, useState } from "react";

import type { BrowserState, ImmutableFlowMetadata } from "../../state/browserState";
import { parseFlowExtras } from "../../protocol";
import { bodyText } from "../inspector/decoders";
import type { BodyText } from "../inspector/decoders";
import { useFlowDetail } from "../inspector/flowDetail";
import type { FlowDetailLoader, FlowDetailResult } from "../inspector/flowDetail";
import { JsonTree, LARGE_TREE_COLLAPSE_THRESHOLD, safeParseJson } from "../inspector/jsonTree";
import { parseAnthropicRequest } from "../inspector/anthropic";
import { ConversationView } from "../inspector/ConversationView";
import { useSearch } from "../search/search";
import type { SearchController, SearchFetcher, SearchMatch } from "../search/search";
import { deriveRowCells } from "./rowSummary";
import "../../styles/shell.css";

export interface PacketListProps {
  browser: BrowserState;
  loadFlowDetail?: FlowDetailLoader;
  searchFetcher?: SearchFetcher;
  onSearchActiveChange?: (active: boolean) => void;
}

type Row =
  | { kind: "header"; sessionId: string | null; count: number }
  | { kind: "flow"; metadata: ImmutableFlowMetadata };

function groupRows(flows: readonly ImmutableFlowMetadata[]): readonly Row[] {
  const rows: Row[] = [];
  let cursor = 0;
  while (cursor < flows.length) {
    const sessionId = flows[cursor].session_id ?? null;
    let end = cursor + 1;
    while (end < flows.length && (flows[end].session_id ?? null) === sessionId) end += 1;
    rows.push({ kind: "header", sessionId, count: end - cursor });
    for (let i = cursor; i < end; i += 1) rows.push({ kind: "flow", metadata: flows[i] });
    cursor = end;
  }
  return rows;
}

function searchStatusLine(state: SearchController["state"]): string | null {
  if (state.status === "loading") return "searching…";
  if (state.status === "results") {
    return `${state.matches.length} ${state.matches.length === 1 ? "match" : "matches"}${state.truncated ? " (truncated)" : ""}`;
  }
  if (state.status === "empty") return "no matches";
  if (state.status === "unavailable") return "search unavailable";
  return null;
}

function SearchBar({ search }: { search: SearchController }) {
  const status = searchStatusLine(search.state);
  return (
    <div className="search-bar">
      <input
        type="search"
        className="search-input"
        placeholder="search bodies"
        aria-label="Search captured bodies"
        aria-describedby="search-status"
        aria-controls="packet-list"
        value={search.query}
        onChange={(event) => search.setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") search.clear();
        }}
      />
      <span id="search-status" className="search-status" role="status" aria-live="polite" aria-atomic="true">{status ?? ""}</span>
    </div>
  );
}

function snippetLabel(matches: readonly SearchMatch[]): string {
  const first = matches[0];
  const prefix = first.field === "request_body" ? "req" : "res";
  const suffix = matches.length > 1 ? `  +${matches.length - 1}` : "";
  return `${prefix}: ${first.snippet}${suffix}`;
}

export function PacketList({ browser, loadFlowDetail, searchFetcher, onSearchActiveChange }: PacketListProps) {
  const [openFlowId, setOpenFlowId] = useState<string | null>(null);
  const search = useSearch(searchFetcher);
  const searchActive = search.state.status !== "idle";
  useEffect(() => {
    onSearchActiveChange?.(searchActive);
  }, [searchActive, onSearchActiveChange]);

  const flows = browser.flows.entries;
  const matchesByFlow = useMemo(() => {
    if (search.state.status !== "results" && search.state.status !== "empty") return null;
    const byFlow = new Map<string, SearchMatch[]>();
    for (const match of search.state.status === "results" ? search.state.matches : []) {
      const existing = byFlow.get(match.flow_id);
      if (existing === undefined) byFlow.set(match.flow_id, [match]);
      else existing.push(match);
    }
    return byFlow;
  }, [search.state]);

  const visibleFlows = useMemo(() => {
    if (matchesByFlow === null) return flows;
    const byId = new Map(flows.map((flow) => [flow.flow_id, flow]));
    return Array.from(matchesByFlow, ([flowId, matches]) => byId.get(flowId) ?? durableFlow(matches[0]));
  }, [flows, matchesByFlow]);
  const rows = useMemo(() => groupRows(visibleFlows), [visibleFlows]);
  const openFlow = openFlowId === null ? undefined : browser.flows.get(openFlowId);
  const detail = useFlowDetail(openFlow?.flow_id ?? null, loadFlowDetail);

  return (
    <div className="packet-pane">
      <SearchBar search={search} />
      {flows.length === 0 ? (
        <p className="packet-empty">no packets captured</p>
      ) : visibleFlows.length === 0 && searchActive ? (
        <p className="packet-empty">no matching packets</p>
      ) : (
        <ol id="packet-list" className="packet-list" aria-label="Captured packets">
          {rows.map((row, index) => {
            if (row.kind === "header") {
              const label = row.sessionId ? `session ${row.sessionId.slice(0, 8)}` : "unassigned";
              return (
                <li
                  key={`hdr-${index}-${row.sessionId ?? "unassigned"}`}
                  className="session-header"
                  aria-hidden="true"
                >
                  <span className="session-header-label">{label}</span>
                  <span className="session-header-count">
                    {row.count} {row.count === 1 ? "flow" : "flows"}
                  </span>
                </li>
              );
            }
            const { metadata } = row;
            const open = openFlow !== undefined && metadata.flow_id === openFlow.flow_id;
            const cells = deriveRowCells(metadata);
            const matches = matchesByFlow?.get(metadata.flow_id);
            return (
              <li key={metadata.flow_id} className="packet">
                <button
                  type="button"
                  className="packet-row"
                  aria-expanded={open}
                  onClick={() => setOpenFlowId(open ? null : metadata.flow_id)}
                >
                  <span className="packet-time">{cells.time}</span>
                  <span className={`packet-badge packet-badge-${cells.badgeKind}`}>{cells.badge}</span>
                  <span className="packet-model">{cells.model}</span>
                  <span className="packet-count">{cells.messages}</span>
                  {matches !== undefined ? (
                    <span className="packet-preview packet-snippet">{snippetLabel(matches)}</span>
                  ) : (
                    <span className={`packet-preview packet-preview-${cells.preview.kind}`}>{cells.preview.text}</span>
                  )}
                  <span className="packet-sizes">{cells.sizes}</span>
                  <span className="packet-duration">{cells.duration}</span>
                  <span className={`packet-status${cells.isError ? " packet-status-error" : ""}`}>{cells.status}</span>
                </button>
                {open && <PacketDetail metadata={metadata} detail={detail} />}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

/**
 * Older search responses contain only ids/snippets. Keep those results
 * visible until the additive durable metadata field is available, while
 * using the real metadata whenever the backend provides it.
 */
function durableFlow(match: SearchMatch): ImmutableFlowMetadata {
  if (match.flow !== undefined) return match.flow as ImmutableFlowMetadata;
  return {
    flow_id: match.flow_id,
    method: "?",
    scheme: "https",
    host: "retained",
    port: "443",
    path: `/${match.flow_id}`,
    request_headers: [],
    request_body: { state: "missing" },
  } as ImmutableFlowMetadata;
}

function responseContentType(metadata: ImmutableFlowMetadata): string | undefined {
  const header = metadata.response_headers?.find((candidate) => candidate.name.toLowerCase() === "content-type");
  return header?.value;
}

function PacketDetail({ metadata, detail }: { metadata: ImmutableFlowMetadata; detail: FlowDetailResult | null }) {
  const overrides = detail?.status === "loaded" ? detail.overrides : undefined;
  const requestBody = overrides?.request_body ?? metadata.request_body;
  const responseBody = overrides?.response_body ?? metadata.response_body;
  const requestDecoded = useMemo(() => bodyText(requestBody), [requestBody]);
  const responseDecoded = useMemo(() => bodyText(responseBody), [responseBody]);
  const extras = useMemo(() => parseFlowExtras(metadata), [metadata]);
  const anthropic = useMemo(() => {
    if (requestDecoded.kind !== "text") return null;
    const parsed = safeParseJson(requestDecoded.text);
    if (!parsed.ok) return null;
    return parseAnthropicRequest(parsed.value);
  }, [requestDecoded]);
  const [rawOverride, setRawOverride] = useState(false);
  const conversation = anthropic !== null && !rawOverride;

  return (
    <div className="packet-detail">
      {anthropic !== null ? (
        <div className="packet-detail-modes">
          <button
            type="button"
            className="packet-mode"
            aria-pressed={conversation}
            onClick={() => setRawOverride(false)}
          >conversation</button>
          <button
            type="button"
            className="packet-mode"
            aria-pressed={!conversation}
            onClick={() => setRawOverride(true)}
          >raw</button>
        </div>
      ) : null}
      {conversation && anthropic !== null ? (
        <ConversationView
          request={anthropic}
          extras={extras}
          responseText={responseDecoded.kind === "text" ? responseDecoded.text : undefined}
          responseContentType={responseContentType(metadata) ?? extras.response_content_type}
        />
      ) : (
        <>
          <RequestBody decoded={requestDecoded} />
          <ResponseBody decoded={responseDecoded} />
        </>
      )}
    </div>
  );
}

function RequestBody({ decoded }: { decoded: BodyText }) {
  if (decoded.kind === "absent") return <p className="packet-nobody">no request body</p>;
  return <DecodedBodyContent decoded={decoded} />;
}

function ResponseBody({ decoded }: { decoded: BodyText }) {
  if (decoded.kind === "absent") return null;
  return (
    <div className="packet-response">
      <span className="packet-response-label">response</span>
      <DecodedBodyContent decoded={decoded} />
    </div>
  );
}

function DecodedBodyContent({ decoded }: { decoded: Extract<BodyText, { kind: "text" }> }) {
  const parsed = useMemo(() => safeParseJson(decoded.text), [decoded.text]);
  if (parsed.ok) return <JsonTree value={parsed.value} startCollapsed={decoded.byteLength > LARGE_TREE_COLLAPSE_THRESHOLD} />;
  return <pre className="packet-text">{decoded.text}</pre>;
}
