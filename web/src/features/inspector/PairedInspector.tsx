import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import type { Header } from "../../protocol";
import { LIFECYCLE_EVENTS_PER_FLOW } from "../../state/browserState";
import { bodyMetadata, decodeBody, defaultBodyMode, type DecodedBody, type SseEvent } from "./decoders";
import { JsonTree, LARGE_TREE_COLLAPSE_THRESHOLD, safeParseJson } from "./jsonTree";
import { inspectorErrorState, lifecycleLabel, lifecyclePhase, orderLifecycle, type InspectorErrorState } from "./lifecycle";
import type { BodyPane, BodySelection, BodyViewMode, InspectableBody, InspectorBodyPanelProps, InspectorFlow, InspectorHeader, InspectorPane, InspectorProps } from "./models";
import { flowSummary } from "./summary";
import "../../styles/inspector.css";

const useInspectorLayoutEffect = typeof document === "undefined" ? useEffect : useLayoutEffect;
const paneOrder: InspectorPane[] = ["request", "response", "error"];
const bodyModes: Array<{ value: BodyViewMode; label: string }> = [
  { value: "json", label: "JSON" },
  { value: "text", label: "Text" },
  { value: "sse", label: "SSE" },
  { value: "hex", label: "Hex" },
];

export function nextBodyTabIndex(current: number, key: string, count: number, orientation: "horizontal" | "vertical" = "horizontal"): number | undefined {
  if (count < 1) return undefined;
  if (key === "Home") return 0;
  if (key === "End") return count - 1;
  const forward = orientation === "horizontal" ? "ArrowRight" : "ArrowDown";
  const backward = orientation === "horizontal" ? "ArrowLeft" : "ArrowUp";
  if (key === forward) return (current + 1) % count;
  if (key === backward) return (current - 1 + count) % count;
  return undefined;
}

export function isBodySelectionAuthorized(selection: BodySelection | null, flowId: string, pane: BodyPane): boolean {
  return selection?.flowId === flowId && selection.pane === pane;
}

/**
 * Deprecated F5 helper preserved so external tooling that still imports it
 * keeps type-checking; the F6 inspector auto-renders bodies and no longer
 * manages focus recovery through this helper.
 */
export function bodyFocusTarget(wasSelected: boolean, selected: boolean): "active-tab" | "inspect-control" | undefined {
  if (selected) return "active-tab";
  if (wasSelected) return "inspect-control";
  return undefined;
}

function bodyFor(flow: InspectorFlow, pane: "request" | "response"): InspectableBody {
  if (pane === "request") return flow.request_body ?? flow.metadata.request_body;
  return flow.response_body ?? flow.metadata.response_body ?? { state: "missing" };
}

function headersFor(flow: InspectorFlow, pane: "request" | "response"): readonly (Header | InspectorHeader)[] {
  if (pane === "request") return flow.request_headers ?? flow.metadata.request_headers;
  return flow.response_headers ?? flow.metadata.response_headers ?? [];
}

function isRedactedHeader(header: Header | InspectorHeader): boolean {
  return "redacted" in header && header.redacted === true || header.value === "[REDACTED]";
}

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 }).format(date);
}

function HeaderList({ headers }: { headers: readonly (Header | InspectorHeader)[] }) {
  if (headers.length === 0) return <p className="inspector-muted">No headers observed.</p>;
  return (
    <div className="inspector-headers" role="table" aria-label="Ordered headers">
      {headers.map((header, index) => (
        <div className="inspector-header-row" role="row" key={`${header.name}-${index}`}>
          <span className="inspector-header-name" role="cell">{header.name}</span>
          <span className={isRedactedHeader(header) ? "inspector-header-value is-redacted" : "inspector-header-value"} role="cell">
            {header.value || <span className="inspector-empty-value">empty value</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

export function InspectorBodyPanel({ body, pane }: InspectorBodyPanelProps) {
  const [mode, setMode] = useState<BodyViewMode>(() => defaultBodyMode(body));
  const metadata = bodyMetadata(body);
  const canDecode = body.state !== "missing" && body.state !== "redacted";
  // Memoize the decoded payload — decodeBody now runs on the full wire
  // ceiling (~3 MiB) so a re-decode on every keystroke or hover would burn
  // real wall-clock time on large captures.
  const decoded = useMemo<DecodedBody | undefined>(
    () => (canDecode ? decodeBody(body, mode) : undefined),
    [body, canDecode, mode],
  );
  const tablistId = `${useId().replaceAll(":", "")}-body-tabs`;
  const panelId = `${tablistId}-panel`;
  const sectionRef = useRef<HTMLElement | null>(null);
  const tabRefs = useRef<Partial<Record<BodyViewMode, HTMLButtonElement | null>>>({});

  const handleModeKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const nextIndex = nextBodyTabIndex(index, event.key, bodyModes.length);
    if (nextIndex === undefined) return;
    event.preventDefault();
    const nextMode = bodyModes[nextIndex].value;
    setMode(nextMode);
    tabRefs.current[nextMode]?.focus();
  };

  return (
    <section ref={sectionRef} className={`inspector-body-panel is-${metadata.state}`} aria-label={`${pane === "request" ? "Request" : "Response"} body`} tabIndex={-1}>
      {canDecode ? (
        <>
          <div className="inspector-mode-row" role="tablist" aria-label={`${pane} body view mode`} aria-orientation="horizontal">
            {bodyModes.map((bodyMode, index) => (
              <button
                key={bodyMode.value}
                ref={(element) => { tabRefs.current[bodyMode.value] = element; }}
                id={`${tablistId}-${bodyMode.value}`}
                className={`inspector-mode-tab ${mode === bodyMode.value ? "is-active" : ""}`}
                type="button"
                role="tab"
                aria-selected={mode === bodyMode.value}
                aria-controls={panelId}
                tabIndex={mode === bodyMode.value ? 0 : -1}
                onClick={() => setMode(bodyMode.value)}
                onKeyDown={(event) => handleModeKeyDown(event, index)}
              >{bodyMode.label}</button>
            ))}
            <span className="inspector-body-hint" title={metadata.contentType}>{metadata.contentType} · {metadata.size}</span>
          </div>
          <div id={panelId} className="inspector-body-panelbody" role="tabpanel" aria-labelledby={`${tablistId}-${mode}`} tabIndex={0}>
            {decoded && <BodyOutput decoded={decoded} />}
          </div>
        </>
      ) : (
        <div className={`inspector-body-empty is-${metadata.state}`} role="status">
          <strong>{metadata.state === "redacted" ? "Withheld by redaction" : "No body bytes retained"}</strong>
          <p>{metadata.state === "redacted" ? "Source marked this content as unavailable." : "Metadata remains, but no payload was captured."}</p>
        </div>
      )}
    </section>
  );
}

function BodyOutput({ decoded }: { decoded: DecodedBody }) {
  const parsed = useMemo(
    () => (decoded.mode === "json" && decoded.fallback === "none" ? safeParseJson(decoded.text) : null),
    [decoded.fallback, decoded.mode, decoded.text],
  );
  const startCollapsed = decoded.byteLength > LARGE_TREE_COLLAPSE_THRESHOLD;
  const warnings = [
    // Only surface the truncated chip when the backend flagged the capture
    // itself as truncated. Decoder-side clamping is gone in F6, so raw
    // "prefix truncated" is only ever true for bodies larger than the wire
    // ceiling — the rare case worth showing a chip for.
    decoded.truncated ? "capture truncated at wire ceiling" : null,
    decoded.invalidEncoding ? "invalid UTF-8" : null,
    decoded.fallback !== "none" ? `fallback: ${decoded.fallback}` : null,
  ].filter(Boolean) as string[];
  return (
    <div className="inspector-output-wrap">
      {warnings.length > 0 && (
        <div className="inspector-output-meta" role="note">
          {warnings.map((warning) => <span key={warning} className="inspector-output-warning">{warning}</span>)}
        </div>
      )}
      {parsed && parsed.ok ? (
        <div className={`inspector-output is-${decoded.mode}`} tabIndex={0} aria-label={`${decoded.mode} body output`}>
          <JsonTree value={parsed.value} startCollapsed={startCollapsed} />
        </div>
      ) : decoded.mode === "sse" && decoded.events !== undefined && decoded.events.length > 0 ? (
        <div className={`inspector-output is-${decoded.mode}`} tabIndex={0} aria-label={`${decoded.mode} body output`}>
          <SseBlocks events={decoded.events} pendingSuffix={decoded.pendingSseSuffix} />
        </div>
      ) : (
        <pre className={`inspector-output is-${decoded.mode}`} tabIndex={0} aria-label={`${decoded.mode} body output`}>{decoded.text}</pre>
      )}
    </div>
  );
}

function SseBlocks({ events, pendingSuffix }: { events: readonly SseEvent[]; pendingSuffix?: string }) {
  return (
    <ol className="inspector-sse-list">
      {events.map((event, index) => (
        <SseBlock key={index} event={event} index={index} />
      ))}
      {pendingSuffix !== undefined && pendingSuffix.length > 0 && (
        <li className="inspector-sse-item is-truncated" data-testid="sse-truncated-frame">
          <div className="inspector-sse-toggle" aria-label="truncated frame">
            <span aria-hidden="true">…</span>
            <span className="inspector-sse-label">truncated frame</span>
            <span className="inspector-sse-meta">no terminator observed</span>
          </div>
          <div className="inspector-sse-body">
            <pre className="inspector-sse-raw">{pendingSuffix}</pre>
          </div>
        </li>
      )}
    </ol>
  );
}

function SseBlock({ event, index }: { event: SseEvent; index: number }) {
  const [collapsed, setCollapsed] = useState<boolean>(index >= 3);
  const label = `event ${String(index + 1).padStart(2, "0")}${event.event ? ` · ${event.event}` : ""}`;
  const parsed = event.data.length > 0 ? safeParseJson(event.data) : null;
  return (
    <li className="inspector-sse-item">
      <button
        type="button"
        className="inspector-sse-toggle"
        aria-expanded={!collapsed}
        onClick={() => setCollapsed((previous) => !previous)}
      >
        <span aria-hidden="true">{collapsed ? "+" : "−"}</span>
        <span className="inspector-sse-label">{label}</span>
        {event.id !== undefined && <span className="inspector-sse-meta">id: {event.id}</span>}
        {event.retry !== undefined && <span className="inspector-sse-meta">retry: {event.retry}</span>}
      </button>
      {!collapsed && (
        <div className="inspector-sse-body">
          {parsed && parsed.ok ? <JsonTree value={parsed.value} /> : <pre className="inspector-sse-raw">{event.data || "(no data)"}</pre>}
          {event.comments.map((comment, commentIndex) => (
            <p key={commentIndex} className="inspector-sse-comment">: {comment}</p>
          ))}
        </div>
      )}
    </li>
  );
}

function LifecycleStrip({ flow, errorState }: { flow: InspectorFlow; errorState: InspectorErrorState }) {
  const lifecycleId = useId().replaceAll(":", "");
  const entries = useMemo(() => orderLifecycle(flow.lifecycle), [flow.lifecycle]);
  const phase = lifecyclePhase(entries);
  return (
    <section className="inspector-lifecycle" aria-labelledby={`${lifecycleId}-heading`}>
      <h3 id={`${lifecycleId}-heading`} className="inspector-lifecycle-heading">Lifecycle</h3>
      <div className="inspector-phase-summary" aria-label="Lifecycle summary">
        <span className={phase.requestEnded ? "is-seen" : ""}>request end {phase.requestEnded ? "seen" : "pending"}</span>
        <span className={phase.responseStarted ? "is-seen" : ""}>response start {phase.responseStarted ? "seen" : "pending"}</span>
        <span className={errorState.hasError || phase.completed ? "is-seen" : ""}>{errorState.hasError ? "ended with error" : phase.completed ? "completed" : "open"}</span>
      </div>
      {flow.lifecycleTruncated === true && (
        <p className="inspector-muted inspector-lifecycle-truncated" role="note">
          Showing the newest {LIFECYCLE_EVENTS_PER_FLOW} events; older observations were dropped from the bounded window.
        </p>
      )}
      {entries.length === 0 ? <p className="inspector-muted inspector-lifecycle-empty">No lifecycle observations attached to this flow.</p> : (
        <ol className="inspector-trace" aria-label="Observed lifecycle events">
          {entries.map((entry, index) => (
            <li className={`inspector-trace-event is-${entry.state}`} key={entry.eventId}>
              <span className="inspector-trace-node" aria-hidden="true" />
              <span className="inspector-trace-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="inspector-trace-label">{lifecycleLabel(entry.state)}</span>
              <span className="inspector-trace-time">{formatTime(entry.occurredAt)}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function ErrorPane({ errorState }: { errorState: InspectorErrorState }) {
  const errorId = useId().replaceAll(":", "");
  return (
    <section className={`inspector-error-panel ${errorState.hasError ? "has-error" : ""}`} aria-labelledby={`${errorId}-heading`}>
      <h3 id={`${errorId}-heading`}>{errorState.hasError ? "Error observed" : "No error recorded"}</h3>
      {errorState.hasError ? <pre className="inspector-error-copy">{errorState.message}</pre> : <p className="inspector-muted">No error event was supplied for this flow.</p>}
    </section>
  );
}

export function PairedInspector({ flow, className = "", compact = false, activePane, onPaneChange }: InspectorProps) {
  const [internalPane, setInternalPane] = useState<InspectorPane>("request");
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const inspectorId = useId().replaceAll(":", "");
  const previousActivePaneRef = useRef<InspectorPane | undefined>(activePane);
  const lastCommittedPaneRef = useRef<InspectorPane>(activePane ?? internalPane);
  const previousCommittedPaneRef = useRef<InspectorPane>(activePane ?? internalPane);
  const previousFlowIdRef = useRef(flow.metadata.flow_id);
  const errorState = inspectorErrorState(flow);

  const pane = activePane ?? (previousActivePaneRef.current === undefined ? internalPane : lastCommittedPaneRef.current);

  useInspectorLayoutEffect(() => {
    if (activePane === undefined) {
      if (previousActivePaneRef.current !== undefined) setInternalPane(lastCommittedPaneRef.current);
      lastCommittedPaneRef.current = pane;
    } else {
      lastCommittedPaneRef.current = activePane;
    }
    previousCommittedPaneRef.current = pane;
    previousFlowIdRef.current = flow.metadata.flow_id;
    previousActivePaneRef.current = activePane;
  }, [activePane, flow.metadata.flow_id, pane]);

  const selectPane = (nextPane: InspectorPane) => {
    if (activePane === undefined) setInternalPane(nextPane);
    onPaneChange?.(nextPane);
  };

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const nextIndex = nextBodyTabIndex(index, event.key, paneOrder.length, "horizontal");
    if (nextIndex === undefined) return;
    event.preventDefault();
    tabRefs.current[nextIndex]?.focus();
    selectPane(paneOrder[nextIndex]);
  };

  const summaryLine = useMemo(() => flowSummary(flow), [flow]);
  return (
    <section data-testid="paired-inspector" className={`paired-inspector ${compact ? "is-compact" : ""} ${className}`.trim()} aria-labelledby={`${inspectorId}-title`}>
      <header className="inspector-header">
        <h1 id={`${inspectorId}-title`} className="inspector-flow-line">
          <span className="inspector-method">{flow.metadata.method}</span>
          <span className="inspector-route">{flow.metadata.host}{flow.metadata.path}</span>
        </h1>
        <code className="inspector-flow-id">{flow.metadata.flow_id}</code>
      </header>
      <p className="inspector-summary" aria-label="Flow summary">{summaryLine}</p>

      <div className="inspector-tabbar" role="tablist" aria-label="Exchange panes" aria-orientation="horizontal" id={`${inspectorId}-exchange-tabs`}>
        {paneOrder.map((item, index) => (
          <button
            key={item}
            ref={(element) => { tabRefs.current[index] = element; }}
            id={`${inspectorId}-exchange-${item}`}
            className={`inspector-pane-tab is-${item} ${pane === item ? "is-active" : ""}`}
            type="button"
            role="tab"
            aria-selected={pane === item}
            aria-controls={`${inspectorId}-exchange-${item}-pane`}
            tabIndex={pane === item ? 0 : -1}
            onClick={() => selectPane(item)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
          >
            <span>{item === "request" ? "Request" : item === "response" ? "Response" : "Error"}</span>
            {item === "error" && <small>{errorState.hasError ? "observed" : "clear"}</small>}
          </button>
        ))}
      </div>

      {paneOrder.map((item) => {
        const isActive = pane === item;
        return (
          <div
            key={item}
            className="inspector-exchange-tabpanel"
            id={`${inspectorId}-exchange-${item}-pane`}
            role="tabpanel"
            tabIndex={isActive ? 0 : -1}
            aria-labelledby={`${inspectorId}-exchange-${item}`}
            hidden={!isActive}
            aria-hidden={isActive ? undefined : "true"}
          >
            {isActive && item === "error" && <ErrorPane errorState={errorState} />}
            {isActive && item !== "error" && (
              <div className="inspector-pane-layout">
                <section className="inspector-column inspector-body-column">
                  <InspectorBodyPanel body={bodyFor(flow, item)} pane={item} />
                </section>
                <section className="inspector-column inspector-metadata-column">
                  <div className="inspector-subsection">
                    <div className="inspector-subheading"><span>Headers</span><span>{headersFor(flow, item).length}</span></div>
                    <HeaderList headers={headersFor(flow, item)} />
                  </div>
                  <dl className="inspector-facts">
                    <div><dt>host</dt><dd>{flow.metadata.host}</dd></div>
                    <div><dt>path</dt><dd>{flow.metadata.path}</dd></div>
                    <div><dt>port</dt><dd>{flow.metadata.port}</dd></div>
                    <div><dt>scheme</dt><dd>{flow.metadata.scheme}</dd></div>
                  </dl>
                </section>
              </div>
            )}
          </div>
        );
      })}

      <LifecycleStrip flow={flow} errorState={errorState} />
    </section>
  );
}

export type { BodyPane, BodySelection, BodyViewMode, InspectableBody, InspectorFlow, InspectorPane } from "./models";
