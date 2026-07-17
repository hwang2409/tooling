import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import type { Header } from "../../protocol";
import { bodyMetadata, decodeBody, type DecodedBody } from "./decoders";
import { lifecycleLabel, lifecyclePhase, orderLifecycle } from "./lifecycle";
import type { BodyPane, BodySelection, BodyViewMode, InspectableBody, InspectorBodyPanelProps, InspectorFlow, InspectorHeader, InspectorPane, InspectorProps } from "./models";
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

function BodyBadge({ body }: { body: InspectableBody }) {
  const metadata = bodyMetadata(body);
  return <span className={`inspector-body-badge is-${metadata.state}`}>{metadata.state}</span>;
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
          <span className="inspector-header-index" aria-label={`header ${index + 1}`}>{String(index + 1).padStart(2, "0")}</span>
        </div>
      ))}
    </div>
  );
}

export function InspectorBodyPanel({ body, pane, selected, onSelect }: InspectorBodyPanelProps) {
  const [mode, setMode] = useState<BodyViewMode>("text");
  const metadata = bodyMetadata(body);
  const decoded = selected ? decodeBody(body, mode) : undefined;
  const bodyId = `${pane}-body-panel`;
  const tablistId = useId();
  const panelId = `${tablistId}-panel`;
  const sectionRef = useRef<HTMLElement | null>(null);
  const inspectRef = useRef<HTMLButtonElement | null>(null);
  const tabRefs = useRef<Partial<Record<BodyViewMode, HTMLButtonElement | null>>>({});
  const wasSelected = useRef(selected);
  const canInspect = body.state !== "missing" && body.state !== "redacted";

  useInspectorLayoutEffect(() => {
    const target = bodyFocusTarget(wasSelected.current, selected);
    if (target === "active-tab") tabRefs.current[mode]?.focus();
    else if (target === "inspect-control") (inspectRef.current ?? sectionRef.current)?.focus();
    wasSelected.current = selected;
  }, [mode, selected]);

  const handleModeKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const nextIndex = nextBodyTabIndex(index, event.key, bodyModes.length);
    if (nextIndex === undefined) return;
    event.preventDefault();
    const nextMode = bodyModes[nextIndex].value;
    setMode(nextMode);
    tabRefs.current[nextMode]?.focus();
  };

  return (
    <section ref={sectionRef} className={`inspector-body-panel ${selected ? "is-selected" : ""}`} aria-labelledby={bodyId} tabIndex={-1}>
      <div className="inspector-section-heading">
        <div>
          <p className="inspector-kicker">{pane === "request" ? "REQUEST BODY" : "RESPONSE BODY"}</p>
          <h3 id={bodyId}>Bounded capture</h3>
        </div>
        <BodyBadge body={body} />
      </div>
      <dl className="inspector-body-meta">
        <div><dt>total</dt><dd>{metadata.size}</dd></div>
        <div><dt>captured</dt><dd>{metadata.captured}</dd></div>
        <div><dt>type</dt><dd>{metadata.contentType}</dd></div>
      </dl>
      {!selected ? (
        <div className="inspector-body-gate">
          <div>
            <strong>{canInspect ? "Body decoding is paused" : metadata.state === "redacted" ? "Body withheld by redaction" : "No body bytes retained"}</strong>
            <p>{canInspect ? "Select this pane to decode the bounded prefix." : metadata.state === "redacted" ? "The source marked this content as unavailable." : "Metadata remains available without a body payload."}</p>
          </div>
          {canInspect && <button ref={inspectRef} className="inspector-action" type="button" onClick={onSelect}>Inspect body <span aria-hidden="true">↗</span></button>}
        </div>
      ) : (
        <div className="inspector-body-view">
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
            <span className="inspector-view-limit">bounded / 64 KiB</span>
          </div>
          <div id={panelId} role="tabpanel" aria-labelledby={`${tablistId}-${mode}`} tabIndex={0}>
            <BodyOutput decoded={decoded!} />
          </div>
        </div>
      )}
    </section>
  );
}

function BodyOutput({ decoded }: { decoded: DecodedBody }) {
  return (
    <div className="inspector-output-wrap">
      <div className="inspector-output-meta">
        <span>{decoded.byteLength.toLocaleString()} decoded bytes</span>
        {decoded.truncated && <span className="inspector-output-warning">prefix truncated</span>}
        {decoded.invalidEncoding && <span className="inspector-output-warning">invalid UTF-8</span>}
        {decoded.fallback !== "none" && <span className="inspector-output-warning">fallback: {decoded.fallback}</span>}
      </div>
      <pre className={`inspector-output is-${decoded.mode}`} tabIndex={0} aria-label={`${decoded.mode} body output`}>{decoded.text}</pre>
      <p className="inspector-copy-note">Derived text is copy-ready from the focused output; raw bytes are never exported automatically.</p>
    </div>
  );
}

function LifecycleStrip({ flow }: { flow: InspectorFlow }) {
  const entries = useMemo(() => orderLifecycle(flow.lifecycle), [flow.lifecycle]);
  const phase = lifecyclePhase(entries);
  return (
    <section className="inspector-lifecycle" aria-labelledby="lifecycle-heading">
      <div className="inspector-section-heading">
        <div>
          <p className="inspector-kicker">OBSERVED ORDER</p>
          <h3 id="lifecycle-heading">Lifecycle trace</h3>
        </div>
        <span className="inspector-lifecycle-note">sequence is authoritative</span>
      </div>
      <div className="inspector-phase-summary" aria-label="Lifecycle summary">
        <span className={phase.requestEnded ? "is-seen" : ""}>request end {phase.requestEnded ? "seen" : "pending"}</span>
        <span className={phase.responseStarted ? "is-seen" : ""}>response start {phase.responseStarted ? "seen" : "pending"}</span>
        <span className={phase.completed ? "is-seen" : ""}>{phase.completed ? "completed" : phase.errored ? "ended with error" : "open"}</span>
      </div>
      {entries.length === 0 ? <p className="inspector-muted inspector-lifecycle-empty">No lifecycle observations attached to this flow.</p> : (
        <ol className="inspector-trace" aria-label="Observed lifecycle events">
          {entries.map((entry, index) => (
            <li className={`inspector-trace-event is-${entry.state}`} key={entry.eventId}>
              <span className="inspector-trace-node" aria-hidden="true" />
              <span className="inspector-trace-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="inspector-trace-label">{lifecycleLabel(entry.state)}</span>
              <span className="inspector-trace-time">{formatTime(entry.occurredAt)}</span>
              <span className="inspector-trace-sequence">#{entry.sequence}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function ErrorPane({ error }: { error?: string }) {
  return (
    <section className={`inspector-error-panel ${error ? "has-error" : ""}`} aria-labelledby="error-heading">
      <p className="inspector-kicker">FLOW OUTCOME</p>
      <h3 id="error-heading">{error ? "Error observed" : "No error recorded"}</h3>
      {error ? <pre className="inspector-error-copy">{error}</pre> : <p className="inspector-muted">No error event was supplied for this flow.</p>}
    </section>
  );
}

export function PairedInspector({ flow, className = "", compact = false, bodySelection, onBodySelect, onPaneChange }: InspectorProps) {
  const [pane, setPane] = useState<InspectorPane>("request");
  const [internalSelectedBody, setInternalSelectedBody] = useState<BodySelection | null>(null);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const tablistId = useId();
  const entries = useMemo(() => orderLifecycle(flow.lifecycle), [flow.lifecycle]);

  useEffect(() => {
    onPaneChange?.(pane);
  }, [onPaneChange, pane]);

  const selectPane = (nextPane: InspectorPane) => {
    setPane(nextPane);
    if (bodySelection === undefined) setInternalSelectedBody(null);
  };

  const selectBody = (bodyPane: BodyPane) => {
    const selection = { flowId: flow.metadata.flow_id, pane: bodyPane };
    if (bodySelection === undefined) setInternalSelectedBody(selection);
    onBodySelect?.(selection);
  };

  const selectedBody = bodySelection === undefined ? internalSelectedBody : bodySelection;
  const isBodySelected = (bodyPane: BodyPane) => isBodySelectionAuthorized(selectedBody, flow.metadata.flow_id, bodyPane);

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | undefined;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % paneOrder.length;
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + paneOrder.length) % paneOrder.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = paneOrder.length - 1;
    if (nextIndex === undefined) return;
    event.preventDefault();
    tabRefs.current[nextIndex]?.focus();
    selectPane(paneOrder[nextIndex]);
  };

  const responseBody = bodyFor(flow, "response");
  return (
    <main className={`paired-inspector ${compact ? "is-compact" : ""} ${className}`.trim()}>
      <header className="inspector-header">
        <div className="inspector-title-lockup">
          <span className="inspector-glyph" aria-hidden="true">↔</span>
          <div>
            <p className="inspector-kicker">PAIRED EXCHANGE / READ ONLY</p>
            <h1>Request / response inspector</h1>
          </div>
        </div>
        <div className="inspector-flow-identity">
          <span className="inspector-method">{flow.metadata.method}</span>
          <span className="inspector-route">{flow.metadata.scheme}://{flow.metadata.host}:{flow.metadata.port}{flow.metadata.path}</span>
          <code>{flow.metadata.flow_id}</code>
        </div>
      </header>

      <div className="inspector-tabbar" role="tablist" aria-label="Exchange panes" id={tablistId}>
        {paneOrder.map((item, index) => (
          <button
            key={item}
            ref={(element) => { tabRefs.current[index] = element; }}
            id={`${tablistId}-${item}`}
            className={`inspector-pane-tab is-${item} ${pane === item ? "is-active" : ""}`}
            type="button"
            role="tab"
            aria-selected={pane === item}
            aria-controls={`${item}-pane`}
            tabIndex={pane === item ? 0 : -1}
            onClick={() => selectPane(item)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
          >
            <span>{item === "request" ? "Request" : item === "response" ? "Response" : "Error"}</span>
            <small>{item === "request" ? "outbound" : item === "response" ? "inbound" : flow.error ? "observed" : "clear"}</small>
          </button>
        ))}
      </div>

      {pane === "error" ? <div id="error-pane" role="tabpanel" tabIndex={0}><ErrorPane error={flow.error} /></div> : (
        <div className="inspector-pane-layout" id={`${pane}-pane`} role="tabpanel" tabIndex={0} aria-labelledby={`${tablistId}-${pane}`}>
          <section className="inspector-column inspector-metadata-column">
            <div className="inspector-section-heading">
              <div>
                <p className="inspector-kicker">{pane === "request" ? "OUTBOUND METADATA" : "INBOUND METADATA"}</p>
                <h2>{pane === "request" ? "Request" : "Response"}</h2>
              </div>
              <span className="inspector-sequence-count">{entries.length} observations</span>
            </div>
            <dl className="inspector-facts">
              <div><dt>host</dt><dd>{flow.metadata.host}</dd></div>
              <div><dt>path</dt><dd>{flow.metadata.path}</dd></div>
              <div><dt>port</dt><dd>{flow.metadata.port}</dd></div>
              <div><dt>scheme</dt><dd>{flow.metadata.scheme}</dd></div>
            </dl>
            <div className="inspector-subsection">
              <div className="inspector-subheading"><span>Headers</span><span>{headersFor(flow, pane).length} ordered</span></div>
              <HeaderList headers={headersFor(flow, pane)} />
            </div>
          </section>
          <section className="inspector-column inspector-body-column">
            <InspectorBodyPanel body={bodyFor(flow, pane)} pane={pane} selected={isBodySelected(pane)} onSelect={() => selectBody(pane)} />
            {pane === "response" && bodyFor(flow, "request").state !== "missing" && <p className="inspector-cross-note">Request body is available in the Request pane.</p>}
            {pane === "request" && responseBody.state !== "missing" && <p className="inspector-cross-note">Response body is available in the Response pane.</p>}
          </section>
        </div>
      )}

      <LifecycleStrip flow={flow} />
    </main>
  );
}

export type { BodyPane, BodySelection, BodyViewMode, InspectableBody, InspectorFlow, InspectorPane } from "./models";
