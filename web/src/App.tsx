import { useId, useState } from "react";

import { useConnection } from "./features/connection/useConnection";
import type { ConnectionStatusName, TransportFactory } from "./features/connection/connectionClient";
import { webSocketTransportFactory } from "./features/connection/wsTransport";
import type { ConnectionViewModel } from "./features/connection/useConnection";
import { FlowWorkspace } from "./features/flows/FlowWorkspace";
import { formatBytes } from "./format";
import "./styles/shell.css";

const statusCopy: Record<ConnectionStatusName, { label: string; detail: string }> = {
  disconnected: { label: "Disconnected", detail: "No source is attached" },
  connecting: { label: "Connecting", detail: "Opening local source" },
  live: { label: "Live", detail: "Receiving source events" },
  reconnecting: { label: "Reconnecting", detail: "Retrying the local source" },
  stale: { label: "Stale", detail: "No events in the last 15 seconds" },
  error: { label: "Source error", detail: "The source needs attention" },
};

export { formatBytes } from "./format";

function formatCursor(value: string): string {
  return value.length > 9 ? `${value.slice(0, 3)}…${value.slice(-4)}` : value;
}

/** Upper bound on remembered seen flow ids. Older entries evict FIFO. */
export const SEEN_FLOW_LIMIT = 1024;

export interface AppProps {
  transportFactory?: TransportFactory;
}

export function App({ transportFactory }: AppProps = {}) {
  // The production default speaks to the local B3 API over its WebSocket
  // stream; tests and previews inject their own transports.
  return <Workbench view={useConnection(transportFactory ?? webSocketTransportFactory())} />;
}

export function Workbench({ view }: { view: ConnectionViewModel }) {
  const {
    browser, latestBrowser, status, followLive, pauseLive, resumeLive, connect, disconnect, retry, requestResync,
  } = view;
  const titleId = useId();
  const [resyncMessage, setResyncMessage] = useState<string | null>(null);
  // seenFlowIds lives on the workbench so a reconnect — which unmounts
  // FlowWorkspace via the empty-state branch when displayed flows drop
  // to zero in follow-live mode — does not throw away the record of
  // which rows the user has already opened. Bounded by SEEN_FLOW_LIMIT
  // with FIFO eviction on the insertion-ordered Set so the working set
  // stays bounded without discarding entries on transient reconnects.
  const [seenFlowIds, setSeenFlowIds] = useState<ReadonlySet<string>>(() => new Set());
  const markFlowSeen = (flowId: string) => {
    setSeenFlowIds((previous) => {
      if (previous.has(flowId)) return previous;
      const next = new Set(previous);
      next.add(flowId);
      while (next.size > SEEN_FLOW_LIMIT) {
        const oldest = next.values().next().value;
        if (oldest === undefined) break;
        next.delete(oldest);
      }
      return next;
    });
  };
  const statusInfo = statusCopy[status.state];
  const isBusy = status.state === "connecting" || status.state === "reconnecting";
  const isRetrying = status.state === "reconnecting";
  const isConnecting = status.state === "connecting";
  const isConnected = status.state === "live" || status.state === "stale";
  const hasError = status.state === "error";
  const retainedCount = browser.flows.size;
  const currentResyncGap = latestBrowser.gap?.sourceEpoch === latestBrowser.sourceEpoch ? latestBrowser.gap : null;
  const resyncAvailable = status.requestResyncAvailable && currentResyncGap !== null;
  const action = isConnected || isBusy ? disconnect : hasError ? retry : connect;
  const memoryBudget = browser.sourceLimits?.max_in_memory_bytes ?? "134217728";
  const displayedCursor = browser.cursor;
  const followLabel = !followLive
    ? "Live paused"
    : status.state === "live"
      ? "Following live"
      : `Follow when ${statusInfo.label.toLowerCase()}`;
  const toggleFollowLive = followLive ? pauseLive : resumeLive;
  const handleResync = () => {
    if (!resyncAvailable) {
      setResyncMessage("No current source gap is available.");
      return;
    }
    const result = requestResync(latestBrowser.sourceEpoch);
    setResyncMessage(result.ok ? null : result.reason === "unsupported" ? "This source cannot request snapshots." : result.reason === "stale-source" ? "The source changed; follow live to resync." : "Snapshot request could not be sent.");
  };

  return (
    <main className="app-shell">
      <a className="skip-link" href={`#${titleId}`}>Skip to workspace</a>
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">⌁</span>
          <div>
            <p className="eyebrow">LOCAL TRAFFIC WORKBENCH</p>
            <h1>mitm-inspector</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <div className={`status-pill status-${status.state}`} role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" />
            <span>{statusInfo.label}</span>
          </div>
          <button className="connect-button" onClick={action}>
            {isRetrying ? "Stop reconnecting" : isConnecting ? "Cancel connection" : isConnected ? "Disconnect" : hasError ? "Try again" : "Connect source"}
          </button>
        </div>
      </header>

      <section className="workspace-frame" aria-labelledby={titleId}>
        <aside className="rail" aria-label="Workspace navigation">
          <div className="rail-label">WORKSPACE</div>
          <a className="rail-item is-active" href={`#${titleId}`} aria-current="page"><span aria-hidden="true">◈</span> Live flows</a>
          <button className="rail-item" disabled><span aria-hidden="true">≡</span> Saved views <small>soon</small></button>
          <div className="rail-divider" />
          <div className="rail-label">SOURCE</div>
          <div className={`source-card source-${status.state}`}>
            <span className="source-icon" aria-hidden="true">mitm</span>
            <div>
              <strong>{browser.sourceId ?? "local proxy"}</strong>
              <small>{statusInfo.detail}</small>
            </div>
          </div>
          {(hasError || isRetrying) && status.error && <p className="source-error">Last failure: {status.error}</p>}
          <div className="rail-divider" />
          <div className="rail-label">RETENTION</div>
          <dl className="rail-metrics">
            <div><dt>kept in memory</dt><dd>{retainedCount}</dd></div>
            <div><dt>body budget</dt><dd>{formatBytes(memoryBudget)}</dd></div>
            <div><dt>cursor</dt><dd>{formatCursor(browser.cursor)}</dd></div>
          </dl>
        </aside>

        <section className="content-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">CAPTURE SURFACE / 01</p>
              <h2 id={titleId} tabIndex={-1}>Live flows</h2>
            </div>
            <div className="heading-meta" aria-label={`${retainedCount} retained flows`}>
              <span className="metric-value">{retainedCount}</span><span>retained</span>
            </div>
          </div>

          <div className="signal-strip" aria-label={`Stream cursor ${displayedCursor}, status ${statusInfo.label}`}>
            <div className={`cursor-beam cursor-${status.state}`} aria-hidden="true">
              <span className="beam-track" /><span className="beam-pulse" />
            </div>
            <div className="signal-copy">
              <span className="signal-label">SOURCE SIGNAL</span>
              <strong>{statusInfo.label}</strong>
              <span>{followLive ? `cursor ${displayedCursor}` : `paused at cursor ${displayedCursor}`}</span>
            </div>
            <div className="signal-stat"><span>messages</span><strong>{browser.counters.receivedMessages}</strong></div>
            <div className="signal-stat"><span>changes</span><strong>{browser.counters.appliedChanges}</strong></div>
            <div className="signal-stat"><span>dropped</span><strong>{browser.counters.droppedMessages}</strong></div>
            <button className={`follow-control ${followLive ? "is-following" : "is-paused"}`} aria-pressed={followLive} onClick={toggleFollowLive}>
              <span aria-hidden="true">{followLive && status.state === "live" ? "↓" : "Ⅱ"}</span>{followLabel}
            </button>
          </div>

          {browser.gap ? (
            <div className="notice notice-warning" role="alert">
              <span className="notice-mark" aria-hidden="true">!</span>
              <div><strong>Stream gap at cursor {browser.gap.received}</strong><p>Waiting for a fresh snapshot from the source. History remains visible until it arrives.</p></div>
              <div className="notice-actions">
                <button className="notice-action" onClick={handleResync} disabled={!resyncAvailable}>
                  {resyncAvailable ? "Request snapshot" : "Snapshot request unavailable"}
                </button>
                {resyncMessage && <span className="notice-result" role="status">{resyncMessage}</span>}
              </div>
            </div>
          ) : isRetrying ? (
            <div className="empty-state" role="status">
              <div className="empty-glyph" aria-hidden="true">↻</div>
              <p className="eyebrow">RETRYING SOURCE / ATTEMPT {status.attempt}</p>
              <h3>Reconnecting to local capture source</h3>
              <p>{status.error ? `Last failure: ${status.error}` : "The source closed. Retrying with backoff."}</p>
              <button className="empty-action" onClick={disconnect}>Stop reconnecting</button>
            </div>
          ) : isConnecting ? (
            <div className="empty-state" role="status">
              <div className="empty-glyph" aria-hidden="true">…</div>
              <p className="eyebrow">OPENING SOURCE</p>
              <h3>Connecting to local capture source</h3>
              <p>{status.error ? `Last failure: ${status.error}` : "Opening the read-only event stream."}</p>
              <button className="empty-action" onClick={disconnect}>Cancel connection</button>
            </div>
          ) : hasError ? (
            <div className="empty-state" role="status">
              <div className="empty-glyph" aria-hidden="true">×</div>
              <p className="eyebrow">SOURCE DID NOT OPEN</p>
              <h3>Connect a local capture source</h3>
              <p>{status.error ?? "The source adapter is unavailable."}</p>
              <button className="empty-action" onClick={retry}>Try the source again</button>
            </div>
          ) : retainedCount === 0 ? (
            <div className="empty-state" role="status">
              <div className="empty-glyph" aria-hidden="true">↯</div>
              <p className="eyebrow">NO CAPTURED FLOWS</p>
              <h3>{isConnected ? "Source connected, waiting for traffic" : "Waiting for a local source"}</h3>
              <p>{isConnected ? "New sanitized flow metadata will appear here as the proxy observes it." : "Connect a local mitmproxy source to begin a read-only session."}</p>
              <div className="empty-hint"><kbd>⌘</kbd><span>Flow search and inspection arrive with the workspace.</span></div>
            </div>
          ) : (
            <FlowWorkspace browser={browser} followLive={followLive} pauseLive={pauseLive} seenFlowIds={seenFlowIds} onFlowSeen={markFlowSeen} />
          )}
        </section>
      </section>

      <footer className="statusbar"><span>READ-ONLY MVP</span><span className="statusbar-separator">/</span><span>memory only</span><span className="statusbar-spacer" /><span>protocol v1</span><span className="statusbar-separator">/</span><span>loopback target</span></footer>
    </main>
  );
}
