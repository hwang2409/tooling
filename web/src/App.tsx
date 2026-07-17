import { useState } from "react";

type ConnectionState = "offline" | "ready";

export function App() {
  const [connection, setConnection] = useState<ConnectionState>("offline");
  const isReady = connection === "ready";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">⌁</span>
          <div>
            <p className="eyebrow">LOCAL TRAFFIC WORKBENCH</p>
            <h1>mitm-inspector</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <span className={`status-pill ${isReady ? "is-ready" : ""}`}>
            <span className="status-dot" />
            {isReady ? "source ready" : "awaiting source"}
          </span>
          <button className="connect-button" onClick={() => setConnection(isReady ? "offline" : "ready")}>
            {isReady ? "Disconnect" : "Connect source"}
          </button>
        </div>
      </header>

      <section className="workspace-frame" aria-label="Traffic workspace">
        <aside className="rail">
          <div className="rail-label">WORKSPACE</div>
          <button className="rail-item is-active"><span>◈</span> Live flows</button>
          <button className="rail-item" disabled><span>≡</span> Saved views <small>soon</small></button>
          <div className="rail-divider" />
          <div className="rail-label">SOURCE</div>
          <div className="source-card">
            <span className="source-icon">mitm</span>
            <div><strong>local proxy</strong><small>{isReady ? "protocol v1" : "not connected"}</small></div>
          </div>
        </aside>

        <section className="content-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">FLOW STREAM</p>
              <h2>Live flows</h2>
            </div>
            <div className="heading-meta"><span className="metric-value">0</span><span>captured</span></div>
          </div>
          <div className="empty-state">
            <div className="empty-glyph" aria-hidden="true">↯</div>
            <h3>{isReady ? "Source connected" : "Waiting for captured flows"}</h3>
            <p>{isReady ? "The read-only workspace is ready for protocol-v1 events." : "Connect a local mitmproxy source to begin a session."}</p>
            <div className="empty-hint"><kbd>⌘</kbd><span>Search and filter arrive with the flow workspace.</span></div>
          </div>
        </section>
      </section>

      <footer className="statusbar"><span>READ-ONLY MVP</span><span className="statusbar-separator">/</span><span>memory only</span><span className="statusbar-spacer" /><span>protocol v1</span><span className="statusbar-separator">/</span><span>loopback target</span></footer>
    </main>
  );
}
