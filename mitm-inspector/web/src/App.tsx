/* eslint-disable no-unused-vars */

import { useEffect, useRef, useState } from "react";

import { useConnection } from "./features/connection/useConnection";
import type { TransportFactory } from "./features/connection/connectionClient";
import { webSocketTransportFactory } from "./features/connection/wsTransport";
import { PacketList } from "./features/flows/PacketList";
import type { FlowDetailLoader } from "./features/inspector/flowDetail";
import type { SearchFetcher } from "./features/search/search";
import { SessionDetail } from "./features/sessions/SessionDetail";
import { SessionList } from "./features/sessions/SessionList";
import { createSessionIndex } from "./features/sessions/sessionSummary";
import type { SessionIndex, SessionKey } from "./features/sessions/sessionSummary";
import type { BrowserState } from "./state/browserState";
import "./styles/shell.css";

type WorkspaceView =
  | { kind: "sessions" }
  | { kind: "session"; sessionKey: SessionKey }
  | { kind: "flows" };

export interface WorkspaceProps {
  browser: BrowserState;
  loadFlowDetail?: FlowDetailLoader;
  searchFetcher?: SearchFetcher;
  onSearchActiveChange?: (active: boolean) => void;
}

/**
 * Session-first shell: home is one row per session; drill-in shows that
 * session's conversation; the raw packet grid stays one switch away so
 * non-Anthropic and debugging flows lose nothing.
 */
export function Workspace({ browser, loadFlowDetail, searchFetcher, onSearchActiveChange }: WorkspaceProps) {
  const [view, setView] = useState<WorkspaceView>({ kind: "sessions" });
  // Stateful incremental index: unchanged sessions keep their summary
  // objects across deltas; only sessions whose flows changed resummarize.
  const indexRef = useRef<SessionIndex | null>(null);
  if (indexRef.current === null) indexRef.current = createSessionIndex();
  const sessions = indexRef.current.update(browser.flows.entries);

  let body;
  if (view.kind === "flows") {
    body = (
      <PacketList
        browser={browser}
        loadFlowDetail={loadFlowDetail}
        searchFetcher={searchFetcher}
        onSearchActiveChange={onSearchActiveChange}
      />
    );
  } else if (view.kind === "session") {
    // Session summaries derive from retained flows, so a fully pruned
    // session simply stops resolving.
    const summary = sessions.find((session) => session.key === view.sessionKey);
    body = summary === undefined ? (
      <div>
        <div className="session-detail-bar">
          <button type="button" className="session-back" onClick={() => setView({ kind: "sessions" })}>← sessions</button>
        </div>
        <p className="packet-empty">session no longer retained</p>
      </div>
    ) : (
      <SessionDetail
        summary={summary}
        onBack={() => setView({ kind: "sessions" })}
        loadFlowDetail={loadFlowDetail}
      />
    );
  } else {
    body = <SessionList sessions={sessions} onOpen={(sessionKey) => setView({ kind: "session", sessionKey })} />;
  }

  return (
    <div className="workspace">
      <nav className="app-nav" aria-label="Views">
        <button
          type="button"
          className="packet-mode"
          aria-pressed={view.kind !== "flows"}
          onClick={() => setView({ kind: "sessions" })}
        >sessions</button>
        <button
          type="button"
          className="packet-mode"
          aria-pressed={view.kind === "flows"}
          onClick={() => setView({ kind: "flows" })}
        >flows</button>
      </nav>
      {body}
    </div>
  );
}

export interface AppProps {
  transportFactory?: TransportFactory;
  loadFlowDetail?: FlowDetailLoader;
  searchFetcher?: SearchFetcher;
}

export function App({ transportFactory, loadFlowDetail, searchFetcher }: AppProps = {}) {
  const view = useConnection(transportFactory ?? webSocketTransportFactory());
  const [searchActive, setSearchActive] = useState(false);
  // Freeze the displayed flow list while a search is being read so live
  // traffic cannot reshuffle the results out from under the user.
  useEffect(() => {
    if (searchActive) view.pauseLive();
    else view.resumeLive();
  }, [searchActive]);
  return (
    <main className="app-shell">
      <Workspace
        browser={view.browser}
        loadFlowDetail={loadFlowDetail}
        searchFetcher={searchFetcher}
        onSearchActiveChange={setSearchActive}
      />
    </main>
  );
}
