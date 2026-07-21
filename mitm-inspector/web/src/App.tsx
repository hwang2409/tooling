/* eslint-disable no-unused-vars */

import { useRef, useState } from "react";

import { PacketList } from "./features/flows/PacketList";
import type { FlowDetailLoader } from "./features/inspector/flowDetail";
import type { SearchFetcher } from "./features/search/search";
import { SessionDetail } from "./features/sessions/SessionDetail";
import type { CandidateParser } from "./features/sessions/SessionDetail";
import type { CanonicalIndex } from "./features/sessions/canonical";
import { SessionList } from "./features/sessions/SessionList";
import { useSessions } from "./features/sessions/sessionApi";
import type { SessionFetcher, SessionDetailFetcher } from "./features/sessions/sessionApi";
import { useSessionDetail } from "./features/sessions/sessionApi";
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
  sessionIndex?: SessionIndex;
  /** Instrumented drill-in instances for tests; SessionDetail defaults its own. */
  candidateParser?: CandidateParser;
  canonicalIndex?: CanonicalIndex;
}

/**
 * Session-first shell: home is one row per session; drill-in shows that
 * session's conversation; the raw packet grid stays one switch away so
 * non-Anthropic and debugging flows lose nothing.
 */
export function Workspace({ browser, loadFlowDetail, searchFetcher, onSearchActiveChange, sessionIndex, candidateParser, canonicalIndex }: WorkspaceProps) {
  const [view, setView] = useState<WorkspaceView>({ kind: "sessions" });
  // One stateful index for the component's lifetime: consecutive deltas
  // update only the sessions owning the changed flow ids (see
  // createSessionIndex); a fresh index here would degrade every delta to a
  // full rebuild.
  const indexRef = useRef<SessionIndex | null>(null);
  if (indexRef.current === null) indexRef.current = sessionIndex ?? createSessionIndex();
  const sessions = indexRef.current.update(browser.flows, browser.flowsUpdate);

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
        sourceEpoch={browser.sourceEpoch}
        candidateParser={candidateParser}
        canonicalIndex={canonicalIndex}
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
  loadFlowDetail?: FlowDetailLoader;
  searchFetcher?: SearchFetcher;
  sessionFetcher?: SessionFetcher;
  sessionDetailFetcher?: SessionDetailFetcher;
}

function HttpWorkspace({ loadFlowDetail, sessionFetcher, sessionDetailFetcher }: Pick<AppProps, "loadFlowDetail" | "sessionFetcher" | "sessionDetailFetcher">) {
  const { sessions, loading, error } = useSessions(sessionFetcher);
  const [selected, setSelected] = useState<SessionKey | null | undefined>(undefined);
  const selectedSummary = selected === undefined ? undefined : sessions.find((session) => session.key === selected);
  const detail = useSessionDetail(selectedSummary?.key, sessionDetailFetcher);
  if (selectedSummary !== undefined) {
    const summary = { ...selectedSummary, flows: detail.flows ?? [] };
    return (
      <SessionDetail
        summary={summary}
        onBack={() => setSelected(undefined)}
        loadFlowDetail={loadFlowDetail}
      />
    );
  }
  if (loading) return <p className="packet-empty">loading sessions…</p>;
  if (error !== null) return <p className="packet-empty">{error}</p>;
  return <SessionList sessions={sessions} onOpen={setSelected} />;
}

export function App({ loadFlowDetail, sessionFetcher, sessionDetailFetcher }: AppProps = {}) {
  return (
    <main className="app-shell">
      <HttpWorkspace loadFlowDetail={loadFlowDetail} sessionFetcher={sessionFetcher} sessionDetailFetcher={sessionDetailFetcher} />
    </main>
  );
}
