/* eslint-disable no-unused-vars */

import { PLACEHOLDER, durationBetween, formatClockTime, shortModel } from "../flows/rowSummary";
import type { SessionKey, SessionSummary } from "./sessionSummary";

export interface SessionListProps {
  sessions: readonly SessionSummary[];
  onOpen: (key: SessionKey) => void;
}

export function sessionLabel(key: SessionKey): string {
  return key === null ? "unassigned" : key.slice(0, 8);
}

export function SessionList({ sessions, onOpen }: SessionListProps) {
  if (sessions.length === 0) {
    return <p className="packet-empty">no sessions captured</p>;
  }
  return (
    <ol className="session-list" aria-label="Captured sessions">
      {sessions.map((session) => {
        const models = session.models.map(shortModel).join(" · ");
        return (
          <li key={session.key === null ? "unassigned" : `s:${session.key}`} className="session-item">
            <button type="button" className="session-row" onClick={() => onOpen(session.key)}>
              <span className="session-time">{formatClockTime(session.startedAt)}</span>
              <span className={`session-id${session.key === null ? " session-id-unassigned" : ""}`}>
                {sessionLabel(session.key)}
              </span>
              <span className="session-count">{session.flowCount}f</span>
              <span className="session-models">{models.length > 0 ? models : PLACEHOLDER}</span>
              <span className={`session-query${session.firstQuery === undefined ? " session-query-none" : ""}`}>
                {session.firstQuery === undefined ? PLACEHOLDER : `“${session.firstQuery}”`}
              </span>
              <span className="session-last">{formatClockTime(session.lastActivity)}</span>
              <span className="session-duration">{durationBetween(session.startedAt, session.lastActivity)}</span>
              <span className={`session-status${session.hasError ? " session-status-error" : ""}`}>
                {session.hasError ? "err" : ""}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
