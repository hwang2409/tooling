/* eslint-disable no-unused-vars */

import { useEffect, useState } from "react";

import { parseProtocolMessage } from "../../protocol";
import type { ImmutableFlowMetadata } from "../../state/browserState";
import type { SessionKey, SessionSummary } from "./sessionSummary";

export const SESSIONS_PATH = "/api/v1/sessions";

export type SessionFetcher = (signal: AbortSignal) => Promise<readonly SessionSummary[]>;
export type SessionDetailFetcher = (key: SessionKey, signal: AbortSignal) => Promise<readonly ImmutableFlowMetadata[]>;

function asSession(value: unknown): SessionSummary | null {
  if (typeof value !== "object" || value === null) return null;
  const raw = value as Record<string, unknown>;
  if (raw.session_id !== null && typeof raw.session_id !== "string") return null;
  if (typeof raw.flow_count !== "number" || !Number.isSafeInteger(raw.flow_count)) return null;
  if (!Array.isArray(raw.models) || !raw.models.every((model) => typeof model === "string")) return null;
  return {
    key: raw.session_id as SessionKey,
    flows: [],
    flowCount: raw.flow_count,
    ...(typeof raw.first_query === "string" ? { firstQuery: raw.first_query } : {}),
    ...(typeof raw.started_at === "string" ? { startedAt: raw.started_at } : {}),
    ...(typeof raw.last_activity === "string" ? { lastActivity: raw.last_activity } : {}),
    models: raw.models,
    hasError: raw.has_error === true,
  };
}

export const fetchSessions: SessionFetcher = async (signal) => {
  const response = await fetch(SESSIONS_PATH, { signal });
  if (!response.ok) throw new Error(`Session list request failed (${response.status}).`);
  const payload = await response.json() as { sessions?: unknown };
  if (!Array.isArray(payload.sessions)) throw new Error("Session list response was malformed.");
  return payload.sessions.map(asSession).filter((session): session is SessionSummary => session !== null);
};

function parseFlow(value: unknown): ImmutableFlowMetadata | null {
  try {
    const parsed = parseProtocolMessage({ protocol_version: "1", type: "flow.metadata", metadata: value });
    if (parsed.kind !== "known" || parsed.message.type !== "flow.metadata") return null;
    return parsed.message.metadata as unknown as ImmutableFlowMetadata;
  } catch {
    return null;
  }
}

export const fetchSessionDetail: SessionDetailFetcher = async (key, signal) => {
  const id = key === null ? "unassigned" : key;
  const response = await fetch(`${SESSIONS_PATH}/${encodeURIComponent(id)}`, { signal });
  if (response.status === 404) return [];
  if (!response.ok) throw new Error(`Session detail request failed (${response.status}).`);
  const payload = await response.json() as { flows?: unknown };
  if (!Array.isArray(payload.flows)) throw new Error("Session detail response was malformed.");
  return payload.flows.map(parseFlow).filter((flow): flow is ImmutableFlowMetadata => flow !== null);
};

export function useSessions(fetcher: SessionFetcher = fetchSessions): {
  readonly sessions: readonly SessionSummary[];
  readonly loading: boolean;
  readonly error: string | null;
} {
  const [sessions, setSessions] = useState<readonly SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetcher(controller.signal)
      .then((value) => { if (!controller.signal.aborted) { setSessions(value); setError(null); setLoading(false); } })
      .catch((reason: unknown) => { if (!controller.signal.aborted) { setError(reason instanceof Error ? reason.message : "Session list request failed."); setLoading(false); } });
    return () => controller.abort();
  }, [fetcher]);
  return { sessions, loading, error };
}

export function useSessionDetail(
  key: SessionKey | undefined,
  fetcher: SessionDetailFetcher = fetchSessionDetail,
): { readonly flows: readonly ImmutableFlowMetadata[] | null; readonly loading: boolean; readonly error: string | null } {
  const [flows, setFlows] = useState<readonly ImmutableFlowMetadata[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (key === undefined) {
      setFlows(null);
      setLoading(false);
      setError(null);
      return;
    }
    const controller = new AbortController();
    setFlows(null);
    setLoading(true);
    setError(null);
    fetcher(key, controller.signal)
      .then((value) => { if (!controller.signal.aborted) { setFlows(value); setLoading(false); } })
      .catch((reason: unknown) => { if (!controller.signal.aborted) { setError(reason instanceof Error ? reason.message : "Session detail request failed."); setLoading(false); } });
    return () => controller.abort();
  }, [fetcher, key]);
  return { flows, loading, error };
}
