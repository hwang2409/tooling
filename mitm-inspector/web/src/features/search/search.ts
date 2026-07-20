/* eslint-disable no-unused-vars */

import { useEffect, useRef, useState } from "react";

import type { FlowMetadata } from "../../protocol";
import { parseProtocolMessage } from "../../protocol";

export const SEARCH_PATH = "/api/v1/search";
export const SEARCH_MIN_QUERY_LENGTH = 2;
export const SEARCH_DEBOUNCE_MS = 250;
export const SEARCH_RESULT_LIMIT = 200;

export interface SearchMatch {
  readonly flow_id: string;
  readonly field: "request_body" | "response_body";
  readonly snippet: string;
  /** Optional durable metadata, added by the retention-backed search API. */
  readonly flow?: FlowMetadata;
}

export type SearchState =
  | { readonly status: "idle" }
  | { readonly status: "loading"; readonly query: string }
  | { readonly status: "results"; readonly query: string; readonly matches: readonly SearchMatch[]; readonly truncated: boolean }
  | { readonly status: "empty"; readonly query: string }
  | { readonly status: "unavailable"; readonly query: string };

export type SearchFetcher = (query: string, signal: AbortSignal) => Promise<SearchState>;

function parseMatches(payload: unknown): { matches: SearchMatch[]; truncated: boolean } | null {
  if (typeof payload !== "object" || payload === null) return null;
  const rawMatches = (payload as { matches?: unknown }).matches;
  if (!Array.isArray(rawMatches)) return null;
  const matches: SearchMatch[] = [];
  for (const raw of rawMatches) {
    if (typeof raw !== "object" || raw === null) continue;
    const match = raw as { flow_id?: unknown; field?: unknown; snippet?: unknown; flow?: unknown };
    if (typeof match.flow_id !== "string" || match.flow_id.length === 0) continue;
    if (match.field !== "request_body" && match.field !== "response_body") continue;
    if (typeof match.snippet !== "string") continue;
    const parsedFlow = match.flow === undefined ? undefined : parseSearchFlow(match.flow);
    if (parsedFlow !== undefined && parsedFlow.flow_id !== match.flow_id) continue;
    matches.push({
      flow_id: match.flow_id,
      field: match.field,
      snippet: match.snippet,
      ...(parsedFlow === undefined ? {} : { flow: parsedFlow }),
    });
  }
  return { matches, truncated: (payload as { truncated?: unknown }).truncated === true };
}

/** Validate optional durable metadata with the same boundary as stream flows. */
function parseSearchFlow(value: unknown): FlowMetadata | undefined {
  try {
    const parsed = parseProtocolMessage({ protocol_version: "1", type: "flow.metadata", metadata: value });
    if (parsed.kind !== "known" || parsed.message.type !== "flow.metadata") return undefined;
    return parsed.message.metadata as unknown as FlowMetadata;
  } catch {
    return undefined;
  }
}

/**
 * Query the local search endpoint. The endpoint may not exist yet (backend
 * unmerged) — every failure mode maps onto the quiet "unavailable" state so
 * typing in the box can never crash the inspector.
 */
export const fetchSearch: SearchFetcher = async (query, signal) => {
  let response: Response;
  try {
    response = await globalThis.fetch(
      `${SEARCH_PATH}?q=${encodeURIComponent(query)}&limit=${SEARCH_RESULT_LIMIT}`,
      { signal },
    );
  } catch (error) {
    if (signal.aborted) throw error;
    return { status: "unavailable", query };
  }
  if (!response.ok) return { status: "unavailable", query };
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return { status: "unavailable", query };
  }
  const parsed = parseMatches(payload);
  if (parsed === null) return { status: "unavailable", query };
  if (parsed.matches.length === 0) return { status: "empty", query };
  return { status: "results", query, matches: parsed.matches, truncated: parsed.truncated };
};

export interface SearchController {
  readonly query: string;
  readonly state: SearchState;
  readonly setQuery: (value: string) => void;
  readonly clear: () => void;
}

/** Debounced full-text search over captured bodies. */
export function useSearch(fetcher: SearchFetcher = fetchSearch): SearchController {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<SearchState>({ status: "idle" });
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    const trimmed = query.trim();
    if (trimmed.length < SEARCH_MIN_QUERY_LENGTH) {
      setState({ status: "idle" });
      return;
    }
    const current = generation.current;
    const controller = new AbortController();
    setState({ status: "loading", query: trimmed });
    const timer = globalThis.setTimeout(() => {
      fetcher(trimmed, controller.signal)
        .then((result) => {
          if (generation.current === current) setState(result);
        })
        .catch(() => {
          if (generation.current === current && !controller.signal.aborted) {
            setState({ status: "unavailable", query: trimmed });
          }
        });
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      globalThis.clearTimeout(timer);
      controller.abort();
    };
  }, [query, fetcher]);

  return {
    query,
    state,
    setQuery,
    clear: () => setQuery(""),
  };
}
