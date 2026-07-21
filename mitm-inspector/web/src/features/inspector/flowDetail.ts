/* eslint-disable no-unused-vars */

import { useEffect, useMemo, useRef, useState } from "react";

import { parseProtocolMessage } from "../../protocol";
import type { BodyDescriptor, FlowMetadata } from "../../protocol";
import type { DeepReadonly } from "../../immutable";
import type { InspectableBody } from "./models";

export const FLOW_DETAIL_PATH_PREFIX = "/api/v1/flows/";

export interface FlowDetailOverrides {
  request_body?: InspectableBody;
  response_body?: InspectableBody;
}

export type FlowDetailResult =
  | { status: "loaded"; overrides: FlowDetailOverrides }
  | { status: "unavailable" }
  | { status: "error"; error: string };

export type FlowDetailLoader = (flowId: string, signal: AbortSignal) => Promise<FlowDetailResult>;

interface BodyEndLike {
  body_side: "request" | "response";
  body: BodyDescriptor;
}

/** Project the retained per-flow messages onto inspector body overrides. */
export function overridesFromDetailMessages(messages: readonly unknown[]): FlowDetailOverrides {
  let metadata: DeepReadonly<FlowMetadata> | null = null;
  let requestEnd: InspectableBody | undefined;
  let responseEnd: InspectableBody | undefined;
  for (const value of messages) {
    let envelope;
    try {
      envelope = parseProtocolMessage(value);
    } catch {
      continue;
    }
    if (envelope.kind !== "known") continue;
    const message = envelope.message as { type?: unknown };
    if (message.type === "flow.metadata") {
      metadata = (envelope.message as unknown as { metadata: DeepReadonly<FlowMetadata> }).metadata;
    } else if (message.type === "body.end") {
      const bodyEnd = envelope.message as unknown as BodyEndLike;
      if (bodyEnd.body_side === "request") requestEnd = bodyEnd.body;
      else responseEnd = bodyEnd.body;
    }
  }
  const overrides: FlowDetailOverrides = {};
  if (metadata !== null) {
    overrides.request_body = metadata.request_body;
    if (metadata.response_body !== undefined) overrides.response_body = metadata.response_body;
  }
  // body.end carries the terminal descriptor even when a later metadata
  // message was evicted; prefer it when present.
  if (requestEnd !== undefined) overrides.request_body = requestEnd;
  if (responseEnd !== undefined) overrides.response_body = responseEnd;
  return overrides;
}

/** Fetch one selected flow's full retained detail from the local API. */
export const fetchFlowDetail: FlowDetailLoader = async (flowId, signal) => {
  let response: Response;
  try {
    response = await globalThis.fetch(`${FLOW_DETAIL_PATH_PREFIX}${encodeURIComponent(flowId)}`, { signal });
  } catch (error) {
    return { status: "error", error: error instanceof Error ? error.message : "Flow detail request failed." };
  }
  if (response.status === 404) return { status: "unavailable" };
  if (!response.ok) return { status: "error", error: `Flow detail request failed (${response.status}).` };
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return { status: "error", error: "Flow detail response was not JSON." };
  }
  const messages = (payload as { messages?: unknown })?.messages;
  if (!Array.isArray(messages)) return { status: "error", error: "Flow detail response was malformed." };
  return { status: "loaded", overrides: overridesFromDetailMessages(messages) };
};

/**
 * Version marker for a flow's observable detail state. The detail cache keys
 * on this alongside the flow id, so an in-flight flow refetches once its
 * terminal fields (status, end time, body descriptors) change — while deltas
 * touching OTHER flows never invalidate it.
 */
export function flowDetailVersion(metadata: DeepReadonly<FlowMetadata>): string {
  const body = (descriptor: DeepReadonly<BodyDescriptor> | undefined): string => {
    if (descriptor === undefined) return "-";
    const size = "size_bytes" in descriptor ? descriptor.size_bytes : "0";
    const captured = "captured_bytes" in descriptor ? descriptor.captured_bytes : "";
    return `${descriptor.state}:${size}:${captured}`;
  };
  const endedAt = (metadata as { ended_at?: unknown }).ended_at;
  return [
    metadata.response_status ?? "",
    typeof endedAt === "string" ? endedAt : "",
    body(metadata.request_body),
    body(metadata.response_body),
  ].join("|");
}

/**
 * Load body detail for the selected flow; falls back to grid metadata on
 * failure. `sourceEpoch` MUST identify the capture incarnation: a new source
 * can legally reuse a flow id (and even its body descriptors), and serving
 * the previous source's cached body would present another capture session's
 * data as current.
 */
export function useFlowDetail(
  flowId: string | null,
  loader: FlowDetailLoader = fetchFlowDetail,
  version?: string,
  sourceEpoch?: number,
): FlowDetailResult | null {
  const [result, setResult] = useState<FlowDetailResult | null>(null);
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const requestKey = flowId === null ? null : `${flowId}\u0000${version ?? ""}\u0000${sourceEpoch ?? ""}`;
  useEffect(() => {
    if (requestKey === null || flowId === null) {
      setResult(null);
      setLoadedFor(null);
      return;
    }
    const controller = new AbortController();
    let active = true;
    setResult(null);
    setLoadedFor(null);
    loader(flowId, controller.signal)
      .then((value) => {
        if (!active) return;
        setResult(value);
        setLoadedFor(requestKey);
      })
      .catch((error: unknown) => {
        if (!active || controller.signal.aborted) return;
        setResult({ status: "error", error: error instanceof Error ? error.message : "Flow detail request failed." });
        setLoadedFor(requestKey);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [flowId, loader, requestKey]);
  return requestKey !== null && loadedFor === requestKey ? result : null;
}

interface VersionedResult {
  readonly version: string;
  readonly result: FlowDetailResult;
}

/**
 * Load detail for a set of flows concurrently (session drill-in). Results
 * key on flow id + detail version: a flow that completes after being opened
 * refetches, results for flows that leave the set are dropped, and a stale
 * version is never surfaced as current.
 */
export function useFlowDetails(
  flows: readonly DeepReadonly<FlowMetadata>[],
  loader: FlowDetailLoader = fetchFlowDetail,
  sourceEpoch?: number,
): ReadonlyMap<string, FlowDetailResult> {
  // The epoch is part of every cached entry's identity: a reconnected source
  // reusing a flow id/version must never surface the old source's body.
  const requests = flows.map((flow) => ({
    flowId: flow.flow_id,
    version: `${sourceEpoch ?? 0}:${flowDetailVersion(flow)}`,
  }));
  const key = requests.map((request) => `${request.flowId}@${request.version}`).join("\u0000");
  const requestsRef = useRef(requests);
  requestsRef.current = requests;
  const resultsRef = useRef(new Map<string, VersionedResult>());
  const inFlightRef = useRef(new Map<string, { version: string; controller: AbortController }>());
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const wanted = requestsRef.current;
    const wantedVersions = new Map(wanted.map((request) => [request.flowId, request.version]));
    for (const [flowId, entry] of resultsRef.current) {
      if (wantedVersions.get(flowId) !== entry.version) resultsRef.current.delete(flowId);
    }
    for (const [flowId, request] of inFlightRef.current) {
      if (wantedVersions.get(flowId) === request.version) continue;
      request.controller.abort();
      inFlightRef.current.delete(flowId);
    }
    const record = (flowId: string, version: string, result: FlowDetailResult) => {
      if (inFlightRef.current.get(flowId)?.version !== version) return;
      inFlightRef.current.delete(flowId);
      if (requestsRef.current.find((request) => request.flowId === flowId)?.version !== version) return;
      resultsRef.current.set(flowId, { version, result });
      setGeneration((value) => value + 1);
    };
    for (const { flowId, version } of wanted) {
      const existing = resultsRef.current.get(flowId);
      if (existing !== undefined && existing.version === version) continue;
      const inFlight = inFlightRef.current.get(flowId);
      if (inFlight !== undefined && inFlight.version === version) continue;
      const controller = new AbortController();
      inFlightRef.current.set(flowId, { version, controller });
      loader(flowId, controller.signal)
        .then((result) => record(flowId, version, result))
        .catch((error: unknown) => {
          if (controller.signal.aborted) {
            if (inFlightRef.current.get(flowId)?.version === version) inFlightRef.current.delete(flowId);
            return;
          }
          record(flowId, version, {
            status: "error",
            error: error instanceof Error ? error.message : "Flow detail request failed.",
          });
        });
    }
    return () => {
      const latestVersions = new Map(requestsRef.current.map((request) => [request.flowId, request.version]));
      for (const [flowId, request] of inFlightRef.current) {
        if (latestVersions.get(flowId) === request.version) continue;
        request.controller.abort();
        inFlightRef.current.delete(flowId);
      }
    };
  }, [key, loader]);

  useEffect(() => () => {
    for (const [flowId, request] of inFlightRef.current) {
      request.controller.abort();
      inFlightRef.current.delete(flowId);
    }
  }, []);

  return useMemo(() => {
    const view = new Map<string, FlowDetailResult>();
    for (const { flowId, version } of requestsRef.current) {
      const entry = resultsRef.current.get(flowId);
      if (entry !== undefined && entry.version === version) view.set(flowId, entry.result);
    }
    return view;
    // generation ticks when a fetch lands; key changes when the wanted set does.
  }, [key, generation]);
}
