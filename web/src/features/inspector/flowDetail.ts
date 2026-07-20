/* eslint-disable no-unused-vars */

import { useEffect, useState } from "react";

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

/** Load body detail for the selected flow; falls back to grid metadata on failure. */
export function useFlowDetail(
  flowId: string | null,
  loader: FlowDetailLoader = fetchFlowDetail,
): FlowDetailResult | null {
  const [result, setResult] = useState<FlowDetailResult | null>(null);
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  useEffect(() => {
    if (flowId === null) {
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
        setLoadedFor(flowId);
      })
      .catch((error: unknown) => {
        if (!active || controller.signal.aborted) return;
        setResult({ status: "error", error: error instanceof Error ? error.message : "Flow detail request failed." });
        setLoadedFor(flowId);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [flowId, loader]);
  return flowId !== null && loadedFor === flowId ? result : null;
}
