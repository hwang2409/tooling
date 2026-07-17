import { formatBytesCompact, formatDurationMs } from "../../format";
import { decodeBody } from "./decoders";
import { durationMsFromLifecycle } from "../flows/gridModel";
import type { InspectableBody, InspectorFlow } from "./models";

/** Best-guess one-line summary of a captured HTTP flow. */
export function flowSummary(flow: InspectorFlow): string {
  const method = flow.metadata.method;
  const host = flow.metadata.host;
  const path = flow.metadata.path;
  const status = normalizedStatus(flow);
  const duration = formatDurationMs(durationMsFromLifecycle(flow.lifecycle ?? []));
  const requestBody = flow.request_body ?? flow.metadata.request_body;
  const responseBody = flow.response_body ?? flow.metadata.response_body;
  const requestSize = formatBytesCompact(sizeOf(requestBody));
  const responseSize = formatBytesCompact(sizeOf(responseBody));

  const anthropicSummary = anthropicMessagesSummary(flow, requestBody, path, method);
  const base = anthropicSummary ?? genericSummary(method, host, path);
  const tail: string[] = [];
  if (anthropicSummary === null) {
    tail.push(`${requestSize} in / ${responseSize} out`);
  } else {
    tail.push(`${requestSize} in / ${responseSize} out`);
  }
  tail.push(duration);
  if (status !== null) tail.push(status);
  return `${base} · ${tail.filter(Boolean).join(" · ")}`;
}

function genericSummary(method: string, host: string, path: string): string {
  return `${method} ${host}${path}`;
}

function sizeOf(body: InspectableBody | undefined): string | undefined {
  if (!body || body.state === "missing" || body.state === "redacted") return undefined;
  if (body.state === "empty") return "0";
  return body.size_bytes;
}

function normalizedStatus(flow: InspectorFlow): string | null {
  const declared = flow.metadata.response_status;
  if (typeof declared === "string" && declared.length > 0) return declared;
  const errored = (flow.lifecycle ?? []).some((event) => event.state === "error") || Boolean(flow.error?.trim());
  return errored ? "err" : null;
}

interface AnthropicMessages {
  model?: string;
  messageCount?: number;
  streaming?: boolean;
}

/**
 * Recognise Anthropic's /v1/messages endpoint and pull out model + msg count
 * from the captured request JSON. Falls back to `null` when the body is not
 * a shape we recognise so the caller can render a generic summary instead.
 */
export function anthropicMessagesSummary(
  flow: InspectorFlow,
  requestBody: InspectableBody | undefined,
  path: string,
  method: string,
): string | null {
  if (method.toUpperCase() !== "POST") return null;
  if (!path.startsWith("/v1/messages")) return null;
  const parsed = parseAnthropicRequest(requestBody);
  const parts = [`POST ${trimPath(path)}`];
  if (parsed.model !== undefined) parts.push(parsed.model);
  if (parsed.messageCount !== undefined) parts.push(`${parsed.messageCount} ${parsed.messageCount === 1 ? "msg" : "msgs"}`);
  if (parsed.streaming) parts.push("stream");
  return parts.join(" · ");
}

function trimPath(path: string): string {
  const questionMark = path.indexOf("?");
  return questionMark === -1 ? path.replace(/^\//, "") : path.slice(0, questionMark).replace(/^\//, "");
}

function parseAnthropicRequest(body: InspectableBody | undefined): AnthropicMessages {
  if (!body || (body.state !== "captured" && body.state !== "truncated")) return {};
  const decoded = decodeBody(body, "json");
  if (decoded.fallback !== "none") return {};
  try {
    const parsed = JSON.parse(decoded.text) as Record<string, unknown>;
    const model = typeof parsed.model === "string" ? parsed.model : undefined;
    const messageCount = Array.isArray(parsed.messages) ? parsed.messages.length : undefined;
    const streaming = parsed.stream === true;
    return { model, messageCount, streaming };
  } catch {
    return {};
  }
}
