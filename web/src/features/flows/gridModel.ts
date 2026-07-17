import { formatBytes } from "../../format";
import type { ImmutableFlowLifecycle, ImmutableFlowMetadata } from "../../state/browserState";
import type { FilterableFlow } from "./filter";

export type FlowPhase = "request" | "response" | "complete" | "error";

const phaseLabels: Record<FlowPhase, string> = {
  request: "await",
  response: "resp",
  complete: "done",
  error: "error",
};

export interface BodyCell {
  readonly text: string;
  readonly truncated: boolean;
}

export interface FlowRow {
  readonly flowId: string;
  readonly method: string;
  readonly scheme: "http" | "https";
  readonly host: string;
  readonly port: string;
  readonly path: string;
  readonly url: string;
  readonly phase: FlowPhase;
  readonly phaseLabel: string;
  readonly requestBody: BodyCell;
  readonly responseBody: BodyCell;
  readonly contentType: string;
  readonly filterable: FilterableFlow;
}

type ImmutableBody = ImmutableFlowMetadata["request_body"];

function bodyCell(body: ImmutableBody | undefined): BodyCell {
  if (body === undefined || body.state === "missing") return { text: "—", truncated: false };
  if (body.state === "truncated") return { text: formatBytes(body.size_bytes), truncated: true };
  return { text: formatBytes(body.size_bytes), truncated: false };
}

function contentTypeOf(body: ImmutableBody | undefined): string | null {
  if (body === undefined) return null;
  const value = body.content_type;
  return value === undefined || value === "" ? null : value;
}

function headerValue(headers: readonly { readonly name: string; readonly value: string }[] | undefined, name: string): string | null {
  if (headers === undefined) return null;
  const match = headers.find((header) => header.name.toLowerCase() === name);
  return match === undefined || match.value === "" ? null : match.value;
}

/** Reduce a full content-type header value to its bare type for display. */
function bareContentType(value: string | null): string | null {
  if (value === null) return null;
  const bare = value.split(";", 1)[0].trim();
  return bare === "" ? value : bare;
}

export function buildFlowRow(
  metadata: ImmutableFlowMetadata,
  lifecycle: readonly ImmutableFlowLifecycle[] = [],
): FlowRow {
  const states = new Set(lifecycle.map((event) => event.state));
  const errored = states.has("error");
  const completed = states.has("flow_completed");
  const responseSeen = metadata.response_headers !== undefined
    || metadata.response_body !== undefined
    || [...states].some((state) => state.startsWith("response"));
  const phase: FlowPhase = errored ? "error" : completed ? "complete" : responseSeen ? "response" : "request";
  const url = `${metadata.scheme}://${metadata.host}:${metadata.port}${metadata.path}`;
  const requestContentType = contentTypeOf(metadata.request_body) ?? headerValue(metadata.request_headers, "content-type");
  const responseContentType = contentTypeOf(metadata.response_body) ?? headerValue(metadata.response_headers, "content-type");

  return {
    flowId: metadata.flow_id,
    method: metadata.method,
    scheme: metadata.scheme,
    host: metadata.host,
    port: metadata.port,
    path: metadata.path,
    url,
    phase,
    phaseLabel: phaseLabels[phase],
    requestBody: bodyCell(metadata.request_body),
    responseBody: bodyCell(metadata.response_body),
    contentType: bareContentType(responseContentType) ?? bareContentType(requestContentType) ?? "—",
    filterable: {
      method: metadata.method,
      scheme: metadata.scheme,
      host: metadata.host,
      port: metadata.port,
      path: metadata.path,
      url,
      requestHeaders: metadata.request_headers,
      responseHeaders: metadata.response_headers ?? [],
      requestContentType,
      responseContentType,
      hasResponse: responseSeen,
      errored,
    },
  };
}
