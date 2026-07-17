export type DecimalString = string;
export type BodyState = "missing" | "empty" | "captured" | "truncated";
export type BodySide = "request" | "response";
export type LifecycleState =
  | "request_started"
  | "request_headers"
  | "request_body"
  | "request_end"
  | "response_started"
  | "response_headers"
  | "response_body"
  | "response_end"
  | "error"
  | "flow_completed";
export type ResyncReason = "cursor_gap" | "history_evicted" | "initial_connect";

export interface Header {
  name: string;
  value: string;
}

export interface MissingBody {
  state: "missing";
  content_type?: string;
}

export interface EmptyBody {
  state: "empty";
  size_bytes: "0";
  content_type?: string;
}

export interface CapturedBody {
  state: "captured";
  size_bytes: DecimalString;
  content_type?: string;
  encoding: "base64";
  data: string;
}

export interface TruncatedBody {
  state: "truncated";
  size_bytes: DecimalString;
  captured_bytes: DecimalString;
  content_type?: string;
  encoding: "base64";
  data: string;
}

export type BodyDescriptor = MissingBody | EmptyBody | CapturedBody | TruncatedBody;

export interface FlowMetadata {
  flow_id: string;
  method: string;
  scheme: "http" | "https";
  host: string;
  port: DecimalString;
  path: string;
  request_headers: Header[];
  response_headers?: Header[];
  request_body: BodyDescriptor;
  response_body?: BodyDescriptor;
}

interface ProtocolBase {
  protocol_version: "1";
  type: string;
  [key: string]: unknown;
}

export interface SourceHello extends ProtocolBase {
  type: "source.hello";
  source_id: string;
  occurred_at: string;
  capabilities: { body_chunks: boolean; redaction: "headers-and-query" };
  limits: { max_body_prefix_bytes: DecimalString; max_in_memory_bytes: DecimalString };
}

export interface FlowMetadataMessage extends ProtocolBase {
  type: "flow.metadata";
  metadata: FlowMetadata;
}

export interface FlowLifecycle extends ProtocolBase {
  type: "flow.lifecycle";
  source_id: string;
  flow_id: string;
  event_id: string;
  occurred_at: string;
  sequence: DecimalString;
  state: LifecycleState;
}

export interface BodyChunk extends ProtocolBase {
  type: "body.chunk";
  flow_id: string;
  body_side: BodySide;
  chunk_index: DecimalString;
  offset_bytes: DecimalString;
  data_base64: string;
}

export interface BodyEnd extends ProtocolBase {
  type: "body.end";
  flow_id: string;
  body_side: BodySide;
  total_bytes: DecimalString;
  body: BodyDescriptor;
}

export interface StreamGap extends ProtocolBase {
  type: "stream.gap";
  expected_sequence: DecimalString;
  actual_sequence: DecimalString;
  dropped_count?: DecimalString;
}

export interface BrowserSnapshot extends ProtocolBase {
  type: "browser.snapshot";
  snapshot_id: string;
  cursor: DecimalString;
  flows: FlowMetadata[];
}

export interface UpsertChange {
  op: "upsert";
  flow: FlowMetadata;
  [key: string]: unknown;
}

export interface RemoveChange {
  op: "remove";
  flow_id: string;
  [key: string]: unknown;
}

export interface BrowserDelta extends ProtocolBase {
  type: "browser.delta";
  cursor: DecimalString;
  changes: Array<UpsertChange | RemoveChange>;
}

export interface BrowserResync extends ProtocolBase {
  type: "browser.resync";
  reason: ResyncReason;
  requested_cursor: DecimalString;
}

export type KnownMessage =
  | SourceHello
  | FlowMetadataMessage
  | FlowLifecycle
  | BodyChunk
  | BodyEnd
  | StreamGap
  | BrowserSnapshot
  | BrowserDelta
  | BrowserResync;

export interface KnownEnvelope {
  kind: "known";
  message: KnownMessage;
}

export interface UnknownEnvelope {
  kind: "unknown";
  original_type: string;
  payload: Record<string, unknown>;
}

export type ParsedMessage = KnownEnvelope | UnknownEnvelope;

export class ProtocolError extends Error {}

export const MAX_U64 = 18_446_744_073_709_551_615n;
const decimalPattern = /^(0|[1-9][0-9]*)$/;
const base64Pattern = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const lifecycleStates = new Set<LifecycleState>([
  "request_started", "request_headers", "request_body", "request_end", "response_started",
  "response_headers", "response_body", "response_end", "error", "flow_completed",
]);
const bodySides = new Set<BodySide>(["request", "response"]);
const resyncReasons = new Set<ResyncReason>(["cursor_gap", "history_evicted", "initial_connect"]);
const knownTypes = new Set([
  "source.hello", "flow.metadata", "flow.lifecycle", "body.chunk", "body.end",
  "stream.gap", "browser.snapshot", "browser.delta", "browser.resync",
]);

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ProtocolError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ProtocolError(`${label} must be a non-empty string`);
  }
  return value;
}

function textValue(value: unknown, label: string): string {
  if (typeof value !== "string") throw new ProtocolError(`${label} must be a string`);
  return value;
}

function decimalValue(value: unknown, label: string): DecimalString {
  const candidate = stringValue(value, label);
  if (!decimalPattern.test(candidate)) throw new ProtocolError(`${label} must be a uint64 decimal string`);
  if (BigInt(candidate) > MAX_U64) throw new ProtocolError(`${label} exceeds uint64`);
  return candidate;
}

function enumValue<T extends string>(value: unknown, values: Set<T>, label: string): T {
  const candidate = stringValue(value, label);
  if (!values.has(candidate as T)) throw new ProtocolError(`${label} is not supported`);
  return candidate as T;
}

function headers(value: unknown, label: string): Header[] {
  if (!Array.isArray(value)) throw new ProtocolError(`${label} must be an ordered list`);
  return value.map((item, index) => {
    const header = record(item, `${label}[${index}]`);
    return {
      name: stringValue(header.name, `${label}[${index}].name`),
      value: textValue(header.value, `${label}[${index}].value`),
    };
  });
}

function decodedBase64(value: unknown, label: string): Uint8Array {
  const data = textValue(value, label);
  if (!base64Pattern.test(data)) throw new ProtocolError(`${label} must be valid base64`);
  try {
    return Uint8Array.from(atob(data), (character) => character.charCodeAt(0));
  } catch (error) {
    throw new ProtocolError(`${label} must be valid base64`, { cause: error });
  }
}

function body(value: unknown, label: string): BodyDescriptor {
  const descriptor = record(value, label);
  const state = stringValue(descriptor.state, `${label}.state`);
  if (!["missing", "empty", "captured", "truncated"].includes(state)) {
    throw new ProtocolError(`${label}.state is not supported`);
  }
  if (descriptor.content_type !== undefined) textValue(descriptor.content_type, `${label}.content_type`);
  if (state === "missing") {
    if (["size_bytes", "captured_bytes", "encoding", "data"].some((key) => key in descriptor)) {
      throw new ProtocolError(`${label} missing state cannot carry body counts or data`);
    }
    return descriptor as unknown as MissingBody;
  }
  const size = decimalValue(descriptor.size_bytes, `${label}.size_bytes`);
  if (state === "empty") {
    if (size !== "0" || ["captured_bytes", "encoding", "data"].some((key) => key in descriptor)) {
      throw new ProtocolError(`${label} empty state must have only size_bytes=0`);
    }
    return descriptor as unknown as EmptyBody;
  }
  if (state === "captured" && "captured_bytes" in descriptor) {
    throw new ProtocolError(`${label} captured state cannot carry captured_bytes`);
  }
  if (descriptor.encoding !== "base64") throw new ProtocolError(`${label}.encoding must be base64`);
  const decoded = decodedBase64(descriptor.data, `${label}.data`);
  if (state === "captured") {
    if (BigInt(decoded.length) !== BigInt(size)) throw new ProtocolError(`${label}.data length does not equal size_bytes`);
    return descriptor as unknown as CapturedBody;
  }
  const captured = decimalValue(descriptor.captured_bytes, `${label}.captured_bytes`);
  if (BigInt(captured) > BigInt(size) || BigInt(decoded.length) > BigInt(captured)) {
    throw new ProtocolError(`${label} truncated prefix exceeds declared counts`);
  }
  return descriptor as unknown as TruncatedBody;
}

function flow(value: unknown, label: string): FlowMetadata {
  const metadata = record(value, label);
  for (const key of ["flow_id", "method", "host", "path"]) stringValue(metadata[key], `${label}.${key}`);
  const scheme = stringValue(metadata.scheme, `${label}.scheme`);
  if (scheme !== "http" && scheme !== "https") throw new ProtocolError(`${label}.scheme is not supported`);
  decimalValue(metadata.port, `${label}.port`);
  headers(metadata.request_headers, `${label}.request_headers`);
  body(metadata.request_body, `${label}.request_body`);
  if (metadata.response_headers !== undefined) headers(metadata.response_headers, `${label}.response_headers`);
  if (metadata.response_body !== undefined) body(metadata.response_body, `${label}.response_body`);
  return metadata as unknown as FlowMetadata;
}

function sourceHello(message: Record<string, unknown>): void {
  stringValue(message.source_id, "source_id");
  stringValue(message.occurred_at, "occurred_at");
  const capabilities = record(message.capabilities, "capabilities");
  if (typeof capabilities.body_chunks !== "boolean") throw new ProtocolError("capabilities.body_chunks must be boolean");
  if (capabilities.redaction !== "headers-and-query") throw new ProtocolError("capabilities.redaction is not supported");
  const limits = record(message.limits, "limits");
  decimalValue(limits.max_body_prefix_bytes, "limits.max_body_prefix_bytes");
  decimalValue(limits.max_in_memory_bytes, "limits.max_in_memory_bytes");
}

function bodyEnd(message: Record<string, unknown>): void {
  const total = decimalValue(message.total_bytes, "total_bytes");
  const descriptor = body(message.body, "body");
  if (descriptor.state === "missing" && total !== "0") throw new ProtocolError("missing body must have total_bytes=0");
  if (descriptor.state !== "missing" && descriptor.size_bytes !== total) throw new ProtocolError("body.size_bytes must equal total_bytes");
}

function streamGap(message: Record<string, unknown>): void {
  const expected = BigInt(decimalValue(message.expected_sequence, "expected_sequence"));
  const actual = BigInt(decimalValue(message.actual_sequence, "actual_sequence"));
  if (actual <= expected) throw new ProtocolError("actual_sequence must be greater than expected_sequence");
  if (message.dropped_count !== undefined && BigInt(decimalValue(message.dropped_count, "dropped_count")) !== actual - expected - 1n) {
    throw new ProtocolError("dropped_count does not match the sequence gap");
  }
}

/** Validate a protocol-v1 message without discarding unknown fields or types. */
export function parseProtocolMessage(value: unknown): ParsedMessage {
  const message = record(value, "message");
  if (message.protocol_version !== "1") throw new ProtocolError("protocol_version must be '1'");
  const type = stringValue(message.type, "type");

  if (type === "source.hello") sourceHello(message);
  else if (type === "flow.metadata") flow(message.metadata, "metadata");
  else if (type === "flow.lifecycle") {
    for (const key of ["source_id", "flow_id", "event_id", "occurred_at"]) stringValue(message[key], key);
    decimalValue(message.sequence, "sequence");
    enumValue(message.state, lifecycleStates, "state");
  } else if (type === "body.chunk") {
    stringValue(message.flow_id, "flow_id");
    enumValue(message.body_side, bodySides, "body_side");
    decimalValue(message.chunk_index, "chunk_index");
    decimalValue(message.offset_bytes, "offset_bytes");
    decodedBase64(message.data_base64, "data_base64");
  } else if (type === "body.end") {
    stringValue(message.flow_id, "flow_id");
    enumValue(message.body_side, bodySides, "body_side");
    bodyEnd(message);
  } else if (type === "stream.gap") streamGap(message);
  else if (type === "browser.snapshot") {
    stringValue(message.snapshot_id, "snapshot_id");
    decimalValue(message.cursor, "cursor");
    if (!Array.isArray(message.flows)) throw new ProtocolError("flows must be an ordered list");
    message.flows.forEach((item, index) => flow(item, `flows[${index}]`));
  } else if (type === "browser.delta") {
    decimalValue(message.cursor, "cursor");
    if (!Array.isArray(message.changes)) throw new ProtocolError("changes must be an ordered list");
    message.changes.forEach((change, index) => {
      const item = record(change, `changes[${index}]`);
      const operation = stringValue(item.op, `changes[${index}].op`);
      if (operation === "upsert") flow(item.flow, `changes[${index}].flow`);
      else if (operation === "remove") stringValue(item.flow_id, `changes[${index}].flow_id`);
      else throw new ProtocolError(`changes[${index}].op is not supported`);
    });
  } else if (type === "browser.resync") {
    enumValue(message.reason, resyncReasons, "reason");
    decimalValue(message.requested_cursor, "requested_cursor");
  }

  if (knownTypes.has(type)) {
    return { kind: "known", message: message as unknown as KnownMessage };
  }
  return { kind: "unknown", original_type: type, payload: message };
}
