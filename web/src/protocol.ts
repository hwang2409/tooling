export type DecimalString = string;

export type BodyState = "missing" | "empty" | "captured" | "truncated";

export interface Header {
  name: string;
  value: string;
}

export interface BodyDescriptor {
  state: BodyState;
  size_bytes?: DecimalString;
  captured_bytes?: DecimalString;
  encoding?: "base64";
  data?: string;
}

export interface FlowMetadata {
  flow_id: string;
  method: string;
  scheme: string;
  host: string;
  port: DecimalString;
  path: string;
  request_headers: Header[];
  response_headers?: Header[];
  request_body: BodyDescriptor;
  response_body?: BodyDescriptor;
}

export interface ProtocolMessage {
  protocol_version: "1";
  type: string;
  [key: string]: unknown;
}

export class ProtocolError extends Error {}

const decimalPattern = /^(0|[1-9][0-9]*)$/;

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

function decimalValue(value: unknown, label: string): DecimalString {
  const candidate = stringValue(value, label);
  if (!decimalPattern.test(candidate)) {
    throw new ProtocolError(`${label} must be an unsigned decimal string`);
  }
  return candidate;
}

function headers(value: unknown, label: string): Header[] {
  if (!Array.isArray(value)) throw new ProtocolError(`${label} must be an ordered list`);
  return value.map((item, index) => {
    const header = record(item, `${label}[${index}]`);
    return {
      name: stringValue(header.name, `${label}[${index}].name`),
      value: stringValue(header.value, `${label}[${index}].value`),
    };
  });
}

function body(value: unknown, label: string): BodyDescriptor {
  const descriptor = record(value, label);
  const state = stringValue(descriptor.state, `${label}.state`);
  if (!["missing", "empty", "captured", "truncated"].includes(state)) {
    throw new ProtocolError(`${label}.state is not supported`);
  }
  if (state !== "missing") decimalValue(descriptor.size_bytes, `${label}.size_bytes`);
  if (state === "truncated") decimalValue(descriptor.captured_bytes, `${label}.captured_bytes`);
  if (state === "captured" || state === "truncated") {
    if (descriptor.encoding !== "base64") throw new ProtocolError(`${label}.encoding must be base64`);
    stringValue(descriptor.data, `${label}.data`);
  }
  return descriptor as unknown as BodyDescriptor;
}

function flow(value: unknown, label: string): FlowMetadata {
  const metadata = record(value, label);
  for (const key of ["flow_id", "method", "scheme", "host", "path"]) {
    stringValue(metadata[key], `${label}.${key}`);
  }
  decimalValue(metadata.port, `${label}.port`);
  headers(metadata.request_headers, `${label}.request_headers`);
  body(metadata.request_body, `${label}.request_body`);
  if (metadata.response_headers !== undefined) headers(metadata.response_headers, `${label}.response_headers`);
  if (metadata.response_body !== undefined) body(metadata.response_body, `${label}.response_body`);
  return metadata as unknown as FlowMetadata;
}

/** Validate a protocol-v1 message without discarding unknown fields or types. */
export function parseProtocolMessage(value: unknown): ProtocolMessage {
  const message = record(value, "message");
  if (message.protocol_version !== "1") throw new ProtocolError("protocol_version must be '1'");
  const type = stringValue(message.type, "type");

  if (type === "source.hello") {
    stringValue(message.source_id, "source_id");
    stringValue(message.occurred_at, "occurred_at");
  } else if (type === "flow.metadata") {
    flow(message.metadata, "metadata");
  } else if (type === "flow.lifecycle") {
    for (const key of ["source_id", "flow_id", "event_id", "occurred_at", "state"]) stringValue(message[key], key);
    decimalValue(message.sequence, "sequence");
  } else if (type === "body.chunk") {
    stringValue(message.flow_id, "flow_id");
    stringValue(message.body_side, "body_side");
    decimalValue(message.chunk_index, "chunk_index");
    decimalValue(message.offset_bytes, "offset_bytes");
    stringValue(message.data_base64, "data_base64");
  } else if (type === "body.end") {
    stringValue(message.flow_id, "flow_id");
    stringValue(message.body_side, "body_side");
    decimalValue(message.total_bytes, "total_bytes");
    body(message.body, "body");
  } else if (type === "stream.gap") {
    decimalValue(message.expected_sequence, "expected_sequence");
    decimalValue(message.actual_sequence, "actual_sequence");
  } else if (type === "browser.snapshot") {
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
    stringValue(message.reason, "reason");
    decimalValue(message.requested_cursor, "requested_cursor");
  }

  return message as ProtocolMessage;
}
