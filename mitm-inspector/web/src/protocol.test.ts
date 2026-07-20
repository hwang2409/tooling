import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";

import schema from "../../contracts/protocol-v1.schema.json";
import {
  isParsedProtocolMessage,
  parseFlowExtras,
  parseProtocolMessage,
  ProtocolError,
  requireParsedProtocolMessage,
} from "./protocol";
import type { ImmutableKnownMessage, ParsedMessage } from "./protocol";

const streamPath = resolve(process.cwd(), "../contracts/fixtures/stream.json");
const conformancePath = resolve(process.cwd(), "../contracts/fixtures/conformance.json");
const streamMessages = JSON.parse(readFileSync(streamPath, "utf8")) as unknown[];
const conformance = JSON.parse(readFileSync(conformancePath, "utf8")) as {
  valid: unknown[];
  invalid: Array<{ name: string; message: unknown }>;
};
const validateSchema = new Ajv2020({ strict: false }).compile(schema);

function known(message: unknown): ImmutableKnownMessage {
  const parsed = parseProtocolMessage(message);
  expect(parsed.kind).toBe("known");
  if (parsed.kind !== "known") throw new Error("expected known message");
  return parsed.message;
}

describe("shared protocol-v1 conformance", () => {
  it("extracts additive flow enrichment tolerantly", () => {
    expect(parseFlowExtras({
      started_at: "2026-01-01T00:00:00Z",
      ended_at: "2026-01-01T00:00:01Z",
      request_body_size: "12",
      content_encoding: { request: "identity", response: 4 },
      summary: { kind: "anthropic_messages", model: "claude", message_count: "2", preview: { source: "user_text", text: "hi" } },
      future: true,
    })).toEqual({
      started_at: "2026-01-01T00:00:00Z",
      ended_at: "2026-01-01T00:00:01Z",
      request_body_size: "12",
      content_encoding: { request: "identity" },
      summary: { kind: "anthropic_messages", model: "claude", message_count: "2", preview: { source: "user_text", text: "hi" } },
    });
    expect(parseFlowExtras({ started_at: "1", request_body_size: "-1", summary: { kind: "future" } })).toEqual({});
    expect(parseFlowExtras({ started_at: "2026-01-01T00:00:00+00:00", ended_at: "2026-01-01T00:00:01+00:00" })).toEqual({
      started_at: "2026-01-01T00:00:00+00:00",
      ended_at: "2026-01-01T00:00:01+00:00",
    });
  });

  it("accepts every positive fixture in the authoritative schema", () => {
    for (const message of [...streamMessages, ...conformance.valid]) {
      expect(validateSchema(message), JSON.stringify(validateSchema.errors)).toBe(true);
      parseProtocolMessage(message);
    }
  });

  it("rejects every named negative fixture in schema and TypeScript validator", () => {
    expect(new Set(conformance.invalid.map((testCase) => testCase.name)).size).toBe(conformance.invalid.length);
    for (const testCase of conformance.invalid) {
      expect(validateSchema(testCase.message), testCase.name).toBe(false);
      expect(() => parseProtocolMessage(testCase.message), testCase.name).toThrow(ProtocolError);
    }
  });

  it("preserves duplicate ordered headers and response-before-request-end ordering", () => {
    const parsed = streamMessages.filter((message) => (message as { type?: string }).type !== "future.message").map(known);
    const metadata = parsed[1].metadata as {
      session_id?: string | null;
      request_headers: Array<{ name: string; value: string }>;
      request_body: { content_type?: string };
    };
    expect(metadata.request_headers.slice(1, 3)).toEqual([
      { name: "x-trace", value: "first" },
      { name: "x-trace", value: "second" },
    ]);
    const conformanceMetadata = (known(conformance.valid[1]).metadata) as typeof metadata;
    expect(conformanceMetadata.session_id).toBe("session-conformance");
    expect(conformanceMetadata.request_headers[2].value).toBe("");
    expect(conformanceMetadata.request_body.content_type).toBe("");
    const nullSession = conformance.valid
      .filter((message): message is { type: string; metadata: { flow_id: string; session_id: string | null } } =>
        typeof message === "object" && message !== null
        && (message as { type?: unknown }).type === "flow.metadata"
        && typeof (message as { metadata?: { flow_id?: unknown } }).metadata?.flow_id === "string"
        && (message as { metadata: { flow_id: string } }).metadata.flow_id === "conformance-null-session")
      .at(0);
    expect(nullSession?.metadata.session_id).toBeNull();
    const lifecycle = parsed.filter((message) => message.type === "flow.lifecycle");
    expect(lifecycle.map((message) => message.state)).toEqual(["response_started", "request_end"]);
  });

  it("tolerates unknown types and additive fields without numeric coercion", () => {
    const future = parseProtocolMessage(conformance.valid.at(-1));
    expect(future.kind).toBe("unknown");
    if (future.kind !== "unknown") throw new Error("expected unknown message");
    expect(future.original_type).toBe("future.additive");
    expect(future.original_type).toBe(future.payload.type);
    expect(future.payload.sequence).toBe("18446744073709551616");
    expect(future.payload.new_field).toEqual({ safe: true });
  });

  it("returns nominal, non-overlapping, recursively immutable values", () => {
    const rawKnown = {
      protocol_version: "1",
      type: "body.chunk",
      flow_id: "f",
      body_side: "request",
      chunk_index: "0",
      offset_bytes: "0",
      data_base64: "",
      extension: { nested: [{ value: "before" }] },
    };
    const parsedKnown = parseProtocolMessage(rawKnown);
    expect(parsedKnown.kind).toBe("known");
    if (parsedKnown.kind !== "known") throw new Error("expected known message");
    expect("payload" in parsedKnown).toBe(false);

    rawKnown.flow_id = "mutated";
    rawKnown.extension.nested[0].value = "after";
    expect(parsedKnown.message.flow_id).toBe("f");
    expect(parsedKnown.message.extension).toEqual({ nested: [{ value: "before" }] });
    expect(Object.isFrozen(parsedKnown)).toBe(true);
    expect(Object.isFrozen(parsedKnown.message)).toBe(true);
    const mutableKnown = parsedKnown.message as unknown as {
      flow_id: string;
      extension: { nested: Array<{ value: string }> };
    };
    expect(() => { mutableKnown.flow_id = "forged"; }).toThrow(TypeError);
    expect(() => { mutableKnown.extension.nested[0].value = "forged"; }).toThrow(TypeError);

    const rawOpaque = {
      protocol_version: "1",
      type: "future.message",
      extension: { values: ["before"] },
    };
    const parsedOpaque = parseProtocolMessage(rawOpaque);
    expect(parsedOpaque.kind).toBe("unknown");
    if (parsedOpaque.kind !== "unknown") throw new Error("expected opaque message");
    expect("message" in parsedOpaque).toBe(false);
    rawOpaque.type = "body.chunk";
    rawOpaque.extension.values[0] = "after";
    expect(parsedOpaque.original_type).toBe(parsedOpaque.payload.type);
    expect(parsedOpaque.original_type).toBe("future.message");
    expect(parsedOpaque.payload.extension).toEqual({ values: ["before"] });
    expect(requireParsedProtocolMessage(parsedOpaque)).toBe(parsedOpaque);
  });

  it("canonicalizes a known message once before validation and branding", () => {
    let reads = 0;
    const raw: Record<string, unknown> = {
      protocol_version: "1",
      type: "body.chunk",
      flow_id: "f",
      chunk_index: "0",
      offset_bytes: "0",
      data_base64: "",
    };
    Object.defineProperty(raw, "body_side", {
      enumerable: true,
      get: () => {
        reads += 1;
        return reads === 1 ? "request" : "sideways";
      },
    });

    const parsed = parseProtocolMessage(raw);
    expect(reads).toBe(1);
    expect(parsed.kind).toBe("known");
    if (parsed.kind !== "known") throw new Error("expected known message");
    expect(parsed.message.body_side).toBe("request");
    expect(requireParsedProtocolMessage(parsed)).toBe(parsed);
  });

  it("canonicalizes an unknown discriminator once before opaque branding", () => {
    let reads = 0;
    const raw: Record<string, unknown> = { protocol_version: "1" };
    Object.defineProperty(raw, "type", {
      enumerable: true,
      get: () => {
        reads += 1;
        return reads === 1 ? "future.accessor" : "body.chunk";
      },
    });

    const parsed = parseProtocolMessage(raw);
    expect(reads).toBe(1);
    expect(parsed.kind).toBe("unknown");
    if (parsed.kind !== "unknown") throw new Error("expected unknown message");
    expect(parsed.original_type).toBe("future.accessor");
    expect(parsed.payload.type).toBe("future.accessor");
    expect(requireParsedProtocolMessage(parsed)).toBe(parsed);
  });

  it.each([
    { kind: "known", message: { protocol_version: "1", type: "source.hello" } },
    {
      kind: "unknown",
      original_type: "future.message",
      payload: { protocol_version: "1", type: "different.future" },
    },
    {
      kind: "unknown",
      original_type: "future.message",
      payload: { protocol_version: "1", type: "body.chunk" },
    },
  ])("rejects forged or spoofed structural envelope %#", (candidate) => {
    const forged = candidate as unknown as ParsedMessage;
    expect(isParsedProtocolMessage(forged)).toBe(false);
    expect(() => requireParsedProtocolMessage(forged)).toThrow(ProtocolError);
  });

  it("rejects a truncated prefix whose count exceeds the total", () => {
    expect(() => parseProtocolMessage({
      protocol_version: "1",
      type: "body.end",
      flow_id: "f",
      body_side: "response",
      total_bytes: "3",
      body: { state: "truncated", size_bytes: "3", captured_bytes: "4", encoding: "base64", data: "AQID" },
    })).toThrow(ProtocolError);
  });

  it("rejects invalid gap ordering and dropped-count arithmetic", () => {
    expect(() => parseProtocolMessage({ protocol_version: "1", type: "stream.gap", expected_sequence: "2", actual_sequence: "1" })).toThrow(ProtocolError);
    expect(() => parseProtocolMessage({ protocol_version: "1", type: "stream.gap", expected_sequence: "9", actual_sequence: "11", dropped_count: "2" })).toThrow(ProtocolError);
  });

  it("bounds response_status to the 100..599 HTTP range in schema and validator", () => {
    const baseFlow = {
      flow_id: "f-status", method: "GET", scheme: "https", host: "h", port: "443", path: "/x",
      request_headers: [], request_body: { state: "missing" }, response_headers: [],
    };
    const flowMessage = (status: string) => ({
      protocol_version: "1", type: "flow.metadata",
      metadata: { ...baseFlow, response_status: status },
    });
    expect(validateSchema(flowMessage("200"))).toBe(true);
    parseProtocolMessage(flowMessage("200"));
    for (const bad of ["0", "1", "99", "600", "999"]) {
      expect(validateSchema(flowMessage(bad)), `schema should reject ${bad}`).toBe(false);
      expect(() => parseProtocolMessage(flowMessage(bad)), `validator should reject ${bad}`).toThrow(ProtocolError);
    }
  });

  it("contains no secret canaries", () => {
    const serialized = JSON.stringify({ streamMessages, conformance }).toLowerCase();
    for (const canary of ["authorization-canary", "x-api-key-canary", "cookie-canary", "query-secret-canary", "contract-secret"]) {
      expect(serialized).not.toContain(canary);
    }
  });
});
