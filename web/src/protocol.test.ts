import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";

import schema from "../../contracts/protocol-v1.schema.json";
import { parseProtocolMessage, ProtocolError } from "./protocol";

const streamPath = resolve(process.cwd(), "../contracts/fixtures/stream.json");
const conformancePath = resolve(process.cwd(), "../contracts/fixtures/conformance.json");
const streamMessages = JSON.parse(readFileSync(streamPath, "utf8")) as unknown[];
const conformance = JSON.parse(readFileSync(conformancePath, "utf8")) as {
  valid: unknown[];
  invalid: Array<{ name: string; message: unknown }>;
};
const validateSchema = new Ajv2020({ strict: false }).compile(schema);

describe("shared protocol-v1 conformance", () => {
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
    const parsed = streamMessages.map(parseProtocolMessage);
    const metadata = parsed[1].metadata as { request_headers: Array<{ name: string; value: string }> };
    expect(metadata.request_headers.slice(1, 3)).toEqual([
      { name: "x-trace", value: "first" },
      { name: "x-trace", value: "second" },
    ]);
    const lifecycle = parsed.filter((message) => message.type === "flow.lifecycle");
    expect(lifecycle.map((message) => message.state)).toEqual(["response_started", "request_end"]);
  });

  it("tolerates unknown types and additive fields without numeric coercion", () => {
    const future = parseProtocolMessage(conformance.valid.at(-1));
    expect(future.type).toBe("future.additive");
    expect(future.sequence).toBe("18446744073709551616");
    expect(future.new_field).toEqual({ safe: true });
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

  it("contains no secret canaries", () => {
    const serialized = JSON.stringify({ streamMessages, conformance }).toLowerCase();
    for (const canary of ["authorization-canary", "x-api-key-canary", "cookie-canary", "query-secret-canary", "contract-secret"]) {
      expect(serialized).not.toContain(canary);
    }
  });
});
