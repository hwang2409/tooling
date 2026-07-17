import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { parseProtocolMessage, ProtocolError } from "./protocol";

const fixturePath = resolve(process.cwd(), "../contracts/fixtures/stream.json");
const messages = JSON.parse(readFileSync(fixturePath, "utf8")) as unknown[];

describe("shared protocol-v1 fixtures", () => {
  it("accepts every fixture and preserves duplicate ordered headers", () => {
    const parsed = messages.map(parseProtocolMessage);
    const metadata = parsed[1].metadata as { request_headers: Array<{ name: string; value: string }> };
    expect(metadata.request_headers.slice(1, 3)).toEqual([
      { name: "x-trace", value: "first" },
      { name: "x-trace", value: "second" },
    ]);
  });

  it("keeps body states distinct and allows response before request end", () => {
    const parsed = messages.map(parseProtocolMessage);
    const lifecycle = parsed.filter((message) => message.type === "flow.lifecycle");
    expect(lifecycle.map((message) => message.state)).toEqual(["response_started", "request_end"]);
    const metadata = parsed[1].metadata as { request_body: { state: string }; response_body: { state: string } };
    const bodyEnd = parsed[5].body as { state: string };
    expect([metadata.request_body.state, metadata.response_body.state, bodyEnd.state]).toEqual([
      "captured",
      "missing",
      "truncated",
    ]);
  });

  it("tolerates unknown types and additive fields without numeric coercion", () => {
    const future = parseProtocolMessage(messages.at(-1));
    expect(future.type).toBe("future.message");
    expect(future.sequence).toBe("9007199254740993");
    expect(future.future_additive_field).toBe(true);
  });

  it("rejects numeric 64-bit values", () => {
    expect(() => parseProtocolMessage({ protocol_version: "1", type: "stream.gap", expected_sequence: 1, actual_sequence: "2" })).toThrow(ProtocolError);
  });

  it("contains no secret canaries", () => {
    const serialized = JSON.stringify(messages).toLowerCase();
    for (const canary of ["authorization-canary", "x-api-key-canary", "cookie-canary", "query-secret-canary", "contract-secret"]) {
      expect(serialized).not.toContain(canary);
    }
  });
});
