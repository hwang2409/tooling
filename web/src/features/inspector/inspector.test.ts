import { Buffer } from "node:buffer";
import { describe, expect, it } from "vitest";

import type { FlowLifecycle } from "../../protocol";
import { bodyMetadata, decodeBase64Bounded, decodeBody, decodeUtf8, gateBodyDecode, hexDump, parseSseEvents } from "./decoders";
import { lifecyclePhase, orderLifecycle } from "./lifecycle";

const encoded = (value: string) => Buffer.from(value, "utf8").toString("base64");

describe("bounded body decoding", () => {
  it("decodes valid base64 without allocating beyond the requested cap", () => {
    const result = decodeBase64Bounded(encoded("0123456789"), 4);
    expect(Array.from(result.bytes)).toEqual([48, 49, 50, 51]);
    expect(result.truncated).toBe(true);
    expect(result.invalid).toBe(false);
  });

  it("rejects malformed base64 and falls back for invalid UTF-8", () => {
    expect(decodeBase64Bounded("not?base64=").invalid).toBe(true);
    const utf8 = decodeUtf8(new Uint8Array([0xff, 0xfe]));
    expect(utf8.invalid).toBe(true);
    expect(utf8.text).toContain("00000000");
  });

  it("pretty-prints JSON and keeps a deterministic text fallback", () => {
    const json = decodeBody({ state: "captured", size_bytes: "7", encoding: "base64", data: encoded('{"a":1}') }, "json");
    expect(json.text).toBe('{\n  "a": 1\n}');
    expect(json.copyText).toBe(json.text);

    const fallback = decodeBody({ state: "captured", size_bytes: "8", encoding: "base64", data: encoded("not-json") }, "json");
    expect(fallback.fallback).toBe("text");
    expect(fallback.text).toContain("Not valid JSON");
    expect(fallback.copyText).toBe("not-json");
  });

  it("marks source and render caps as truncation", () => {
    const result = decodeBody({ state: "truncated", size_bytes: "999", captured_bytes: "10", encoding: "base64", data: encoded("0123456789") }, "text", 4);
    expect(result.truncated).toBe(true);
    expect(result.byteLength).toBe(4);
    expect(bodyMetadata({ state: "truncated", size_bytes: "999", captured_bytes: "10", encoding: "base64", data: encoded("0123456789") }).captured).toBe("10 B");
  });

  it("handles missing, empty, redacted, and invalid body states visibly", () => {
    expect(bodyMetadata({ state: "missing" }).state).toBe("missing");
    expect(bodyMetadata({ state: "empty", size_bytes: "0" }).size).toBe("0 B");
    expect(bodyMetadata({ state: "redacted", reason: "policy" }).state).toBe("redacted");
    expect(decodeBody({ state: "redacted" }, "text").text).toContain("not available");
    expect(decodeBody({ state: "captured", size_bytes: "3", encoding: "base64", data: "wat?" }, "text").invalidEncoding).toBe(true);
  });

  it("renders bounded hex with stable offsets", () => {
    expect(hexDump(new Uint8Array([0, 31, 32, 65, 255]))).toContain("00000000");
    expect(hexDump(new Uint8Array([0, 31, 32, 65, 255]))).toContain("00 1f 20 41 ff");
    const bounded = decodeBody({ state: "captured", size_bytes: "5000", encoding: "base64", data: encoded("x".repeat(5000)) }, "hex");
    expect(bounded.byteLength).toBe(4096);
    expect(bounded.truncated).toBe(true);
  });

  it("does not decode a body until the pane is explicitly selected", () => {
    const body = { state: "captured" as const, size_bytes: "5", encoding: "base64" as const, data: encoded("hello") };
    const gated = gateBodyDecode(body, false, "text");
    expect(gated.selected).toBe(false);
    if (!gated.selected) expect(gated).not.toHaveProperty("decoded");
    const opened = gateBodyDecode(body, true, "text");
    expect(opened.selected).toBe(true);
    if (opened.selected) expect(opened.decoded.text).toBe("hello");
  });
});

describe("SSE framing", () => {
  it("preserves comments, multiline data, event IDs, retry, and unknown fields", () => {
    const events = parseSseEvents(": keepalive\nid: 7\nevent: message\ndata: first\ndata: second\nretry: 1500\nextension: yes\n\n");
    expect(events).toEqual([{
      data: "first\nsecond",
      event: "message",
      id: "7",
      retry: 1500,
      comments: ["keepalive"],
      fields: [
        { name: "id", value: "7" },
        { name: "event", value: "message" },
        { name: "data", value: "first" },
        { name: "data", value: "second" },
        { name: "retry", value: "1500" },
        { name: "extension", value: "yes" },
      ],
    }]);
  });

  it("accepts an event without a colon and drops NUL-containing IDs", () => {
    const events = parseSseEvents("id: bad\0id\ndata\n\n");
    expect(events[0]?.id).toBeUndefined();
    expect(events[0]?.fields).toEqual([{ name: "id", value: "bad\0id" }, { name: "data", value: "" }]);
  });
});

describe("lifecycle ordering", () => {
  const event = (state: FlowLifecycle["state"], sequence: string, index: number): FlowLifecycle => ({
    protocol_version: "1",
    type: "flow.lifecycle",
    source_id: "source",
    flow_id: "flow",
    event_id: `event-${index}`,
    occurred_at: `2026-07-17T12:00:0${index}.000Z`,
    sequence,
    state,
  });

  it("follows observed sequence even when response starts before request end", () => {
    const ordered = orderLifecycle([event("request_end", "9", 2), event("response_started", "8", 1), event("response_end", "10", 3)]);
    expect(ordered.map((item) => item.state)).toEqual(["response_started", "request_end", "response_end"]);
    expect(lifecyclePhase(ordered)).toEqual({ requestEnded: true, responseStarted: true, completed: false, errored: false });
  });
});
