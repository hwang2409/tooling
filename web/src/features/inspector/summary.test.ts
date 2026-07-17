import { describe, expect, it } from "vitest";

import type { InspectorFlow } from "./models";
import { flowSummary } from "./summary";

function toBase64(text: string): string {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function byteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

function flow(overrides: Partial<InspectorFlow> = {}): InspectorFlow {
  const base: InspectorFlow = {
    metadata: {
      flow_id: "flow-1",
      method: "GET",
      scheme: "https",
      host: "api.example.test",
      port: "443",
      path: "/things",
      request_headers: [],
      request_body: { state: "missing" },
    },
    lifecycle: [],
  };
  return { ...base, ...overrides, metadata: { ...base.metadata, ...(overrides.metadata ?? {}) } };
}

describe("flowSummary", () => {
  it("assembles the generic shape as method path · status · duration with no host or bytes", () => {
    const summary = flowSummary(flow({
      metadata: {
        flow_id: "flow-1", method: "GET", scheme: "https", host: "api.example.test", port: "443", path: "/things",
        request_headers: [], request_body: { state: "empty", size_bytes: "0" },
        response_body: { state: "captured", size_bytes: "1024", encoding: "base64", data: "" },
        response_status: "200",
      },
      lifecycle: [
        { protocol_version: "1", type: "flow.lifecycle", source_id: "s", flow_id: "flow-1", event_id: "e1", occurred_at: "2026-01-01T00:00:00.000Z", sequence: "1", state: "request_started" },
        { protocol_version: "1", type: "flow.lifecycle", source_id: "s", flow_id: "flow-1", event_id: "e2", occurred_at: "2026-01-01T00:00:00.250Z", sequence: "2", state: "response_end" },
      ],
    }));
    expect(summary).toBe("GET /things · 200 · 250ms");
    expect(summary).not.toContain("api.example.test");
    expect(summary).not.toContain("in / ");
    expect(summary).not.toContain("out");
  });

  it("recognises Anthropic /v1/messages, extracts model, message count, and stream flag", () => {
    const requestJson = JSON.stringify({ model: "claude-sonnet-4-6", messages: [{ role: "user", content: "hi" }, { role: "user", content: "again" }], stream: true });
    const requestBase64 = toBase64(requestJson);
    const requestSize = String(byteLength(requestJson));
    const responseJson = "data: {\"type\":\"message_start\"}\n\n";
    const responseBase64 = toBase64(responseJson);
    const responseSize = String(byteLength(responseJson));

    const summary = flowSummary(flow({
      metadata: {
        flow_id: "flow-1", method: "POST", scheme: "https", host: "api.anthropic.com", port: "443", path: "/v1/messages",
        request_headers: [], request_body: { state: "captured", size_bytes: requestSize, encoding: "base64", data: requestBase64, content_type: "application/json" },
        response_body: { state: "captured", size_bytes: responseSize, encoding: "base64", data: responseBase64, content_type: "text/event-stream" },
        response_status: "200",
      },
      lifecycle: [
        { protocol_version: "1", type: "flow.lifecycle", source_id: "s", flow_id: "flow-1", event_id: "e1", occurred_at: "2026-01-01T00:00:00.000Z", sequence: "1", state: "request_started" },
        { protocol_version: "1", type: "flow.lifecycle", source_id: "s", flow_id: "flow-1", event_id: "e2", occurred_at: "2026-01-01T00:00:01.400Z", sequence: "2", state: "response_end" },
      ],
    }));
    expect(summary.startsWith("POST v1/messages")).toBe(true);
    expect(summary).toContain("claude-sonnet-4-6");
    expect(summary).toContain("2 msgs");
    expect(summary).toContain("stream");
    expect(summary).toContain("1.4s");
    expect(summary).toContain("200");
  });

  it("marks errored flows with the err status token when no status code was captured", () => {
    const summary = flowSummary(flow({
      lifecycle: [
        { protocol_version: "1", type: "flow.lifecycle", source_id: "s", flow_id: "flow-1", event_id: "e1", occurred_at: "2026-01-01T00:00:00Z", sequence: "1", state: "error" },
      ],
    }));
    expect(summary).toContain("err");
    expect(summary).not.toContain("api.example.test");
  });

  it("omits the status segment entirely from the generic shape when unknown and no error observed", () => {
    const summary = flowSummary(flow({
      metadata: { flow_id: "flow-1", method: "GET", scheme: "https", host: "api.example.test", port: "443", path: "/things", request_headers: [], request_body: { state: "missing" } },
      lifecycle: [
        { protocol_version: "1", type: "flow.lifecycle", source_id: "s", flow_id: "flow-1", event_id: "e1", occurred_at: "2026-01-01T00:00:00Z", sequence: "1", state: "request_started" },
      ],
    }));
    expect(summary).toBe("GET /things · —");
  });

  it("falls back to the generic shape when the request body is missing", () => {
    const summary = flowSummary(flow({
      metadata: {
        flow_id: "flow-1", method: "POST", scheme: "https", host: "api.anthropic.com", port: "443", path: "/v1/messages",
        request_headers: [], request_body: { state: "missing" },
      },
    }));
    expect(summary.startsWith("POST v1/messages")).toBe(true);
    expect(summary).not.toContain("msg");
    expect(summary).not.toContain("model");
  });
});
