import { Buffer } from "node:buffer";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { FlowLifecycle } from "../../protocol";
import * as decoderModule from "./decoders";
import { bodyFocusTarget, InspectorBodyPanel, isBodySelectionAuthorized, nextBodyTabIndex, PairedInspector } from "./PairedInspector";
import { bodyMetadata, decodeBase64Bounded, decodeBody, decodeUtf8, gateBodyDecode, hexDump, parseSseEvents } from "./decoders";
import { lifecyclePhase, orderLifecycle } from "./lifecycle";
import type { InspectableBody, InspectorFlow } from "./models";

const encoded = (value: string) => Buffer.from(value, "utf8").toString("base64");

describe("bounded body decoding", () => {
  it("decodes valid base64 without allocating beyond the requested cap", () => {
    const result = decodeBase64Bounded(encoded("0123456789"), 4);
    expect(Array.from(result.bytes)).toEqual([48, 49, 50, 51]);
    expect(result.truncated).toBe(true);
    expect(result.invalid).toBe(false);
  });

  it("bounds multi-megabyte valid base64 before validation and decoding", () => {
    const multiMegabyte = Buffer.alloc(8 * 1024 * 1024, 0x78).toString("base64");
    expect(() => decodeBase64Bounded(multiMegabyte)).not.toThrow();
    const result = decodeBase64Bounded(multiMegabyte);
    expect(result.bytes.byteLength).toBe(64 * 1024);
    expect(result.truncated).toBe(true);
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

  it("does not dispatch comment/id/retry-only blocks, persists ID and retry, and drops EOF data", () => {
    expect(parseSseEvents(": heartbeat\n\n")).toEqual([]);
    expect(parseSseEvents("id: 7\n\nretry: 1500\n\ndata: complete\n\n")).toEqual([expect.objectContaining({ data: "complete", id: "7", retry: 1500 })]);
    expect(parseSseEvents("id: 7\ndata: incomplete")).toEqual([]);
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

  it("preserves observed input order for equal sequences", () => {
    const ordered = orderLifecycle([event("response_body", "12", 2), event("request_body", "12", 1)]);
    expect(ordered.map((item) => item.state)).toEqual(["response_body", "request_body"]);
  });
});

describe("PairedInspector rendering and interaction contracts", () => {
  const capturedBody: InspectableBody = { state: "captured", size_bytes: "5", encoding: "base64", data: encoded("hello") };
  const flow: InspectorFlow = {
    metadata: {
      flow_id: "flow-a",
      method: "POST",
      scheme: "https",
      host: "api.example.test",
      port: "443",
      path: "/trace/<script>alert(1)</script>",
      request_headers: [
        { name: "x-trace", value: "first" },
        { name: "x-trace", value: "[REDACTED]" },
        { name: "x-empty", value: "" },
      ],
      request_body: capturedBody,
    },
    response_body: { state: "redacted", reason: "policy" },
    error: "<img src=x onerror=alert(1)>",
  };

  it("renders the initial flow with metadata and gated body content", () => {
    const markup = renderToStaticMarkup(createElement(PairedInspector, { flow }));
    expect(markup).toContain("Body decoding is paused");
    expect(markup).not.toContain(">hello<");
    expect(markup).toContain("x-trace");
    expect(markup).toContain("[REDACTED]");
    expect(markup).toContain("empty value");
    expect(markup).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
    expect(markup).not.toContain("<script>alert(1)</script>");
  });

  it("renders selected body output through the component and exposes complete body tabs", () => {
    const decodeSpy = vi.spyOn(decoderModule, "decodeBody");
    const markup = renderToStaticMarkup(createElement(PairedInspector, { flow, bodySelection: { flowId: "flow-a", pane: "request" } }));
    expect(decodeSpy).toHaveBeenCalledWith(capturedBody, "text");
    expect(markup).toContain(">hello</pre>");
    expect(markup).toContain('role="tablist"');
    expect(markup).toContain('aria-orientation="horizontal"');
    expect(markup).toContain('role="tabpanel"');
    expect(markup).toContain('aria-controls=');
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('tabindex="-1"');
    decodeSpy.mockRestore();
  });

  it("gates synchronously when the flow ID changes under a stale selection", () => {
    const nextFlow: InspectorFlow = { ...flow, metadata: { ...flow.metadata, flow_id: "flow-b" } };
    const markup = renderToStaticMarkup(createElement(PairedInspector, { flow: nextFlow, bodySelection: { flowId: "flow-a", pane: "request" } }));
    expect(markup).toContain("Body decoding is paused");
    expect(markup).not.toContain(">hello</pre>");
  });

  it("keeps redacted, empty, and missing states explicit in rendered body panels", () => {
    for (const body of [{ state: "redacted" as const }, { state: "empty" as const, size_bytes: "0" as const }, { state: "missing" as const }]) {
      const markup = renderToStaticMarkup(createElement(InspectorBodyPanel, { body, pane: "response", selected: false, onSelect: () => undefined }));
      expect(markup).toMatch(/is-(redacted|empty|missing)/);
    }
  });

  it("authorizes selection by both current flow ID and body pane", () => {
    const selection = { flowId: "flow-a", pane: "request" as const };
    expect(isBodySelectionAuthorized(selection, "flow-a", "request")).toBe(true);
    expect(isBodySelectionAuthorized(selection, "flow-b", "request")).toBe(false);
    expect(isBodySelectionAuthorized(selection, "flow-a", "response")).toBe(false);
    expect(isBodySelectionAuthorized(null, "flow-a", "request")).toBe(false);
  });

  it("uses horizontal roving-tab keyboard behavior and deterministic focus recovery", () => {
    expect(nextBodyTabIndex(0, "ArrowRight", 4)).toBe(1);
    expect(nextBodyTabIndex(0, "ArrowLeft", 4)).toBe(3);
    expect(nextBodyTabIndex(0, "ArrowDown", 4)).toBeUndefined();
    expect(nextBodyTabIndex(1, "Home", 4)).toBe(0);
    expect(nextBodyTabIndex(1, "End", 4)).toBe(3);
    expect(nextBodyTabIndex(0, "ArrowDown", 4, "vertical")).toBe(1);
    expect(nextBodyTabIndex(0, "ArrowRight", 4, "vertical")).toBeUndefined();
    expect(bodyFocusTarget(false, true)).toBe("active-tab");
    expect(bodyFocusTarget(true, false)).toBe("inspect-control");
    expect(bodyFocusTarget(false, false)).toBeUndefined();
  });
});
