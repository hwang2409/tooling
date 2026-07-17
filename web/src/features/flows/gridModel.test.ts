import { describe, expect, it } from "vitest";

import type { ImmutableFlowLifecycle, ImmutableFlowMetadata } from "../../state/browserState";
import { buildFlowRow } from "./gridModel";

function metadata(overrides: Partial<ImmutableFlowMetadata> = {}): ImmutableFlowMetadata {
  return {
    flow_id: "flow-1",
    method: "GET",
    scheme: "https",
    host: "api.example.test",
    port: "8443",
    path: "/v1/messages?cursor=1",
    request_headers: [{ name: "accept", value: "application/json" }],
    request_body: { state: "missing" },
    ...overrides,
  };
}

function lifecycle(state: ImmutableFlowLifecycle["state"], sequence: string): ImmutableFlowLifecycle {
  return {
    protocol_version: "1",
    type: "flow.lifecycle",
    source_id: "src",
    flow_id: "flow-1",
    event_id: `event-${sequence}`,
    occurred_at: "2026-01-01T00:00:00Z",
    sequence,
    state,
  };
}

describe("buildFlowRow", () => {
  it("builds the full url with explicit port and keeps decimal-string ports intact", () => {
    const row = buildFlowRow(metadata({ port: "18446744073709551615" }));
    expect(row.url).toBe("https://api.example.test:18446744073709551615/v1/messages?cursor=1");
    expect(row.port).toBe("18446744073709551615");
  });

  it("shows awaiting phase when no response evidence exists", () => {
    const row = buildFlowRow(metadata(), [lifecycle("request_started", "1"), lifecycle("request_end", "2")]);
    expect(row.phase).toBe("request");
    expect(row.filterable.hasResponse).toBe(false);
  });

  it("marks response phase from lifecycle even when response starts before request end", () => {
    const row = buildFlowRow(metadata(), [lifecycle("request_started", "1"), lifecycle("response_started", "2")]);
    expect(row.phase).toBe("response");
    expect(row.filterable.hasResponse).toBe(true);
  });

  it("marks response phase from metadata response fields without lifecycle", () => {
    const row = buildFlowRow(metadata({ response_headers: [] }));
    expect(row.phase).toBe("response");
  });

  it("prefers error over completion regardless of observed order", () => {
    const row = buildFlowRow(metadata(), [
      lifecycle("flow_completed", "9"),
      lifecycle("error", "3"),
      lifecycle("response_end", "8"),
    ]);
    expect(row.phase).toBe("error");
    expect(row.filterable.errored).toBe(true);
  });

  it("marks completed flows as done", () => {
    const row = buildFlowRow(metadata(), [lifecycle("response_end", "8"), lifecycle("flow_completed", "9")]);
    expect(row.phase).toBe("complete");
    expect(row.phaseLabel).toBe("done");
  });

  it("formats body cells with BigInt-exact sizes and truncation markers", () => {
    const row = buildFlowRow(metadata({
      request_body: { state: "captured", size_bytes: "18446744063223267327", encoding: "base64", data: "" },
      response_body: { state: "truncated", size_bytes: "2097152", captured_bytes: "1048576", encoding: "base64", data: "" },
    }));
    expect(row.requestBody).toEqual({ text: "17592186034415 MiB", truncated: false });
    expect(row.responseBody).toEqual({ text: "2 MiB", truncated: true });
  });

  it("shows a dash for missing and absent bodies", () => {
    const row = buildFlowRow(metadata());
    expect(row.requestBody.text).toBe("—");
    expect(row.responseBody.text).toBe("—");
  });

  it("derives content type from body descriptors first, then headers, response side preferred", () => {
    const fromBodies = buildFlowRow(metadata({
      request_body: { state: "empty", size_bytes: "0", content_type: "application/json; charset=utf-8" },
      response_body: { state: "empty", size_bytes: "0", content_type: "text/event-stream" },
    }));
    expect(fromBodies.contentType).toBe("text/event-stream");
    expect(fromBodies.filterable.requestContentType).toBe("application/json");

    const fromHeaders = buildFlowRow(metadata({
      request_headers: [{ name: "Content-Type", value: "application/xml; charset=utf-8" }],
    }));
    expect(fromHeaders.contentType).toBe("application/xml");

    const none = buildFlowRow(metadata());
    expect(none.contentType).toBe("—");
    expect(none.filterable.requestContentType).toBeNull();
    expect(none.filterable.responseContentType).toBeNull();
  });
});
