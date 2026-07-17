import { describe, expect, it } from "vitest";

import { overridesFromDetailMessages } from "./flowDetail";

function metadataMessage(withResponse: boolean, path = "/v1/messages") {
  return {
    protocol_version: "1",
    type: "flow.metadata",
    metadata: {
      flow_id: "flow-1",
      method: "POST",
      scheme: "https",
      host: "api.example.test",
      port: "443",
      path,
      request_headers: [{ name: "authorization", value: "[REDACTED]" }],
      request_body: { state: "captured", size_bytes: "4", encoding: "base64", data: "Ym9keQ==" },
      ...(withResponse
        ? {
            response_headers: [{ name: "content-type", value: "application/json" }],
            response_body: { state: "empty", size_bytes: "0" },
          }
        : {}),
    },
  };
}

function bodyEndMessage(side: "request" | "response", data: string) {
  return {
    protocol_version: "1",
    type: "body.end",
    flow_id: "flow-1",
    body_side: side,
    total_bytes: "4",
    body: { state: "captured", size_bytes: "4", encoding: "base64", data },
  };
}

describe("overridesFromDetailMessages", () => {
  it("uses the newest metadata for headers and bodies", () => {
    const overrides = overridesFromDetailMessages([
      metadataMessage(false),
      metadataMessage(true, "/v1/updated"),
    ]);
    expect(overrides.request_headers?.[0].name).toBe("authorization");
    expect(overrides.response_headers?.[0].value).toBe("application/json");
    expect(overrides.request_body).toEqual({ state: "captured", size_bytes: "4", encoding: "base64", data: "Ym9keQ==" });
    expect(overrides.response_body).toEqual({ state: "empty", size_bytes: "0" });
  });

  it("prefers terminal body.end descriptors over metadata bodies", () => {
    const overrides = overridesFromDetailMessages([
      metadataMessage(true),
      bodyEndMessage("response", "ZnVsbA=="),
    ]);
    expect(overrides.response_body).toEqual({
      state: "captured",
      size_bytes: "4",
      encoding: "base64",
      data: "ZnVsbA==",
    });
  });

  it("ignores invalid entries and returns empty overrides when nothing parses", () => {
    const overrides = overridesFromDetailMessages([{ nonsense: true }, 42, null]);
    expect(overrides).toEqual({});
  });
});
