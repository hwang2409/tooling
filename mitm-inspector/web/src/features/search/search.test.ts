import { describe, expect, it, vi } from "vitest";

import { fetchSearch } from "./search";

describe("body search", () => {
  it("parses result matches and truncation without coercing fields", async () => {
    const fetchMock = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify({
      matches: [
        { flow_id: "flow-a", field: "response_body", snippet: "launch notes" },
        { flow_id: "future", field: "headers", snippet: "misleading" },
        { flow_id: "bad-snippet", field: "request_body", snippet: 4 },
      ],
      truncated: true,
    }), { status: 200, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(fetchSearch("launch", new AbortController().signal)).resolves.toEqual({
      status: "results",
      query: "launch",
      matches: [{ flow_id: "flow-a", field: "response_body", snippet: "launch notes" }],
      truncated: true,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/search?q=launch&limit=200",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("accepts additive durable flow metadata without requiring it", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify({
      matches: [{
        flow_id: "durable-flow",
        field: "request_body",
        snippet: "durable hit",
        flow: {
          flow_id: "durable-flow",
          method: "POST",
          scheme: "https",
          host: "retained.example.test",
          port: "443",
          path: "/v1/messages",
          request_headers: [],
          request_body: { state: "missing" },
        },
      }],
    }), { status: 200 })));
    const result = await fetchSearch("durable", new AbortController().signal);
    expect(result.status).toBe("results");
    if (result.status === "results") expect(result.matches[0].flow?.host).toBe("retained.example.test");
  });

  it("represents empty and unavailable backend states quietly", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockResolvedValueOnce(new Response(JSON.stringify({ matches: [] }), { status: 200 })));
    await expect(fetchSearch("zz", new AbortController().signal)).resolves.toEqual({ status: "empty", query: "zz" });

    vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockRejectedValue(new Error("offline")));
    await expect(fetchSearch("zz", new AbortController().signal)).resolves.toEqual({ status: "unavailable", query: "zz" });
  });
});
