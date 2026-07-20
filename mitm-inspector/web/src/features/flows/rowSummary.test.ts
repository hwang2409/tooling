import { describe, expect, it } from "vitest";

import { deriveRowCells, durationBetween, formatBytes, formatDuration } from "./rowSummary";
import type { ImmutableFlowMetadata } from "../../state/browserState";

function flow(overrides: Record<string, unknown> = {}): ImmutableFlowMetadata {
  return {
    flow_id: "flow-a",
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: "/v1/messages",
    request_headers: [],
    request_body: { state: "empty", size_bytes: "0" },
    ...overrides,
  } as ImmutableFlowMetadata;
}

describe("row summary", () => {
  it("formats a messages flow with distinctive cells", () => {
    const cells = deriveRowCells(flow({
      started_at: "2026-01-01T12:34:56.000Z",
      ended_at: "2026-01-01T12:35:19.400Z",
      request_body_size: "475000",
      response_body_size: "6100",
      response_status: "200",
      summary: {
        kind: "anthropic_messages",
        model: "claude-sonnet-4-20250514",
        message_count: "7",
        preview: { source: "user_text", text: "find the launch notes" },
      },
    }));
    expect(cells.badge).toBe("msg");
    expect(cells.model).toBe("sonnet-4-20250514");
    expect(cells.messages).toBe("7m");
    expect(cells.preview).toEqual({ kind: "user", text: "“find the launch notes”" });
    expect(cells.sizes).toBe("475K→6.1K");
    expect(cells.duration).toBe("23.4s");
    expect(cells.status).toBe("200");
    expect(cells.isError).toBe(false);
  });

  it("uses quiet placeholders when optional enrichment is absent", () => {
    const cells = deriveRowCells(flow());
    expect(cells.time).toBe("—");
    expect(cells.model).toBe("—");
    expect(cells.messages).toBe("—");
    expect(cells.preview).toEqual({ kind: "none", text: "—" });
    expect(cells.sizes).toBe("—");
    expect(cells.duration).toBe("—");
  });

  it("keeps count-token and generic flows distinguishable", () => {
    expect(deriveRowCells(flow({ summary: { kind: "anthropic_count_tokens", count_tokens_result: "12" } })).badge).toBe("cnt");
    const generic = deriveRowCells(flow({ method: "GET" }));
    expect(generic.badge).toBe("GET api.example.test/v1/messages");
    expect(generic.preview).toEqual({ kind: "none", text: "—" });
  });

  it("does not turn invalid or negative timings into misleading values", () => {
    expect(formatBytes("not-a-number")).toBe("—");
    expect(formatDuration(-1)).toBe("—");
    expect(durationBetween("2026-01-01T00:00:01Z", "2026-01-01T00:00:00Z")).toBe("—");
  });

  it("carries rounded duration seconds into minutes", () => {
    expect(formatDuration(59_949)).toBe("59.9s");
    expect(formatDuration(59_950)).toBe("1m00s");
    expect(formatDuration(119_499)).toBe("1m59s");
    expect(formatDuration(119_500)).toBe("2m00s");
  });
});
