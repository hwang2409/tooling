import { describe, expect, it } from "vitest";

import type { JsonValue } from "./jsonTree";
import { assembleSse, parseAnthropicRequest, parseAnthropicResponse, splitSseFrames } from "./anthropic";

describe("anthropic conversation model", () => {
  it("normalises shorthand content and preserves unknown fields", () => {
    const request: JsonValue = {
      model: "claude-sonnet-4-20250514",
      stream: true,
      output_config: { effort: "high", format: { type: "json_schema", schema: { type: "object" } }, future: true },
      system: [{ type: "text", text: "be concise", cache_control: { type: "ephemeral" } }],
      tools: [{ name: "lookup", description: "look things up", input_schema: { type: "object" }, vendor_extension: true }],
      messages: [{ role: "user", content: "hello", future_message_key: { keep: true } }],
      future_request_key: "still reachable",
    };
    const parsed = parseAnthropicRequest(request);
    expect(parsed?.messages[0].blocks[0]).toMatchObject({ kind: "text", text: "hello" });
    expect(parsed?.messages[0].extra).toEqual([["future_message_key", { keep: true }]]);
    expect(parsed?.extra).toEqual([["future_request_key", "still reachable"]]);
    expect(parsed?.effort).toBe("high");
    expect(parsed?.outputConfig).toEqual({ effort: "high", format: { type: "json_schema", schema: { type: "object" } }, future: true });
    expect(parsed?.tools[0].raw).toEqual({
      name: "lookup",
      description: "look things up",
      input_schema: { type: "object" },
      vendor_extension: true,
    });
  });

  it("keeps malformed and future members as raw inspectable values", () => {
    const parsed = parseAnthropicRequest({
      system: null,
      tools: [null],
      messages: [null, {
        role: "user",
        content: [
          null,
          { type: "future_block", future: true },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "a", future_source: { keep: true } } },
        ],
      }],
      max_tokens: "not-a-number",
      output_config: "future-format-shape",
    } as JsonValue);
    expect(parsed?.system[0]).toEqual({ kind: "unknown", raw: null });
    expect(parsed?.tools[0].raw).toBeNull();
    expect(parsed?.messages[0].blocks[0]).toEqual({ kind: "unknown", raw: null });
    expect(parsed?.messages[1].blocks[0]).toEqual({ kind: "unknown", raw: null });
    expect(parsed?.messages[1].blocks[1]).toEqual({ kind: "unknown", raw: { type: "future_block", future: true } });
    expect(parsed?.messages[1].blocks[2]).toMatchObject({ kind: "image", extra: [["source", { type: "base64", media_type: "image/png", data: "a", future_source: { keep: true } }]] });
    expect(parsed?.outputConfig).toBe("future-format-shape");
    expect(parsed?.extra).toEqual([["max_tokens", "not-a-number"]]);

    const extendedThinking = { type: "enabled", budget_tokens: 64, future_thinking: { marker: "thinking-marker" } };
    const thinkingRequest = parseAnthropicRequest({
      thinking: extendedThinking,
      system: [{ type: "text", text: "system" }],
      messages: [{ role: "user", content: "hello" }],
    } as JsonValue);
    expect(thinkingRequest?.thinking).toEqual(extendedThinking);

    const response = parseAnthropicResponse({
      model: "response-model-marker",
      role: "response-role-marker",
      content: [{ type: "text", text: "done" }],
      stop_reason: "end_turn",
    } as JsonValue);
    expect(response?.model).toBe("response-model-marker");
    expect(response?.role).toBe("response-role-marker");

    const malformedResponse = parseAnthropicResponse({
      model: 4,
      role: { future: true },
      content: [{ type: "text", text: "done" }],
      stop_reason: null,
      usage: { input_tokens: 3, future_usage: { keep: true } },
    } as JsonValue);
    expect(malformedResponse?.extra).toEqual([
      ["model", 4],
      ["role", { future: true }],
      ["stop_reason", null],
      ["usage", { input_tokens: 3, future_usage: { keep: true } }],
    ]);
  });

  it("reassembles thinking, text, and tool input SSE deltas while retaining raw frames", () => {
    const sse = [
      'event: message_start\ndata: {"type":"message_start","message":{"model":"claude","role":"assistant","usage":{"input_tokens":4}}}',
      'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}',
      'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}',
      'event: content_block_start\ndata: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool-1","name":"lookup","input":{}}}',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":\\"x\\"}"}}',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
      'event: message_stop\ndata: {"type":"message_stop"}',
      'event: content_block_delta\ndata: {"type":"content_block_delta"',
    ].join("\n\n");
    const frames = splitSseFrames(sse);
    expect(frames.at(-1)?.complete).toBe(false);
    const assembled = assembleSse(sse);
    expect(assembled.model).toBe("claude");
    expect(assembled.stopReason).toBe("end_turn");
    expect(assembled.usage).toEqual({ input_tokens: 4, output_tokens: 2 });
    expect(assembled.blocks).toMatchObject([
      { kind: "thinking", thinking: "reason", signature: "sig" },
      { kind: "text", text: "answer" },
      { kind: "tool_use", input: { q: "x" } },
    ]);
    expect(assembled.frames).toHaveLength(11);
  });
});
