// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { JsonValue } from "./jsonTree";
import { ConversationView } from "./ConversationView";
import { parseAnthropicRequest } from "./anthropic";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounts: Array<{ root: ReturnType<typeof createRoot>; container: HTMLDivElement }> = [];

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("ConversationView", () => {
  it("renders each supported block and exposes unknown fields", async () => {
    const request = parseAnthropicRequest({
      model: "claude-sonnet-4-20250514",
      stream: false,
      max_tokens: 128,
      thinking: { type: "enabled", budget_tokens: 64 },
      system: [{ type: "text", text: "system instruction", cache_control: { type: "ephemeral" } }],
      tools: [{ name: "lookup", input_schema: { type: "object", properties: { q: { type: "string" } } } }],
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "hello", future_field: "kept" },
          { type: "thinking", thinking: "private reasoning", signature: "signed" },
          { type: "tool_use", id: "tool-1", name: "lookup", input: { q: "x" } },
          { type: "tool_result", tool_use_id: "tool-1", content: "result" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
          { type: "future_block", future_key: true },
        ],
      }],
    } as JsonValue);
    if (request === null) throw new Error("request fixture did not parse");

    const response = JSON.stringify({
      model: "claude-sonnet-4-20250514",
      role: "assistant",
      content: [{ type: "text", text: "done" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 3, output_tokens: 2 },
    });
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    await act(async () => root.render(<ConversationView request={request} extras={{}} responseText={response} />));

    expect(container.querySelector(".conv")?.textContent).toContain("claude-sonnet-4-20250514");
    const system = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-collapse-head"))
      .find((button) => button.textContent?.includes("system (1)"));
    expect(system).toBeDefined();
    await act(async () => system?.click());
    expect(container.textContent).toContain("system instruction");
    expect(container.textContent).toContain("lookup");
    expect(container.textContent).toContain("hello");
    expect(container.textContent).toContain("private reasoning");
    expect(container.textContent).toContain("tool_result");
    expect(container.querySelector("img")?.getAttribute("src")).toBe("data:image/png;base64,iVBORw0KGgo=");
    expect(container.textContent).toContain("done");
    expect(container.textContent).toContain("input_tokens 3");
    expect(container.textContent).toContain("future_block");

    const more = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-more .conv-collapse-head"))
      .find((button) => button.textContent?.includes("more field"));
    expect(more).toBeDefined();
    await act(async () => more?.click());
    expect(container.textContent).toContain("future_field");
  });
});
