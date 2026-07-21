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
      output_config: { effort: "high", format: { type: "json_schema", schema: { type: "object" } } },
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
    expect(container.textContent).toContain("output_config");
    const outputConfig = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-collapse-head"))
      .find((button) => button.textContent?.includes("output_config"));
    expect(outputConfig).toBeDefined();
    await act(async () => outputConfig?.click());
    const outputTreeToggle = outputConfig?.parentElement?.querySelector<HTMLButtonElement>(".json-toggle");
    expect(outputTreeToggle).toBeDefined();
    await act(async () => outputTreeToggle?.click());
    const outputNestedToggles = outputConfig?.parentElement?.querySelectorAll<HTMLButtonElement>(".json-toggle");
    await act(async () => outputNestedToggles?.[outputNestedToggles.length - 1]?.click());
    expect(container.textContent).toContain("json_schema");
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

  it("renders message text as markdown by default with the plain text one toggle away", async () => {
    const request = parseAnthropicRequest({
      model: "claude",
      messages: [
        { role: "user", content: [{ type: "text", text: "# Heading\n\nwith **bold** text" }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "# not markdown" }] },
      ],
    } as JsonValue);
    if (request === null) throw new Error("request fixture did not parse");
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    await act(async () => root.render(<ConversationView request={request} extras={{}} />));

    expect(container.querySelector(".conv-message h1")?.textContent).toBe("Heading");
    expect(container.querySelector(".conv-message strong")?.textContent).toBe("bold");
    // Tool output is data, not prose: it must stay literal even in markdown mode.
    expect(container.querySelector(".conv-card-tool-result h1")).toBeNull();
    expect(container.querySelector(".conv-card-tool-result")?.textContent).toContain("# not markdown");

    const plain = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-text-modes .packet-mode"))
      .find((button) => button.textContent === "plain");
    expect(plain).toBeDefined();
    await act(async () => plain?.click());
    // No information loss: the exact source text is one toggle away.
    expect(container.querySelector(".conv-message h1")).toBeNull();
    expect(container.textContent).toContain("# Heading\n\nwith **bold** text");
  });

  it("renders tool_use as a labeled <pre><code> code block with pretty-printed JSON", async () => {
    const request = parseAnthropicRequest({
      model: "claude",
      messages: [{
        role: "assistant",
        content: [{ type: "tool_use", id: "toolu_A1", name: "bash", input: { command: "echo hi", timeout: 5 } }],
      }],
    } as JsonValue);
    if (request === null) throw new Error("request fixture did not parse");
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    await act(async () => root.render(<ConversationView request={request} extras={{}} />));

    const card = container.querySelector(".conv-card-tool-use");
    expect(card).not.toBeNull();
    // Caption line: kind + tool name.
    expect(card?.querySelector(".conv-card-kind")?.textContent).toBe("tool_use");
    expect(card?.querySelector(".conv-card-name")?.textContent).toBe("bash");
    // Body is a real <pre><code>, not a JsonTree.
    const pre = card?.querySelector("pre.conv-pre");
    expect(pre).not.toBeNull();
    expect(pre?.querySelector("code")).not.toBeNull();
    expect(card?.querySelector(".json-tree")).toBeNull();
    // Pretty-printed JSON: 2-space indent, one key per line.
    const body = pre?.textContent ?? "";
    expect(body).toContain('"command": "echo hi"');
    expect(body).toContain('"timeout": 5');
    expect(body).toMatch(/\n {2}"command"/);
  });

  it("renders tool_result payload as a labeled <pre><code> code block", async () => {
    const request = parseAnthropicRequest({
      model: "claude",
      messages: [{
        role: "user",
        content: [{ type: "tool_use", id: "toolu_B2", name: "bash", input: { command: "ls" } }, {
          type: "tool_result",
          tool_use_id: "toolu_B2",
          content: [{ type: "text", text: "line one\nline two" }],
        }],
      }],
    } as JsonValue);
    if (request === null) throw new Error("request fixture did not parse");
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    await act(async () => root.render(<ConversationView request={request} extras={{}} />));

    const card = container.querySelector(".conv-card-tool-result");
    expect(card).not.toBeNull();
    // Caption line: kind + linked tool name.
    expect(card?.querySelector(".conv-card-kind")?.textContent).toBe("↳ tool_result");
    expect(card?.querySelector(".conv-card-name")?.textContent).toBe("bash");
    // Payload is a real <pre><code> preserving whitespace, not prose.
    const pre = card?.querySelector("pre.conv-pre");
    expect(pre).not.toBeNull();
    expect(pre?.querySelector("code")?.textContent).toBe("line one\nline two");
    expect(card?.querySelector(".conv-prose")).toBeNull();
  });

  it("keeps lifted, malformed, and future conversation values visible", async () => {
    const request = parseAnthropicRequest({
      model: "request-model",
      thinking: { type: "enabled", budget_tokens: 64, future_thinking: "thinking-marker" },
      system: null,
      messages: [{
        role: "user",
        content: [{ type: "future_block", future_block_marker: "future-block-marker" }],
        future_message: "future-message-marker",
      }],
    } as JsonValue);
    if (request === null) throw new Error("request fixture did not parse");

    const response = JSON.stringify({
      model: "response-model-marker",
      role: "response-role-marker",
      content: [{ type: "text", text: "done" }],
      future_response: "future-response-marker",
    });
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    await act(async () => root.render(<ConversationView request={request} extras={{}} responseText={response} />));

    const thinking = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-collapse-head"))
      .find((button) => button.textContent?.includes("thinking details"));
    expect(thinking).toBeDefined();
    await act(async () => thinking?.click());
    const system = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-collapse-head"))
      .find((button) => button.textContent?.includes("system (1)"));
    expect(system).toBeDefined();
    await act(async () => system?.click());

    for (let attempt = 0; attempt < 10; attempt += 1) {
      const closedMore = Array.from(container.querySelectorAll<HTMLButtonElement>(".conv-more .conv-collapse-head"))
        .find((button) => button.getAttribute("aria-expanded") === "false");
      if (closedMore === undefined) break;
      await act(async () => closedMore.click());
    }

    expect(container.textContent).toContain("thinking-marker");
    expect(container.textContent).toContain("null");
    expect(container.textContent).toContain("response-model-marker");
    expect(container.textContent).toContain("response-role-marker");
    expect(container.textContent).toContain("future-block-marker");
    expect(container.textContent).toContain("future-message-marker");
    expect(container.textContent).toContain("future-response-marker");
  });
});
