// @vitest-environment jsdom

import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { JsonTree, collapsedSummary, safeParseJson, shouldStartCollapsed } from "./jsonTree";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounts: Array<{ root: ReturnType<typeof createRoot>; container: HTMLDivElement }> = [];

async function mount(node: ReactNode): Promise<HTMLDivElement> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => root.render(node));
  mounts.push({ root, container });
  return container;
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("safeParseJson", () => {
  it("returns the parsed value when the input is valid JSON", () => {
    expect(safeParseJson('{"a":1}')).toEqual({ ok: true, value: { a: 1 } });
  });
  it("surfaces the parser error when the input is not JSON", () => {
    const result = safeParseJson("not json");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error.length).toBeGreaterThan(0);
  });
});

describe("shouldStartCollapsed", () => {
  it("keeps small shapes expanded", () => {
    expect(shouldStartCollapsed({ a: 1, b: 2 })).toBe(false);
    expect(shouldStartCollapsed([1, 2, 3])).toBe(false);
  });
  it("collapses deeply nested structures", () => {
    expect(shouldStartCollapsed({ a: { b: { c: { d: 1 } } } })).toBe(true);
  });
  it("collapses large arrays and objects", () => {
    expect(shouldStartCollapsed(Array.from({ length: 20 }, (_, index) => index))).toBe(true);
    const wideObject = Object.fromEntries(Array.from({ length: 20 }, (_, index) => [`k${index}`, index]));
    expect(shouldStartCollapsed(wideObject)).toBe(true);
  });
});

describe("collapsedSummary", () => {
  it("counts array items and object keys", () => {
    expect(collapsedSummary([1, 2, 3])).toBe("[ 3 items ]");
    expect(collapsedSummary([1])).toBe("[ 1 item ]");
    expect(collapsedSummary({ a: 1, b: 2 })).toBe("{ 2 keys }");
    expect(collapsedSummary({ a: 1 })).toBe("{ 1 key }");
  });
});

describe("<JsonTree>", () => {
  it("renders keys and quoted string scalars", async () => {
    const container = await mount(<JsonTree value={{ greeting: "hi", n: 3 }} />);
    expect(container.textContent).toContain("\"greeting\"");
    expect(container.textContent).toContain("\"hi\"");
    expect(container.textContent).toContain("3");
    expect(container.querySelector(".json-key")?.textContent).toBe("\"greeting\"");
  });

  it("toggles children when the collapse button is clicked", async () => {
    const container = await mount(<JsonTree value={{ nested: { a: 1 } }} />);
    const toggle = container.querySelector<HTMLButtonElement>(".json-toggle");
    expect(toggle).toBeTruthy();
    expect(container.textContent).toContain("\"a\"");
    await act(async () => toggle?.click());
    expect(toggle?.getAttribute("aria-expanded")).toBe("false");
  });

  it("renders empty containers on a single line", async () => {
    const container = await mount(<JsonTree value={{ items: [] }} />);
    expect(container.textContent).toContain("[]");
  });
});
