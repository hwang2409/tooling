// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { MarkdownProse } from "./Markdown";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounts: Array<{ root: ReturnType<typeof createRoot>; container: HTMLDivElement }> = [];

async function mountMarkdown(text: string): Promise<HTMLDivElement> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mounts.push({ root, container });
  await act(async () => root.render(<MarkdownProse text={text} />));
  return container;
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("MarkdownProse", () => {
  it("renders headings, lists, emphasis, inline code, fenced code, and links", async () => {
    const container = await mountMarkdown(
      [
        "# Title",
        "",
        "Some **bold** and *italic* and `inline()` text.",
        "",
        "- first",
        "- second",
        "",
        "```py",
        "print('fenced')",
        "```",
        "",
        "[docs](https://example.test/docs)",
      ].join("\n"),
    );
    expect(container.querySelector("h1")?.textContent).toBe("Title");
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("em")?.textContent).toBe("italic");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("p > code")?.textContent).toBe("inline()");
    expect(container.querySelector("pre code")?.textContent).toContain("print('fenced')");
    const link = container.querySelector("a");
    expect(link?.getAttribute("href")).toBe("https://example.test/docs");
    expect(link?.getAttribute("target")).toBe("_blank");
    expect(link?.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it("never passes raw HTML through: script/style/event-handler markup renders inert", async () => {
    const container = await mountMarkdown(
      'before <script>window.__markdown_pwned = true;</script> <img src="x" onerror="window.__markdown_pwned = true"> after',
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect((globalThis as unknown as Record<string, unknown>).__markdown_pwned).toBeUndefined();
    // Surrounding prose still renders.
    expect(container.textContent).toContain("before");
    expect(container.textContent).toContain("after");
  });

  it("strips javascript: URLs from markdown links", async () => {
    const container = await mountMarkdown("[click me](javascript:window.__markdown_pwned=true)");
    const link = container.querySelector("a");
    expect(link).not.toBeNull();
    expect(link?.getAttribute("href") ?? "").not.toContain("javascript:");
  });

  it("strips javascript: and data: URLs from markdown images", async () => {
    const container = await mountMarkdown("![x](javascript:window.__markdown_pwned=true)");
    const image = container.querySelector("img");
    expect(image?.getAttribute("src") ?? "").not.toContain("javascript:");
  });
});
