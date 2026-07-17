import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const HEX_PATTERN = /#[0-9a-fA-F]{6}\b/g;

function readFile(relative: string): string {
  return readFileSync(resolve(process.cwd(), relative), "utf8");
}

function hexesIn(content: string): string[] {
  return (content.match(HEX_PATTERN) ?? []).map((value) => value.toLowerCase());
}

const tokenCss = readFile("src/styles/tokens.css");
const tokenHexes = new Set(hexesIn(tokenCss));

const restrictedSources: Array<{ label: string; path: string }> = [
  { label: "shell.css", path: "src/styles/shell.css" },
  { label: "flows.css", path: "src/styles/flows.css" },
  { label: "inspector.css", path: "src/styles/inspector.css" },
  { label: "index.html", path: "index.html" },
];

describe("palette discipline", () => {
  for (const source of restrictedSources) {
    it(`${source.label} references only palette tokens (no stray chroma)`, () => {
      const stray = hexesIn(readFile(source.path)).filter((hex) => !tokenHexes.has(hex));
      expect(stray, `stray hex colours in ${source.label}`).toEqual([]);
    });
  }

  it("does not paint the redaction badge with the danger red token", () => {
    const inspector = readFile("src/styles/inspector.css");
    const rule = /\.inspector-body-badge\.is-redacted\s*\{[^}]*\}/.exec(inspector)?.[0] ?? "";
    expect(rule).not.toContain("--inspector-red");
    expect(rule).not.toContain("--danger");
  });
});
