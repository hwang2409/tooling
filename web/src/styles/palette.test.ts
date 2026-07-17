import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const HEX_PATTERN = /#[0-9a-fA-F]{3,8}\b/g;
const RGB_PATTERN = /(?:rgb|hsl)a?\s*\(/gi;

/**
 * Explicit allowlist of approved colour tokens. Anything outside this set —
 * even a value newly declared inside tokens.css — must be flagged so the
 * b/w-plus-danger-only palette discipline stays enforceable long-term.
 * Update this list DELIBERATELY when a new token is introduced.
 */
const ALLOWED_HEXES = new Set<string>([
  // Light-theme neutrals + danger.
  "#111111",
  "#3a3a3a",
  "#565656",
  "#6b6b6b",
  "#ffffff",
  "#fafafa",
  "#f4f4f4",
  "#e5e5e5",
  "#cccccc",
  "#efefef",
  "#f0f0f0",
  "#b3261e",
  "#fce8e6",
  // Dark-theme neutrals + danger.
  "#f0f0f0",
  "#d0d0d0",
  "#b0b0b0",
  "#a5a5a5",
  "#141414",
  "#1c1c1c",
  "#0e0e0e",
  "#2a2a2a",
  "#3d3d3d",
  "#262626",
  "#f18b83",
  "#3e2323",
  "#f5f5f5",
].map((hex) => hex.toLowerCase()));

/**
 * Explicit banlist of previously-shipped chroma tokens. Even if a future
 * mistake somehow slips into the allowlist, hitting one of these hexes must
 * still fail the discipline check.
 */
const BANNED_HEXES = new Set<string>([
  "#0f6e66", // f4-era teal accent
  "#8a5200", // f4-era amber accent
  "#e6f2f0", // f4-era teal-soft
  "#fff2dc", // f4-era amber-soft
  "#101317", // legacy theme-color navy
].map((hex) => hex.toLowerCase()));

const SCANNED_SOURCES: Array<{ label: string; path: string }> = [
  { label: "tokens.css", path: "src/styles/tokens.css" },
  { label: "shell.css", path: "src/styles/shell.css" },
  { label: "flows.css", path: "src/styles/flows.css" },
  { label: "inspector.css", path: "src/styles/inspector.css" },
  { label: "index.html", path: "index.html" },
];

function readFile(relative: string): string {
  return readFileSync(resolve(process.cwd(), relative), "utf8");
}

function normalisedHexesIn(content: string): string[] {
  return (content.match(HEX_PATTERN) ?? []).map((value) => value.toLowerCase());
}

describe("palette discipline", () => {
  for (const source of SCANNED_SOURCES) {
    it(`${source.label} references only the explicit allowlist of neutrals + danger`, () => {
      const content = readFile(source.path);
      const stray = normalisedHexesIn(content).filter((hex) => !ALLOWED_HEXES.has(hex));
      expect(stray, `stray hex colours in ${source.label}`).toEqual([]);
      const banned = normalisedHexesIn(content).filter((hex) => BANNED_HEXES.has(hex));
      expect(banned, `banned hex colours in ${source.label}`).toEqual([]);
      const rgbLikeMatches = content.match(RGB_PATTERN) ?? [];
      expect(rgbLikeMatches, `rgb()/hsl() are not permitted in ${source.label}`).toEqual([]);
    });
  }

  it("does not paint the redaction badge with the danger red token", () => {
    const inspector = readFile("src/styles/inspector.css");
    const rule = /\.inspector-body-badge\.is-redacted\s*\{[^}]*\}/.exec(inspector)?.[0] ?? "";
    expect(rule).not.toContain("--inspector-red");
    expect(rule).not.toContain("--danger");
  });

  it("keeps the F4-era chroma tokens out of the allowlist", () => {
    for (const banned of BANNED_HEXES) {
      expect(ALLOWED_HEXES.has(banned), `banned ${banned} must not be allowlisted`).toBe(false);
    }
  });
});
