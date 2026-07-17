import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");

function themeBlock(selector: string): string {
  const escaped = selector.replace(/[[\]"]/g, (character) => `\\${character}`);
  const match = css.match(new RegExp(`${escaped}\\s*\\{([\\s\\S]*?)\\}`));
  if (!match) throw new Error(`missing selector ${selector}`);
  return match[1];
}

function token(block: string, name: string): string {
  const value = block.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`))?.[1];
  if (!value) throw new Error(`missing token --${name}`);
  return value;
}

function luminance(hex: string): number {
  const channels = [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255);
  const linear = channels.map((channel) => channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

function contrast(foreground: string, background: string): number {
  const light = Math.max(luminance(foreground), luminance(background));
  const dark = Math.min(luminance(foreground), luminance(background));
  return (light + 0.05) / (dark + 0.05);
}

const foregrounds = ["ink", "ink-soft", "muted", "faint", "amber", "teal", "danger"] as const;
const backgrounds = ["paper", "paper-deep"] as const;
const themes: Array<{ label: string; selector: string }> = [
  { label: "light", selector: ":root" },
  { label: "dark", selector: '[data-theme="dark"]' },
];

describe("console token contrast", () => {
  for (const theme of themes) {
    const block = themeBlock(theme.selector);
    it(`keeps normal telemetry text WCAG AA on both instrument surfaces (${theme.label})`, () => {
      for (const foreground of foregrounds) {
        for (const background of backgrounds) {
          expect(contrast(token(block, foreground), token(block, background)), `${foreground} on ${background} (${theme.label})`).toBeGreaterThanOrEqual(4.5);
        }
      }
    });

    it(`keeps signal colors legible inside their tinted status surfaces (${theme.label})`, () => {
      expect(contrast(token(block, "amber"), token(block, "amber-soft")), `amber-soft (${theme.label})`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(token(block, "teal"), token(block, "teal-soft")), `teal-soft (${theme.label})`).toBeGreaterThanOrEqual(4.5);
    });
  }

  it("collapses the accent tokens onto the neutral ramp (light theme)", () => {
    const block = themeBlock(":root");
    expect(token(block, "teal")).toBe(token(block, "ink"));
    expect(token(block, "amber")).toBe(token(block, "muted"));
  });
});
