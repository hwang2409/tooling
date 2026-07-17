import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");

function token(name: string): string {
  const value = css.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`))?.[1];
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

describe("console token contrast", () => {
  it("keeps normal telemetry text WCAG AA on both instrument surfaces", () => {
    const foregrounds = ["ink", "ink-soft", "muted", "faint", "amber", "teal", "danger"];
    const backgrounds = ["paper", "paper-deep"];
    for (const foreground of foregrounds) {
      for (const background of backgrounds) {
        expect(contrast(token(foreground), token(background)), `${foreground} on ${background}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps signal colors legible inside their tinted status surfaces", () => {
    expect(contrast(token("amber"), token("amber-soft"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast(token("teal"), token("teal-soft"))).toBeGreaterThanOrEqual(4.5);
  });
});
