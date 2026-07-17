import { describe, expect, it } from "vitest";

import { formatBytes, formatBytesCompact, formatDurationMs } from "./format";

describe("formatBytes", () => {
  it("returns dashed placeholder for empty input", () => {
    expect(formatBytes(undefined)).toBe("—");
    expect(formatBytes("not-a-number")).toBe("—");
  });
  it("selects the largest unit that keeps the count above one", () => {
    expect(formatBytes("512")).toBe("512 B");
    expect(formatBytes("2048")).toBe("2 KiB");
    expect(formatBytes(String(3 * 1024 * 1024))).toBe("3 MiB");
  });
});

describe("formatBytesCompact", () => {
  it("returns dashed placeholder for missing or malformed input", () => {
    expect(formatBytesCompact(undefined)).toBe("—");
    expect(formatBytesCompact(null)).toBe("—");
    expect(formatBytesCompact("")).toBe("—");
    expect(formatBytesCompact("noise")).toBe("—");
    expect(formatBytesCompact("-1")).toBe("—");
  });

  it("keeps small counts compact", () => {
    expect(formatBytesCompact("0")).toBe("0B");
    expect(formatBytesCompact("128")).toBe("128B");
    expect(formatBytesCompact("1023")).toBe("1023B");
  });

  it("shifts to K / M / G at 1024 boundaries with one decimal", () => {
    expect(formatBytesCompact("1024")).toBe("1.0K");
    expect(formatBytesCompact("4301")).toBe("4.2K");
    expect(formatBytesCompact("1153434")).toBe("1.1M");
    expect(formatBytesCompact(String(102 * 1024))).toBe("102K");
    expect(formatBytesCompact(String(3 * 1024 * 1024 * 1024))).toBe("3.0G");
  });
});

describe("formatDurationMs", () => {
  it("returns dashed placeholder for missing input", () => {
    expect(formatDurationMs(undefined)).toBe("—");
    expect(formatDurationMs(null)).toBe("—");
    expect(formatDurationMs(-4)).toBe("—");
    expect(formatDurationMs(Number.NaN)).toBe("—");
  });

  it("keeps sub-second durations in ms", () => {
    expect(formatDurationMs(12)).toBe("12ms");
    expect(formatDurationMs(940)).toBe("940ms");
  });

  it("crosses to seconds and minutes with a decimal below ten", () => {
    expect(formatDurationMs(1400)).toBe("1.4s");
    expect(formatDurationMs(45_000)).toBe("45s");
    expect(formatDurationMs(180_000)).toBe("3.0m");
    expect(formatDurationMs(3_600_000)).toBe("1.0h");
  });
});
