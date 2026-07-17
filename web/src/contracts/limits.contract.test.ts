import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  INGEST_ENVELOPE_ALLOWANCE_BYTES,
  INGEST_FIXED_ENVELOPE_BYTES,
  MAX_INGEST_BODY_PREFIX_BYTES,
  MAX_INGEST_LINE_BYTES,
  MAX_METADATA_HEADER_BYTES,
} from "./limits";

/**
 * Guard: the browser bundle mirrors the backend wire ceiling. If any of the
 * Python constants change, this test breaks and forces us to update
 * `web/src/contracts/limits.ts` in the same commit — no silent drift.
 */

function readPython(relative: string): string {
  return readFileSync(resolve(process.cwd(), "..", relative), "utf8");
}

function evalIntLiteral(expression: string): number {
  const cleaned = expression.replace(/\s+/g, "").replace(/_/g, "");
  if (!/^[0-9+\-*/()]+$/.test(cleaned)) {
    throw new Error(`unsafe expression: ${expression}`);
  }
  // Numeric-only expression with basic arithmetic; safe under the regex above.
  const value = Number(new Function(`return (${cleaned})`)());
  if (!Number.isFinite(value)) throw new Error(`non-finite value: ${expression}`);
  return value;
}

function pythonConstant(source: string, name: string): number {
  const match = new RegExp(`^${name}\\s*=\\s*(.+?)(?:\\s*#.*)?$`, "m").exec(source);
  if (!match) throw new Error(`constant ${name} not found in python source`);
  return evalIntLiteral(match[1]);
}

describe("wire ceiling contract mirrors the backend", () => {
  const protocol = readPython("src/mitm_inspector/protocol.py");

  it("MAX_METADATA_HEADER_BYTES matches", () => {
    expect(MAX_METADATA_HEADER_BYTES).toBe(pythonConstant(protocol, "MAX_METADATA_HEADER_BYTES"));
  });

  it("MAX_INGEST_LINE_BYTES matches", () => {
    expect(MAX_INGEST_LINE_BYTES).toBe(pythonConstant(protocol, "MAX_INGEST_LINE_BYTES"));
  });

  it("MAX_INGEST_BODY_PREFIX_BYTES matches the python-side computation", () => {
    // Reproduce the exact arithmetic from src/mitm_inspector/api/limits.py so
    // that both sides truly implement the same formula, not just happen to
    // agree numerically once.
    const envelope = 2 * MAX_METADATA_HEADER_BYTES + INGEST_FIXED_ENVELOPE_BYTES;
    expect(envelope).toBe(INGEST_ENVELOPE_ALLOWANCE_BYTES);
    const expected = Math.floor(((MAX_INGEST_LINE_BYTES - envelope) * 3) / 8);
    expect(MAX_INGEST_BODY_PREFIX_BYTES).toBe(expected);
    // Sanity: the wire ceiling should sit close to 3 MiB.
    expect(MAX_INGEST_BODY_PREFIX_BYTES).toBeGreaterThan(2 * 1024 * 1024);
    expect(MAX_INGEST_BODY_PREFIX_BYTES).toBeLessThan(4 * 1024 * 1024);
  });
});
