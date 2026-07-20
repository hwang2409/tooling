import { Buffer } from "node:buffer";
import { describe, expect, it } from "vitest";

import { MAX_INGEST_BODY_PREFIX_BYTES } from "../../contracts/limits";
import { bodyText, decodeBase64Bounded, decodeUtf8, DEFAULT_BODY_LIMIT } from "./decoders";

const encoded = (value: string) => Buffer.from(value, "utf8").toString("base64");

describe("decodeBase64Bounded", () => {
  it("decodes valid base64 without allocating beyond the requested cap", () => {
    const result = decodeBase64Bounded(encoded("0123456789"), 4);
    expect(Array.from(result.bytes)).toEqual([48, 49, 50, 51]);
    expect(result.truncated).toBe(true);
    expect(result.invalid).toBe(false);
  });

  it("decodes multi-megabyte payloads whole up to the wire ceiling", () => {
    const halfMegabyte = Buffer.alloc(512 * 1024, 0x78).toString("base64");
    const result = decodeBase64Bounded(halfMegabyte);
    expect(result.bytes.byteLength).toBe(512 * 1024);
    expect(result.truncated).toBe(false);
    expect(DEFAULT_BODY_LIMIT).toBe(MAX_INGEST_BODY_PREFIX_BYTES);
  });

  it("flags malformed base64 instead of decoding garbage", () => {
    const result = decodeBase64Bounded("not-base64!!");
    expect(result.invalid).toBe(true);
    expect(result.bytes.byteLength).toBe(0);
  });

  it("rejects a negative or unsafe limit", () => {
    expect(() => decodeBase64Bounded(encoded("x"), -1)).toThrow(RangeError);
    expect(() => decodeBase64Bounded(encoded("x"), Number.MAX_SAFE_INTEGER + 1)).toThrow(RangeError);
  });
});

describe("decodeUtf8", () => {
  it("decodes valid UTF-8 text", () => {
    expect(decodeUtf8(new TextEncoder().encode("héllo"))).toEqual({ text: "héllo", invalid: false });
  });

  it("falls back to a hex dump on invalid UTF-8", () => {
    const result = decodeUtf8(new Uint8Array([0xff, 0xfe, 0x41]));
    expect(result.invalid).toBe(true);
    expect(result.text).toContain("|");
    expect(result.text).toContain("ff fe 41");
  });
});

describe("bodyText", () => {
  it("treats missing and empty bodies as absent", () => {
    expect(bodyText(undefined)).toEqual({ kind: "absent" });
    expect(bodyText({ state: "missing" })).toEqual({ kind: "absent" });
    expect(bodyText({ state: "empty", size_bytes: "0" })).toEqual({ kind: "absent" });
  });

  it("treats a zero-byte truncated descriptor (projection stripped shape) as absent, never empty text", () => {
    const stripped = bodyText({ state: "truncated", size_bytes: "100", captured_bytes: "0", encoding: "base64", data: "" });
    expect(stripped).toEqual({ kind: "absent" });
    expect(bodyText({ state: "captured", size_bytes: "0", encoding: "base64", data: "" })).toEqual({ kind: "absent" });
  });

  it("decodes a captured body to text with its byte length", () => {
    const text = JSON.stringify({ hello: "world" });
    const result = bodyText({ state: "captured", size_bytes: String(text.length), encoding: "base64", data: encoded(text) });
    expect(result).toEqual({ kind: "text", text, byteLength: text.length });
  });

  it("decodes a truncated body prefix as text", () => {
    const result = bodyText({ state: "truncated", size_bytes: "100", captured_bytes: "7", encoding: "base64", data: encoded("{\"trunc") });
    expect(result).toEqual({ kind: "text", text: "{\"trunc", byteLength: 7 });
  });

  it("labels redacted and undecodable bodies instead of hiding them", () => {
    expect(bodyText({ state: "redacted" })).toEqual({ kind: "text", text: "(body redacted)", byteLength: 0 });
    expect(bodyText({ state: "captured", size_bytes: "4", encoding: "base64", data: "!!!" })).toEqual({
      kind: "text",
      text: "(invalid base64 body)",
      byteLength: 0,
    });
  });
});
