import { MAX_INGEST_BODY_PREFIX_BYTES } from "../../contracts/limits";
import type { InspectableBody } from "./models";

/**
 * Decode ceiling equals the backend wire ceiling — the biggest body the
 * ingest listener will ever accept, so bodies are always surfaced whole.
 */
export const DEFAULT_BODY_LIMIT = MAX_INGEST_BODY_PREFIX_BYTES;

export interface Base64DecodeResult {
  bytes: Uint8Array;
  invalid: boolean;
  truncated: boolean;
}

export interface TextDecodeResult {
  text: string;
  invalid: boolean;
}

const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function toHex(bytes: Uint8Array): string {
  const lines: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 16) {
    const row = bytes.subarray(offset, offset + 16);
    const hex = Array.from(row, (byte) => byte.toString(16).padStart(2, "0")).join(" ");
    const padded = hex.padEnd(47, " ");
    const ascii = Array.from(row, (byte) => byte >= 0x20 && byte <= 0x7e ? String.fromCharCode(byte) : ".").join("");
    lines.push(`${offset.toString(16).padStart(8, "0")}  ${padded}  |${ascii}|`);
  }
  return lines.join("\n");
}

export function decodeBase64Bounded(value: string, limit = DEFAULT_BODY_LIMIT): Base64DecodeResult {
  if (!Number.isSafeInteger(limit) || limit < 0) throw new RangeError("limit must be a non-negative safe integer");
  const boundedLimit = limit;
  const inputLimit = Math.ceil(boundedLimit / 3) * 4;
  const inputWasTruncated = value.length > inputLimit;
  const boundedValue = inputWasTruncated ? value.slice(0, inputLimit) : value;
  if (boundedValue.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(boundedValue)) {
    return { bytes: new Uint8Array(), invalid: true, truncated: false };
  }

  const decodedLength = boundedValue.length / 4 * 3 - (boundedValue.endsWith("==") ? 2 : boundedValue.endsWith("=") ? 1 : 0);
  const output = new Uint8Array(Math.min(decodedLength, boundedLimit));
  let write = 0;
  for (let index = 0; index < boundedValue.length; index += 4) {
    const a = BASE64_ALPHABET.indexOf(boundedValue[index]);
    const b = BASE64_ALPHABET.indexOf(boundedValue[index + 1]);
    const c = boundedValue[index + 2] === "=" ? 0 : BASE64_ALPHABET.indexOf(boundedValue[index + 2]);
    const d = boundedValue[index + 3] === "=" ? 0 : BASE64_ALPHABET.indexOf(boundedValue[index + 3]);
    const block = (a << 18) | (b << 12) | (c << 6) | d;
    if (write < output.length) output[write++] = (block >> 16) & 0xff;
    if (boundedValue[index + 2] !== "=" && write < output.length) output[write++] = (block >> 8) & 0xff;
    if (boundedValue[index + 3] !== "=" && write < output.length) output[write++] = block & 0xff;
  }
  return { bytes: output, invalid: false, truncated: inputWasTruncated || decodedLength > boundedLimit };
}

export function decodeUtf8(bytes: Uint8Array): TextDecodeResult {
  try {
    return { text: new TextDecoder("utf-8", { fatal: true }).decode(bytes), invalid: false };
  } catch {
    return { text: toHex(bytes), invalid: true };
  }
}

export type BodyText =
  | { kind: "absent" }
  | { kind: "text"; text: string; byteLength: number };

/**
 * Reduce a captured body descriptor to displayable text. Absent covers
 * missing and zero-byte bodies; everything else decodes to text (invalid
 * UTF-8 falls back to a hex dump, which still renders as plain text).
 */
export function bodyText(body: InspectableBody | undefined): BodyText {
  if (body === undefined || body.state === "missing" || body.state === "empty") return { kind: "absent" };
  if (body.state === "redacted") return { kind: "text", text: "(body redacted)", byteLength: 0 };
  const base64 = decodeBase64Bounded(body.data);
  if (base64.invalid) return { kind: "text", text: "(invalid base64 body)", byteLength: 0 };
  const utf8 = decodeUtf8(base64.bytes);
  return { kind: "text", text: utf8.text, byteLength: base64.bytes.length };
}
